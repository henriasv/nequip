# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.

import torch

from nequip.data import AtomicDataDict
from ._ghost_exchange_base import GhostExchangeModule


# Native ``pair_nequip`` per-layer ghost exchange (multi-GPU Route 2b / "Design A").
#
# The ML-IAP ghost exchange (``_ghost_exchange_lmp_mliap.py``) calls Python methods on a
# ``lmp_data`` object and therefore only works under eager execution -- its own source notes
# it "can't use custom ops ... because of complications with the ``lmp_data`` type". That is
# exactly what cannot survive AOTInductor (the ``.pt2`` is a standalone C++ artifact with no
# Python interpreter).
#
# This path is designed for the AOTInductor ``pair_nequip`` target instead: the exchange is a
# registered custom operator (``nequip_lammps::ghost_exchange``) so it can be traced into, and
# called from, the compiled artifact -- exactly as OpenEquivariance's ``libtorch_tp_jit`` op
# is. Crucially the op carries NO LAMMPS handle as an argument (that was the obstacle for the
# ML-IAP op). Instead the native LAMMPS pair style (``pair_nequip_allegro``) installs the real
# implementation -- a halo ``Comm::forward_comm`` / ``reverse_comm`` of per-node feature
# vectors -- via a C++ registration that reaches the active ``Pair`` through a thread-local
# pointer. In the absence of a LAMMPS comm context (single rank, ASE, export tracing, eager
# development) the op is the identity: on a single rank there are no inter-rank ghost atoms to
# fill, so ``ntotal == nlocal`` and the padded ghost block is empty.


def _register_ghost_exchange_ops():
    """Register the ``nequip_lammps`` exchange operators (identity defaults).

    The native pair style (``pair_nequip_allegro``) overrides these in-process with the real
    ``Comm::forward_comm`` / ``reverse_comm`` halo implementations; absent a LAMMPS comm
    context they are the identity.
    """

    @torch.library.custom_op("nequip_lammps::ghost_exchange", mutates_args=())
    def ghost_exchange(node_features: torch.Tensor) -> torch.Tensor:
        # Forward per-layer ghost-feature halo exchange; default identity implementation.
        return node_features.clone()

    @ghost_exchange.register_fake
    def _(node_features: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(node_features)

    @torch.library.custom_op("nequip_lammps::ghost_exchange_reverse", mutates_args=())
    def ghost_exchange_reverse(grad_features: torch.Tensor) -> torch.Tensor:
        # Reverse (transpose) of the exchange, used in the force/backward pass; default identity.
        return grad_features.clone()

    @ghost_exchange_reverse.register_fake
    def _(grad_features: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(grad_features)

    def _setup_context(ctx, inputs, output):
        # The exchange transpose is shape-preserving and needs no saved state.
        pass

    def _backward(ctx, grad):
        return torch.ops.nequip_lammps.ghost_exchange_reverse(grad)

    ghost_exchange.register_autograd(_backward, setup_context=_setup_context)

    # === M10 async-overlap split of the forward halo ===
    # `ghost_exchange` does one blocking `Comm::forward_comm`. To hide that latency behind the
    # owned-source TP-scatter (which needs only owned features, available before the halo), the
    # async multi-rank target splits it into two registered ops the native pair style implements:
    #   * `ghost_exchange_start(full)` -> records a "owned features ready" marker on the model
    #     stream (before the owned-edge TP is launched) and returns `full` unchanged (ghost rows
    #     still zero); no comm. Identity by default / single-rank.
    #   * `ghost_exchange_finish(full)` -> runs the halo on the Kokkos comm stream (overlapping the
    #     owned-edge TP still in flight on the model stream) and returns `full` with ghost rows
    #     filled. Identity by default / single-rank.
    # Both are functional (no in-place mutation), so AOT export + the in-model force/backward pass
    # behave exactly like the single-op path. `finish`'s backward is the same reverse halo as
    # `ghost_exchange`'s; `start`'s backward is the identity.
    @torch.library.custom_op("nequip_lammps::ghost_exchange_start", mutates_args=())
    def ghost_exchange_start(node_features: torch.Tensor) -> torch.Tensor:
        return node_features.clone()

    @ghost_exchange_start.register_fake
    def _(node_features: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(node_features)

    ghost_exchange_start.register_autograd(
        lambda ctx, grad: grad, setup_context=_setup_context
    )

    @torch.library.custom_op("nequip_lammps::ghost_exchange_finish", mutates_args=())
    def ghost_exchange_finish(node_features: torch.Tensor) -> torch.Tensor:
        return node_features.clone()

    @ghost_exchange_finish.register_fake
    def _(node_features: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(node_features)

    ghost_exchange_finish.register_autograd(_backward, setup_context=_setup_context)


# Register once per process. Under ``torch.package`` framework code is imported inside an
# isolated namespace, so this module can legitimately be imported more than once (the fork copy
# used by the compile tool AND the bundled copy carried by a packaged model). The operators are
# global, so guard against the resulting double-registration.
try:
    _register_ghost_exchange_ops()
except RuntimeError:
    pass


class PairNequIPGhostExchangeModule(GhostExchangeModule):
    """Native ``pair_nequip`` per-layer ghost exchange via ``nequip_lammps::ghost_exchange``.

    Mirrors :class:`LAMMPSMLIAPGhostExchangeModule` but (a) takes the owned count as a *backed*
    tensor dimension (``num_local_nodes_marker.shape[0]`` / the owned feature block) rather than
    a Python ``lmp_data`` object, and (b) performs the exchange through a registered custom
    operator so it survives AOTInductor compilation. The owned-only `[nlocal]` features are
    re-expanded to `[ntotal]` with ``slice_scatter`` before the exchange (see ``forward``).
    """

    def forward(
        self, data: AtomicDataDict.Type, ghost_included: bool = False
    ) -> AtomicDataDict.Type:
        # Truncate-to-nlocal: the incoming features span only the `nlocal` owned nodes (every
        # per-node op upstream was sliced to owned atoms). Re-expand them to `ntotal` so the
        # native pair style's `forward_comm` can fill the ghost rows from their owners on
        # neighboring ranks, ready for the TP-scatter that consumes the feature halo. Absent a
        # LAMMPS comm context (single rank / ASE / export tracing) the op is the identity and
        # the ghost rows stay zero — and since the single-rank case has `nghost == 0` the
        # expansion is itself a no-op there.
        node_features = data[self.field]
        # `ntotal` from a never-truncated full-size node tensor (positions persist for forces).
        ntotal = data[AtomicDataDict.POSITIONS_KEY].shape[0]
        if ghost_included:
            # already `ntotal`-wide: take the owned block (marker carries the backed `nlocal`).
            # Literal key (not AtomicDataDict.NUM_LOCAL_NODES_MARKER_KEY): injection-safe when
            # repacked into an older model whose bundled AtomicDataDict lacks the attribute.
            nlocal = data["num_local_nodes_marker"].shape[0]
            owned = node_features[:nlocal]
        else:
            owned = node_features
        # Expand owned `[nlocal, F]` -> `[ntotal, F]` (ghost rows zero) via `slice_scatter`, NOT
        # `cat` with a `(ntotal - nlocal, F)` zeros block. A `cat`/`zeros((ntotal-nlocal,))`
        # triggers the 0/1 size specialization in `torch.export` and bakes a `nghost >= 1` (or
        # `nghost <= 0`) guard, which then fails for either the single-rank correctness gate
        # (`nghost == 0`) or multi-rank runs (`nghost > 0`). Scattering into an `ntotal`-sized
        # buffer instead only defers a `nlocal <= ntotal` assert, satisfiable at both ends.
        full = owned.new_zeros((ntotal,) + tuple(owned.shape[1:]))
        full = torch.slice_scatter(full, owned, dim=0, start=0, end=owned.shape[0])
        data[self.field] = torch.ops.nequip_lammps.ghost_exchange(full)
        return data

    def forward_start(
        self, data: AtomicDataDict.Type, ghost_included: bool = False
    ) -> AtomicDataDict.Type:
        """M10 async forward halo, phase 1: expand owned features to ``ntotal`` and record the
        "owned features ready" marker (no comm yet).

        Mirrors the expand in :meth:`forward` but calls ``ghost_exchange_start`` instead of the
        blocking ``ghost_exchange``: the ghost rows stay zero and the native pair style records a
        stream event so :meth:`forward_finish` can begin the halo without waiting for the
        owned-source TP-scatter that runs in between (the overlap). See ``interaction_block``.
        """
        node_features = data[self.field]
        ntotal = data[AtomicDataDict.POSITIONS_KEY].shape[0]
        if ghost_included:
            nlocal = data["num_local_nodes_marker"].shape[0]
            owned = node_features[:nlocal]
        else:
            owned = node_features
        full = owned.new_zeros((ntotal,) + tuple(owned.shape[1:]))
        full = torch.slice_scatter(full, owned, dim=0, start=0, end=owned.shape[0])
        data[self.field] = torch.ops.nequip_lammps.ghost_exchange_start(full)
        return data

    def forward_finish(
        self, data: AtomicDataDict.Type, ghost_included: bool = False
    ) -> AtomicDataDict.Type:
        """M10 async forward halo, phase 2: complete the halo (fill ghost rows).

        ``data[field]`` is the ``ntotal``-wide tensor from :meth:`forward_start` (owned rows valid,
        ghost rows zero). ``ghost_exchange_finish`` runs the per-layer ``Comm::forward_comm`` on
        the Kokkos comm stream and returns it with the ghost rows filled, ready for the
        ghost-source TP-scatter.
        """
        data[self.field] = torch.ops.nequip_lammps.ghost_exchange_finish(data[self.field])
        return data
