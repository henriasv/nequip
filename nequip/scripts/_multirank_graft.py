# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
"""Version-robust multirank re-bundle: derive the truncate-to-nlocal / ghost-exchange overlay from
a packaged model's OWN bundled ``nn`` source, instead of injecting a fixed model-era overlay.

Why derive-from-own-source: a ``.nequip.zip`` is unpickled (its ``__init__`` never re-runs), so the
re-bundled source must be compatible with the model's *frozen* instances. Injecting a different-era
``nn`` is unbounded whack-a-mole — a newer source references attributes the old instances lack (the
norm submodule ``avg_num_neighbors_norm``; the ZBL ``cutoff``; ...). Starting from the package's OWN
bundled source and applying only the *bounded, additive* multirank feature-set keeps the model's own
norm / ZBL / imports and is therefore attribute-safe AND import-safe by construction.

The grafts (validated end-to-end: a fine-tuned OAM-M packaged by a mliap-only fork era compiled to a
correct ``pair_nequip_multirank`` ``.pt2`` == single-rank ground truth):
  1. ``interaction_block``: read ``nlocal`` from the ``num_local_nodes_marker`` input (pair_nequip
     path; the mliap-only era only had ``_get_mliap_num_local``).
  2. ``graph_model.forward``: pass the marker through (the saved ``model_input_fields`` whitelist
     would otherwise drop it -> no truncation -> the energy sum includes ghosts).
  3. ``_ghost_exchange_base``: add the ``enable_PairNequIPGhostExchange`` modifier; plus ship the real
     ``_ghost_exchange_pair.py`` (the era predated it) so the no-op exchange becomes the real one.
  4. ``graph_model.metadata``: collect ``_nequip_custom_ops_libs`` so the OEQ op ``.so`` is embedded
     in the ``.pt2`` (Design B); else the run dies "no schema for libtorch_tp_jit::jit_conv_forward".
  5. ``pair_potential`` (ZBL): scatter over ``ntotal`` then truncate to ``nlocal`` (``dim_size=nlocal``
     is an out-of-bounds write on ghost-centered edges).
  6. ``atomwise`` (``AtomwiseReduce``): align ``BATCH_KEY`` to the truncated field so the per-frame
     energy sum is owned-only.

Each graft is idempotent (skipped if already present) and graceful (an absent anchor logs a warning
and leaves the file unchanged — the pair style's runtime multirank-vs-single-rank guard is the
backstop). Anchors are stable code patterns shared across the eras we re-bundle (pre-marker bundles).
"""
import logging
import pathlib

logger = logging.getLogger(__name__)


# --- per-file grafts: (idempotency_substring, anchor, replacement) --------------------------------
# An anchor absent from a file is logged and skipped (graceful). `replacement` reproduces the
# validated overlay's code for that graft.

_GRAFT_IB = (
    "num_local_nodes_marker",
    """        if AtomicDataDict.LMP_MLIAP_DATA_KEY in data:
            num_local_nodes = self._get_mliap_num_local(data)
        else:
            num_local_nodes = AtomicDataDict.num_nodes(data)""",
    """        if AtomicDataDict.LMP_MLIAP_DATA_KEY in data:
            num_local_nodes = self._get_mliap_num_local(data)
        elif "num_local_nodes_marker" in data:
            # Native multi-rank pair_nequip (truncate-to-nlocal): owned count = size-0 of the marker
            # input (a backed dynamic dim, never via .item(), so it AOT-exports cleanly). Literal key
            # keeps this injection-safe for this older model's bundled AtomicDataDict.
            num_local_nodes = data["num_local_nodes_marker"].shape[0]
        else:
            num_local_nodes = AtomicDataDict.num_nodes(data)""",
)

_GRAFT_GM_FORWARD = (
    'new_data["num_local_nodes_marker"]',
    """        for k in self.model_input_fields:
            if k in data:
                new_data[k] = data[k]
        return self.model(new_data)""",
    """        for k in self.model_input_fields:
            if k in data:
                new_data[k] = data[k]
        # Pass the truncate-to-nlocal marker through even when absent from the saved
        # model_input_fields whitelist (a loaded model would otherwise filter it out and every layer
        # would fall back to the all-ntotal no-truncation path). Literal key = injection-safe.
        if "num_local_nodes_marker" in data:
            new_data["num_local_nodes_marker"] = data["num_local_nodes_marker"]
        return self.model(new_data)""",
)

_GRAFT_GM_METADATA = (
    "nequip_custom_ops_libs",
    """            out[R_MAX_KEY] = str(max(cutoff_values))

        return out""",
    """            out[R_MAX_KEY] = str(max(cutoff_values))

        # Collect custom-op libs (e.g. OpenEquivariance's libtorch_tp_jit.so) so the compile tool
        # embeds them into the self-contained .pt2 (Design B) and the pair style dlopens them at
        # pair_coeff. The model's own OEQ modules already carry `_nequip_custom_ops_libs`; this older
        # era's graph_model just predated the collection step. Literal key = injection-safe.
        _custom_ops_libs = set()
        for _m in self.model.modules():
            _custom_ops_libs.update(getattr(_m, "_nequip_custom_ops_libs", ()))
        if _custom_ops_libs:
            out["nequip_custom_ops_libs"] = " ".join(sorted(_custom_ops_libs))

        return out""",
)

_GRAFT_ATOMWISE = (
    "[: field.size(0)]",
    """        if AtomicDataDict.BATCH_KEY in data:
            result = scatter(
                field,
                data[AtomicDataDict.BATCH_KEY],
                dim=0,""",
    """        if AtomicDataDict.BATCH_KEY in data:
            # Truncate-to-nlocal: `field` spans only the nlocal owned nodes while BATCH_KEY spans
            # ntotal; align it so the per-frame energy sum is owned-only. No-op when field is ntotal.
            _batch = data[AtomicDataDict.BATCH_KEY][: field.size(0)]
            result = scatter(
                field,
                _batch,
                dim=0,""",
)

_GRAFT_ZBL_DIMSIZE = (
    "POSITIONS_KEY].shape[0]  # ntotal (truncate-to-nlocal ZBL)",
    """        if AtomicDataDict.PER_ATOM_ENERGY_KEY in data:
            num_nodes = data[AtomicDataDict.PER_ATOM_ENERGY_KEY].size(0)
        else:
            num_nodes = AtomicDataDict.num_nodes(data)""",
    """        if AtomicDataDict.PER_ATOM_ENERGY_KEY in data:
            # Truncate-to-nlocal: scatter ZBL over ALL ntotal nodes (positions never truncated) then
            # truncate before adding; dim_size=nlocal is an OOB write on ghost-centered edges.
            num_nodes = data[AtomicDataDict.POSITIONS_KEY].shape[0]  # ntotal (truncate-to-nlocal ZBL)
        else:
            num_nodes = AtomicDataDict.num_nodes(data)""",
)

_GRAFT_ZBL_ADD = (
    "atomic_eng[:_nlocal]",
    # Anchor on the ZBL scatter (zbl_edge_eng / edge_center are unique to the ZBL forward) so the
    # truncate-before-add lands on the RIGHT forward -- the `if PER_ATOM_ENERGY: atomic_eng + ...`
    # add line alone is not unique (other AtomwiseOperation forwards share it).
    """        atomic_eng = scatter(
            zbl_edge_eng,
            edge_center,
            dim=0,
            dim_size=num_nodes,
        )
        if AtomicDataDict.PER_ATOM_ENERGY_KEY in data:
            atomic_eng = atomic_eng + data[AtomicDataDict.PER_ATOM_ENERGY_KEY]""",
    """        atomic_eng = scatter(
            zbl_edge_eng,
            edge_center,
            dim=0,
            dim_size=num_nodes,
        )
        if AtomicDataDict.PER_ATOM_ENERGY_KEY in data:
            _nlocal = data[AtomicDataDict.PER_ATOM_ENERGY_KEY].size(0)
            atomic_eng = atomic_eng[:_nlocal] + data[AtomicDataDict.PER_ATOM_ENERGY_KEY]""",
)

_GRAFT_GEB_CLASSMETHOD = (
    "enable_PairNequIPGhostExchange",
    """        return replace_submodules(model, cls, factory)""",
    """        return replace_submodules(model, cls, factory)

    @model_modifier(persistent=True, private=True)
    @classmethod
    def enable_PairNequIPGhostExchange(cls, model):
        \"\"\"Enable native pair_nequip per-layer ghost exchange (registered custom op) for the
        multi-rank pair_nequip AOTInductor target. Swaps only the no-op exchange submodule; the
        model's own norm/instances are untouched.\"\"\"
        from ._ghost_exchange_pair import PairNequIPGhostExchangeModule

        def factory(old):
            return PairNequIPGhostExchangeModule(field=old.field, irreps_in=old.irreps_in)

        return replace_submodules(model, cls, factory)""",
)

_GRAFTS = {
    "interaction_block.py": [_GRAFT_IB],
    "graph_model.py": [_GRAFT_GM_FORWARD, _GRAFT_GM_METADATA],
    "atomwise.py": [_GRAFT_ATOMWISE],
    "pair_potential.py": [_GRAFT_ZBL_DIMSIZE, _GRAFT_ZBL_ADD],
    "_ghost_exchange_base.py": [_GRAFT_GEB_CLASSMETHOD],
    # _ghost_exchange_pair.py is a NEW file (the era predated it); supplied from installed nequip.
}


def graft_multirank_source(fname: str, src: str) -> str:
    """Apply this file's additive multirank grafts to a package's OWN bundled ``nn`` source.

    Idempotent and graceful: a graft already present (idempotency substring found) or whose anchor
    is absent leaves ``src`` unchanged (the latter logs a warning).
    """
    for marker, anchor, replacement in _GRAFTS.get(fname, ()):
        if marker in src:
            continue  # already grafted
        if anchor not in src:
            logger.warning(
                "pair_nequip multirank graft: anchor for a graft in %s not found in the bundled "
                "source; leaving it unchanged. The pair style's runtime multirank-vs-single-rank "
                "guard is the backstop.",
                fname,
            )
            continue
        src = src.replace(anchor, replacement, 1)
    return src


def _installed_pair_exchange_source() -> str:
    """The installed ``_ghost_exchange_pair.py`` source (self-contained; the older era lacks it)."""
    import nequip.nn._ghost_exchange_pair as _pair

    return pathlib.Path(_pair.__file__).read_text(encoding="utf-8")


def derive_multirank_overlay(zip_path, prefix, nn_files, out_dir) -> pathlib.Path:
    """Write an era-matched multirank ``nn`` overlay derived from the package's OWN bundled source.

    For each file in ``nn_files``: read the bundled copy from the ``.nequip.zip`` (member
    ``{prefix}/nequip/nn/{f}``) and apply :func:`graft_multirank_source`; for ``_ghost_exchange_pair``
    (absent from a pre-pair-exchange bundle) write the installed source. Returns ``out_dir``.
    """
    import zipfile

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = {n for n in zf.namelist()}
        for f in nn_files:
            member = f"{prefix}/nequip/nn/{f}"
            if member in members:
                src = zf.read(member).decode("utf-8")
                src = graft_multirank_source(f, src)
            elif f == "_ghost_exchange_pair.py":
                src = _installed_pair_exchange_source()
            else:
                logger.warning(
                    "pair_nequip multirank: %s absent from bundle and no fallback; skipping.", f
                )
                continue
            (out_dir / f).write_text(src, encoding="utf-8")
    return out_dir
