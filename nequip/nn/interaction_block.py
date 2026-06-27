# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
"""Interaction Block"""

import torch

from e3nn.o3._irreps import Irreps
from e3nn.o3._linear import Linear
from e3nn.o3._tensor_product._sub import FullyConnectedTensorProduct

from nequip.data import AtomicDataDict

from ._graph_mixin import GraphModuleMixin
from .mlp import ScalarMLPFunction
from ._ghost_exchange_base import NoOpGhostExchangeModule
from ._tp_scatter_base import TensorProductScatter
from .norm import AvgNumNeighborsNorm

from typing import Optional, Sequence, Union, Dict


class InteractionBlock(GraphModuleMixin, torch.nn.Module):
    use_sc: bool

    def __init__(
        self,
        irreps_in,
        irreps_out,
        radial_mlp_depth: int = 1,
        radial_mlp_width: int = 8,
        use_sc: bool = True,
        is_first_layer: bool = False,
        type_names: Optional[Sequence[str]] = None,
        avg_num_neighbors: Optional[Union[float, Dict[str, float]]] = None,
    ) -> None:
        """InteractionBlock.

        Args:
            irreps_in: input irreps
            irreps_out: output irreps
            radial_mlp_depth (int): number of radial layers
            radial_mlp_width (int): number of hidden neurons in radial function
            use_sc (bool): use self-connection or not
            is_first_layer (bool): whether to use first layer (default ``False``)
            avg_num_neighbors (float/Dict[str, float]): global (float) or per-type (dict) average number of neighbors
            type_names (List[str]): list of type names
        """
        super().__init__()

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                AtomicDataDict.EDGE_EMBEDDING_KEY,
                AtomicDataDict.EDGE_ATTRS_KEY,
                AtomicDataDict.NODE_FEATURES_KEY,
                AtomicDataDict.NODE_ATTRS_KEY,
            ],
            my_irreps_in={
                AtomicDataDict.EDGE_EMBEDDING_KEY: Irreps(
                    [
                        (
                            irreps_in[AtomicDataDict.EDGE_EMBEDDING_KEY].num_irreps,
                            (0, 1),
                        )
                    ]  # (0, 1) is even (invariant) scalars. We are forcing the EDGE_EMBEDDING to be invariant scalars so we can use a dense network
                )
            },
            irreps_out={AtomicDataDict.NODE_FEATURES_KEY: irreps_out},
        )

        # === normalization module ===
        self.avg_num_neighbors_norm = AvgNumNeighborsNorm(
            avg_num_neighbors=avg_num_neighbors, type_names=type_names
        )

        self.use_sc = use_sc

        feature_irreps_in = self.irreps_in[AtomicDataDict.NODE_FEATURES_KEY]
        feature_irreps_out = self.irreps_out[AtomicDataDict.NODE_FEATURES_KEY]
        irreps_edge_attr = self.irreps_in[AtomicDataDict.EDGE_ATTRS_KEY]

        # - Build modules -
        self.linear_1 = Linear(
            irreps_in=feature_irreps_in,
            irreps_out=feature_irreps_in,
            internal_weights=True,
            shared_weights=True,
        )

        irreps_mid = []
        instructions = []

        for i, (mul, ir_in) in enumerate(feature_irreps_in):
            for j, (_, ir_edge) in enumerate(irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in feature_irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))

        # We sort the output irreps of the tensor product so that we can simplify them
        # when they are provided to the second o3.Linear
        irreps_mid = Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        # Permute the output indexes of the instructions to match the sorted irreps:
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        self.tp_scatter = TensorProductScatter(
            feature_irreps_in,
            irreps_edge_attr,
            irreps_mid,
            instructions,
        )

        # init_irreps already confirmed that the edge embeddding is all invariant scalars
        self.edge_mlp = ScalarMLPFunction(
            input_dim=self.irreps_in[AtomicDataDict.EDGE_EMBEDDING_KEY].num_irreps,
            output_dim=self.tp_scatter.tp.weight_numel,
            hidden_layers_depth=radial_mlp_depth,
            hidden_layers_width=radial_mlp_width,
            nonlinearity="silu",  # hardcode SiLU
            bias=False,
            forward_weight_init=True,
        )

        self.linear_2 = Linear(
            # irreps_mid has uncoallesed irreps because of the uvu instructions,
            # but there's no reason to treat them seperately for the Linear
            # Note that normalization of o3.Linear changes if irreps are coallesed
            # (likely for the better)
            irreps_in=irreps_mid.simplify(),
            irreps_out=feature_irreps_out,
            internal_weights=True,
            shared_weights=True,
        )

        self.sc = None
        if self.use_sc:
            self.sc = FullyConnectedTensorProduct(
                feature_irreps_in,
                self.irreps_in[AtomicDataDict.NODE_ATTRS_KEY],
                feature_irreps_out,
            )

        self.ghost_exchange = NoOpGhostExchangeModule(
            field=AtomicDataDict.NODE_FEATURES_KEY, irreps_in=self.irreps_in
        )

        self.is_first_layer = is_first_layer

    @torch.jit.unused
    def _get_mliap_num_local(self, data: AtomicDataDict.Type) -> int:
        return data[AtomicDataDict.LMP_MLIAP_DATA_KEY].nlocal

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if AtomicDataDict.LMP_MLIAP_DATA_KEY in data:
            num_local_nodes = self._get_mliap_num_local(data)
        elif "num_local_nodes_marker" in data:
            # Native multi-rank `pair_nequip` (truncate-to-nlocal): the owned count is the
            # size-0 of the marker input — a *backed* dynamic dim read as a tensor dimension,
            # never via `.item()`, so it carries no data-dependent (unbacked) size and
            # AOT-exports cleanly. Every per-node op below is then sliced to owned atoms only
            # (cutting the redundant ghost-node compute of the all-`ntotal` formulation); the
            # per-layer `ghost_exchange` re-expands owned features back to `ntotal` (filling
            # ghost rows from their owners on other ranks) right before the TP-scatter.
            # Literal key (not AtomicDataDict.NUM_LOCAL_NODES_MARKER_KEY): injection-safe when
            # this source is repacked into an older model whose AtomicDataDict lacks it.
            num_local_nodes = data["num_local_nodes_marker"].shape[0]
        else:
            # Plain single-rank `pair_nequip` / ASE: no ghosts, so `nlocal == ntotal` and the
            # truncations below are no-ops. Backed `num_nodes` keeps the graph export-clean.
            num_local_nodes = AtomicDataDict.num_nodes(data)

        x = data[AtomicDataDict.NODE_FEATURES_KEY]

        # truncate if not first layer
        if not self.is_first_layer:
            x = x[:num_local_nodes]

        if self.sc is not None:
            node_attrs = data[AtomicDataDict.NODE_ATTRS_KEY]
            # truncate if not first layer
            if not self.is_first_layer:
                node_attrs = node_attrs[:num_local_nodes]
            sc = self.sc(x, node_attrs)

        x = self.linear_1(x)

        # normalize before TP-scatter
        data[AtomicDataDict.NODE_FEATURES_KEY] = x
        data = self.avg_num_neighbors_norm(data)
        x = data[AtomicDataDict.NODE_FEATURES_KEY]

        # === comms for ghost-exchange ===
        # only done if not first layer
        # because initial embedding include ghosts since atom types come with ghosts
        if not self.is_first_layer:
            data[AtomicDataDict.NODE_FEATURES_KEY] = x
            if "num_owned_edges_marker" in data:
                # Async-overlap (M10 Stage 2): expand owned features to `ntotal` and record the
                # "owned features ready" marker, but DEFER the halo `forward_comm` to `forward_finish`
                # below — so the owned-source TP-scatter (which reads only owned features) runs while
                # the halo is in flight. Ghost rows are zero here; the owned-source TP does not read
                # them. Literal key (not the AtomicDataDict attribute): injection-safe under repack.
                data = self.ghost_exchange.forward_start(data, ghost_included=False)
            else:
                data = self.ghost_exchange(data, ghost_included=False)
            x = data[AtomicDataDict.NODE_FEATURES_KEY]

        # === TP and scatter ===
        edge_attr = data[AtomicDataDict.EDGE_ATTRS_KEY]
        edge_weight = self.edge_mlp(data[AtomicDataDict.EDGE_EMBEDDING_KEY])
        edge_dst = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_src = data[AtomicDataDict.EDGE_INDEX_KEY][1]

        # Async-overlap edge split (M10): on non-first layers of the async multi-rank target the
        # native pair style emits the edge list owned-source-first (edges with `edge_src < nlocal`)
        # and supplies `num_owned_edges_marker` whose dim-0 is the *backed* count of those edges.
        # The owned-source TP-scatter needs only owned features `x[edge_src]` — available BEFORE the
        # feature halo arrives — so it can run while the halo is in flight (the overlap is wired in
        # Stage 2; here the `ghost_exchange` above is still blocking). Splitting is exact: `scatter`
        # is additive over edges and both calls share the same `x` (hence the same `dim_size`), so
        # `tp_scatter(owned) + tp_scatter(ghost)` equals one scatter over all edges up to scatter-add
        # reassociation. Literal key (not AtomicDataDict.NUM_OWNED_EDGES_MARKER_KEY): injection-safe
        # when this source is repacked into an older model whose AtomicDataDict lacks the attribute.
        if (not self.is_first_layer) and ("num_owned_edges_marker" in data):
            s = data["num_owned_edges_marker"].shape[0]
            # Owned-source edges first: their TP-scatter reads only owned features `x[edge_src]`
            # (`edge_src < nlocal`), available BEFORE the halo. This runs on the model stream while
            # the deferred halo is still in flight (Stage 2: `forward_start` above issued no comm;
            # `forward_finish` below completes it). `x` here is the `ntotal`-wide pre-halo tensor.
            x_owned = self.tp_scatter(
                x=x,
                edge_attr=edge_attr[:s],
                edge_weight=edge_weight[:s],
                edge_dst=edge_dst[:s],
                edge_src=edge_src[:s],
            )
            # Complete the feature halo now (overlapped with `x_owned` above). After this `x` is the
            # ghost-filled `ntotal` tensor; the ghost-source TP-scatter (`edge_src >= nlocal`) needs
            # those rows. Splitting is exact (scatter is additive over edges; owned rows of the pre-
            # and post-halo tensors are identical), so the sum reproduces the single-scatter result.
            data[AtomicDataDict.NODE_FEATURES_KEY] = x
            data = self.ghost_exchange.forward_finish(data, ghost_included=False)
            x = data[AtomicDataDict.NODE_FEATURES_KEY]
            x_ghost = self.tp_scatter(
                x=x,
                edge_attr=edge_attr[s:],
                edge_weight=edge_weight[s:],
                edge_dst=edge_dst[s:],
                edge_src=edge_src[s:],
            )
            x = x_owned + x_ghost
        else:
            x = self.tp_scatter(
                x=x,
                edge_attr=edge_attr,
                edge_weight=edge_weight,
                edge_dst=edge_dst,
                edge_src=edge_src,
            )
        x = x[:num_local_nodes]

        x = self.linear_2(x)

        if self.sc is not None:
            x = x + sc

        data[AtomicDataDict.NODE_FEATURES_KEY] = x
        return data
