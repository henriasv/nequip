# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch

from nequip.data import AtomicDataDict
from ._graph_mixin import GraphModuleMixin
from .atomwise import AtomwiseReduce, PerTypeScaleShift
from .mlp import ScalarMLP

from typing import Dict, List, Optional, Union


class MultiHeadReadout(GraphModuleMixin, torch.nn.Module):
    """Multi-head readout that branches into per-head readout pathways and selects
    outputs based on ``HEAD_KEY``.

    Each head has its own ``ScalarMLP`` (readout) and ``PerTypeScaleShift``
    (per-type scales/shifts). All heads produce per-atom energies, which are
    then selected per-atom based on the head index. A shared ``AtomwiseReduce``
    sums the selected per-atom energies to total energy.

    If ``HEAD_KEY`` is absent from the data, head 0 is used (backward compat).

    Args:
        head_names: list of head name strings (e.g. ``["HF", "MP2"]``)
        type_names: list of atom type names
        readout_mlp_hidden_layers_depth: depth of hidden layers in readout MLP
        readout_mlp_hidden_layers_width: width of hidden layers in readout MLP
        readout_mlp_nonlinearity: nonlinearity for readout MLP
        per_head_energy_scales: dict mapping head_name to scales (float or Dict[str, float])
        per_head_energy_shifts: dict mapping head_name to shifts (float or Dict[str, float])
        per_type_energy_scales_trainable: whether scales are trainable
        per_type_energy_shifts_trainable: whether shifts are trainable
        irreps_in: input irreps dict
    """

    def __init__(
        self,
        head_names: List[str],
        type_names: List[str],
        readout_mlp_hidden_layers_depth: int = 0,
        readout_mlp_hidden_layers_width: Optional[int] = None,
        readout_mlp_nonlinearity: Optional[str] = "silu",
        per_head_energy_scales: Optional[
            Dict[str, Optional[Union[float, Dict[str, float]]]]
        ] = None,
        per_head_energy_shifts: Optional[
            Dict[str, Optional[Union[float, Dict[str, float]]]]
        ] = None,
        per_type_energy_scales_trainable: bool = False,
        per_type_energy_shifts_trainable: bool = False,
        irreps_in=None,
    ):
        super().__init__()
        assert len(head_names) >= 1, "At least one head must be provided"
        self.head_names = head_names
        self.num_heads = len(head_names)

        if per_head_energy_scales is None:
            per_head_energy_scales = {}
        if per_head_energy_shifts is None:
            per_head_energy_shifts = {}

        # Build per-head readout + scale/shift modules
        heads = {}
        for head_name in head_names:
            readout = ScalarMLP(
                output_dim=1,
                hidden_layers_depth=readout_mlp_hidden_layers_depth,
                hidden_layers_width=readout_mlp_hidden_layers_width,
                nonlinearity=readout_mlp_nonlinearity,
                bias=False,
                forward_weight_init=True,
                field=AtomicDataDict.NODE_FEATURES_KEY,
                out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
                irreps_in=irreps_in,
            )
            scale_shift = PerTypeScaleShift(
                type_names=type_names,
                field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
                out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
                scales=per_head_energy_scales.get(head_name, None),
                shifts=per_head_energy_shifts.get(head_name, None),
                scales_trainable=per_type_energy_scales_trainable,
                shifts_trainable=per_type_energy_shifts_trainable,
                irreps_in=readout.irreps_out,
            )
            heads[head_name] = torch.nn.ModuleDict(
                {"readout": readout, "scale_shift": scale_shift}
            )

        self.heads = torch.nn.ModuleDict(heads)

        # Shared reduce
        # Get irreps_in for reduce from any head's scale_shift output
        any_head = heads[head_names[0]]
        self.reduce = AtomwiseReduce(
            irreps_in=any_head["scale_shift"].irreps_out,
            reduce="sum",
            field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
        )

        # Set up irreps
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out=self.reduce.irreps_out,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # Get head indices per frame
        if AtomicDataDict.HEAD_KEY in data:
            frame_heads = data[AtomicDataDict.HEAD_KEY].view(-1)
        else:
            # Default to head 0 if no HEAD_KEY (backward compat / inference)
            n_frames = AtomicDataDict.num_frames(data)
            frame_heads = torch.zeros(
                n_frames,
                dtype=torch.long,
                device=data[AtomicDataDict.POSITIONS_KEY].device,
            )

        # Broadcast to per-atom head indices
        if AtomicDataDict.BATCH_KEY in data:
            node_heads = frame_heads[data[AtomicDataDict.BATCH_KEY]]
        else:
            # Single frame, all atoms belong to same head
            n_atoms = AtomicDataDict.num_nodes(data)
            node_heads = frame_heads[0].expand(n_atoms)

        # Run each head and collect per-atom energies
        head_outputs = []
        for head_name in self.head_names:
            head_modules = self.heads[head_name]
            # Make a copy of data so each head operates independently
            head_data = data.copy()
            # If PerHeadConvNetLayer provided per-head features, use them
            per_head_key = f"_per_head_features_{head_name}"
            if per_head_key in data:
                head_data[AtomicDataDict.NODE_FEATURES_KEY] = data[per_head_key]
            head_data = head_modules["readout"](head_data)
            head_data = head_modules["scale_shift"](head_data)
            head_outputs.append(head_data[AtomicDataDict.PER_ATOM_ENERGY_KEY])

        # Stack and select: [n_atoms, n_heads, 1]
        stacked = torch.stack(head_outputs, dim=1)  # [n_atoms, n_heads, 1]
        arange = torch.arange(stacked.shape[0], device=stacked.device)
        selected = stacked[arange, node_heads, :]  # [n_atoms, 1]

        data[AtomicDataDict.PER_ATOM_ENERGY_KEY] = selected

        # Reduce to total energy (for the selected head)
        data = self.reduce(data)

        # Store per-head total energies as explicit named outputs.
        # This ensures torch.compile preserves them in the computation graph
        # (unnamed intermediates like _all_head_per_atom_energies get pruned).
        # training_step uses these for per-head force computation via autograd.
        for i, head_name in enumerate(self.head_names):
            head_per_atom = stacked[:, i, :]
            head_reduce_data = data.copy()
            head_reduce_data[AtomicDataDict.PER_ATOM_ENERGY_KEY] = head_per_atom
            head_reduce_data = self.reduce(head_reduce_data)
            data[f"_total_energy_{head_name}"] = (
                head_reduce_data[AtomicDataDict.TOTAL_ENERGY_KEY]
            )

        # Clean up internal per-head feature keys
        for head_name in self.head_names:
            data.pop(f"_per_head_features_{head_name}", None)

        return data
