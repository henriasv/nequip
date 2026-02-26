# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
"""Utility to extract a single head (or sum of heads) from a trained multi-head model."""
import copy

import torch

from nequip.data import AtomicDataDict
from nequip.nn import (
    GraphModel,
    SequentialGraphNetwork,
    ScalarMLP,
    PerTypeScaleShift,
    AtomwiseReduce,
    MultiHeadReadout,
)
from nequip.nn._graph_mixin import GraphModuleMixin
from nequip.nn.mlp import ScalarMLPFunction, ScalarLinearLayer
from nequip.utils import find_first_of_type

from e3nn.o3._irreps import Irreps


def _is_instance_by_name(obj, cls):
    """Check isinstance, falling back to class name for torch.package compatibility."""
    if isinstance(obj, cls):
        return True
    return type(obj).__name__ == cls.__name__ and hasattr(obj, "__module__")


def _fuse_linear_readouts(shared: ScalarMLP, correction: ScalarMLP) -> ScalarMLP:
    """Fuse a shared readout and a correction readout into a single ScalarMLP.

    Only works when both are depth-0 (single linear layer, no bias).
    The fused output is: ``shared(x) + correction(x) = x @ (W_s * a_s + W_c * a_c)``.
    """
    shared_fn = shared.mlp_module
    correction_fn = correction.mlp_module

    assert shared_fn.num_layers == 1 and correction_fn.num_layers == 1, (
        "_fuse_linear_readouts only supports depth-0 (single linear layer) readouts"
    )
    assert not shared_fn.bias and not correction_fn.bias, (
        "_fuse_linear_readouts only supports bias=False readouts"
    )

    # Get the single linear layer from each
    shared_linear = shared_fn.mlp[0]
    correction_linear = correction_fn.mlp[0]

    # Compute combined effective weight: W_s * alpha_s + W_c * alpha_c
    combined_weight = (
        shared_linear.weight.data * shared_linear.alpha
        + correction_linear.weight.data * correction_linear.alpha
    )

    # Create a new ScalarMLP with the same config as the shared one
    fused = ScalarMLP(
        output_dim=1,
        hidden_layers_depth=0,
        nonlinearity=None,  # depth=0 means no nonlinearity applied anyway
        bias=False,
        forward_weight_init=True,
        field=shared.field,
        out_field=shared.out_field,
        irreps_in=shared.irreps_in,
    )

    # Set the fused weights: store combined weight with alpha=1.0
    fused_linear = fused.mlp_module.mlp[0]
    fused_linear.weight.data.copy_(combined_weight)
    fused_linear.alpha.fill_(1.0)

    return fused


class SharedPlusCorrectionReadout(GraphModuleMixin, torch.nn.Module):
    """Wrapper that sums shared readout + per-head correction into a single forward pass.

    Used by :func:`extract_head` when the shared readout has hidden layers
    (depth > 0) and weight fusion is not possible.
    """

    field: str
    out_field: str

    def __init__(self, shared: ScalarMLP, correction: ScalarMLP):
        super().__init__()
        self.shared = shared
        self.correction = correction
        self.field = shared.field
        self.out_field = shared.out_field
        self._init_irreps(
            irreps_in=shared.irreps_in,
            irreps_out=shared.irreps_out,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        shared_data = data.copy()
        shared_data = self.shared(shared_data)
        shared_energy = shared_data[self.out_field]

        correction_data = data.copy()
        correction_data = self.correction(correction_data)
        correction_energy = correction_data[self.out_field]

        data[self.out_field] = shared_energy + correction_energy
        return data


def extract_head(model: GraphModel, head_name: str) -> GraphModel:
    """Extract a single head from a multi-head model into a standalone single-head model.

    The returned model does not require ``HEAD_KEY`` in input data and produces
    identical outputs to the multi-head model for the specified head.

    For shared-readout models, the shared readout and per-head correction are
    fused into a single readout (weight fusion for depth-0 MLPs, or a wrapper
    module for deeper MLPs).

    Args:
        model: A :class:`~nequip.nn.GraphModel` containing a
            :class:`~nequip.nn.MultiHeadReadout`.
        head_name: Name of the head to extract (must be one of the
            ``head_names`` used when building the multi-head model).

    Returns:
        A new :class:`~nequip.nn.GraphModel` with the ``MultiHeadReadout``
        replaced by the extracted head's ``ScalarMLP`` +
        ``PerTypeScaleShift`` + ``AtomwiseReduce``.

    Raises:
        ValueError: If no ``MultiHeadReadout`` is found or ``head_name``
            is not in the model.
    """
    model = copy.deepcopy(model)

    # Find the SequentialGraphNetwork inside the model
    # The structure is: GraphModel -> ForceStressOutput -> SequentialGraphNetwork
    # or GraphModel -> SequentialGraphNetwork
    seq_net = None
    mhr = None
    mhr_name = None

    # Walk the model tree to find MultiHeadReadout
    def _find_mhr(module, parent_name=""):
        nonlocal seq_net, mhr, mhr_name
        if _is_instance_by_name(module, MultiHeadReadout):
            mhr = module
            mhr_name = parent_name
            return
        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            _find_mhr(child, full_name)

    _find_mhr(model)

    if mhr is None:
        raise ValueError(
            "No MultiHeadReadout found in model. "
            "Is this a multi-head model?"
        )

    if head_name not in mhr.head_names:
        raise ValueError(
            f"Head '{head_name}' not found. "
            f"Available heads: {mhr.head_names}"
        )

    # Find the SequentialGraphNetwork that contains the MultiHeadReadout
    # Navigate to it by finding the parent
    def _find_seq_and_replace(module):
        if _is_instance_by_name(module, SequentialGraphNetwork):
            for name, child in module.named_children():
                if _is_instance_by_name(child, MultiHeadReadout):
                    return module, name
        for name, child in module.named_children():
            result = _find_seq_and_replace(child)
            if result is not None:
                return result
        return None

    result = _find_seq_and_replace(model)
    if result is None:
        raise ValueError(
            "Could not find SequentialGraphNetwork containing MultiHeadReadout"
        )
    seq_net, multihead_key = result

    # Extract head modules
    head_modules = mhr.heads[head_name]
    correction_readout = head_modules["readout"]
    scale_shift = head_modules["scale_shift"]

    # Build the readout for the extracted head
    if mhr.shared_readout_mode:
        shared = mhr.shared_readout
        # Try weight fusion for depth-0 linear readouts
        if (
            shared.mlp_module.num_layers == 1
            and correction_readout.mlp_module.num_layers == 1
            and not shared.mlp_module.bias
            and not correction_readout.mlp_module.bias
        ):
            readout = _fuse_linear_readouts(shared, correction_readout)
        else:
            readout = SharedPlusCorrectionReadout(shared, correction_readout)
    else:
        readout = correction_readout

    # Create AtomwiseReduce matching the original
    reduce = AtomwiseReduce(
        irreps_in=scale_shift.irreps_out,
        reduce="sum",
        field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
    )

    # Replace: remove multihead_readout, insert individual modules
    # We need to rebuild the OrderedDict of the SequentialGraphNetwork
    new_modules = {}
    for name, child in seq_net.named_children():
        if name == multihead_key:
            # Replace with individual head modules
            new_modules["per_atom_energy_readout"] = readout
            new_modules["per_type_energy_scale_shift"] = scale_shift
            new_modules["total_energy_sum"] = reduce
        else:
            new_modules[name] = child

    # Build new SequentialGraphNetwork
    new_seq = SequentialGraphNetwork(new_modules)

    # Replace in the model hierarchy
    # Find where seq_net sits
    def _replace_seq(parent, old_seq, new_seq):
        for name, child in parent.named_children():
            if child is old_seq:
                setattr(parent, name, new_seq)
                return True
            if _replace_seq(child, old_seq, new_seq):
                return True
        return False

    _replace_seq(model, seq_net, new_seq)

    # Update model's irreps
    model._init_irreps(
        irreps_in=model.irreps_in,
        irreps_out=new_seq.irreps_out if hasattr(model, 'model') and hasattr(model.model, 'irreps_out') else model.irreps_out,
    )

    return model


class SummedHeadsReadout(GraphModuleMixin, torch.nn.Module):
    """Runs multiple heads' readout+scale_shift pipelines and sums per-atom energies.

    Each head's pipeline (readout MLP → PerTypeScaleShift) is run independently,
    producing a per-atom energy tensor. All per-atom energies are then summed
    element-wise. This enables deploying a single model that computes e.g.
    ``E_base + E_delta`` from a multi-head training run.

    Args:
        head_pipelines: list of ``(readout, scale_shift)`` tuples, where each
            readout is a :class:`~nequip.nn.ScalarMLP` (or fused/wrapper) and
            each scale_shift is a :class:`~nequip.nn.PerTypeScaleShift`.
    """

    def __init__(self, head_pipelines: list):
        super().__init__()
        self.head_readouts = torch.nn.ModuleList(
            [r for r, _ in head_pipelines]
        )
        self.head_scale_shifts = torch.nn.ModuleList(
            [s for _, s in head_pipelines]
        )
        # Use first pipeline's irreps for the module
        first_readout = head_pipelines[0][0]
        last_scale_shift = head_pipelines[0][1]
        self._init_irreps(
            irreps_in=first_readout.irreps_in,
            irreps_out=last_scale_shift.irreps_out,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        total = None
        for readout, scale_shift in zip(
            self.head_readouts, self.head_scale_shifts
        ):
            head_data = data.copy()
            head_data = readout(head_data)
            head_data = scale_shift(head_data)
            e = head_data[AtomicDataDict.PER_ATOM_ENERGY_KEY]
            total = e if total is None else total + e

        data[AtomicDataDict.PER_ATOM_ENERGY_KEY] = total
        return data


def extract_summed_heads(model: GraphModel, head_names: list) -> GraphModel:
    """Extract multiple heads from a multi-head model and sum their outputs.

    The returned model runs each requested head's readout + scale_shift pipeline
    and sums the resulting per-atom energies. This is useful for delta-learning
    workflows where the deployed prediction is e.g.
    ``E_base + E_delta = E_target``.

    The returned model does not require ``HEAD_KEY`` in input data.

    Args:
        model: A :class:`~nequip.nn.GraphModel` containing a
            :class:`~nequip.nn.MultiHeadReadout`.
        head_names: List of head names to sum (must all be present in the model).

    Returns:
        A new :class:`~nequip.nn.GraphModel` with the ``MultiHeadReadout``
        replaced by a :class:`SummedHeadsReadout` + ``AtomwiseReduce``.

    Raises:
        ValueError: If no ``MultiHeadReadout`` is found, or any head name
            is not in the model.
    """
    model = copy.deepcopy(model)

    # Find MultiHeadReadout
    mhr = None

    def _find_mhr(module, parent_name=""):
        nonlocal mhr
        if _is_instance_by_name(module, MultiHeadReadout):
            mhr = module
            return
        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            _find_mhr(child, full_name)

    _find_mhr(model)

    if mhr is None:
        raise ValueError(
            "No MultiHeadReadout found in model. "
            "Is this a multi-head model?"
        )

    for hn in head_names:
        if hn not in mhr.head_names:
            raise ValueError(
                f"Head '{hn}' not found. "
                f"Available heads: {mhr.head_names}"
            )

    # Find the SequentialGraphNetwork containing the MultiHeadReadout
    def _find_seq_and_replace(module):
        if _is_instance_by_name(module, SequentialGraphNetwork):
            for name, child in module.named_children():
                if _is_instance_by_name(child, MultiHeadReadout):
                    return module, name
        for name, child in module.named_children():
            result = _find_seq_and_replace(child)
            if result is not None:
                return result
        return None

    result = _find_seq_and_replace(model)
    if result is None:
        raise ValueError(
            "Could not find SequentialGraphNetwork containing MultiHeadReadout"
        )
    seq_net, multihead_key = result

    # Build per-head (readout, scale_shift) pipelines
    head_pipelines = []
    for hn in head_names:
        head_modules = mhr.heads[hn]
        correction_readout = head_modules["readout"]
        scale_shift = head_modules["scale_shift"]

        if mhr.shared_readout_mode:
            shared = mhr.shared_readout
            if (
                shared.mlp_module.num_layers == 1
                and correction_readout.mlp_module.num_layers == 1
                and not shared.mlp_module.bias
                and not correction_readout.mlp_module.bias
            ):
                readout = _fuse_linear_readouts(shared, correction_readout)
            else:
                readout = SharedPlusCorrectionReadout(shared, correction_readout)
        else:
            readout = correction_readout

        head_pipelines.append((readout, scale_shift))

    summed = SummedHeadsReadout(head_pipelines)

    # Create AtomwiseReduce
    reduce = AtomwiseReduce(
        irreps_in=head_pipelines[0][1].irreps_out,
        reduce="sum",
        field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
    )

    # Replace MultiHeadReadout in the SequentialGraphNetwork
    new_modules = {}
    for name, child in seq_net.named_children():
        if name == multihead_key:
            new_modules["summed_heads_readout"] = summed
            new_modules["total_energy_sum"] = reduce
        else:
            new_modules[name] = child

    new_seq = SequentialGraphNetwork(new_modules)

    # Replace in the model hierarchy
    def _replace_seq(parent, old_seq, new_seq):
        for name, child in parent.named_children():
            if child is old_seq:
                setattr(parent, name, new_seq)
                return True
            if _replace_seq(child, old_seq, new_seq):
                return True
        return False

    _replace_seq(model, seq_net, new_seq)

    model._init_irreps(
        irreps_in=model.irreps_in,
        irreps_out=new_seq.irreps_out if hasattr(model, 'model') and hasattr(model.model, 'irreps_out') else model.irreps_out,
    )

    return model
