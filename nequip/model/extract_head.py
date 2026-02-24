# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
"""Utility to extract a single head (or sum of heads) from a trained multi-head model."""
import copy

import torch

from nequip.data import AtomicDataDict
from nequip.nn import (
    GraphModel,
    SequentialGraphNetwork,
    AtomwiseReduce,
    MultiHeadReadout,
)
from nequip.nn._graph_mixin import GraphModuleMixin


def _is_instance_by_name(obj, cls):
    """Check isinstance, falling back to class name for torch.package compatibility."""
    if isinstance(obj, cls):
        return True
    return type(obj).__name__ == cls.__name__ and hasattr(obj, "__module__")


def extract_head(model: GraphModel, head_name: str) -> GraphModel:
    """Extract a single head from a multi-head model into a standalone single-head model.

    The returned model does not require ``HEAD_KEY`` in input data and produces
    identical outputs to the multi-head model for the specified head.

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
    readout = head_modules["readout"]
    scale_shift = head_modules["scale_shift"]

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
        # We must clone tensors that readout/scale_shift will overwrite,
        # so that AOT tracing doesn't see aliased writes through shared
        # dict references. The readout writes PER_ATOM_ENERGY_KEY and
        # scale_shift reads/writes it, so we need a fresh dict per head.
        total = None
        for readout, scale_shift in zip(
            self.head_readouts, self.head_scale_shifts
        ):
            head_data = {k: v for k, v in data.items()}
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
        readout = head_modules["readout"]
        scale_shift = head_modules["scale_shift"]
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
