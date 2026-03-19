# Tests for per-head l_max feature
import copy

import pytest
import torch

from nequip.data import AtomicDataDict
from nequip.data.transforms import (
    ChemicalSpeciesToAtomTypeMapper,
    NeighborListTransform,
    HeadStamper,
)
from nequip.data.dataset import EMTTestDataset
from nequip.model import NequIPGNNModel
from nequip.model.extract_head import extract_head
from nequip.nn import MultiHeadReadout
from nequip.nn.per_head_convnetlayer import PerHeadConvNetLayer
from nequip.utils import find_first_of_type
from nequip.utils.global_state import set_global_state


@pytest.fixture(autouse=True)
def setup_global_state():
    set_global_state(allow_tf32=False)


R_MAX = 5.0
TYPE_NAMES = ["Cu", "Al"]
SEED = 42


def _build_model(per_head_l_max=None, l_max=2, num_layers=3):
    return NequIPGNNModel(
        seed=SEED,
        model_dtype="float64",
        type_names=TYPE_NAMES,
        r_max=R_MAX,
        l_max=l_max,
        parity=True,
        num_layers=num_layers,
        num_features=8,
        radial_mlp_depth=1,
        radial_mlp_width=8,
        avg_num_neighbors=10.0,
        head_names=["full", "reduced"],
        per_head_l_max=per_head_l_max,
        per_type_energy_scales={"full": 1.0, "reduced": 1.0},
        per_type_energy_shifts={
            "full": {"Cu": 0.0, "Al": 0.0},
            "reduced": {"Cu": 0.0, "Al": 0.0},
        },
    )


def _make_data(head_index=0):
    transforms = [
        ChemicalSpeciesToAtomTypeMapper(
            model_type_names=TYPE_NAMES,
            chemical_species_to_atom_type_map={"Cu": "Cu"},
        ),
        NeighborListTransform(r_max=R_MAX),
        HeadStamper(head_index=head_index),
    ]
    ds = EMTTestDataset(
        transforms=transforms,
        element="Cu",
        num_frames=3,
        supercell=(2, 2, 2),
        seed=SEED,
    )
    return ds[0]


def test_model_builds_with_per_head_l_max():
    """Model with per_head_l_max should build and contain PerHeadConvNetLayer."""
    model = _build_model(per_head_l_max={"full": 2, "reduced": 0})
    # Should contain PerHeadConvNetLayer
    phc = find_first_of_type(model, PerHeadConvNetLayer)
    assert phc is not None


def test_forward_produces_finite_output():
    """Both heads should produce finite energies and forces."""
    model = _build_model(per_head_l_max={"full": 2, "reduced": 0})
    data = _make_data(head_index=0)

    out0 = model(data.copy())
    assert torch.isfinite(out0[AtomicDataDict.TOTAL_ENERGY_KEY]).all()
    assert torch.isfinite(out0[AtomicDataDict.FORCE_KEY]).all()

    data1 = data.copy()
    data1[AtomicDataDict.HEAD_KEY] = torch.tensor([1], dtype=torch.long)
    out1 = model(data1)
    assert torch.isfinite(out1[AtomicDataDict.TOTAL_ENERGY_KEY]).all()
    assert torch.isfinite(out1[AtomicDataDict.FORCE_KEY]).all()


def test_different_l_max_different_output():
    """Heads with different l_max should produce different energies."""
    model = _build_model(per_head_l_max={"full": 2, "reduced": 0})
    data = _make_data(head_index=0)

    out0 = model(data.copy())
    data1 = data.copy()
    data1[AtomicDataDict.HEAD_KEY] = torch.tensor([1], dtype=torch.long)
    out1 = model(data1)

    assert not torch.allclose(
        out0[AtomicDataDict.TOTAL_ENERGY_KEY],
        out1[AtomicDataDict.TOTAL_ENERGY_KEY],
    )


def test_weight_subset_nesting():
    """Lower l_max heads should use a subset of higher l_max heads' weights."""
    model = _build_model(per_head_l_max={"full": 2, "reduced": 0})
    phc = find_first_of_type(model, PerHeadConvNetLayer)

    full_idx = set(getattr(phc, "_weight_indices_full").tolist())
    reduced_idx = set(getattr(phc, "_weight_indices_reduced").tolist())

    assert reduced_idx.issubset(full_idx), (
        "Reduced l_max head weights should be a subset of full l_max head weights"
    )
    assert len(reduced_idx) < len(full_idx), (
        "Reduced l_max head should use fewer weights"
    )


def test_backward_pass():
    """Backward pass should work with per_head_l_max."""
    model = _build_model(per_head_l_max={"full": 2, "reduced": 0})
    data = _make_data(head_index=0)

    out = model(data.copy())
    loss = out[AtomicDataDict.TOTAL_ENERGY_KEY].sum()
    loss.backward()

    # Check that shared edge MLP has gradients
    phc = find_first_of_type(model, PerHeadConvNetLayer)
    edge_mlp_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in phc.edge_mlp.parameters()
    )
    assert edge_mlp_has_grad, "Shared edge MLP should receive gradients"


def test_backward_compat_without_per_head_l_max():
    """Model without per_head_l_max should not contain PerHeadConvNetLayer."""
    model = _build_model(per_head_l_max=None)
    phc = find_first_of_type(model, PerHeadConvNetLayer)
    assert phc is None


def test_extract_head_with_per_head_l_max():
    """Extracted head should match multi-head output."""
    model = _build_model(per_head_l_max={"full": 2, "reduced": 0})
    data = _make_data(head_index=0)

    # Multi-head outputs
    out_h0 = model(data.copy())
    data_h1 = data.copy()
    data_h1[AtomicDataDict.HEAD_KEY] = torch.tensor([1], dtype=torch.long)
    out_h1 = model(data_h1)

    # Extract and compare
    extracted_full = extract_head(model, "full")
    extracted_reduced = extract_head(model, "reduced")

    assert find_first_of_type(extracted_full, MultiHeadReadout) is None
    assert find_first_of_type(extracted_reduced, MultiHeadReadout) is None

    out_ex_full = extracted_full(data.copy())
    out_ex_reduced = extracted_reduced(data.copy())

    torch.testing.assert_close(
        out_h0[AtomicDataDict.TOTAL_ENERGY_KEY],
        out_ex_full[AtomicDataDict.TOTAL_ENERGY_KEY],
    )
    torch.testing.assert_close(
        out_h1[AtomicDataDict.TOTAL_ENERGY_KEY],
        out_ex_reduced[AtomicDataDict.TOTAL_ENERGY_KEY],
    )
    torch.testing.assert_close(
        out_h0[AtomicDataDict.FORCE_KEY],
        out_ex_full[AtomicDataDict.FORCE_KEY],
    )
    torch.testing.assert_close(
        out_h1[AtomicDataDict.FORCE_KEY],
        out_ex_reduced[AtomicDataDict.FORCE_KEY],
    )


def test_per_head_l_max_validation():
    """per_head_l_max exceeding backbone l_max should raise."""
    with pytest.raises(AssertionError, match="exceeds backbone l_max"):
        _build_model(per_head_l_max={"full": 3, "reduced": 0}, l_max=2)


def test_per_head_l_max_requires_head_names():
    """per_head_l_max without head_names should raise."""
    with pytest.raises(AssertionError, match="requires.*head_names"):
        NequIPGNNModel(
            seed=42,
            model_dtype="float64",
            type_names=TYPE_NAMES,
            r_max=R_MAX,
            l_max=2,
            num_layers=3,
            num_features=8,
            avg_num_neighbors=10.0,
            per_head_l_max={"a": 2, "b": 0},
        )


def test_same_l_max_same_output():
    """Heads with the same l_max should produce the same output (before readout)."""
    model = _build_model(per_head_l_max={"full": 2, "reduced": 2})
    phc = find_first_of_type(model, PerHeadConvNetLayer)

    full_idx = set(getattr(phc, "_weight_indices_full").tolist())
    reduced_idx = set(getattr(phc, "_weight_indices_reduced").tolist())

    assert full_idx == reduced_idx, (
        "Heads with same l_max should use exactly the same weight indices"
    )
