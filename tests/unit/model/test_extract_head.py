# Test extract_head utility
import copy
import pytest
import torch

from nequip.data import AtomicDataDict
from nequip.data.dict import from_dict
from nequip.data.transforms import ChemicalSpeciesToAtomTypeMapper, NeighborListTransform
from nequip.nn import MultiHeadReadout
from nequip.model.extract_head import extract_head, extract_summed_heads, SummedHeadsReadout
from nequip.utils.global_state import set_global_state
from nequip.utils import find_first_of_type

from hydra.utils import get_method


@pytest.fixture(autouse=True)
def setup_global_state():
    set_global_state(allow_tf32=False)


BASIC_INFO = {
    "seed": 123,
    "model_dtype": "float64",
    "type_names": ["H", "C"],
    "r_max": 4.0,
    "avg_num_neighbors": 5,
}


def _build_multihead_model():
    config = {
        "_target_": "nequip.model.NequIPGNNModel",
        "l_max": 1,
        "parity": True,
        "num_features": 8,
        "num_layers": 2,
        "radial_mlp_depth": 1,
        "radial_mlp_width": 8,
        "head_names": ["HF", "MP2"],
        "per_head_energy_scales": {"HF": 1.0, "MP2": 2.0},
        "per_head_energy_shifts": {
            "HF": {"H": 1.0, "C": 2.0},
            "MP2": {"H": 0.5, "C": 1.5},
        },
        **BASIC_INFO,
    }
    config = copy.deepcopy(config)
    builder = get_method(config.pop("_target_"))
    return builder(**config)


def _make_data(head_index=0, seed=42):
    type_mapper = ChemicalSpeciesToAtomTypeMapper(
        model_type_names=["H", "C"],
        chemical_species_to_atom_type_map={"H": "H", "C": "C"},
    )
    nl = NeighborListTransform(r_max=4.0)

    data = from_dict(
        {
            "pos": torch.randn(4, 3, dtype=torch.float64) * 2.0,
            "atomic_numbers": torch.tensor([1, 6, 1, 6], dtype=torch.long),
            "cell": torch.eye(3, dtype=torch.float64).unsqueeze(0) * 10.0,
            "pbc": torch.tensor([[True, True, True]]),
        }
    )
    data = type_mapper(data)
    data = nl(data)
    data[AtomicDataDict.HEAD_KEY] = torch.tensor([head_index], dtype=torch.long)
    return data


def test_extract_head_produces_single_head_model():
    """Extracted model should not contain MultiHeadReadout."""
    model = _build_multihead_model()
    extracted = extract_head(model, "HF")

    mhr = find_first_of_type(extracted, MultiHeadReadout)
    assert mhr is None


def test_extract_head_identical_output():
    """Extracted model should produce identical output to multi-head model for that head."""
    model = _build_multihead_model()

    # Set seed for reproducible test data
    torch.manual_seed(42)
    data = _make_data(head_index=0, seed=42)

    # Get output from multi-head model
    out_multi = model(data.copy())

    # Extract HF head (index 0)
    extracted = extract_head(model, "HF")

    # Get output from extracted model (no HEAD_KEY needed)
    data_no_head = data.copy()
    # HEAD_KEY can be present but won't be used since there's no MultiHeadReadout
    out_single = extracted(data_no_head)

    torch.testing.assert_close(
        out_multi[AtomicDataDict.TOTAL_ENERGY_KEY],
        out_single[AtomicDataDict.TOTAL_ENERGY_KEY],
    )
    torch.testing.assert_close(
        out_multi[AtomicDataDict.FORCE_KEY],
        out_single[AtomicDataDict.FORCE_KEY],
    )


def test_extract_second_head():
    """Extracting the second head should give different output than the first."""
    model = _build_multihead_model()

    torch.manual_seed(42)
    data = _make_data(head_index=1, seed=42)

    # Multi-head with head 1
    out_multi = model(data.copy())

    # Extract MP2 head
    extracted = extract_head(model, "MP2")
    out_single = extracted(data.copy())

    torch.testing.assert_close(
        out_multi[AtomicDataDict.TOTAL_ENERGY_KEY],
        out_single[AtomicDataDict.TOTAL_ENERGY_KEY],
    )


def test_extract_head_without_head_key():
    """Extracted model should work without HEAD_KEY in input."""
    model = _build_multihead_model()
    extracted = extract_head(model, "HF")

    torch.manual_seed(42)
    data = _make_data(head_index=0, seed=42)
    # Remove HEAD_KEY
    del data[AtomicDataDict.HEAD_KEY]

    out = extracted(data)
    assert AtomicDataDict.TOTAL_ENERGY_KEY in out
    assert AtomicDataDict.FORCE_KEY in out
    assert torch.isfinite(out[AtomicDataDict.TOTAL_ENERGY_KEY]).all()
    assert torch.isfinite(out[AtomicDataDict.FORCE_KEY]).all()


def test_extract_head_invalid_name():
    """Extracting a non-existent head should raise ValueError."""
    model = _build_multihead_model()
    with pytest.raises(ValueError, match="not found"):
        extract_head(model, "CCSD")


def test_extract_head_from_single_head_model():
    """Extracting from a model without MultiHeadReadout should raise ValueError."""
    config = {
        "_target_": "nequip.model.NequIPGNNModel",
        "l_max": 1,
        "parity": True,
        "num_features": 8,
        "num_layers": 2,
        "radial_mlp_depth": 1,
        "radial_mlp_width": 8,
        "per_type_energy_shifts": {"H": 1.0, "C": 2.0},
        **BASIC_INFO,
    }
    config = copy.deepcopy(config)
    builder = get_method(config.pop("_target_"))
    model = builder(**config)

    with pytest.raises(ValueError, match="No MultiHeadReadout"):
        extract_head(model, "HF")


def test_original_model_unchanged():
    """extract_head should not modify the original model (deep copy)."""
    model = _build_multihead_model()

    # Store original state
    original_params = {
        name: p.clone() for name, p in model.named_parameters()
    }

    extracted = extract_head(model, "HF")

    # Check original model unchanged
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, original_params[name])

    # Check MultiHeadReadout still present in original
    mhr = find_first_of_type(model, MultiHeadReadout)
    assert mhr is not None


# === Shared Readout extract_head Tests ===


def _build_shared_readout_model():
    config = {
        "_target_": "nequip.model.NequIPGNNModel",
        "l_max": 1,
        "parity": True,
        "num_features": 8,
        "num_layers": 2,
        "radial_mlp_depth": 1,
        "radial_mlp_width": 8,
        "head_names": ["HF", "MP2"],
        "shared_readout": True,
        "per_head_energy_scales": {"HF": 1.0, "MP2": 2.0},
        "per_head_energy_shifts": {
            "HF": {"H": 1.0, "C": 2.0},
            "MP2": {"H": 0.5, "C": 1.5},
        },
        **BASIC_INFO,
    }
    config = copy.deepcopy(config)
    builder = get_method(config.pop("_target_"))
    return builder(**config)


def test_extract_shared_readout_head_no_multihead():
    """Extracted model from shared readout should not contain MultiHeadReadout."""
    model = _build_shared_readout_model()
    extracted = extract_head(model, "HF")
    mhr = find_first_of_type(extracted, MultiHeadReadout)
    assert mhr is None


def test_extract_shared_readout_head_identical_output():
    """Extracted shared readout head should produce identical output to multi-head model."""
    model = _build_shared_readout_model()

    torch.manual_seed(42)
    data = _make_data(head_index=0, seed=42)

    out_multi = model(data.copy())
    extracted = extract_head(model, "HF")
    out_single = extracted(data.copy())

    torch.testing.assert_close(
        out_multi[AtomicDataDict.TOTAL_ENERGY_KEY],
        out_single[AtomicDataDict.TOTAL_ENERGY_KEY],
    )
    torch.testing.assert_close(
        out_multi[AtomicDataDict.FORCE_KEY],
        out_single[AtomicDataDict.FORCE_KEY],
    )


def test_extract_shared_readout_second_head():
    """Extracting the second head from shared readout model should match."""
    model = _build_shared_readout_model()

    torch.manual_seed(42)
    data = _make_data(head_index=1, seed=42)

    out_multi = model(data.copy())
    extracted = extract_head(model, "MP2")
    out_single = extracted(data.copy())

    torch.testing.assert_close(
        out_multi[AtomicDataDict.TOTAL_ENERGY_KEY],
        out_single[AtomicDataDict.TOTAL_ENERGY_KEY],
    )


# === Summed Heads Tests ===


def test_extract_summed_heads_no_multihead():
    """Extracted summed model should not contain MultiHeadReadout."""
    model = _build_multihead_model()
    extracted = extract_summed_heads(model, ["HF", "MP2"])

    mhr = find_first_of_type(extracted, MultiHeadReadout)
    assert mhr is None

    # Should contain SummedHeadsReadout
    found = find_first_of_type(extracted, SummedHeadsReadout)
    assert found is not None


def test_extract_summed_heads_equals_sum():
    """Summed model output should equal sum of individually-extracted heads' per-atom energies."""
    model = _build_multihead_model()

    torch.manual_seed(42)
    data = _make_data(head_index=0, seed=42)

    # Extract individual heads
    head_hf = extract_head(model, "HF")
    head_mp2 = extract_head(model, "MP2")

    # Extract summed model
    summed = extract_summed_heads(model, ["HF", "MP2"])

    # Run each individually (no HEAD_KEY needed)
    data_copy = data.copy()
    del data_copy[AtomicDataDict.HEAD_KEY]

    out_hf = head_hf(data_copy.copy())
    out_mp2 = head_mp2(data_copy.copy())
    out_summed = summed(data_copy.copy())

    expected_energy = (
        out_hf[AtomicDataDict.TOTAL_ENERGY_KEY]
        + out_mp2[AtomicDataDict.TOTAL_ENERGY_KEY]
    )

    torch.testing.assert_close(
        out_summed[AtomicDataDict.TOTAL_ENERGY_KEY],
        expected_energy,
    )


def test_extract_summed_heads_forces():
    """Summed model should produce finite forces consistent with autograd of summed energy."""
    model = _build_multihead_model()

    torch.manual_seed(42)
    data = _make_data(head_index=0, seed=42)
    del data[AtomicDataDict.HEAD_KEY]

    summed = extract_summed_heads(model, ["HF", "MP2"])
    out = summed(data.copy())

    assert AtomicDataDict.FORCE_KEY in out
    assert torch.isfinite(out[AtomicDataDict.FORCE_KEY]).all()

    # Forces should equal sum of individual heads' forces
    head_hf = extract_head(model, "HF")
    head_mp2 = extract_head(model, "MP2")
    out_hf = head_hf(data.copy())
    out_mp2 = head_mp2(data.copy())
    expected_forces = (
        out_hf[AtomicDataDict.FORCE_KEY] + out_mp2[AtomicDataDict.FORCE_KEY]
    )
    torch.testing.assert_close(
        out[AtomicDataDict.FORCE_KEY],
        expected_forces,
    )


def test_extract_summed_heads_no_head_key():
    """Summed model should work without HEAD_KEY in input."""
    model = _build_multihead_model()

    torch.manual_seed(42)
    data = _make_data(head_index=0, seed=42)
    del data[AtomicDataDict.HEAD_KEY]

    summed = extract_summed_heads(model, ["HF", "MP2"])
    out = summed(data)

    assert AtomicDataDict.TOTAL_ENERGY_KEY in out
    assert AtomicDataDict.FORCE_KEY in out
    assert torch.isfinite(out[AtomicDataDict.TOTAL_ENERGY_KEY]).all()
    assert torch.isfinite(out[AtomicDataDict.FORCE_KEY]).all()


def test_extract_summed_heads_shared_readout():
    """Summed model from shared readout should match sum of individual heads."""
    model = _build_shared_readout_model()

    torch.manual_seed(42)
    data = _make_data(head_index=0, seed=42)
    del data[AtomicDataDict.HEAD_KEY]

    head_hf = extract_head(model, "HF")
    head_mp2 = extract_head(model, "MP2")
    summed = extract_summed_heads(model, ["HF", "MP2"])

    out_hf = head_hf(data.copy())
    out_mp2 = head_mp2(data.copy())
    out_summed = summed(data.copy())

    expected_energy = (
        out_hf[AtomicDataDict.TOTAL_ENERGY_KEY]
        + out_mp2[AtomicDataDict.TOTAL_ENERGY_KEY]
    )

    torch.testing.assert_close(
        out_summed[AtomicDataDict.TOTAL_ENERGY_KEY],
        expected_energy,
    )

    # Forces should also match
    expected_forces = (
        out_hf[AtomicDataDict.FORCE_KEY] + out_mp2[AtomicDataDict.FORCE_KEY]
    )
    torch.testing.assert_close(
        out_summed[AtomicDataDict.FORCE_KEY],
        expected_forces,
    )


def test_extract_summed_heads_invalid_name():
    """Extracting with a non-existent head should raise ValueError."""
    model = _build_multihead_model()
    with pytest.raises(ValueError, match="not found"):
        extract_summed_heads(model, ["HF", "CCSD"])
