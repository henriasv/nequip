# Test multi-head model building and forward pass
import copy
import pytest
import torch

from nequip.data import AtomicDataDict
from nequip.data.dict import from_dict
from nequip.data.transforms import ChemicalSpeciesToAtomTypeMapper, NeighborListTransform
from nequip.nn import MultiHeadReadout, ForceStressOutput
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
    "per_type_energy_shifts": {"H": 1.0, "C": 2.0},
}


def _build_model(extra_config=None):
    config = {
        "_target_": "nequip.model.NequIPGNNModel",
        "l_max": 1,
        "parity": True,
        "num_features": 8,
        "num_layers": 2,
        "radial_mlp_depth": 1,
        "radial_mlp_width": 8,
        **BASIC_INFO,
    }
    if extra_config:
        config.update(extra_config)
    config = copy.deepcopy(config)
    builder = get_method(config.pop("_target_"))
    return builder(**config)


def _make_data(head_index=0, num_atoms=4, seed=42):
    """Create test data with positions, types, edges, and HEAD_KEY."""
    rng = torch.Generator().manual_seed(seed)
    type_mapper = ChemicalSpeciesToAtomTypeMapper(
        model_type_names=["H", "C"],
        chemical_species_to_atom_type_map={"H": "H", "C": "C"},
    )
    nl = NeighborListTransform(r_max=4.0)

    data = from_dict(
        {
            "pos": torch.randn(num_atoms, 3, generator=rng, dtype=torch.float64) * 2.0,
            "atomic_numbers": torch.tensor([1, 6, 1, 6], dtype=torch.long),
            "cell": torch.eye(3, dtype=torch.float64).unsqueeze(0) * 10.0,
            "pbc": torch.tensor([[True, True, True]]),
        }
    )
    data = type_mapper(data)
    data = nl(data)
    data[AtomicDataDict.HEAD_KEY] = torch.tensor([head_index], dtype=torch.long)
    return data


def test_multihead_model_builds():
    """Model with head_names should build without error and contain MultiHeadReadout."""
    model = _build_model(
        {
            "head_names": ["HF", "MP2"],
            "per_head_energy_scales": {"HF": 1.0, "MP2": 1.0},
            "per_head_energy_shifts": {"HF": {"H": 1.0, "C": 2.0}, "MP2": {"H": 0.5, "C": 1.0}},
        }
    )
    # Check that MultiHeadReadout is in the model
    mhr = find_first_of_type(model, MultiHeadReadout)
    assert mhr is not None


def test_multihead_model_forward():
    """Multi-head model forward pass produces correct output keys."""
    model = _build_model(
        {
            "head_names": ["HF", "MP2"],
            "per_head_energy_scales": {"HF": 1.0, "MP2": 1.0},
            "per_head_energy_shifts": {"HF": {"H": 1.0, "C": 2.0}, "MP2": {"H": 0.5, "C": 1.0}},
        }
    )
    data = _make_data(head_index=0)
    out = model(data)

    assert AtomicDataDict.TOTAL_ENERGY_KEY in out
    assert AtomicDataDict.FORCE_KEY in out


def test_multihead_forces_correct():
    """Forces from multi-head model should be finite and have correct shape."""
    model = _build_model(
        {
            "head_names": ["HF", "MP2"],
            "per_head_energy_scales": {"HF": 1.0, "MP2": 1.0},
            "per_head_energy_shifts": {"HF": {"H": 1.0, "C": 2.0}, "MP2": {"H": 0.5, "C": 1.0}},
        }
    )
    data = _make_data(head_index=0)
    out = model(data)

    forces = out[AtomicDataDict.FORCE_KEY]
    assert forces.shape == (4, 3)
    assert torch.isfinite(forces).all()


def test_multihead_different_heads_different_output():
    """Different head indices should produce different energies."""
    model = _build_model(
        {
            "head_names": ["HF", "MP2"],
            "per_head_energy_scales": {"HF": 1.0, "MP2": 1.0},
            "per_head_energy_shifts": {"HF": {"H": 1.0, "C": 2.0}, "MP2": {"H": 0.5, "C": 1.0}},
        }
    )
    data0 = _make_data(head_index=0, seed=42)
    data1 = _make_data(head_index=1, seed=42)

    out0 = model(data0)
    out1 = model(data1)

    # Different heads should give different energies (different shifts at minimum)
    assert not torch.allclose(
        out0[AtomicDataDict.TOTAL_ENERGY_KEY],
        out1[AtomicDataDict.TOTAL_ENERGY_KEY],
    )


def test_no_head_names_is_unchanged():
    """Model without head_names should be identical to current behavior."""
    model = _build_model()
    # Should NOT have MultiHeadReadout
    mhr = find_first_of_type(model, MultiHeadReadout)
    assert mhr is None

    # Should still produce output
    data = _make_data(head_index=0)
    out = model(data)
    assert AtomicDataDict.TOTAL_ENERGY_KEY in out
    assert AtomicDataDict.FORCE_KEY in out


def test_force_stress_output_wraps_multihead():
    """ForceStressOutput should wrap the multi-head model correctly."""
    model = _build_model(
        {
            "head_names": ["HF", "MP2"],
            "per_head_energy_scales": {"HF": 1.0, "MP2": 1.0},
            "per_head_energy_shifts": {"HF": {"H": 1.0, "C": 2.0}, "MP2": {"H": 0.5, "C": 1.0}},
        }
    )
    fso = find_first_of_type(model, ForceStressOutput)
    assert fso is not None
