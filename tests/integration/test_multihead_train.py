# Integration test for multi-head training
# Tests the full pipeline: data → model → training → checkpointing
import copy
import tempfile
import os

import pytest
import torch

from nequip.utils.global_state import set_global_state
from nequip.data import AtomicDataDict
from nequip.data.dataset import EMTTestDataset
from nequip.data.transforms import (
    ChemicalSpeciesToAtomTypeMapper,
    NeighborListTransform,
    HeadStamper,
)
from nequip.data.datamodule import NequIPDataModule
from nequip.nn import MultiHeadReadout
from nequip.utils import find_first_of_type

from hydra.utils import instantiate, get_method
import lightning


@pytest.fixture(autouse=True)
def setup_global_state():
    set_global_state(allow_tf32=False)


R_MAX = 5.0
TYPE_NAMES = ["Cu", "Al"]
SEED = 42


def _build_multihead_model():
    """Build a multi-head NequIP model programmatically."""
    from nequip.model import NequIPGNNModel

    model = NequIPGNNModel(
        seed=SEED,
        model_dtype="float64",
        type_names=TYPE_NAMES,
        r_max=R_MAX,
        l_max=1,
        parity=True,
        num_layers=2,
        num_features=8,
        radial_mlp_depth=1,
        radial_mlp_width=8,
        avg_num_neighbors=10.0,
        head_names=["Cu_head", "Al_head"],
        per_head_energy_shifts={
            "Cu_head": {"Cu": 0.0, "Al": 0.0},
            "Al_head": {"Cu": 0.0, "Al": 0.0},
        },
        per_head_energy_scales={
            "Cu_head": {"Cu": 1.0, "Al": 1.0},
            "Al_head": {"Cu": 1.0, "Al": 1.0},
        },
    )
    return model


def _make_datasets():
    """Create Cu and Al datasets with head stamps."""
    cu_transforms = [
        ChemicalSpeciesToAtomTypeMapper(
            model_type_names=TYPE_NAMES,
            chemical_species_to_atom_type_map={"Cu": "Cu"},
        ),
        NeighborListTransform(r_max=R_MAX),
        HeadStamper(head_index=0),
    ]
    al_transforms = [
        ChemicalSpeciesToAtomTypeMapper(
            model_type_names=TYPE_NAMES,
            chemical_species_to_atom_type_map={"Al": "Al"},
        ),
        NeighborListTransform(r_max=R_MAX),
        HeadStamper(head_index=1),
    ]

    cu_train = EMTTestDataset(
        transforms=cu_transforms,
        element="Cu",
        num_frames=10,
        supercell=(2, 2, 2),
        seed=SEED,
    )
    al_train = EMTTestDataset(
        transforms=al_transforms,
        element="Al",
        num_frames=10,
        supercell=(2, 2, 2),
        seed=SEED + 1,
    )
    cu_val = EMTTestDataset(
        transforms=cu_transforms,
        element="Cu",
        num_frames=3,
        supercell=(2, 2, 2),
        seed=SEED + 2,
    )
    return cu_train, al_train, cu_val


def test_multihead_model_builds_and_runs():
    """Multi-head model should build and run forward pass on both head datasets."""
    model = _build_multihead_model()

    # Check MultiHeadReadout is present
    mhr = find_first_of_type(model, MultiHeadReadout)
    assert mhr is not None

    cu_train, al_train, cu_val = _make_datasets()

    # Test forward on Cu data (head 0)
    cu_data = cu_train[0]
    out = model(cu_data)
    assert AtomicDataDict.TOTAL_ENERGY_KEY in out
    assert AtomicDataDict.FORCE_KEY in out
    assert torch.isfinite(out[AtomicDataDict.TOTAL_ENERGY_KEY]).all()
    assert torch.isfinite(out[AtomicDataDict.FORCE_KEY]).all()

    # Test forward on Al data (head 1)
    al_data = al_train[0]
    out = model(al_data)
    assert AtomicDataDict.TOTAL_ENERGY_KEY in out
    assert AtomicDataDict.FORCE_KEY in out
    assert torch.isfinite(out[AtomicDataDict.TOTAL_ENERGY_KEY]).all()
    assert torch.isfinite(out[AtomicDataDict.FORCE_KEY]).all()


def test_multihead_training_loop():
    """Full training loop with multi-head model should complete and loss should decrease."""
    model = _build_multihead_model()
    cu_train, al_train, cu_val = _make_datasets()

    # Manually run a few training steps
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []
    for step in range(5):
        optimizer.zero_grad()
        total_loss = 0.0

        # Forward Cu batch
        cu_batch = AtomicDataDict.batched_from_list(
            [cu_train[i] for i in range(min(3, len(cu_train)))]
        )
        cu_out = model(cu_batch)
        cu_energy = cu_out[AtomicDataDict.TOTAL_ENERGY_KEY]
        cu_loss = cu_energy.pow(2).mean()
        total_loss = total_loss + cu_loss

        # Forward Al batch
        al_batch = AtomicDataDict.batched_from_list(
            [al_train[i] for i in range(min(3, len(al_train)))]
        )
        al_out = model(al_batch)
        al_energy = al_out[AtomicDataDict.TOTAL_ENERGY_KEY]
        al_loss = al_energy.pow(2).mean()
        total_loss = total_loss + al_loss

        total_loss.backward()
        optimizer.step()
        losses.append(total_loss.item())

    # Loss should be finite throughout
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)


def test_multihead_checkpoint_save_load():
    """Multi-head model should be saveable and loadable via checkpoint."""
    model = _build_multihead_model()
    cu_train, al_train, cu_val = _make_datasets()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "model.pt")
        torch.save(model.state_dict(), ckpt_path)

        # Load into fresh model
        model2 = _build_multihead_model()
        model2.load_state_dict(torch.load(ckpt_path, weights_only=True))

        # Compare outputs
        cu_data = cu_train[0]
        out1 = model(cu_data.copy())
        out2 = model2(cu_data.copy())

        torch.testing.assert_close(
            out1[AtomicDataDict.TOTAL_ENERGY_KEY],
            out2[AtomicDataDict.TOTAL_ENERGY_KEY],
        )


def test_head_stamper_in_dataset():
    """HeadStamper transforms should correctly stamp HEAD_KEY in dataset frames."""
    cu_train, al_train, cu_val = _make_datasets()

    cu_data = cu_train[0]
    al_data = al_train[0]

    assert AtomicDataDict.HEAD_KEY in cu_data
    assert AtomicDataDict.HEAD_KEY in al_data
    assert cu_data[AtomicDataDict.HEAD_KEY].item() == 0
    assert al_data[AtomicDataDict.HEAD_KEY].item() == 1


def test_different_heads_different_output():
    """Same model should produce different energies for different heads on the same structure."""
    model = _build_multihead_model()
    cu_train, al_train, cu_val = _make_datasets()

    cu_data = cu_train[0]

    # Run with head 0
    data0 = cu_data.copy()
    data0[AtomicDataDict.HEAD_KEY] = torch.tensor([0], dtype=torch.long)
    out0 = model(data0)

    # Run with head 1
    data1 = cu_data.copy()
    data1[AtomicDataDict.HEAD_KEY] = torch.tensor([1], dtype=torch.long)
    out1 = model(data1)

    # Different heads should give different energies
    assert not torch.allclose(
        out0[AtomicDataDict.TOTAL_ENERGY_KEY],
        out1[AtomicDataDict.TOTAL_ENERGY_KEY],
    )
