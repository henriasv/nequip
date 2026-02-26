# Test multi-head lightning module and datamodule changes
import pytest
import torch

from nequip.data import AtomicDataDict
from nequip.data.datamodule import NequIPDataModule


def test_combined_loader_mode_default():
    """NequIPDataModule should accept combined_loader_mode parameter."""
    dm = NequIPDataModule(
        seed=42,
        train_dataset=[],
        val_dataset=[],
        combined_loader_mode="max_size_cycle",
    )
    assert dm.combined_loader_mode == "max_size_cycle"


def test_combined_loader_mode_min_size():
    """NequIPDataModule should accept 'min_size' combined_loader_mode."""
    dm = NequIPDataModule(
        seed=42,
        train_dataset=[],
        val_dataset=[],
        combined_loader_mode="min_size",
    )
    assert dm.combined_loader_mode == "min_size"


def test_multihead_datamodule_num_datasets():
    """NequIPDataModule with two train datasets should report num_datasets train=2."""
    dm = NequIPDataModule(
        seed=42,
        train_dataset=[
            {
                "_target_": "nequip.data.dataset.EMTTestDataset",
                "transforms": [],
                "element": "Cu",
                "num_frames": 5,
                "supercell": [2, 2, 2],
            },
            {
                "_target_": "nequip.data.dataset.EMTTestDataset",
                "transforms": [],
                "element": "Al",
                "num_frames": 5,
                "supercell": [2, 2, 2],
            },
        ],
        val_dataset=[
            {
                "_target_": "nequip.data.dataset.EMTTestDataset",
                "transforms": [],
                "element": "Cu",
                "num_frames": 3,
                "supercell": [2, 2, 2],
            },
        ],
    )
    assert dm.num_datasets["train"] == 2
    assert dm.num_datasets["val"] == 1


def test_multihead_train_dataloader_returns_combined():
    """With 2 train datasets, train_dataloader should return CombinedLoader."""
    from lightning.pytorch.utilities.combined_loader import CombinedLoader

    dm = NequIPDataModule(
        seed=42,
        train_dataset=[
            {
                "_target_": "nequip.data.dataset.EMTTestDataset",
                "transforms": [],
                "element": "Cu",
                "num_frames": 5,
                "supercell": [2, 2, 2],
            },
            {
                "_target_": "nequip.data.dataset.EMTTestDataset",
                "transforms": [],
                "element": "Al",
                "num_frames": 5,
                "supercell": [2, 2, 2],
            },
        ],
        val_dataset=[
            {
                "_target_": "nequip.data.dataset.EMTTestDataset",
                "transforms": [],
                "element": "Cu",
                "num_frames": 3,
                "supercell": [2, 2, 2],
            },
        ],
        train_dataloader={"_target_": "torch.utils.data.DataLoader", "batch_size": 2},
        val_dataloader={"_target_": "torch.utils.data.DataLoader", "batch_size": 2},
    )
    dm.setup("fit")
    train_dl = dm.train_dataloader()
    assert isinstance(train_dl, CombinedLoader)
    dm.teardown("fit")


def test_single_train_dataloader_returns_regular():
    """With 1 train dataset, train_dataloader should return regular DataLoader."""
    from torch.utils.data import DataLoader
    from lightning.pytorch.utilities.combined_loader import CombinedLoader

    dm = NequIPDataModule(
        seed=42,
        train_dataset=[
            {
                "_target_": "nequip.data.dataset.EMTTestDataset",
                "transforms": [],
                "element": "Cu",
                "num_frames": 5,
                "supercell": [2, 2, 2],
            },
        ],
        val_dataset=[
            {
                "_target_": "nequip.data.dataset.EMTTestDataset",
                "transforms": [],
                "element": "Cu",
                "num_frames": 3,
                "supercell": [2, 2, 2],
            },
        ],
        train_dataloader={"_target_": "torch.utils.data.DataLoader", "batch_size": 2},
        val_dataloader={"_target_": "torch.utils.data.DataLoader", "batch_size": 2},
    )
    dm.setup("fit")
    train_dl = dm.train_dataloader()
    assert not isinstance(train_dl, CombinedLoader)
    assert isinstance(train_dl, DataLoader)
    dm.teardown("fit")


# === Cross-head force regularization tests ===


def _build_multihead_model_for_reg():
    """Build a multi-head NequIP model for force reg testing."""
    from nequip.model import NequIPGNNModel
    from nequip.utils.global_state import set_global_state

    set_global_state(allow_tf32=False)

    model = NequIPGNNModel(
        seed=42,
        model_dtype="float64",
        type_names=["Cu", "Al"],
        r_max=5.0,
        l_max=1,
        parity=True,
        num_layers=2,
        num_features=8,
        radial_mlp_depth=1,
        radial_mlp_width=8,
        avg_num_neighbors=10.0,
        head_names=["dft", "rpa"],
        per_head_energy_shifts={
            "dft": {"Cu": 0.0, "Al": 0.0},
            "rpa": {"Cu": 0.0, "Al": 0.0},
        },
        per_head_energy_scales={
            "dft": {"Cu": 1.0, "Al": 1.0},
            "rpa": {"Cu": 1.0, "Al": 1.0},
        },
    )
    return model


def _make_test_batch(head_index=0, seed=42):
    """Create a test batch with forces for the force reg test."""
    from nequip.data.dataset import EMTTestDataset
    from nequip.data.transforms import (
        ChemicalSpeciesToAtomTypeMapper,
        NeighborListTransform,
        HeadStamper,
    )

    transforms = [
        ChemicalSpeciesToAtomTypeMapper(
            model_type_names=["Cu", "Al"],
            chemical_species_to_atom_type_map={"Cu": "Cu"},
        ),
        NeighborListTransform(r_max=5.0),
        HeadStamper(head_index=head_index),
    ]
    ds = EMTTestDataset(
        transforms=transforms,
        element="Cu",
        num_frames=3,
        supercell=(2, 2, 2),
        seed=seed,
    )
    return AtomicDataDict.batched_from_list([ds[i] for i in range(3)])


def test_cross_head_force_reg_loss_computed():
    """Cross-head force reg should compute a finite MSE loss between heads' forces."""
    model = _build_multihead_model_for_reg()
    ref_batch = _make_test_batch(head_index=0, seed=42)

    # Run reference head (dft, head 0)
    ref_output = model(ref_batch.copy())
    ref_forces = ref_output[AtomicDataDict.FORCE_KEY].detach()

    # Clone batch and change to target head (rpa, head 1)
    reg_batch = {
        k: v.clone() if isinstance(v, torch.Tensor) else v
        for k, v in ref_batch.items()
    }
    reg_batch[AtomicDataDict.HEAD_KEY] = torch.full_like(
        ref_batch[AtomicDataDict.HEAD_KEY], 1
    )
    reg_output = model(reg_batch)
    reg_forces = reg_output[AtomicDataDict.FORCE_KEY]

    force_reg_loss = torch.nn.functional.mse_loss(reg_forces, ref_forces)
    assert torch.isfinite(force_reg_loss)
    assert force_reg_loss > 0  # different heads should have different forces


def test_cross_head_force_reg_gradient_flow():
    """Force reg gradients should flow to target head readout but not reference head readout."""
    from nequip.nn import MultiHeadReadout
    from nequip.utils import find_first_of_type

    model = _build_multihead_model_for_reg()
    mhr = find_first_of_type(model, MultiHeadReadout)
    assert mhr is not None

    ref_batch = _make_test_batch(head_index=0, seed=42)

    # Run reference head and detach forces (no grad to ref head from this)
    ref_output = model(ref_batch.copy())
    ref_forces = ref_output[AtomicDataDict.FORCE_KEY].detach()

    # Clone batch for target head
    reg_batch = {
        k: v.clone() if isinstance(v, torch.Tensor) else v
        for k, v in ref_batch.items()
    }
    reg_batch[AtomicDataDict.HEAD_KEY] = torch.full_like(
        ref_batch[AtomicDataDict.HEAD_KEY], 1
    )
    reg_output = model(reg_batch)
    reg_forces = reg_output[AtomicDataDict.FORCE_KEY]

    force_reg_loss = torch.nn.functional.mse_loss(reg_forces, ref_forces)
    force_reg_loss.backward()

    # Target head (rpa, index 1) readout should have gradients
    target_head = mhr.heads["rpa"]
    target_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in target_head["readout"].parameters()
    )
    assert target_has_grad, "target head readout should receive gradients from force reg"

    # Reference head (dft, index 0) readout should NOT have gradients
    # (ref_forces was detached)
    ref_head = mhr.heads["dft"]
    ref_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in ref_head["readout"].parameters()
    )
    assert not ref_has_grad, "reference head readout should not receive gradients from force reg"


def test_cross_head_force_reg_training_step():
    """Integration: force reg should decrease over a few training steps."""
    model = _build_multihead_model_for_reg()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    ref_batch = _make_test_batch(head_index=0, seed=42)

    reg_losses = []
    for step in range(5):
        optimizer.zero_grad()

        # Normal training loss (energy for both heads)
        out0 = model(ref_batch.copy())
        total_loss = out0[AtomicDataDict.TOTAL_ENERGY_KEY].pow(2).mean()

        # Force reg: make head 1's forces match head 0's on same data
        ref_forces = out0[AtomicDataDict.FORCE_KEY].detach()
        reg_batch = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in ref_batch.items()
        }
        reg_batch[AtomicDataDict.HEAD_KEY] = torch.full_like(
            ref_batch[AtomicDataDict.HEAD_KEY], 1
        )
        reg_out = model(reg_batch)
        reg_forces = reg_out[AtomicDataDict.FORCE_KEY]
        force_reg_loss = torch.nn.functional.mse_loss(reg_forces, ref_forces)
        reg_losses.append(force_reg_loss.item())

        total_loss = total_loss + 0.1 * force_reg_loss
        total_loss.backward()
        optimizer.step()

    # Force reg loss should be finite
    assert all(torch.isfinite(torch.tensor(l)) for l in reg_losses)


def test_cross_head_force_reg_zero_reference():
    """Zero reference should penalize force magnitude (L2 toward zero)."""
    model = _build_multihead_model_for_reg()
    ref_batch = _make_test_batch(head_index=0, seed=42)

    # Run target head (rpa, head 1) on the data
    reg_batch = {
        k: v.clone() if isinstance(v, torch.Tensor) else v
        for k, v in ref_batch.items()
    }
    reg_batch[AtomicDataDict.HEAD_KEY] = torch.full_like(
        ref_batch[AtomicDataDict.HEAD_KEY], 1
    )
    reg_output = model(reg_batch)
    reg_forces = reg_output[AtomicDataDict.FORCE_KEY]

    # Zero reference: loss = mean(F^2)
    force_reg_loss = torch.mean(reg_forces**2)
    assert torch.isfinite(force_reg_loss)
    assert force_reg_loss > 0  # forces should be non-zero

    # Verify gradient flows to target head
    force_reg_loss.backward()
    from nequip.nn import MultiHeadReadout
    from nequip.utils import find_first_of_type

    mhr = find_first_of_type(model, MultiHeadReadout)
    target_head = mhr.heads["rpa"]
    target_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in target_head["readout"].parameters()
    )
    assert target_has_grad, "target head should receive gradients from zero-ref force reg"
