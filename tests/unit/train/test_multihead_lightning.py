# Test multi-head lightning module and datamodule changes
import pytest

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
