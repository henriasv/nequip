# Test PySCFMultiHeadTestDataset
import pytest
import torch

pyscf = pytest.importorskip("pyscf")

from nequip.data import AtomicDataDict
from nequip.data.dataset.pyscf_test_data import PySCFMultiHeadTestDataset


@pytest.fixture
def h2o_hf_dataset():
    return PySCFMultiHeadTestDataset(
        molecule="H2O",
        method="hf",
        head_index=0,
        num_frames=2,
        seed=42,
    )


@pytest.fixture
def h2o_mp2_dataset():
    return PySCFMultiHeadTestDataset(
        molecule="H2O",
        method="mp2",
        head_index=1,
        num_frames=2,
        seed=42,
    )


def test_dataset_length(h2o_hf_dataset):
    assert len(h2o_hf_dataset) == 2


def test_dataset_correct_shapes(h2o_hf_dataset):
    """Datasets should generate correct shapes for H2O (3 atoms)."""
    data = h2o_hf_dataset[0]
    assert data[AtomicDataDict.POSITIONS_KEY].shape == (3, 3)
    assert data[AtomicDataDict.FORCE_KEY].shape == (3, 3)
    assert data[AtomicDataDict.TOTAL_ENERGY_KEY].numel() == 1


def test_forces_are_finite(h2o_hf_dataset):
    """Forces should be finite."""
    for i in range(len(h2o_hf_dataset)):
        data = h2o_hf_dataset[i]
        assert torch.isfinite(data[AtomicDataDict.FORCE_KEY]).all()


def test_head_key_stamped(h2o_hf_dataset, h2o_mp2_dataset):
    """HEAD_KEY should be stamped correctly."""
    data_hf = h2o_hf_dataset[0]
    data_mp2 = h2o_mp2_dataset[0]
    assert data_hf[AtomicDataDict.HEAD_KEY].item() == 0
    assert data_mp2[AtomicDataDict.HEAD_KEY].item() == 1


def test_hf_mp2_energies_differ(h2o_hf_dataset, h2o_mp2_dataset):
    """HF and MP2 energies should differ on the same geometry."""
    e_hf = h2o_hf_dataset[0][AtomicDataDict.TOTAL_ENERGY_KEY]
    e_mp2 = h2o_mp2_dataset[0][AtomicDataDict.TOTAL_ENERGY_KEY]
    # Same seed, same geometry -> different theory level -> different energy
    assert not torch.allclose(e_hf, e_mp2)


def test_ch4_dataset():
    """CH4 molecule dataset should work."""
    ds = PySCFMultiHeadTestDataset(
        molecule="CH4",
        method="hf",
        head_index=0,
        num_frames=1,
        seed=42,
    )
    assert len(ds) == 1
    data = ds[0]
    assert data[AtomicDataDict.POSITIONS_KEY].shape == (5, 3)
    assert data[AtomicDataDict.FORCE_KEY].shape == (5, 3)
    assert torch.isfinite(data[AtomicDataDict.FORCE_KEY]).all()
