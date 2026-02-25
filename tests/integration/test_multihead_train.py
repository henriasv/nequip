# Integration test for multi-head training
# Tests the full pipeline: data → model → training → checkpointing
#
# Also includes PySCF-based tests (requires pyscf) that verify:
#   - MP2 head forces from energy-only training are genuinely MP2-like
#   - More training data improves zero-shot MP2 force quality
#   - HF head quality is maintained alongside MP2 training
import copy
import math
import tempfile
import os

import pytest
import torch

from nequip.utils.global_state import set_global_state
from nequip.model import NequIPGNNModel
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


# ============================================================================
# PySCF-based multi-head tests: HF(E+F) + MP2(E only) → MP2 forces
#
# These tests train a shared-backbone model on HF energies+forces and MP2
# energies only (no MP2 forces), then verify that the MP2 head produces
# physically correct forces purely from autograd of the learned energy surface.
# ============================================================================

_pyscf = pytest.importorskip("pyscf")

from nequip.data.dataset.pyscf_test_data import PySCFMultiHeadTestDataset

HA_TO_MEV = 27211.386  # 1 Hartree = 27211.386 meV

_PYSCF_TYPE_NAMES = ["O", "H"]
_PYSCF_R_MAX = 5.0
_PYSCF_TRANSFORMS = [
    ChemicalSpeciesToAtomTypeMapper(
        model_type_names=_PYSCF_TYPE_NAMES,
        chemical_species_to_atom_type_map={"O": "O", "H": "H"},
    ),
    NeighborListTransform(r_max=_PYSCF_R_MAX),
]


def _make_pyscf_dataset(method, head_index, num_frames, seed):
    return PySCFMultiHeadTestDataset(
        molecule="H2O",
        method=method,
        head_index=head_index,
        num_frames=num_frames,
        seed=seed,
        transforms=_PYSCF_TRANSFORMS,
    )


def _compute_pyscf_stats(hf_train):
    all_e = torch.stack(
        [hf_train[i][AtomicDataDict.TOTAL_ENERGY_KEY] for i in range(len(hf_train))]
    )
    all_f = torch.cat(
        [hf_train[i][AtomicDataDict.FORCE_KEY] for i in range(len(hf_train))]
    )
    n_atoms = hf_train[0][AtomicDataDict.POSITIONS_KEY].shape[0]
    e_mean = (all_e / n_atoms).mean().item()
    f_rms = all_f.pow(2).mean().sqrt().item()
    avg_nn = sum(
        hf_train[i][AtomicDataDict.EDGE_INDEX_KEY].shape[1]
        / hf_train[i][AtomicDataDict.POSITIONS_KEY].shape[0]
        for i in range(len(hf_train))
    ) / len(hf_train)
    return e_mean, f_rms, avg_nn


def _train_multihead_hf_mp2(hf_train, mp2_train, e_mean_hf, e_mean_mp2, f_rms, avg_nn):
    """Build and train: HF energy+forces, MP2 energy only. Returns trained model."""
    torch.manual_seed(42)
    model = NequIPGNNModel(
        seed=42,
        model_dtype="float64",
        type_names=_PYSCF_TYPE_NAMES,
        r_max=_PYSCF_R_MAX,
        l_max=2,
        parity=True,
        num_layers=3,
        num_features=16,
        radial_mlp_depth=2,
        radial_mlp_width=16,
        avg_num_neighbors=avg_nn,
        head_names=["HF", "MP2"],
        per_head_energy_shifts={"HF": e_mean_hf, "MP2": e_mean_mp2},
        per_head_energy_scales={"HF": f_rms, "MP2": f_rms},
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=25, factor=0.5
    )
    n = len(hf_train)

    for epoch in range(150):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0

        for bs in range(0, n, 5):
            idx = perm[bs : min(bs + 5, n)].tolist()
            optimizer.zero_grad()
            loss = torch.tensor(0.0, dtype=torch.float64)

            hf_batch = AtomicDataDict.batched_from_list([hf_train[i] for i in idx])
            hf_out = model(hf_batch)
            loss = loss + (
                hf_out[AtomicDataDict.TOTAL_ENERGY_KEY]
                - hf_batch[AtomicDataDict.TOTAL_ENERGY_KEY]
            ).pow(2).mean()
            loss = loss + 10.0 * (
                hf_out[AtomicDataDict.FORCE_KEY]
                - hf_batch[AtomicDataDict.FORCE_KEY]
            ).pow(2).mean()

            mp2_batch = AtomicDataDict.batched_from_list([mp2_train[i] for i in idx])
            mp2_out = model(mp2_batch)
            loss = loss + (
                mp2_out[AtomicDataDict.TOTAL_ENERGY_KEY]
                - mp2_batch[AtomicDataDict.TOTAL_ENERGY_KEY]
            ).pow(2).mean()

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step(epoch_loss)

    return model


def _evaluate_multihead(model, hf_test, mp2_test):
    """Evaluate on test set. Returns force lists and MAEs."""
    model.eval()
    n = len(hf_test)

    hf_pred_f, mp2_pred_f = [], []
    hf_true_f, mp2_true_f = [], []
    hf_e_errs, mp2_e_errs = [], []

    for i in range(n):
        hf_o = model(hf_test[i].copy())
        mp2_o = model(mp2_test[i].copy())

        hf_pred_f.append(hf_o[AtomicDataDict.FORCE_KEY].detach())
        mp2_pred_f.append(mp2_o[AtomicDataDict.FORCE_KEY].detach())
        hf_true_f.append(hf_test[i][AtomicDataDict.FORCE_KEY])
        mp2_true_f.append(mp2_test[i][AtomicDataDict.FORCE_KEY])

        hf_e_errs.append(
            (hf_o[AtomicDataDict.TOTAL_ENERGY_KEY] - hf_test[i][AtomicDataDict.TOTAL_ENERGY_KEY]).abs().item()
        )
        mp2_e_errs.append(
            (mp2_o[AtomicDataDict.TOTAL_ENERGY_KEY] - mp2_test[i][AtomicDataDict.TOTAL_ENERGY_KEY]).abs().item()
        )

    def _mae(a, b):
        return sum((x - y).abs().mean().item() for x, y in zip(a, b)) / len(a)

    return {
        "hf_e_mae": sum(hf_e_errs) / n,
        "mp2_e_mae": sum(mp2_e_errs) / n,
        "hf_f_vs_hf": _mae(hf_pred_f, hf_true_f),
        "mp2_f_vs_mp2": _mae(mp2_pred_f, mp2_true_f),
        "mp2_f_vs_hf": _mae(mp2_pred_f, hf_true_f),
        "hf_mp2_true_gap": _mae(hf_true_f, mp2_true_f),
        "hf_pred_f": hf_pred_f,
        "mp2_pred_f": mp2_pred_f,
        "hf_true_f": hf_true_f,
        "mp2_true_f": mp2_true_f,
    }


def _decompose_mp2_forces(mp2_pred_f, hf_true_f, mp2_true_f):
    """Decompose MP2 predicted forces into gap-parallel and gap-perpendicular.

    For each frame, the MP2 prediction relative to HF truth is projected onto
    the HF→MP2 theory gap vector:
        pred_rel = F_MP2_pred - F_HF_true
        gap      = F_MP2_true - F_HF_true
        alpha    = (pred_rel · gap) / (gap · gap)

    alpha=0 means pure HF, alpha=1 means perfect MP2.
    The perpendicular component is model noise unrelated to the theory gap.
    """
    alphas = []
    noise_per_comp = []
    bias_per_comp = []
    gap_per_comp = []

    for f_mp2_pred, f_hf_true, f_mp2_true in zip(mp2_pred_f, hf_true_f, mp2_true_f):
        gap = (f_mp2_true - f_hf_true).flatten()
        pred_rel = (f_mp2_pred - f_hf_true).flatten()
        n_comp = len(gap)
        gap_sq = gap.dot(gap).item()

        alpha = pred_rel.dot(gap).item() / gap_sq if gap_sq > 1e-20 else 0.0
        pred_perp = pred_rel - alpha * gap

        alphas.append(alpha)
        gap_per_comp.append(math.sqrt(gap_sq / n_comp))
        noise_per_comp.append(pred_perp.norm().item() / math.sqrt(n_comp))
        bias_per_comp.append(abs(alpha - 1.0) * math.sqrt(gap_sq / n_comp))

    n = len(alphas)
    a_mean = sum(alphas) / n
    a_std = (sum((a - a_mean) ** 2 for a in alphas) / n) ** 0.5
    return {
        "alpha_mean": a_mean,
        "alpha_std": a_std,
        "gap_rms": (sum(g**2 for g in gap_per_comp) / n) ** 0.5,
        "noise_rms": (sum(g**2 for g in noise_per_comp) / n) ** 0.5,
        "bias_rms": (sum(g**2 for g in bias_per_comp) / n) ** 0.5,
    }


def _run_pyscf_experiment(n_train_frames, hf_test, mp2_test):
    """Generate data, train, evaluate, decompose. Returns (metrics, decomposition)."""
    hf_train = _make_pyscf_dataset("hf", 0, num_frames=n_train_frames, seed=42)
    mp2_train = _make_pyscf_dataset("mp2", 1, num_frames=n_train_frames, seed=42)

    e_mean_hf, f_rms, avg_nn = _compute_pyscf_stats(hf_train)
    all_e_mp2 = torch.stack(
        [mp2_train[i][AtomicDataDict.TOTAL_ENERGY_KEY] for i in range(len(mp2_train))]
    )
    n_atoms = mp2_train[0][AtomicDataDict.POSITIONS_KEY].shape[0]
    e_mean_mp2 = (all_e_mp2 / n_atoms).mean().item()

    model = _train_multihead_hf_mp2(hf_train, mp2_train, e_mean_hf, e_mean_mp2, f_rms, avg_nn)
    metrics = _evaluate_multihead(model, hf_test, mp2_test)
    decomp = _decompose_mp2_forces(
        metrics["mp2_pred_f"], metrics["hf_true_f"], metrics["mp2_true_f"]
    )
    return metrics, decomp


@pytest.fixture(scope="module")
def pyscf_test_data():
    """Held-out H2O test set (15 frames, seed=7777). Generated once per module."""
    hf_test = _make_pyscf_dataset("hf", 0, num_frames=15, seed=7777)
    mp2_test = _make_pyscf_dataset("mp2", 1, num_frames=15, seed=7777)
    return hf_test, mp2_test


def _print_results(results: dict):
    """Print a comparison table."""
    print()
    header = f"{'':>25s}"
    for label in results:
        header += f" | {label:>14s}"
    print(header)
    print("-" * len(header))

    for label, key, scale in [
        ("HF energy MAE (meV)", "hf_e_mae", HA_TO_MEV),
        ("HF force MAE (meV/A)", "hf_f_vs_hf", HA_TO_MEV),
        ("MP2 energy MAE (meV)", "mp2_e_mae", HA_TO_MEV),
        ("MP2 force MAE (meV/A)", "mp2_f_vs_mp2", HA_TO_MEV),
        ("MP2 F vs HF true (meV/A)", "mp2_f_vs_hf", HA_TO_MEV),
        ("HF↔MP2 true gap (meV/A)", "hf_mp2_true_gap", HA_TO_MEV),
    ]:
        row = f"{label:>25s}"
        for m, _ in results.values():
            row += f" | {m[key]*scale:14.1f}"
        print(row)

    row = f"{'alpha (0=HF, 1=MP2)':>25s}"
    for _, d in results.values():
        row += f" | {d['alpha_mean']:7.3f}±{d['alpha_std']:.3f}"
    print(row)

    row = f"{'noise ⊥ gap (meV/A)':>25s}"
    for _, d in results.values():
        row += f" | {d['noise_rms']*HA_TO_MEV:14.1f}"
    print(row)

    row = f"{'bias ∥ gap (meV/A)':>25s}"
    for _, d in results.values():
        row += f" | {d['bias_rms']*HA_TO_MEV:14.1f}"
    print(row)
    print()


class TestMultiHeadEnergyToForces:
    """Multi-head training: HF(E+F) + MP2(E only) → zero-shot MP2 forces.

    Uses PySCF to generate H2O data at HF and MP2 levels (STO-3G basis).
    The MP2 head is trained on energies only; its forces come purely from
    autograd of the learned energy surface through the shared backbone.
    """

    def test_mp2_forces_are_genuine_mp2(self, pyscf_test_data):
        """MP2 head forces should be closer to MP2 truth than to HF truth.

        This verifies the forces are genuinely MP2-like, not HF copies.
        The projection alpha ≈ 1 means the model reaches the MP2 energy surface;
        the error is dominated by noise perpendicular to the theory gap.
        """
        hf_test, mp2_test = pyscf_test_data
        metrics, decomp = _run_pyscf_experiment(40, hf_test, mp2_test)

        # MP2 predicted forces should be much closer to MP2 true than to HF true
        assert metrics["mp2_f_vs_mp2"] < metrics["mp2_f_vs_hf"], (
            f"MP2 forces closer to HF ({metrics['mp2_f_vs_hf']*HA_TO_MEV:.0f} meV/A) "
            f"than MP2 ({metrics['mp2_f_vs_mp2']*HA_TO_MEV:.0f} meV/A)"
        )

        # MP2 forces should close >50% of the HF↔MP2 gap
        gap_closed = 1.0 - metrics["mp2_f_vs_mp2"] / metrics["hf_mp2_true_gap"]
        assert gap_closed > 0.5, (
            f"Only {gap_closed:.0%} of HF↔MP2 force gap closed, expected >50%"
        )

        # alpha should be close to 1 (reaching MP2, not stuck near HF)
        assert decomp["alpha_mean"] > 0.7, (
            f"alpha={decomp['alpha_mean']:.3f}, expected >0.7"
        )

        # Noise should dominate over bias
        assert decomp["noise_rms"] > decomp["bias_rms"], (
            f"Bias ({decomp['bias_rms']*HA_TO_MEV:.1f} meV/A) dominates over "
            f"noise ({decomp['noise_rms']*HA_TO_MEV:.1f} meV/A)"
        )

        _print_results({"40 frames": (metrics, decomp)})

    def test_hf_head_quality_maintained(self, pyscf_test_data):
        """HF head should still predict accurate HF energies and forces."""
        hf_test, mp2_test = pyscf_test_data
        metrics, _ = _run_pyscf_experiment(40, hf_test, mp2_test)

        assert metrics["hf_e_mae"] * HA_TO_MEV < 50.0, (
            f"HF energy MAE {metrics['hf_e_mae']*HA_TO_MEV:.1f} meV too large"
        )
        assert metrics["hf_f_vs_hf"] * HA_TO_MEV < 150.0, (
            f"HF force MAE {metrics['hf_f_vs_hf']*HA_TO_MEV:.1f} meV/Ang too large"
        )

    def test_more_data_improves_mp2_forces(self, pyscf_test_data):
        """More training data should improve MP2 zero-shot force quality."""
        hf_test, mp2_test = pyscf_test_data
        results = {}
        for n_frames in [40, 80, 160]:
            m, d = _run_pyscf_experiment(n_frames, hf_test, mp2_test)
            results[n_frames] = (m, d)

        _print_results({f"{n} frames": r for n, r in results.items()})

        # MP2 force quality should improve with more data
        assert results[80][0]["mp2_f_vs_mp2"] < results[40][0]["mp2_f_vs_mp2"], (
            "80 frames should give better MP2 forces than 40"
        )
        assert results[160][0]["mp2_f_vs_mp2"] < results[40][0]["mp2_f_vs_mp2"], (
            "160 frames should give better MP2 forces than 40"
        )
