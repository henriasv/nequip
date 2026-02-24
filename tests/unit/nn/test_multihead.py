# Test MultiHeadReadout module
import torch
import pytest

from nequip.data import AtomicDataDict
from nequip.nn import MultiHeadReadout, ScalarMLP, PerTypeScaleShift, AtomwiseReduce
from nequip.utils.global_state import set_global_state


@pytest.fixture(autouse=True)
def setup_global_state():
    set_global_state(allow_tf32=False)


def _make_data(
    num_atoms=5,
    feature_dim=8,
    head_index=0,
    num_types=2,
    seed=0,
    include_head_key=True,
    num_frames=1,
):
    """Create a minimal data dict suitable for MultiHeadReadout input."""
    rng = torch.Generator().manual_seed(seed)
    if num_frames == 1:
        batch = torch.zeros(num_atoms, dtype=torch.long)
        num_nodes_tensor = torch.tensor([num_atoms], dtype=torch.long)
    else:
        # Split atoms evenly across frames
        atoms_per_frame = num_atoms // num_frames
        batch = torch.cat(
            [torch.full((atoms_per_frame,), i, dtype=torch.long) for i in range(num_frames)]
        )
        num_nodes_tensor = torch.full((num_frames,), atoms_per_frame, dtype=torch.long)

    data = {
        AtomicDataDict.NODE_FEATURES_KEY: torch.randn(
            num_atoms, feature_dim, generator=rng
        ),
        AtomicDataDict.ATOM_TYPE_KEY: torch.randint(
            0, num_types, (num_atoms,), generator=rng
        ),
        AtomicDataDict.POSITIONS_KEY: torch.randn(num_atoms, 3, generator=rng),
        AtomicDataDict.BATCH_KEY: batch,
        AtomicDataDict.NUM_NODES_KEY: num_nodes_tensor,
    }
    if include_head_key:
        if num_frames == 1:
            data[AtomicDataDict.HEAD_KEY] = torch.tensor(
                [head_index], dtype=torch.long
            )
        else:
            # head_index can be a list for multi-frame
            if isinstance(head_index, list):
                data[AtomicDataDict.HEAD_KEY] = torch.tensor(
                    head_index, dtype=torch.long
                )
            else:
                data[AtomicDataDict.HEAD_KEY] = torch.full(
                    (num_frames,), head_index, dtype=torch.long
                )
    return data


def _make_multihead(feature_dim=8, num_types=2, head_names=None):
    """Create a MultiHeadReadout with given params."""
    if head_names is None:
        head_names = ["HF", "MP2"]
    irreps_in = {
        AtomicDataDict.NODE_FEATURES_KEY: f"{feature_dim}x0e",
        AtomicDataDict.PER_ATOM_ENERGY_KEY: "1x0e",
    }
    return MultiHeadReadout(
        head_names=head_names,
        type_names=[f"type{i}" for i in range(num_types)],
        irreps_in=irreps_in,
    )


def test_output_shapes():
    """Output shapes should be correct."""
    mhr = _make_multihead()
    data = _make_data(num_atoms=5, head_index=0)
    out = mhr(data)

    assert AtomicDataDict.PER_ATOM_ENERGY_KEY in out
    assert AtomicDataDict.TOTAL_ENERGY_KEY in out
    assert out[AtomicDataDict.PER_ATOM_ENERGY_KEY].shape == (5, 1)
    assert out[AtomicDataDict.TOTAL_ENERGY_KEY].shape == (1, 1)


def test_different_heads_different_energies():
    """Different head indices should produce different per-atom energies."""
    mhr = _make_multihead()

    data0 = _make_data(num_atoms=5, head_index=0, seed=42)
    data1 = _make_data(num_atoms=5, head_index=1, seed=42)

    out0 = mhr(data0)
    out1 = mhr(data1)

    # With random weights, different heads should give different results
    assert not torch.allclose(
        out0[AtomicDataDict.PER_ATOM_ENERGY_KEY],
        out1[AtomicDataDict.PER_ATOM_ENERGY_KEY],
    )


def test_backward_pass():
    """Backward pass should produce valid gradients for the selected head only."""
    mhr = _make_multihead()

    data = _make_data(num_atoms=5, head_index=0, seed=42)
    # Enable grad for node features
    data[AtomicDataDict.NODE_FEATURES_KEY].requires_grad_(True)

    out = mhr(data)
    total_energy = out[AtomicDataDict.TOTAL_ENERGY_KEY]
    total_energy.sum().backward()

    grad = data[AtomicDataDict.NODE_FEATURES_KEY].grad
    assert grad is not None
    assert torch.isfinite(grad).all()


def test_no_head_key_defaults_to_zero():
    """If HEAD_KEY is absent, should default to head 0."""
    mhr = _make_multihead()

    data_no_head = _make_data(num_atoms=5, head_index=0, seed=42, include_head_key=False)
    data_head0 = _make_data(num_atoms=5, head_index=0, seed=42, include_head_key=True)

    out_no = mhr(data_no_head)
    out_h0 = mhr(data_head0)

    torch.testing.assert_close(
        out_no[AtomicDataDict.PER_ATOM_ENERGY_KEY],
        out_h0[AtomicDataDict.PER_ATOM_ENERGY_KEY],
    )


def test_batched_mixed_heads():
    """Batched data with mixed heads should produce correct results."""
    mhr = _make_multihead()

    # Create 2-frame batch with different heads
    num_atoms = 6  # 3 per frame
    data = _make_data(
        num_atoms=num_atoms,
        head_index=[0, 1],
        num_frames=2,
        seed=42,
    )
    out = mhr(data)

    assert out[AtomicDataDict.PER_ATOM_ENERGY_KEY].shape == (num_atoms, 1)
    assert out[AtomicDataDict.TOTAL_ENERGY_KEY].shape == (2, 1)


def test_single_head_multihead():
    """Single-head MultiHeadReadout should work correctly."""
    mhr = _make_multihead(head_names=["single"])
    data = _make_data(num_atoms=5, head_index=0, seed=42)
    out = mhr(data)

    assert AtomicDataDict.PER_ATOM_ENERGY_KEY in out
    assert AtomicDataDict.TOTAL_ENERGY_KEY in out


def test_single_head_matches_standard():
    """Single-head MultiHeadReadout should match standard ScalarMLP + PerTypeScaleShift + AtomwiseReduce."""
    feature_dim = 8
    num_types = 2
    type_names = [f"type{i}" for i in range(num_types)]

    irreps_in = {
        AtomicDataDict.NODE_FEATURES_KEY: f"{feature_dim}x0e",
        AtomicDataDict.PER_ATOM_ENERGY_KEY: "1x0e",
    }

    # Build single-head MultiHeadReadout
    mhr = MultiHeadReadout(
        head_names=["sole"],
        type_names=type_names,
        irreps_in=irreps_in,
    )

    # Build standard pipeline
    readout = ScalarMLP(
        output_dim=1,
        hidden_layers_depth=0,
        nonlinearity="silu",
        bias=False,
        forward_weight_init=True,
        field=AtomicDataDict.NODE_FEATURES_KEY,
        out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        irreps_in=irreps_in,
    )
    scale_shift = PerTypeScaleShift(
        type_names=type_names,
        field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        irreps_in=readout.irreps_out,
    )
    reduce = AtomwiseReduce(
        irreps_in=scale_shift.irreps_out,
        reduce="sum",
        field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
    )

    # Copy weights from multi-head to standard
    mhr_readout = mhr.heads["sole"]["readout"]
    mhr_ss = mhr.heads["sole"]["scale_shift"]

    readout.load_state_dict(mhr_readout.state_dict())
    scale_shift.load_state_dict(mhr_ss.state_dict())

    # Compare outputs
    data_mhr = _make_data(num_atoms=5, head_index=0, seed=99)
    data_std = _make_data(num_atoms=5, head_index=0, seed=99)

    out_mhr = mhr(data_mhr)

    data_std = readout(data_std)
    data_std = scale_shift(data_std)
    data_std = reduce(data_std)

    torch.testing.assert_close(
        out_mhr[AtomicDataDict.PER_ATOM_ENERGY_KEY],
        data_std[AtomicDataDict.PER_ATOM_ENERGY_KEY],
    )
    torch.testing.assert_close(
        out_mhr[AtomicDataDict.TOTAL_ENERGY_KEY],
        data_std[AtomicDataDict.TOTAL_ENERGY_KEY],
    )
