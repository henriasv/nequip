# Test HEAD_KEY registration and data flow
import torch

from nequip.data import AtomicDataDict
from nequip.data._key_registry import _GRAPH_FIELDS, _LONG_FIELDS
from nequip.data.dict import from_dict


def test_head_key_in_graph_fields():
    """HEAD_KEY should be registered as a graph field."""
    assert AtomicDataDict.HEAD_KEY in _GRAPH_FIELDS


def test_head_key_in_long_fields():
    """HEAD_KEY should be registered as a long field."""
    assert AtomicDataDict.HEAD_KEY in _LONG_FIELDS


def test_head_key_value():
    """HEAD_KEY should have the expected string value."""
    assert AtomicDataDict.HEAD_KEY == "head"


def _make_frame(head_index, num_atoms=3, seed=0):
    """Helper to make a simple data dict with HEAD_KEY."""
    rng = torch.Generator().manual_seed(seed)
    data = {
        "pos": torch.randn(num_atoms, 3, generator=rng),
        "atomic_numbers": torch.ones(num_atoms, dtype=torch.long),
        AtomicDataDict.HEAD_KEY: torch.tensor([head_index], dtype=torch.long),
    }
    return from_dict(data)


def test_batched_from_list_concatenates_head_key():
    """batched_from_list should concatenate HEAD_KEY correctly across frames."""
    frame0 = _make_frame(head_index=0, seed=0)
    frame1 = _make_frame(head_index=1, seed=1)
    frame2 = _make_frame(head_index=0, seed=2)

    batched = AtomicDataDict.batched_from_list([frame0, frame1, frame2])
    assert AtomicDataDict.HEAD_KEY in batched
    # from_dict reshapes graph fields to (N_frames, 1), so batched is (3, 1)
    torch.testing.assert_close(
        batched[AtomicDataDict.HEAD_KEY],
        torch.tensor([[0], [1], [0]], dtype=torch.long),
    )


def test_frame_from_batched_extracts_head_key():
    """frame_from_batched should extract HEAD_KEY correctly."""
    frame0 = _make_frame(head_index=0, seed=0)
    frame1 = _make_frame(head_index=1, seed=1)

    batched = AtomicDataDict.batched_from_list([frame0, frame1])

    extracted0 = AtomicDataDict.frame_from_batched(batched, 0)
    extracted1 = AtomicDataDict.frame_from_batched(batched, 1)

    torch.testing.assert_close(
        extracted0[AtomicDataDict.HEAD_KEY],
        torch.tensor([[0]], dtype=torch.long),
    )
    torch.testing.assert_close(
        extracted1[AtomicDataDict.HEAD_KEY],
        torch.tensor([[1]], dtype=torch.long),
    )
