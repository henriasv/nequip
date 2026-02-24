# Test HeadStamper transform
import torch

from nequip.data import AtomicDataDict
from nequip.data.dict import from_dict
from nequip.data.transforms import HeadStamper


def _make_frame(num_atoms=3, seed=0):
    """Helper to make a simple data dict without HEAD_KEY."""
    rng = torch.Generator().manual_seed(seed)
    data = {
        "pos": torch.randn(num_atoms, 3, generator=rng),
        "atomic_numbers": torch.ones(num_atoms, dtype=torch.long),
    }
    return from_dict(data)


def test_stamps_correct_head_index():
    """HeadStamper should add HEAD_KEY with the correct index."""
    stamper = HeadStamper(head_index=2)
    frame = _make_frame()
    result = stamper(frame)
    assert AtomicDataDict.HEAD_KEY in result
    assert result[AtomicDataDict.HEAD_KEY].item() == 2


def test_stamps_zero():
    """HeadStamper with head_index=0 should stamp 0."""
    stamper = HeadStamper(head_index=0)
    frame = _make_frame()
    result = stamper(frame)
    assert result[AtomicDataDict.HEAD_KEY].item() == 0


def test_head_stamper_dtype():
    """HEAD_KEY should be long dtype."""
    stamper = HeadStamper(head_index=1)
    frame = _make_frame()
    result = stamper(frame)
    assert result[AtomicDataDict.HEAD_KEY].dtype == torch.long


def test_batching_different_head_stamps():
    """Batching two datasets with different head stamps produces correct HEAD_KEY tensor."""
    stamper0 = HeadStamper(head_index=0)
    stamper1 = HeadStamper(head_index=1)

    frame0 = stamper0(_make_frame(seed=0))
    frame1 = stamper1(_make_frame(seed=1))
    frame2 = stamper0(_make_frame(seed=2))

    # HeadStamper stamps (1,) tensors, so batching gives (3,)
    batched = AtomicDataDict.batched_from_list([frame0, frame1, frame2])
    expected = torch.tensor([0, 1, 0], dtype=torch.long)
    torch.testing.assert_close(batched[AtomicDataDict.HEAD_KEY], expected)


def test_overwrites_existing_head_key():
    """HeadStamper should overwrite an existing HEAD_KEY."""
    stamper0 = HeadStamper(head_index=0)
    stamper1 = HeadStamper(head_index=1)
    frame = _make_frame()
    frame = stamper0(frame)
    assert frame[AtomicDataDict.HEAD_KEY].item() == 0
    frame = stamper1(frame)
    assert frame[AtomicDataDict.HEAD_KEY].item() == 1
