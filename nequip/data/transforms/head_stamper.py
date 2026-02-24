# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch

from nequip.data import AtomicDataDict


class HeadStamper(torch.nn.Module):
    """Stamps ``HEAD_KEY`` onto each data dict with a fixed head index.

    Add to a dataset's ``transforms`` list to assign all frames from that
    dataset to a particular head.

    Args:
        head_index (int): integer head index to stamp onto each frame
    """

    def __init__(self, head_index: int):
        super().__init__()
        self.head_index = head_index

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data[AtomicDataDict.HEAD_KEY] = torch.tensor(
            [self.head_index], dtype=torch.long
        )
        return data
