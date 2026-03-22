import torch
import torch.nn as nn


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk=None):
        # seq can be a tensor or a dict of tensors
        if isinstance(seq, dict):
            vals = list(seq.values())
            if len(vals) == 0:
                raise ValueError("AvgReadout received an empty dict.")
            seq = torch.cat(vals, dim=0)

        if msk is None:
            return torch.mean(seq, dim=0)
        else:
            msk = msk.unsqueeze(-1)
            return torch.sum(seq * msk, dim=0) / torch.sum(msk)