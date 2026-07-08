from typing import Optional

import torch
from torch import Tensor


def _empty_iou_matrix(
    preds: torch.Tensor,
    target: torch.Tensor,
) -> Optional[torch.Tensor]:
    if preds.numel() == 0:  # if no boxes are predicted
        return torch.zeros(target.shape[0], target.shape[0], device=target.device, dtype=torch.float32)
    if target.numel() == 0:  # if no boxes are true
        return torch.zeros(preds.shape[0], preds.shape[0], device=preds.device, dtype=torch.float32)
    return None
