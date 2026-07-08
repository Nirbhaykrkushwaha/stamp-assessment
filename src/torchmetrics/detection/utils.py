from typing import Callable, Dict, List, Tuple

import torch
from torch import Tensor


def _extract_boxes_and_labels(
    p: Dict[str, Tensor],
    t: Dict[str, Tensor],
    get_safe_item_values: Callable[[Tensor], Tensor],
    groundtruth_labels: List[Tensor],
) -> Tuple[Tensor, Tensor]:
    det_boxes = get_safe_item_values(p["boxes"])
    gt_boxes = get_safe_item_values(t["boxes"])
    groundtruth_labels.append(t["labels"])
    return det_boxes, gt_boxes


def _label_eq_matrix(
    det_boxes: Tensor,
    gt_boxes: Tensor,
    pl: Tensor,
    tl: Tensor,
    iou_matrix: Tensor,
) -> Tensor:
    if det_boxes.numel() > 0 and gt_boxes.numel() > 0:
        return pl.unsqueeze(1) == tl.unsqueeze(0)  # N x M
    return torch.eye(iou_matrix.shape[0], dtype=bool, device=iou_matrix.device)
