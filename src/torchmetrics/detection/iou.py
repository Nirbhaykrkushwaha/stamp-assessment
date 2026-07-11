from typing import Any, Dict, List

from torch import Tensor

from torchmetrics.detection.helpers import (
    _fix_empty_tensors,
    _input_validator,
)
from torchmetrics.functional.detection.iou import (
    _iou_compute,
    _iou_update,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat


class IntersectionOverUnion(Metric):
    """Intersection over Union (IoU) metric."""

    @staticmethod
    def _iou_compute_fn(*args: Any, **kwargs: Any) -> Tensor:
        return _iou_compute(*args, **kwargs)

    @staticmethod
    def _iou_update_fn(*args: Any, **kwargs: Any) -> Tensor:
        return _iou_update(*args, **kwargs)

    def update(
        self,
        preds: List[Dict[str, Tensor]],
        target: List[Dict[str, Tensor]],
    ) -> None:
        """Update state with predictions and targets."""

        _input_validator(preds, target, ignore_score=True)

        for p, t in zip(preds, target):
            det_boxes = self._get_safe_item_values(p["boxes"])
            gt_boxes = self._get_safe_item_values(t["boxes"])

            self.groundtruth_labels.append(t["labels"])

            iou_matrix = self._iou_update_fn(
                det_boxes,
                gt_boxes,
                self.iou_threshold,
                self._invalid_val,
            )

            if self.respect_labels:
                label_eq = (
                    p["labels"].unsqueeze(1)
                    == t["labels"].unsqueeze(0)
                )
                iou_matrix[~label_eq] = self._invalid_val

            self.iou_matrix.append(iou_matrix)
