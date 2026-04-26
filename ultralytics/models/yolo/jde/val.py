# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from ultralytics.data import converter
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, RANK, nms, ops
from ultralytics.utils.metrics import DetMetrics


class JDEValidator(DetectionValidator):
    """A class extending the DetectionValidator for validating Joint Detection and Embedding models.

    This validator specializes in evaluating models that simultaneously output object detections
    and feature embeddings. It extends DetectionValidator by adding support for processing and
    evaluating embedding outputs.

    Attributes:
        args (dict): Configuration arguments for the validator.
        metrics (DetMetrics): Metrics object for evaluating detection model performance.

    Methods:
        init_metrics: Initialize evaluation metrics for JDE model.
        postprocess: Apply non-maximum suppression to JDE model predictions.

    Examples:
        >>> from ultralytics.models.yolo.jde import JDEValidator
        >>> args = dict(model="yolo26n-jde.pt", data="coco8.yaml")
        >>> validator = JDEValidator(args=args)
        >>> validator(model=args["model"])
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None) -> None:
        """Initialize JDEValidator and set task to 'jde'.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): DataLoader to be used for validation.
            save_dir (str | Path, optional): Directory to save results.
            args (dict, optional): Arguments containing validation parameters.
            _callbacks (dict, optional): Dictionary of callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "jde"

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize evaluation metrics for JDE model validation.

        Args:
            model (torch.nn.Module): JDE model to validate.
        """
        super().init_metrics(model)
        # Additional embedding metrics can be initialized here in the future
        # e.g., embedding similarity metrics, clustering metrics, retrieval metrics, etc.

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Apply Non-maximum suppression to JDE model predictions.

        Args:
            preds (torch.Tensor): Raw predictions from the JDE model with shape
                (batch_size, num_detections, 4 + nc + emb_size) where emb_size is
                the embedding dimension.

        Returns:
            (list[dict[str, torch.Tensor]]): Processed predictions after NMS, where each dict
                contains 'bboxes', 'conf', 'cls', and 'extra' (embeddings) tensors.
        """
        outputs = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            nc=self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
            rotated=self.args.task == "obb",
        )
        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]} for x in outputs]
