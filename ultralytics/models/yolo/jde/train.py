# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from typing import Any

from ultralytics.models import yolo
from ultralytics.nn.tasks import JDEModel
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.patches import override_configs


class JDETrainer(yolo.detect.DetectionTrainer):
    """A class extending the DetectionTrainer class for training Joint Detection and Embedding models.

    This trainer specializes in training YOLO models that simultaneously output object detections
    and feature embeddings, enabling applications like multi-object tracking and person re-identification.

    Attributes:
        loss_names (tuple): Names of the loss components used during training including
            box_loss, cls_loss, dfl_loss, and emb_loss.

    Methods:
        get_model: Return JDEModel initialized with specified config and weights.
        get_validator: Return an instance of JDEValidator for validation of JDE models.

    Examples:
        >>> from ultralytics.models.yolo.jde import JDETrainer
        >>> args = dict(model="yolo26n-jde.pt", data="coco8.yaml", epochs=3)
        >>> trainer = JDETrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """Initialize a JDETrainer object for training Joint Detection and Embedding models.

        Args:
            cfg (dict, optional): Default configuration dictionary containing training parameters.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (dict, optional): Dictionary of callback functions to be executed during training.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "jde"
        super().__init__(cfg, overrides, _callbacks)
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "emb_loss"

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """Return JDEModel initialized with specified config and weights.

        Args:
            cfg (str | dict, optional): Model configuration. Can be a path to a YAML config file,
                a dictionary containing configuration parameters, or None to use default configuration.
            weights (str | Path, optional): Path to pretrained weights file. If None, random
                initialization is used.
            verbose (bool): Whether to display model information during initialization.

        Returns:
            (JDEModel): Initialized JDE model instance.
        """
        model = JDEModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a JDEValidator for JDE model validation.

        Returns:
            (JDEValidator): Validator instance configured for JDE model evaluation.
        """
        return yolo.jde.JDEValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def auto_batch(self):
        """Get optimal batch size by calculating memory occupation of model.

        Returns:
            (int): Optimal batch size for JDE model training.
        """
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4  # 4 for mosaic augmentation
        del train_dataset  # free memory
        return super().auto_batch(max_num_obj)
