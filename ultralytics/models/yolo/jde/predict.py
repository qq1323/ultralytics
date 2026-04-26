# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import ops


class JDEPredictor(DetectionPredictor):
    """A class extending the DetectionPredictor for Joint Detection and Embedding inference.

    This predictor specializes in JDE models that simultaneously output object detections
    and feature embeddings. It extends DetectionPredictor by properly handling embedding
    outputs in the Results objects.

    Attributes:
        args (namespace): Configuration arguments for the predictor.
        model (nn.Module): The JDE model used for inference.
        batch (list): Batch of images and metadata for processing.

    Methods:
        construct_result: Create a single Result object with embedding outputs.

    Examples:
        >>> from ultralytics.models.yolo.jde import JDEPredictor
        >>> args = dict(model="yolo26n-jde.pt", source="path/to/images")
        >>> predictor = JDEPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def construct_result(self, pred, img, orig_img, img_path):
        """Construct a single Results object from one image prediction with embeddings.

        Args:
            pred (torch.Tensor): Predicted boxes, scores and embeddings with shape
                (N, 6 + emb_size) where N is the number of detections. Columns 0-3
                are bounding boxes, 4 is confidence, 5 is class, and 6+ are embeddings.
            img (torch.Tensor): Preprocessed image tensor used for inference.
            orig_img (np.ndarray): Original image before preprocessing.
            img_path (str): Path to the original image file.

        Returns:
            (Results): Results object containing the original image, image path,
                class names, scaled bounding boxes, and embeddings for each detection.
        """
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], embeds=pred[:, 6:])
