from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances

from tridentnet import add_tridentnet_config
from utils import slice_image

def parse_detectron2_inference(inference: Instances) -> list[dict]:
    """
    Converts the raw detectron2 inference format into a more readable format.
    """
    parsed = []
    fields = inference.get_fields()
    for i in range(len(inference)):
        parsed.append({
            "bbox": fields["pred_boxes"].tensor.cpu().numpy().tolist()[i],
            "score": fields["scores"].cpu().numpy().tolist()[i],
            "pred_class": fields["pred_classes"].cpu().numpy().tolist()[i]
        })
    
    return parsed

class TridentNetPredictor(DefaultPredictor):
    """
    Predictor class for TridentNet object detection.
    Callable; takes an OpenCV-format image as input, and outputs COCO-format predictions.
    See the parent class, DefaultPredictor, for more details.
    """
    def __init__(self, config_file: str, opts: list[str], confidence_threshold: float = 0.5):
        # get default detectron2 config
        cfg = get_cfg()
        # configure tridentnet (it's not inside by default)
        add_tridentnet_config(cfg)
        # add configs from config file and options
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(opts)
        # Set score_threshold for builtin models
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
        cfg.freeze()

        super().__init__(cfg)

    def __call__(self, original_image):
        raw_predictions = super().__call__(original_image)
        # process the predictions into a JSON-compatible format
        predictions = parse_detectron2_inference(raw_predictions["instances"])
        return predictions
    
class TridentNetSlicePredictor(TridentNetPredictor):
    """
    TridentNetPredictor that slices the image into smaller pieces first, for better detection of small objects.
    """
    def __init__(self, config_file: str, opts: list[str], confidence_threshold: float = 0.5, segment_size: int = 640, 
                 overlap_portion: float = 0.5):
        super().__init__(config_file, opts, confidence_threshold)
        self.segment_size = segment_size
        self.overlap_portion = overlap_portion

    def preprocess(self, original_image):
        # Preprocessing step: slices the image into multiple slices, for feeding to the model
        slices = slice_image(original_image, self.segment_size, self.overlap_portion)
        return slices
    
    def __call__(self, original_image):
        slices = self.preprocess(original_image)
        predictions = []
        for imgslice in slices:
            slice_preds = super().__call__(imgslice.img) # bboxes are (x1, y1, x2, y2)
            for pred in slice_preds:
                # add x-offset
                pred["bbox"][0] += imgslice.x_offset
                pred["bbox"][2] += imgslice.x_offset
                # add y-offset
                pred["bbox"][1] += imgslice.y_offset
                pred["bbox"][3] += imgslice.y_offset

                # add to the final predictions
                predictions.append(pred)

        return predictions