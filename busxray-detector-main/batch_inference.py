'''
This script runs inference on a set of images, using the segmenting algorithm.
'''

import argparse
from pathlib import Path

import cv2, orjson

from inference import inference, draw_annotations, save_annotations
from tridentnet_predictor import TridentNetPredictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--img-source", help="source folder for the images", type=str)
    parser.add_argument("-o", "--output", help="folder to output the annotated images", type=str, required=False)
    parser.add_argument("-c", "--config-file", type=str)
    parser.add_argument("-w", "--model-weights", type=str)
    args = parser.parse_args()

    predictor = TridentNetPredictor(
        config_file=args.config_file,
        opts=["MODEL.WEIGHTS", args.model_weights]
    )

    images_to_predictions = {}

    for file in Path(args.img_source).glob("*.jpg"):
        print("=" * 16)
        print("Processing", str(file))
        cv2_img = cv2.imread(str(file))
        _, nms_preds = inference(cv2_img, predictor=predictor, segment_size=640, crop_image=False, IOU_THRESHOLD=0.5, display=False)

        images_to_predictions[file.name] = nms_preds

        if args.output:
            save_annotations(cv2_img, nms_preds, save=str(Path(args.output) / file.name))
            with open((Path(args.output) / file.name).with_suffix(".json"), "wb") as f:
                f.write(orjson.dumps(nms_preds, option=orjson.OPT_INDENT_2))
        else:
            draw_annotations(cv2_img, nms_preds)

    if args.output:
        with open(Path(args.output) / "inference.json", "wb") as f:
            f.write(orjson.dumps(images_to_predictions, option=orjson.OPT_INDENT_2))