'''
This script runs inference on a set of images, using the segmenting algorithm.
'''

import argparse
from pathlib import Path

import cv2

from inference import inference, draw_annotations, save_annotations
from tridentnet_predictor import TridentNetPredictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--img-source", help="source folder for the images", type=str)
    parser.add_argument("-o", "--output", help="location to output the annotated images", type=str, required=False)
    parser.add_argument("-c", "--config-file", type=str)
    parser.add_argument("-w", "--model-weights", type=str)
    args = parser.parse_args()

    predictor = TridentNetPredictor(
        config_file=args.config_file,
        opts=["MODEL.WEIGHTS", args.model_weights]
    )

    for file in Path(args.img_source).glob("*.jpg"):
        print("=" * 16)
        print("Processing", str(file))
        cv2_img = cv2.imread(str(file))
        _, nms_preds = inference(cv2_img, predictor=predictor, segment_size=640, crop_image=False, IOU_THRESHOLD=0.5, display=False)

        if args.output:
            save_annotations(cv2_img, nms_preds, save=str(Path(args.output) / file.name))
        else:
            draw_annotations(cv2_img, nms_preds)
