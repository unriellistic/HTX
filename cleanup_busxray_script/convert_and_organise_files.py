"""
Combines segment_bus_images_v3.py, xml2yolo.py, consolidate_segmented_files.py, and split_train_test_val.py.
"""
import xml2yolo as x2y
import consolidate_segmented_files as csf
import split_train_test_val as sttv
import argparse, general_scripts as gs, os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", help="directory to the image and annotation files", default=r"D:\BusXray\scanbus_training\temp")
    args = parser.parse_args()
    # Change this to the folder that contains all the images and xml files. 
    basepath, pathname = gs.path_leaf(args.root_dir)
    target_dir = os.path.join(basepath, f"segmented_{pathname}")
    output_folder = os.path.join(basepath, f"output_{pathname}")

    # The class
    CLASSES = ["cig", "guns", "human", "knives", "drugs", "exp"]
    x2y.convert_xml_to_yolo(args.root_dir, classes=CLASSES)
    csf.consolidate_files(args.root_dir, target_dir)
    sttv.split_data(input_folder=target_dir, output_folder=output_folder, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1, seed=42)