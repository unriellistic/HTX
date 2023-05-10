"""
Combines segment_bus_images_v3.py, xml2yolo.py, consolidate_segmented_files.py, and split_train_test_val.py.
"""
import segment_bus_images_v3 as sbi
import xml2yolo as x2y
import consolidate_segmented_files as csf
import split_train_test_val as sttv
import argparse, general_scripts as gs, os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", help="directory to the image and annotation files", default=r"D:\BusXray\scanbus_training\temp")
    parser.add_argument("--overlap-portion", help="fraction of each segment that should overlap adjacent segments. from 0 to 1", default=0.5)
    parser.add_argument("--segment-size", help="size of each segment", default=640)
    parser.add_argument("--cutoff-threshold", help="cutoff threshold to determine whether to exclude annotation from the new segment", default=0.3)
    parser.add_argument("--special-items", help="a list of string items to supercede the threshold set", default=['cig', 'human'])
    args = parser.parse_args()
    # Change this to the folder that contains all the images and xml files. 
    basepath, pathname = gs.path_leaf(args.root_dir)
    target_dir = os.path.join(basepath, f"segmented_{pathname}")
    output_folder = os.path.join(basepath, f"output_{pathname}")

    # The class
    CLASSES = ["cig", "guns", "human", "knives", "drugs", "exp"]

    sbi.bulk_image_analysis_of_info_loss_and_segment_annotation(args_root_dir=args.root_dir,
                                                                args_overlap_portion=args.overlap_portion,
                                                                args_cutoff_threshold=args.cutoff_threshold,
                                                                args_segment_size=args.segment_size,
                                                                args_special_items=args.special_items)
    x2y.convert_xml_to_yolo(args.root_dir, classes=CLASSES)
    csf.consolidate_files(args.root_dir, target_dir)
    sttv.split_data(input_folder=target_dir, output_folder=output_folder, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1, seed=42)