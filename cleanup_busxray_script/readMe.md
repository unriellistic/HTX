# Clean busxray images
**Pre-processing tool for segmenting images**
## About
A pre-processing tool that contains several functions:
1. Compile files from exp folder
2. Crops excess black and white space
3. Segmenting big image into smaller images (segments)
4. Compiles segmented images and it's respective labels into `train/test/val` folder

## Usage
To run the pre-processing on the images. Run the scripts in this order: 
1. `python compile_annotations_busxray.py` (Optional: To compile files from exp folder)
2. `python crop_bus_images_v2.py` (Optional: To crop away excess black and white space)
3. `python segment_bus_images_v3.py`
4. `python convert_and_organise_files.py`

The current folder should contain a "exp" folder which contains sub-folders in a running order, and each subfolder contains a copy of the image and it's label in Pascal VOC format (.xml files).

For in-depth explanation of each script, can look below or at the explanation given within the script.

## Scripts

### 1. Compile files from exp folder
<span style="font-size: smaller;">Note: Run this file only if you need to consolidate all images and annotation from [exp] folder into one folder, if such a folder already exist, no need run this.</span>

To run the `compile_annotations_busxray.py` script:
```shell
python compile_annotations_busxray.py
```
*Note: The image and corresponding label must have `annotated` in their name. The script finds the `annotated` keyword and copies to `--target-dir`.  Look for this line* 
```python
if "annotated" in file:
```

Optional arguments:
|Parameter|Default|Description|
|---------|-------|-----------|
|--root-dir|./exp|path to `exp` folder|
|--target-dir|./conpile_annotations|path to store compiled image and labels|

#### Parameters
**Parameter `--root-dir`**  
Path to the `exp` folder which contains subfolders that contains the images and xml files.

It'll check the folder specified at `--root-dir` for subfolders. If none is specified, it'll check the default `exp` folder.

**Parameter `--target-dir`**  
Path to folder to compile the images and xml files in.

It'll check and create a new directory specified at `--target-dir`. If none is specified, it'll create a folder called `compiled_annotations` at the current directory.

#### Command examples:
If you run straight from the thumbdrive:  
```shell
python compile_annotations_busxray.py
```
If you run from other source:  
```shell
python compile_annotations_busxray.py --root-dir path/to/exp/exp --target-dir path/to/compiled_annotations/compiled_annotations
```

### 2. Crops excess black and white space
**If not necessary to crop images, skip this step.**

To run the `crop_bus_images_v2.py` script:
```shell
python crop_bus_images_v2.py
```
*Note: This function is designed for busxray images only. To customise it for other domains, look at code for more details*

Optional arguments:
|Parameter|Default|Description|
|---------|-------|-----------|
|--root-dir-images|./compiled_annotations|path to images.|
|--root-dir-annotations|./compiled_annotations|path to annotations.|
|--target-dir|./annotations_adjusted|path to new folder to store adjusted images.|
|--recursive-search|False|if true, will search both image and root dir recursively.|
|--store|False|if true, will save both image and label at the directory it was at.|
|--display|False|if true, it displays the all annotated images in the `--target-dir` after the script finishes running.|
|--display-path|required=False|specify path to display a single image file.|

#### Parameters
**Parameter `--root-dir-images`**  
Path variable.  
It'll check the folder specified at `--root-dir-images` for image files. If no path is specified, it'll check the `./compiled_annotations` folder.

**Parameter `--root-dir-annotations`**  
Path variable.  
It'll check the folder specified at `--root-dir-annotations` for annotation files (.xml). If no path is specified, it'll check the `./compiled_annotations` folder.

**Parameter `--target-dir`**  
Path variable.  
It'll check and create a new directory specified at `--target-dir`. If none is specified, it'll create a folder `./annotations_adjusted` at the current directory and store the adjusted image and xml files in there.

**Parameter `--recursive-search`**  
When `--recursive-search` is inputted, script will search recursively into all the subdirs specified at `--root-dir-images` and `--root-dir-annotations`. The `labels` can be in different folders from the `images`, but the path structure will have to be identical to image path structure.

Example:  
|image folder directory|label folder directory|Correct or Wrong|
|----------------------|----------------------|----------------|
|images/01/01_annotated.jpg|labels/01/01_annotated.xml|Correct. Same subdirectory path. <image/label>/01/<image/label> |
|images/01/01_annotated.jpg|labels/02/01_annotated.xml|Wrong. Different subdirectory path. 01 != 02. <image/label>/**01**/<image/label>|
|images/02/01_annotated.jpg|labels/02/01_annotated.xml|Correct. Same <image/label>/02/<image/label> subdirectory path|
|images/02/abc/01_annotated.jpg|labels/02/abc/01_annotated.xml|Correct. Same <image/label>/02/abc/<image/label> subdirectory path|
|images/02/abc/01_annotated.jpg|labels/02/abc/02_annotated.xml|Wrong. Different label file name. labels/02/abc/0**2**_annotated.xml|

**Parameter `--store`**  
A boolean variable.  
If true, saves the image and labels at the directory where it was found.

**Parameter `--display`**  
An `on/off flag` optional argument.  
If inputted, it'll display the cropped annotated images after. This is for testing and debugging the cropping algorithm.

**Parameter `--display-path`**  
Path to a singular image file.  
To display a single image without running the cropping function. This is for testing and debugging the cropping algorithm.
 
#### Command examples:
If you run straight from the thumbdrive:  
```shell
python crop_bus_images.py
```
If you run from other source:  
```shell
python crop_bus_images_v2.py --root-dir path/to/compiled_annotations/compiled_annotations --target-dir path/to/adjusted_annotations/adjusted_annotations
```


### 3. Segmenting big image into smaller images (segments)
To run the segmenting script:
```shell
python segment_bus_images_v3.py
```
Optional arguments:
|Parameter |Default |Description |
|----------|--------|------------|
|--root-dir |./annotations_adjusted|path to root dir.|
|--overlap-portion|0.5|the amount in fraction of each segment that should overlap adjacent segments. From 0 to 1.|
|--segment-size|640|size of each square segment in pixel width.|
|--cutoff-threshold|0.3|`--cutoff-threshold` to determine whether to exclude annotation that has an area less than `--cutoff-threshold` of it's original size from the new segment|
|--special-items|['cig', 'human']|a list of string items to supercede the threshold set.|
|--special-items-threshold|0.1|`--special-item-threshold` to determine whether to exclude annotation that has an area less than `--special-item-threshold` of it's original size from the new segment|

#### Parameters
**Parameter `--root-dir`**  
Path variable.  
It'll check the folder specified at `--root-dir` for the **adjusted** image and xml files. If none is specified, it'll check the `./annotations_adjusted` folder.

**Parameter `--overlap-portion`**  
A float value.  
Specify the float value for percentage of overlap. Default=0.5 (50%) of image.

**Parameter `--segment-size`**  
An integer value.  
Specify the integer value for image size. default=640 (640x640 pixel image)

**Parameter `--cutoff-threshold`**  
Can experiment with the `threshold value` to see which results in the least information loss while maximising model performance.

**Parameter `--special-items`**  
The items that are to be included in this parameter consist of classes that have features that are highly distinctive even if majority of the object is cut out. And thus, it would still be highly recognisable by the model even if a small percentage of the object is shown.

**Parameter `--special-items-threshold`**  
Can experiment with the special items `threshold value` to see which results in the least information loss while maximising model performance.

#### Detailed Explanation
There's two main functions being performed in this script:

**Segmenting + Pascal VOC adjustment**  

This function segments images into segments and adjusts the [Pascal VOC annotation](https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/#pascal_voc) file relative to the segments.  
E.g. Notice the change in the label coordinates before and after segmentation:   
![diagram of segmenting](https://github.com/AlphaeusNg/HTX/assets/78588510/9558146a-a2ea-4849-9813-11901fb0c9ee)  
The *first pistol object* had no change to it's `xmin` and `ymin` since it's segment starts from the top-left.  
The *second sword object* had a change in it's `xmin`. Previously it was `130` since it was at the far-right in the original image. But after segmenting, the `xmin` became `0` since it's now at the left edge of the image.  
The *third gun object* had no change to it's `xmin`, but had a change to the `ymin`. Was at the far-bottom in the original image, but is now flushed to the top in the segmented image.

**Cleaning up image artefacts**

During the segmentation process, sometimes one part of the image contains a very tiny bounding box from an object.

E.g.  
![diagram for explaining cleaning function](https://github.com/AlphaeusNg/HTX/assets/78588510/742a52c5-5b43-43ee-bdb0-5be20157b0bb)

Take note of the purple arrows. Note that in one segment of the image, there was a small green gun sticking through. We do not want to train the model on this small feature, and thus we performed cleaning whereby we crop the segment into such that the gun gets cropped out and the model doesn't train on that tiny feature.

It crops out features that have a total area less than the `--cutoff-threshold` value set, while ensuring minimal information loss. The function currently finds the best plane to cut while minimising information loss; if 2 or more planes have zero information loss when cut, script selects plane that has the least amount of area being cut.

#### Command examples:
If you run straight from the thumbdrive: 
```shell
python segment_bus_images.py
```
If you run from other source:  
```shell
python segment_bus_images_v3.py --root-dir path/to/annotations_adjustedfolder/annotations_adjusted
```

**After generating the try-it-out example thumbdrive:**

Should have 34 files in a folder called `compiled_annotations`, consist of 17 jpg, and 17 xml.  
Should have 51 items in a fodler called `annotations_adjusted`, consist of 34 files (17 jpg, 17 xml) and 17 folders, and each folder contains divided segments in 640x640 images with 320 increment.  
```
E.g. in adjusted_355_annotated_segment:
segment_0_0.jpg, segment_0_320.jpg, segment_0_640.jpg, ..., segment_0_1999.jpg,
segment_320_0.jpg, segment_320_320.jpg, segment_320_640.jpg, ..., segment_320_1999.jpg,
.
.
.
segment_960_0.jpg, segment_960_320.jpg, segment_960_640.jpg, ..., segment_960_1999.jpg,
segment_1150_0.jpg, segment_1150_320.jpg, segment_1150_640.jpg, ..., segment_1150_1999.jpg,
```

**Correct results:**  
```
{
    "Percentage threshold value set": 0.3,
    "Overall total num of annotation": 1291,
    "Overall total num of reject": 190,
    "Overall % of reject": "14.72%",
    "Overall total num of passed": 1101,
    "Overall % of passed": "85.28%",
    "Info loss info": {
        "Overall % of info loss": "0.19%",
        "list of images with loss": {
            "adjusted_1833_annotated_segmented -> 0.303": [
                "segment_320_640.jpg -> 1.55",
                "segment_640_640.jpg -> 1.11",
                "segment_640_960.jpg -> 2.92",
                "segment_960_640.jpg -> 3.59",
                "segment_960_960.jpg -> 2.95"
            ],
            "adjusted_1835_annotated_segmented -> 0.47": [
                "segment_1077_320.jpg -> 18.81"
            ],
            "adjusted_1837_annotated_segmented -> 0.262": [
                "segment_640_1280.jpg -> 10.47"
            ],
            "adjusted_1838_annotated_segmented -> 0.005": [
                "segment_960_1969.jpg -> 0.22"
            ],
            "adjusted_357_annotated_segmented -> 0.723": [
                "segment_640_1920.jpg -> 1.89",
                "segment_640_1972.jpg -> 2.12",
                "segment_960_640.jpg -> 24.9"
            ],
            "adjusted_358_annotated_segmented -> 0.576": [
                "segment_640_640.jpg -> 24.1",
                "segment_640_960.jpg -> -4.66",
                "segment_960_640.jpg -> 3.61"
            ],
            "adjusted_359_annotated_segmented -> 0.905": [
                "segment_320_640.jpg -> 9.96",
                "segment_640_640.jpg -> 26.24"
            ],
            "adjusted_360_annotated_segmented -> 0.069": [
                "segment_1147_640.jpg -> 0.48",
                "segment_960_640.jpg -> 2.27"
            ]
        }
    },
    ...
    "adjusted_1833_annotated_segmented": {
            "image's total annotation": 82,
            "image's total reject": 14,
            "image's total info loss": 0.30300000000000005,
            "image's segment info": {
                "segment_0_0.jpg": {
                    "num_of_reject": 0,
                    "num_of_total": 1,
                    "segment_info_loss": 0.0
                },
                ...,
                "segment_320_640.jpg": {
                    "num_of_reject": 1,
                    "num_of_total": 4,
                    "segment_info_loss": 1.55
                },
                ...,
                "segment_640_640.jpg": {
                    "num_of_reject": 1,
                    "num_of_total": 8,
                    "segment_info_loss": 1.11
                },
                "segment_640_960.jpg": {
                    "num_of_reject": 3,
                    "num_of_total": 8,
                    "segment_info_loss": 2.92
                },
                ...,
                "segment_960_640.jpg": {
                    "num_of_reject": 1,
                    "num_of_total": 6,
                    "segment_info_loss": 3.59
                },
                "segment_960_960.jpg": {
                    "num_of_reject": 1,
                    "num_of_total": 6,
                    "segment_info_loss": 2.95
                }
            }
        },
```

### 4. Compiles segmented images and it's respective labels into `train/test/val` folder
To run the `convert_and_organise_files.py` script:
```shell
python convert_and_organise_files.py
```
Optional arguments:
|Parameter|Default|Description|
|---------|-------|-----------|
|--root-dir|./annotations_adjusted|path to segmented files.|
|--train|0.8|value for train folder split.|
|--test|0.1|value for test folder split.|
|--valid|0.1|value for validation folder split.|
|--seed|42|value for randomiser seed.|

#### Parameters
**Parameters `--root-dir`**  
A path variable.  
It'll check the folder specified at `--root-dir` for the **adjusted and segmented** image and xml files. If none is specified, it'll check the `./annotations_adjusted` folder.

**Parameters `--train`**  
A float value.  
Indicate the ratio split for the training dataset.

**Parameters `--test`**  
A float value.  
Indicate the ratio split for the test dataset.

**Parameters `--valid`**  
A float value.  
Indicate the ratio split for the validation dataset.

**Parameters `--seed`**  
An Integer value.  
Indicates the random seed number.

#### Explanation
The function will save the output in a folder called "*output_&lt;`name of dir`&gt;*".

It'll be in the YOLOv7 format:
```
images
  |->
    train
    test
    validation
labels
  |->
    train
    test
    validation
```
There'll be 3 subdirs in `images` and `labels` each containing randomly splitted data.

**Breakdown of script function**

The `convert_and_organise_files.py` script calls 2 other scripts:

`xml2yolo.py`: Converts .xml files into .txt files (YOLO format). It'll recursively search into all sub-folders in ROOT_DIR and create a converted copy of the XML file into txt, and store it in the same directory where it was found. If no corresponding .xml file is found, creates an empty .txt file.  
`split_train_test_val.py`: It'll recursively search into all sub-folders in `--root-dir` and copy over the files that have "cleaned" in the name into the new `--target-dir`, rename the image and label files to their respective original images. It also splits each image folders into train/test/val folders, ensuring that all segmented image from a single image belongs to either train/test/val folder. 
<small>_Note_: This is to ensure that the `test` folder contains new images.</small>

*Example:*

| Example | Original file name                                                                           | New file name                                                                     |
|---------|----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| 1.      | adjusted_PA8506K Higer 49 seats-clean-610-1 DualEnergy_segmented//segment_0_0_cleaned.tiff   | adjusted_PA8506K Higer 49 seats-clean-610-1 DualEnergy_segment_0_0_cleaned.tiff   |
| 2.      | adjusted_PA8506K Higer 49 seats-clean-610-1 DualEnergy_segmented//segment_0_320_cleaned.tiff | adjusted_PA8506K Higer 49 seats-clean-610-1 DualEnergy_segment_0_320_cleaned.tiff | 
| 3.      | adjusted_&lt;image_name&gt;_segmented//segment_640_320_cleaned.jpg                           | adjusted_&lt;image_name&gt;_segment_640_320_cleaned.jpg                           |
| 4.      | adjusted_&lt;image_name&gt;_segmented//segment_640_320_cleaned.txt                           | adjusted_&lt;image_name&gt;_segment_640_320_cleaned.txt                           |

## Run everything together

To run the default arguments in `compile_annotations_busxray.py`, `crop_bus_images_v2.py`, `segment_bus_images_v3.py`, and `convert_and_organise_files.py` together:

Navigate to folder containing the *exp folder* and the *scripts* and run:
```shell
python compile_annotations_busxray.py & python crop_bus_images_v2.py & python segment_bus_images_v3.py & convert_and_organise_files.py
```

## Additional Information on try-it-out examples:

| ID   | time              | content                                                                    | detected                                                               | actual_results                     | remarks                 |  
|------|-------------------|----------------------------------------------------------------------------|------------------------------------------------------------------------|------------------------------------|-------------------------|
| 353  | 1125              | clean                                                                      | 1 x fp (gun)                                                           |                                    |                         |
| 54   | 1138              | clean                                                                      |
| 355  | 1146              | 10 cigs	                                                                   | 9 cig 1 exp                                                            | 9TP,1FP                            | (1 cig detected as exp) |
| 356  | 1212              | 10 cigs (8 carton, 2 unpacked)                                             | 3 cig                                                                  | 3TP,7FN                            |
| 357  | 1257              | 10 cigs                                                                    | cigs same as 356, 5 guns on aisle, 2 rear seats	(G GL T ASG SMG SG M4) | 1 human	1 cig 1 human				2TP, 15FN |
| 358  | 1343              | 4 cigs (3 carton, 1 unpacked), yong chun knives, and drugs in back storage | 3 guns 1 human 4 knives 1 drug 1 exp                                   |
| 359  | 1416              | cigs, 2 fongyun, 1 yong chun                                               | 1 human 1 knive                                                        |
| 360  | 1441              | guns (ASG SG M4 GL), fongyun sword, drug, cigs                             | 1 human 5 knives 1 exp                                                 |
| 1830 | 1059              | clean                                                                      | 3 gunss, 1 knives                                                      |
| 1831 | 1106              | clean                                                                      | 1 guns, 1 knives                                                       |
| 1832 | 1119              | 6 guns, 1 cig, 1 drug                                                      | 1 knives                                                               |
| 1833 | 1131              | 6 guns, 2 knives, 1 cig, 1 drug, 1 human                                   | 3 cigs, 7 guns                                                         |
| 1834 | 1150              | 6 guns, 2 knives, 2 cig, 1 drug, 1 human                                   | 2 cigs, 5 guns, 1 human, 1 knives                                      |
| 1835 | 1209              | 6 guns, 3 knives, 2 cig, 1 drug, 1 human                                   | 7 guns, 1 human, 2 knives, 1 drugs                                     |
| 1836 | 1227              | 6 guns, 4 knives, 1 cig, 1 drug, 1 human                                   | 3 cigs, 4 guns, 1 human, 1 exp                                         |
| 1837 | 1246              | 6 guns, 3 knives, 1 cig, 1 drug, 1 human                                   | 6 guns, 1 human, 2 knives                                              |
| 1838 | 1328              | 6 guns, 3 knives, 2 cig, 1 drug                                            | 7 guns                                                                 |
| 1839 | 1347              | 6 guns, 1 knives, 1 drug, 5 cig                                            | 5 cigs, 7 guns                                                         |
| 1840 | 1411              | 6 guns, 1 knives, 1 drug, 1 cig                                            | 4 guns                                                                 |
| 1841 ||||| duplicate of 1840 |
| 1842 | 1435              | 6 guns, 1 drug, 1 cig, 1 human                                             | 4 gunss, 1 knives, 1 drugs, 1 exp                                      |
| 1843 | 1453              | 6 guns, 1 drug, 1 cig, 1 human                                             | 4 guns, 2 knives                                                       |


