# Clean busxray images
A pre-processing tool that:
1. Crops excess black and white space
2. Segments it via cropping away artefacts with the least information loss

## Usage
To run the pre-processing on the images. Run the scripts in this order: 
1. ```compile_annotations_busxray.py```
2. ```crop_bus_images_v2.py```
3. ```segment_bus_images_v3.py```
4. ```xml2yolo.py``` (manually go into script and change variable)
5. ```consolidate_segmented_files.py``` (manually go into script and change variable)

The current folder should contain a "exp" folder which contains sub-folders of each image.

For in-depth explanation of each script, can look below or at the explanation given within the script.

To run both the cropping and segmented function:
```
python crop_bus_images_v2.py --root-dir-images "D:\BusXray\scanbus_training\master_file_for_both_clean_and_threat_images_dualenergy" --root-dir-annotations "D:\BusXray\scanbus_training\master_file_for_both_clean_and_threat_images_dualenergy" --target-dir "D:\BusXray\scanbus_training\adjusted_master_file_for_both_clean_and_threat_images_dualenergy" & python crop_bus_images_v2.py --root-dir-images "D:\BusXray\scanbus_training\master_file_for_both_clean_and_threat_images_monochrome" --root-dir-annotations "D:\BusXray\scanbus_training\master_file_for_both_clean_and_threat_images_monochrome" --target-dir "D:\BusXray\scanbus_training\adjusted_master_file_for_both_clean_and_threat_images_monochrome" & python segment_bus_images_v3.py --root-dir "D:\BusXray\scanbus_training\adjusted_master_file_for_both_clean_and_threat_images_dualenergy" & python segment_bus_images_v3.py --root-dir "D:\BusXray\scanbus_training\adjusted_master_file_for_both_clean_and_threat_images_monochrome"
```
To run `xml2yolo.py` and `consolidate_segmented_files.py`:

Go to each folder and change their respective **<ROOT_DIR>** and **<TARGET_DIR>** variables

## Detailed explanation
### 1) compile_annotations_busxray.py
<span style="font-size: smaller;">Note: Run this file only if you need to consolidate all images and annotation from [exp] folder into one folder, if such a folder already exist, no need run this.</span>

#### Command to run:
```python compile_annotations_busxray.py```
[--root-dir /path/to/rootdirectory]
[--target-dir /path/to/annotationdirectory]

##### Meaning
root directory: Path to the subfolders to your images and xml files  
annotation directory: Path to compile the images and xml file in

Arguments explanation:  
--root-dir: It'll check the folder specified at --root-dir for subfolders. If none is specified, it'll check the "exp" folder.
--annotation-dir: It'll check and create a new directory specified at --annotation-dir. If none is specified, it'll create a folder called "annotations" at the current directory.

Command examples:  
If you run straight from the thumbdrive:
python compile_annotations_busxray.py
If you run from other source:
python compile_annotations_busxray.py --root-dir "<path to exp>\exp" --annotation-dir "<path to annotations>\annotations"

### 2) crop_bus_images.py
#### Command to run  
`python crop_bus_images.py`

To display the already cropped images without running the cropping function:  
`python crop_bus_images.py --display-only`

**Additional arguments:**  
[--root-dir /path/to/root directory]  
[--target-dir /path/to/target directory]  
[--display-path /path/to/image file]

##### Meaning
root directory: Path to the folder containing the compiled images and xml files.  
target directory:  Path to compile the adjusted images and xml file in.  
image file: Path to a singular image.  

##### Arguments explanation
`--root-dir`: It'll check the folder specified at --root-dir for image and xml files. If none is specified, it'll check the "annotations" folder.  
`--target-dir`: It'll check and create a new directory specified at --annotation-dir. If none is specified, it'll create a folder called "annotations_adjusted" at the current directory and store the adjusted image and xml files in there.  
`--display`: an optional argument that can be specified to allow the display of the cropped annotated image after.
    To use, just include the '--display-only' in, no need to specify any path, it'll take path from --target-dir and display all images there.  
`--display-path`: specifies a singular image file to display after adjustments.  

##### Command examples
If you run straight from the thumbdrive:  
`python crop_bus_images.py`  
If you run from other source:  
```
python crop_bus_images_v2.py --root-dir-images "D:\BusXray\scanbus_training\master_file_for_both_clean_and_threat_images_dualenergy" --root-dir-annotations "D:\BusXray\scanbus_training\master_file_for_both_clean_and_threat_images_dualenergy" --target-dir "D:\BusXray\scanbus_training\adjusted_master_file_for_both_clean_and_threat_images_dualenergy" & python crop_bus_images_v2.py --root-dir-images "D:\BusXray\scanbus_training\master_file_for_both_clean_and_threat_images_monochrome" --root-dir-annotations "D:\BusXray\scanbus_training\master_file_for_both_clean_and_threat_images_monochrome" --target-dir "D:\BusXray\scanbus_training\adjusted_master_file_for_both_clean_and_threat_images_monochrome"
```

### 3) segment_bus_images.py
**Command to run:**

**Additional arguments:**
--root-dir <root directory>
--overlap-portion <specify float overlap value>
--segment-size <specify integer pixel size>

Meaning:
<root directory>: Path to the folder containing the adjusted images and xml files
<specify overlap value>:  Specify the float value for percentage of overlap. default=0.5 (50%)
<specify integer pixel size>: Specify the integer value for image size. default=640 (640x640 image)

Arguments explanation:
--root-dir: It'll check the folder specified at --root-dir for adjusted image and xml files. If none is specified, it'll check the "annotations_adjusted" folder.

Command examples:
If you run straight from the thumbdrive:
python segment_bus_images.py 

If you run from other source:
python segment_bus_images_v3.py --root-dir "D:\BusXray\scanbus_training\adjusted_master_file_for_both_clean_and_threat_images_dualenergy" & python segment_bus_images_v3.py --root-dir "D:\BusXray\scanbus_training\adjusted_master_file_for_both_clean_and_threat_images_monochrome"

If you want to change segmented size:
python segment_bus_images.py --segment-size 1080
-> result in 1080x1080 images

If you want to change overlap size:
python segment_bus_images.py --overlap-portion 0.2
-> reduce overlap from 50% to 20%

**After generating the try-it-out example thumbdrive:**  
Should have 34 files in a folder called "annotations", consist of 17 jpg, and 17 xml.  
Should have 51 items in a fodler called "annotations_adjusted", consist of 34 files (17 jpg, 17 xml) and 17 folders, and each folder contains divided segments in 640x640 images with 320 increment.  
	E.g. in adjusted_355_annotated_segment:
	segment_0_0.jpg, segment_0_320.jpg, segment_0_640.jpg, ..., segment_0_1999.jpg,
	segment_320_0.jpg, segment_320_320.jpg, segment_320_640.jpg, ..., segment_320_1999.jpg,
	.
	.
	.
	segment_960_0.jpg, segment_960_320.jpg, segment_960_640.jpg, ..., segment_960_1999.jpg,
	segment_1150_0.jpg, segment_1150_320.jpg, segment_1150_640.jpg, ..., segment_1150_1999.jpg,

**Correct results:**  
for 1832:  
adjusted_1832_annotated_segmented": {
            "image's total annotation": 42,
            "image's total reject": 6,
            "image's total info loss": 0.0,
            "image's segment info": { ...
			}  

for 1833:  
adjusted_1833_annotated_segmented": {
            "image's total annotation": 82,
            "image's total reject": 17,
            "image's total info loss": 1.6285,
            "image's segment info": {
			"segment_320_640.jpg": {
                    "num_of_reject": 1,
                    "num_of_total": 4,
                    "info_loss": 56.99
			}  
### 4) xml2yolo.py
**Command to run:**  
Open up script, change variable "ROOT_DIR" to the correct directory. It'll recursively search into all sub-folders in ROOT_DIR and create a converted copy of the XML file into txt, and store it in the same directory where it was found.

#######################################################################################################################
5) consolidate_segmented_files.py
Command to run:
Open up script, change variable "ROOT_DIR" to the correct directory and "TARGET_DIR" to the new directory you want to store it in. It'll recursively search into all sub-folders in ROOT_DIR and copy over the files that have "cleaned" in the name into the new TARGET_DIR, while renaming the image and label files to their respective original images.
E.g. 
adjusted_PA8506K Higer 49 seats-clean-610-1 DualEnergy_segmented -> segment_0_0_cleaned.tiff
adjusted_PA8506K Higer 49 seats-clean-610-1 DualEnergy_segmented -> segment_0_320_cleaned.tiff
becomes
PA8506K Higer 49 seats-clean-610-1 DualEnergy_segment_0_0_cleaned.tiff
PA8506K Higer 49 seats-clean-610-1 DualEnergy_segment_0_320_cleaned.tiff


Additional information on the try-it-out example file IDs:

| ID | time | content | detected | actual_results | remarks |  
| -- | ---- | ------- | -------- | -------------- | ------- |
|353 | 1125 |clean    |1 x fp (gun)|              |         |
|54	 | 1138	|clean||||
|355	|1146|	10 cigs	|9 cig 1 exp|9TP,1FP|(1 cig detected as exp)|
356	1212		10 cigs (8 carton, 2 unpacked)									3 cig								3TP,7FN	
357	1257		10 cigs, cigs same as 356, 5 guns on aisle, 2 rear seats	(G GL T ASG SMG SG M4), 1 human	1 cig 1 human				2TP, 15FN	
358	1343		4 cigs (3 carton, 1 unpacked) yong chun knives and drugs in back storage		3 guns 1 human 4 knives 1 drug 1 exp		...
359	1416		cigs, 2 fongyun, 1 yong chun										1 human 1 knive						...
360	1441		guns (ASG SG M4 GL), fongyun sword, drug, cigs							1 human 5 knives 1 exp


ID 	time		content													detected							actual_results		remarks
1830	1059		clean														3 gunss, 1 knives
1831	1106		clean														1 guns, 1 knives
1832	1119		6 guns, 1 cig, 1 drug											1 knives
1833	1131		6 guns, 2 knives, 1 cig, 1 drug, 1 human								3 cigs, 7 guns
1834	1150		6 guns, 2 knives, 2 cig, 1 drug, 1 human								2 cigs, 5 guns, 1 human, 1 knives
1835	1209		6 guns, 3 knives, 2 cig, 1 drug, 1 human								7 guns, 1 human, 2 knives, 1 drugs
1836	1227		6 guns, 4 knives, 1 cig, 1 drug, 1 human								3 cigs, 4 guns, 1 human, 1 exp
1837	1246		6 guns, 3 knives, 1 cig, 1 drug, 1 human								6 guns, 1 human, 2 knives
1838	1328		6 guns, 3 knives, 2 cig, 1 drug									7 guns
1839	1347		6 guns, 1 knives, 1 drug, 5 cig									5 cigs, 7 guns
1840	1411		6 guns, 1 knives, 1 drug, 1 cig									4 guns
1841	duplicate of 1840
1842	1435		6 guns, 1 drug, 1 cig, 1 human									4 gunss, 1 knives, 1 drugs, 1 exp
1843	1453		6 guns, 1 drug, 1 cig, 1 human									4 guns, 2 knives


