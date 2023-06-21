"""
To quickly check through whether names in excel match names in tif for SeeTrue excel sheet
"""

from general_scripts import general_scripts as gs
import os
from tqdm import tqdm
from pathlib import Path
import pandas as pd

image_path = Path(r"D:\Eyefox Data\CAG\SeeTrue\Phase 2 Data\01feb")
image_filename = gs.load_files(image_path, "A.tif")
for index, filename in enumerate(image_filename):
    _, filename = gs.path_leaf(filename)
    filename = filename.split("_")[2]
    image_filename[index] = filename

# Open the Excel file
excel_path = Path(r"D:\Eyefox Data\CAG\SeeTrue\Phase 2 Data\EYEFOX_STATS_OVERALL.xlsx")
excel_file = pd.ExcelFile(excel_path)

# Read the desired sheet into a DataFrame
sheet_name = '01FEB'  # Replace with the name of your sheet
df = excel_file.parse(sheet_name)

# Get the specific column from the DataFrame
column_name = 'Image name'  # Replace with the name of your column
column_data = df[column_name].astype(str)

# Print the column data
print(column_data)
no_data_found = []
for data in column_data:
    if data not in image_filename:
        no_data_found.append(data)

print("no_data_found:", no_data_found)
