"""
A script that returns the number of detections for each folder in the directory you specify.
For e.g. if folder json test files contains 3 other folders, LAG + Threat, LAG only, No LAG + No Threat, then it'll automatically run through all those folders

Changes made:
- line 88: changed "while counter < Numfiles - 1:" TO "while counter < Numfiles:"
- line 89 - 100, added a try-except portion to check if there's an out-of-array error, only access "res[counter]" and not "res[counter+1]". The function will only perform on the last remaining single view1 or view2 image
"""

# python function to count the number of detections in each folder for Eyefox trial @ Tuas checkpoint bus arrival hall
# usage:  python countfunc2.py --source <folder>
# example: python countfunc2.py --source 17

# countfunc3: modifying the function to exclude counting of LAG in the detection
# countfunc3Alp: a build-up of countfunc3 with the addition of doing it for all folders + saving the information into an excel file.

# importing the relevant modules
import os
from os import listdir, getcwd
import json
import argparse
import pandas as pd
import openpyxl
import math

# specifying the required arguments for the function
parser = argparse.ArgumentParser(description='Counting the number of image sets and detections from the Eyefox archive')
parser.add_argument("--source", help="Path to the folder that contains the image folder", type=str, required=True)

# For debugging
argv = ["--source", "json test files"]
args = parser.parse_args(argv)

# When running actual code via cmd prompt
# args = parser.parse_args()
dirs = args.source

print("dirs:", dirs)
# converting the string into a list by using whitespace as a delimiter 
# dirs = list(dirs.split(" "))
print('Folder to processed = ',dirs)

# printing the version of the json package
print('Json module verion is ', json.__version__)

## List the directory to run the script
#dirs = ['17']

cwd = getcwd()
info = []
cwd = cwd + '\\' + dirs
# Iterate current directory
for file in os.listdir(cwd):
   
   full_dir_path = cwd + '\\' + file
   print("full_dir_path:", full_dir_path)

   # check only text files
   print("file:", file)
   if os.path.isdir(full_dir_path):
      print("folder found:", file)

      # list to store json files
      res = []
      # list to store json files with only 1 view
      res_single = []
      # list to store json files with 2 views
      res_double = []
      # list to store json files with detections on both views
      res_twodetect = []
      # counter for counting number of detections
      detections = 0
      # counter for counting number of sets of images (consisting of 1 side and 1 top view)
      setnum = 0
      # counter for looping through the list of file names
      counter = 0
      
      # Iterate current directory
      for json_files in os.listdir(full_dir_path):
         # check only text files
         if json_files.endswith('.json'):
            res.append(json_files)

      res = sorted(res)
      print(res[0:4])
      Numfiles = len(res)
      print('Number of Json files is ',Numfiles)

      # looping through all json files
      while counter < Numfiles:
         try:
            fname1 = res[counter]
            fname2 = res[counter+1]
            # Compare the filenames fname1 and fname2 to check if both files are from the same set
            # Stripping out the view info and extension
            fname1trunc = fname1[0:15]
            fname2trunc = fname2[0:15]
         except:
            print("Reached the last file, just use fname1")
            fname1 = res[counter]
            fname1trunc = fname1[0:15]
            fname2trunc = "poop" # To ensure that both file name will not be the same
         
         # display counter to monitor progress of loop - print every 5,000 files
         if counter%5000 == 0:
            print('Counter = ', counter)
            
         if fname1trunc == fname2trunc:
            # both fname1 and fname2 are from the same set
            fname1 = full_dir_path + '\\' + fname1
            fname2 = full_dir_path + '\\' + fname2
            
            # List of json files that has 2 views
            res_double.append(fname1trunc)
                  
            # open fname1
            with open(fname1, 'r') as f:
               try:
                  data = json.load(f)

                  # assign the list in "rois" to the variable data1
                  data1 = data["rois"]
                  if len(data1):
                     # looping through all entries in rois
                     # if LAG is detected, ignore the detection (skip LAG)
                     # xx = 1 if there is at least 1 non-LAG detection
                     for ii in range(len(data1)):
                        tempvar = data1[ii]['meta']['display_label']
                        if tempvar!='LAG':
                           xx = 1
                           break
                        else:
                           xx = 0
                  else:
                     xx = 0
               except:
                  print("No details found in JSON file")
                     
            # open fname2
            with open(fname2, 'r') as j:
               data = json.load(j)
               
               # assign the list in "rois" to the variable data1
               data2 = data["rois"]
               if len(data2):
                  # looping through all entries in rois
                  # if LAG is detected, ignore the detection (skip LAG)
                  # xx = 1 if there is at least 1 non-LAG detection
                  for jj in range(len(data2)):
                     tempvar = data2[jj]['meta']['display_label']
                     if tempvar!='LAG':
                        yy = 1
                        break
                     else:
                        yy = 0
               else:
                  yy = 0
                           
            if xx + yy > 0:
               detections = detections + 1
               if xx + yy == 2:
                  # list of json files with detections on both views
                  res_twodetect.append(fname1trunc)
                  #print(fname1trunc)
            
            # updating the loop variables
            counter = counter + 2
            setnum = setnum + 1
         else:
            # both fname1 and fname2 are from different set
            fname1 = full_dir_path + '\\' + fname1
            
            # List of json files that has only 1 view
            res_single.append(fname1trunc)
                  
            # open fname1
            with open(fname1, 'r') as f:
               data = json.load(f)
               
               # assign the list in "rois" to the variable data1
               data1 = data["rois"]
               if len(data1):
                  # looping through all entries in rois
                  # if LAG is detected, ignore the detection (skip LAG)
                  # xx = 1 if there is at least 1 non-LAG detection
                  for ii in range(len(data1)):
                     tempvar = data1[ii]['meta']['display_label']
                     if tempvar!='LAG':
                        xx = 1
                        break
                     else:
                        xx = 0     	  
               else:
                  xx = 0  
                     
            if xx > 0:
               detections = detections + 1
            
            # updating the loop variables
            counter = counter + 1
            setnum = setnum + 1
         
         #if counter % 1000 == 0:
         #   print('Counter = ', counter)   

      print('Number of detection = ', detections)
      print('Number of sets = ', setnum)
      print('Number of sets with detections on both views = ', len(res_twodetect))

      print('*****************')
      print('Number of single views = ', len(res_single))
      print('Number of double views = ', len(res_double))
      
      # Stores each folder information into a list
      info.append([file, detections, setnum, round(detections*100/setnum, 2), len(res_twodetect), len(res_single), len(res_double)])

# Stores info list into pd and then convert to excel format.
df = pd.DataFrame(info, columns=['Date','Number of detection','Number of sets','Percentage of non-LAG', 'Number of sets with detections on both views','Number of single views', 'Number of double views'])
print(df)
df.to_excel('LAG_stats.xlsx', sheet_name='data', index=False)