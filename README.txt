download trainset/ at 
download testset/ at 


FILES AND FOLDERS AND ASSUMPTIONS
====================================
SLAM.py         : Main python script
                    - ASSUMES the following files and folders exits in the SAME directory
                        - myquaternion.py    [FILE]
                        - load_data.py       [FILE]
                        - p3_utils.py        [FILE]
                        - "trainset/" folder [FOLDER]
                        - "testset/" folder  [FOLDER]
                    - To specify the source folder and dataset to use, change the 
                      following lines of code:
                        - [Line 32]: Change the variable <dataset> to the suffix of the dataset
                                     e.g. if dataset has suffix 2, do:
                                            dataset = str(2)
                                     e.g. if dataset has NO suffix, do:
                                            dataset = str()  
                        - [Line 35]: Change the variable <folder_prefix> to "test" or "train"
                                     e.g. to use folder "testset", do:
                                            folder_prefix = "test" (NOTE THE ABSENCE OF "set")
                                     e.g. to use folder "trainset", do:
                                            folder_prefix = "train" (NOTE THE ABSENCE OF "set")
myquaternion.py : [From Project 2] Contains quaternion function implementations
load_data.py    : as provided for project
py_utils.py     : as provided for project



EXECUTION INSTRUCTIONS
========================
Make sure all requirements above are met, then run "python SLAM.py" from terminal

WHAT SCRIPT DOES AND OUTPUTS
-----------------------------
- Reads lidar data (.mat files) from "<folder_prefix>set"/lidar/" folder. 
- Reads joint data (.mat files) from "<folder_prefix>set"/joint/" folder. 
- Reads depth data (.mat files) from "<folder_prefix>set"/cam/" folder. 
- Reads RGB data (.mat files) from "<folder_prefix>set"/cam/" folder. 
        - CAUTION:
            - For datasets with several RGB files, the script will load ALL RGB files to
              memory at once which may slow down computer if running on less than 16GB RAM

- For the specified <dataset>
     - Launches one window with the following three images
          - SLAM map with body pose plotted
          - Texture map
          - SLAM + Texture plot
     - Prints to terminal
          - Step being performed
          - Progress status
     
- CLOSE DISPLAYED IMAGE TO QUIT PROGRAM

         



