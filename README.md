# CMPUT 428/615 Final Project: Automatic Control of Robotic Arm with Vision 

*Authors: Jeramy Luo and Haruto Tanaka*   

This is a repository for implementation of our final course project. For the details explanation on the background and logics/theories, please refer to a [report](https://docs.google.com/document/d/1FystrSGz50eCrUJ6BVAPnOpcD9QNgFxlwvj33sWsYLY/edit). 

## File/Directory Structure 
-----
- data/ - Contains various data for experimenting ray estimation (Haruto)
- src/ - Source and testing scripts for ray estimation (Haruto)
- uvs_kinvoa/ - Source files for the visual servoing (Jeramy)  

*Note that there are duplicated files between src and uvs_kinova directory, since we tested both asynchronously. If you found any duplicated files in uvs_kinova, please refer to the src one, which are the recently updated file (e.g. image_utils.py exists both in src/ and uvs_kinova/, but refer to one in src/, please).*  

## Environment 
----- 
Followings are the package versions (within src/):  
- python 3.8.3
- pandas 1.0.5
- numpy 1.18.5
- opencv-python 4.5.5.62

## How to run the files 
----- 
### Within the src/ directory 
Simply run the ```.py``` test scripts by invoking 
```
python <file_name>.py
```  
The file names that you can test out are: 

- ```get_focal_len.py```: Focal length estimation from images of ruler. It will prompt to select two points on the image along the ruler. Note that two points must have 5 cm interval on the ruler scale by default (you can change this by changing CENTIMETER variable in the file).
- ```test_intrinsic.py```: It shows the sequence of checkerboard images, then computes/saves the intrinsic parameters). **Caution**: For each checkerboard image, please hit any key on a keyboard to proceed to next image. Once you get to the end, windows will disappear and results will be shown on the command prompt. 
- ```test_proj.py```: The script will prompt user to click on the single location where they want to track. Once you clicked on single point and hit ```q``` key, it will display the result of back projection method on the test video. 
- ```test_ray.py```: The script will prompt user to click on the single location where they want to track. Once you clicked on single point and hit ```q``` key, it will display the result of focal length method on the test video.

## Bug Report 
-----
If you noticed any bugs, please let us know or post an issue for it. 