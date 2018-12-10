## CE264 Multi-view Stereo
An implementation of plane-sweeping 2-view stereo technique. Second project for Fall 2018 CE264-Computer Vision at UC Santa Cruz

## Motivation
Implemented plane sweeping 2-view stereo after calibrating the camera for finding intrinsic matrix and rotational and translational 
vectors. Also performed SIFT to find matching points and found the depth of each pixel. Resultant depth image is also plotted.

## OS, Libraries and Language
The code can be compiled in any operating system MAC-OS, Windows. The code is written in python3 and Libraries used are:
1. Opencv
2. Numpy
3. PIL 
4. sklearn
5. Glob

## Scripts 
The project consist of three python notebooks belonging to each part of the project. The notebooks kernels can be refreshed and 
again run to verify the result of the code.
The scripts name and their functionality are decribed below :
1. Calibration: 
   Find the intrinsic matrix and radial distortion matrix of the camera.
2. MultiviewStereo:
   This script first determines the matching points and plots the epipolar lines. Also it finds the depth of the image and plot
   the resultant depth image.

## Files 
1. CE264MultiviewStereo             : parent directory
2. CalibImages                      : Consist of 10 images of chessboard taken at different angles to determine intrinsic camera matrix. 
3. StereoImages                     : Two images depicting situation as if taken by left and right camera to introduce paralax.

## Results 

Raw images:

![alt text](https://github.com/trungnguyencs/CE264MultiViewStereo/blob/master/Results/input.png "Title")

HDR Image Result:
* Method 1:
![alt text](https://github.com/trungnguyencs/CE264HDR/blob/master/ToneMappedImages/hdr1.jpg "Title")
* Method 2:
![alt text](https://github.com/trungnguyencs/CE264HDR/blob/master/ToneMappedImages/hdr2.jpg "Title")
* Method 3:
![alt text](https://github.com/trungnguyencs/CE264HDR/blob/master/ToneMappedImages/hdr_fusion_mertens.jpg "Title")
