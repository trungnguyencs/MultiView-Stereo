## CE264 Multi-view Stereo
An implementation of plane-sweeping 2-view stereo technique.

## Motivation
Implemented plane sweeping 2-view stereo after calibrating the camera for finding intrinsic matrix and rotational and translational 
vectors. Also performed SIFT to find points of interest and match them, then use the results to find the depth of each pixel. Resultant depth image is also plotted.

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

* Left & right image:

<img src="https://github.com/trungnguyencs/CE264MultiViewStereo/blob/master/StereoImages/01.jpeg" width="425"/> <img src="https://github.com/trungnguyencs/CE264MultiViewStereo/blob/master/StereoImages/02.jpeg" width="425"/> 
* Feature matching:
![alt text](https://github.com/trungnguyencs/CE264MultiViewStereo/blob/master/Results/feature_matching.png "Title")
* Epilines:
![alt text](https://github.com/trungnguyencs/CE264MultiViewStereo/blob/master/Results/epilines.png "Title")
* Projection:

<img src="https://github.com/trungnguyencs/CE264MultiViewStereo/blob/master/Results/projection1.png" width="425"/> <img src="https://github.com/trungnguyencs/CE264MultiViewStereo/blob/master/Results/projection2.png" width="425"/> 
* Image warping:

![alt text](https://github.com/trungnguyencs/CE264MultiViewStereo/blob/master/Results/warping0.png "Title")
* Depth map:

![alt text](https://github.com/trungnguyencs/CE264MultiViewStereo/blob/master/Results/depth.png "Title")

