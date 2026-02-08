"""
The goal of this project is to estimate and find the pose to the maximal accuracy, and to be as fast as possible. Our second goal it to optimize, then run the code on Raspberry Pi 0.

This will be the first version of the code. The first version will run on any two random given images. 
Since there is no camera calibration, we will compute the Fundamental matrix. 

The second version will be optimized for speed and accuracy.

The third version will change the Pipeline so that instead of computing the Fundamental matrix, it will compute the Essential matrix directly,
since we will have an actual camera to work with.

The fourth version will be optimized to run on Raspberry Pi 0.

"""