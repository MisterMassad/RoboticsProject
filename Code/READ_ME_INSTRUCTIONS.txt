The project should be run from the CMD.

Inputs:
    
	1. Image 1
	2. Image 2
	3. Calibration matrix (camera intrinsics) as a .npz file

Notes about Calibration:
	- Preferred key name is "K". (Capital letter K).
	- If "K" is not present, the program will use the first 3x3 matrix.

Usage: 

	python sift_pose.py "path/to/img1.jpg" "path/to/img2.jpg" "path/to/calibration.npz"

Output:

	1. Rotation Matrix (3x3)
	2. Translation Vector (3x1 - Up to scale)