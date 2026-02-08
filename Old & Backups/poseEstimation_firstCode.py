"""
The goal of this project is to estimate and find the pose to the maximal accuracy, and to be as fast as possible. Our second goal it to optimize, then run the code on Raspberry Pi 0.

This will be the first version of the code. The first version will run on any two random given images. 
Since there is no camera calibration, we will compute the Fundamental matrix. 

The second version will be optimized for speed and accuracy.

The third version will change the Pipeline so that instead of computing the Fundamental matrix, it will compute the Essential matrix directly,
since we will have an actual camera to work with.

The fourth version will be optimized to run on Raspberry Pi 0.

"""

import sys
import cv2
import numpy as np

"""
Pipeline for Version 1:

Step 1: Image Acquisition
- Obtain two images from a dataset or camera.
- Load the images to the program.
- Convert to gray-scale
- Resize, if necessary, to reduce computation time. (We can skip this step in this version).

Step 2: Feature Detection and Matching
- Use feature detection algorithms (I will use ORB as it's the fastest) to detect keypoints in both images.
- Extract descriptors from the detected keypoints.
- Match the descriptors between the two images using a suitable matching algorithm.
- Filter matches using techniques like Lowe's ratio test to retain only good matches.

Step 3: Estimate the Fundamental Matrix
- Use the matched keypoints to estimate the Fundamental matrix using RANSAC to handle outliers.
- Validate the estimated Fundamental matrix.
- Compute the epipolar lines and visualize them on the images to verify the accuracy of the Fundamental matrix.

Step 4: Pose Estimation
- Decompose the Fundamental matrix to obtain the relative camera pose (rotation and translation) between the two images.
- Use the matched keypoints and the Fundamental matrix to compute the Essential matrix if camera intrinsics are known.
- Extract rotation and translation from the Essential matrix.
- Validate the estimated pose by checking the reprojection error.

Step 5: Output Results
- Display the matched keypoints on both images.
- Print the estimated Fundamental matrix and camera pose (rotation and translation).
- Save the results to files for further analysis.
"""

def load_and_preprocess_image(image_path, max_width=900):
    """
    Loads an image from the path, then converst to grayscale, and resize if necessary (only if it's wider than max_width).

    image_path: str: Path to the image file.
    max_width: int: Maximum width threshold of the image.

    Returns:
    img: np.ndarray: The original (possibly resized) image.
    gray_img: np.ndarray: The grayscale version of the image.
    """
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        sys.exit(1)
        
    # Resize if necessary (I did this to speed up processing for large images, it's an optimization step)
    height, width = img.shape[:2]
    if width > max_width:
        scale = max_width / width
        new_width = (int(width * scale), int(height * scale))
        img = cv2.resize(img, new_width, interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img, gray_img


# Step 2: Feature Detection using ORB

def detect_orb_features(gray_img, n_features=2500):
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(gray_img, None)
    return keypoints, descriptors


# Step 3: Match descriptors between two images
def match_descriptors(desc1, desc2, ratio_thresh=0.75):

    if desc1 is None or desc2 is None:
        print("At least one of the descriptors is None. No match found!")
        return []
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) # Using Hamming distance for ORB, because it's a binary feature descriptor.

    knn_matches = bf.knnMatch(desc1, desc2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for m, n in knn_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return good_matches

def extract_matched_points(kpts1, kpts2, matches):
    """
    Given keypoints and matches, extract matched point coordinates.

    Returns:
        pts1, pts2: Nx2 float32 arrays of matched points.
    """
    pts1 = []
    pts2 = []

    for m in matches:
        pt1 = kpts1[m.queryIdx].pt  # (x, y) in image 1
        pt2 = kpts2[m.trainIdx].pt  # (x, y) in image 2
        pts1.append(pt1)
        pts2.append(pt2)

    pts1 = np.array(pts1, dtype=np.float32)
    pts2 = np.array(pts2, dtype=np.float32)
    return pts1, pts2


"""
We need the camera intrinsics to compute the Essential matrix.
However, since this is the first version, we will only compute the Fundamental matrix.
Then, later, when we have access to the drone and cameras, we will compute the Essential matrix directly.
Which will of-course improve accuracy, and improve speed as well because we can skip the Fundamental matrix computation step.

"""

# Calculate the Fundamental matrix using RANSAC

# def estimate_fundamental_matrix(pts1, pts2):
#     if pts1.shape[0] < 8 or pts2.shape[0] < 8:
#         print("Not enough points to estimate the Fundamental matrix. Need at least 8.")
#         return None, None
    
#     F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.99)

#     if F is None or F.shape != (3, 3):
#         print("Failed to compute a valid Fundamental matrix.")
#         return None, None

#     return F, mask


# Now, we can draw the inlier matches and epipolar lines later once we have the Fundamental matrix.

def draw_inlier_matches(img1_color, img2_color, kpts1, kpts2, matches, inlier_mask, max_to_draw=50):
    """
    Draws only inlier matches for visualization.
    """
    inlier_matches = [m for m, inl in zip(matches, inlier_mask.ravel()) if inl]

    print(f"[INFO] Inlier matches after RANSAC: {len(inlier_matches)}")

    if len(inlier_matches) == 0:
        print("[WARN] No inlier matches to draw.")
        return

    matches_to_draw = inlier_matches[:max_to_draw]

    img_matches = cv2.drawMatches(
        img1_color, kpts1,
        img2_color, kpts2,
        matches_to_draw,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    cv2.imshow("Inlier matches after RANSAC (Fundamental Matrix)", img_matches)
    print("[INFO] Press any key in the image window to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def guess_intrinsics(image_shape, f_factor=0.9):    
    """
    Guess camera intrinsics based on image shape.
    f_factor: float: Fraction of the image width to use as focal length.
    """

    h, w = image_shape
    f = f_factor * max(h, w)  # Focal length in pixels
    cx = w / 2.0 # Principal point x-coordinate
    cy = h / 2.0 # Principal point y-coordinate
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Print K
    print("[INFO] Guessed camera intrinsics K:")
    print(K)
    return K


def estimate_pose_from_matches(pts1, pts2, K):
    """
    Estimate the relative pose (R, t) from matched points and camera intrinsics.
    """
    if pts1.shape[0] < 8:
        print("[ERROR] Need at least 8 point correspondences to estimate Essential matrix.")
        return None, None, None
    

    # Estimate Essential Matrix with RANSAC
    E, mask_E = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    if E is None or E.shape != (3, 3):
        print("[ERROR] Failed to compute a valid Essential matrix.")
        return None, None, None
    
    # Recover pose from Essential matrix
    num_inliers, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K)
    print(f"[INFO] Number of inliers used for pose recovery: {num_inliers}")

    return R, t, pose_mask

def main(img1_path, img2_path):
    # --- Load images ---
    img1_color, img1_gray = load_and_preprocess_image(img1_path)
    img2_color, img2_gray = load_and_preprocess_image(img2_path)

    print(f"[INFO] Image 1 shape: {img1_gray.shape}")
    print(f"[INFO] Image 2 shape: {img2_gray.shape}")

    # --- Detect ORB features ---
    kpts1, desc1 = detect_orb_features(img1_gray)
    kpts2, desc2 = detect_orb_features(img2_gray)
    print(f"[INFO] Image 1: {len(kpts1)} keypoints")
    print(f"[INFO] Image 2: {len(kpts2)} keypoints")

    # --- Match descriptors ---
    good_matches = match_descriptors(desc1, desc2, ratio_thresh=0.75)
    print(f"[INFO] Good matches after ratio test: {len(good_matches)}")

    if len(good_matches) < 8:
        print("[ERROR] Not enough matches to estimate pose (need >= 8).")
        return

    # --- Extract matched points ---
    pts1, pts2 = extract_matched_points(kpts1, kpts2, good_matches)
    print(f"[INFO] pts1 shape: {pts1.shape}, pts2 shape: {pts2.shape}")

    # --- Guess intrinsics from image shape (approximate K) ---
    K = guess_intrinsics(img1_gray.shape)
    print("[INFO] Approximated intrinsics K:")
    print(K)

    # --- Estimate pose from matches + K ---
    R, t, pose_mask = estimate_pose_from_matches(pts1, pts2, K)
    if R is None:
        return

    print("[INFO] Rotation R (camera 1 -> camera 2):")
    print(R)
    print("[INFO] Translation direction t (up to scale):")
    print(t.ravel())

    # Optionally visualize only inlier matches according to pose_mask
    inlier_matches = [m for m, inl in zip(good_matches, pose_mask.ravel()) if inl]
    print(f"[INFO] Inlier matches used for pose: {len(inlier_matches)}")

    max_to_draw = min(50, len(inlier_matches))
    matches_to_draw = inlier_matches[:max_to_draw]

    matches_img = cv2.drawMatches(
        img1_color, kpts1,
        img2_color, kpts2,
        matches_to_draw,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    cv2.imshow("Inlier matches used for pose", matches_img)
    print("[INFO] Press any key in the image window to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python relative_pose_step4_pose.py path/to/img1.jpg path/to/img2.jpg")
        sys.exit(1)

    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    main(img1_path, img2_path)