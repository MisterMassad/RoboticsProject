
import sys
import cv2
import numpy as np

"""
Pipeline:

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

# Step 1: Image Acquisition and Preprocessing

def preprocess_bgr(image_bgr, max_width=None):
    """
    Resize (if necessary, it's also optional) and convert BGR image to grayscale.
    
    Arguments:
        1) image_bgr (np.ndarray): Input BGR image. (H, W, 3)
        2) max_width (int or None): If specified, resize the image to this width if it's wider.
        
    Returns:
        image_bgr_resized (np.ndarray): Resized BGR image. (H, W, 3)
        image_gray (np.ndarray): Grayscale image. (H, W)
        s (float): Scale factor applied during resize. 1.0 if no resize.
    """
    
    if image_bgr is None:
        print("Error: Input image is None.")
        return None, None, 1.0
    
    # Obtain the height and width of the input image
    height, width = image_bgr.shape[:2]
    s = 1.0
    
    
    if max_width is not None and width > max_width:
        s = max_width / width # Scale factor, obtained from width ratio which is calculated as new_width / old_width which means intuitively that if the image is wider than max_width, we need to scale it down.
        new_size = (int(width * s), int(height * s)) # (width, height)
        image_bgr = cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)
        
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return image_bgr, image_gray, s

# Step 2: Feature Detection using ORB

def detect_orb_features(img_gray, nfeatures=2500, fastThreshold=20, nlevels=8):
    orb = cv2.ORB_create(
        nfeatures=int(nfeatures),
        fastThreshold=int(fastThreshold),
        nlevels=int(nlevels)
    )
    
    keypoints, descriptors = orb.detectAndCompute(img_gray, None)
    return keypoints, descriptors

# Step 3: Match descriptors between two images

def match_descriptors(desc1, desc2, ratio_thresh=0.75, max_matches=500, use_symmetry = True):
    """
    Match descriptors between two images using BFMatcher and apply Lowe's ratio test.
    In this function, I did implement symmetric matching as an option. We can turn it off if needed. --------------------------------------> If use_symmetry is set to False, then it will only do one-way matching from desc1 to desc2.
    
    Arguments:
        desc1 (np.ndarray): Descriptors from image 1.
        desc2 (np.ndarray): Descriptors from image 2.
        ratio_thresh (float): Ratio threshold for Lowe's ratio test.
        max_matches (int): Maximum number of matches to return.
        
    Returns:
        good_matches (list of cv2.DMatch): Filtered good matches.
    """
    
    if desc1 is None or desc2 is None:
        print("At least one of the descriptors is None. No match found!")
        return []
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) # Using Hamming distance for ORB, because it's a binary feature descriptor.
    
    # KNN 1 to 2 matching
    knn12 = bf.knnMatch(desc1, desc2, k=2)
    good12 = []
    for pair in knn12:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good12.append(m)
            
    if not use_symmetry:
        good12 = sorted(good12, key=lambda m: m.distance)
        return good12[:max_matches]

    # KNN 2->1
    knn21 = bf.knnMatch(desc2, desc1, k=2)
    good21 = []
    for pair in knn21:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good21.append(m)
            
    # Build a set of mutual pairs
    pairs21 = set((m.trainIdx, m.queryIdx) for m in good21)  # (idx_in_desc1, idx_in_desc2)

    mutual = []
    for m in good12:
        if (m.queryIdx, m.trainIdx) in pairs21:
            mutual.append(m)

    mutual = sorted(mutual, key=lambda m: m.distance)
    return mutual[:max_matches]

# ============================ Efficient Match Descriptors Function ============================ 
def match_descriptors_efficient(desc1, desc2, ratio_thresh=0.75):

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

# ============================ End of Efficient Match Descriptors Function ============================


# Step 4: Extract matched keypoint coordinates - Use the descriptor matches to get the point coordinates.

def extract_matched_points(kpts1, kpts2, matches):
    """
    Given keypoints and matches, extract matched point coordinates.

    Returns:
        pts1, pts2: Nx2 float32 arrays of matched points.
    """
    
    pts1 = np.array([kpts1[m.queryIdx].pt for m in matches], dtype=np.float32)
    pts2 = np.array([kpts2[m.trainIdx].pt for m in matches], dtype=np.float32)
    
    return pts1, pts2


# Step 5: Find the Essential Matrix

def estimate_essential_matrix(pts1, pts2, K, prob=0.999, threshold=1.0, debug=False):
    """
    Estimate the Essential Matrix using RANSAC.
    
    Arguments:
        pts1, pts2: Nx2 arrays of matched pixel coordinates.
        K: 3x3 camera intrinsic matrix. (Given as input)
        prob: RANSAC confidence/success probability.
        threshold: RANSAC reprojection threshold in pixels.
    
    Returns:
        E: 3x3 Essential matrix or **stacked** candidates.
        mask: Inlier mask from RANSAC.
    """
    
    if pts1.shape[0] < 5:
        print("Not enough points to estimate the Essential matrix. Need at least 5.")
        return None, None
    
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=prob, threshold=threshold)
    
    if E is None or mask is None: # We don't check if E.shape is (3, 3) because it can return multiple candidates stacked vertically.
        print("Failed to compute a valid Essential matrix.")
        print(f"E shape: {None if E is None else E.shape}, mask: {None if mask is None else mask.shape}")
        return None, None
    
    #### Debug info  --------------------------------------- ######
    num_inliers = int(inlier_mask.sum())

    if debug:
        print(f"[E] Inliers: {num_inliers}/{len(pts1)} "
            f"({num_inliers/len(pts1):.2f})")
        
    ##### --------------------------------------- ######

    
    inlier_mask = mask.ravel().astype(bool) # Flatten the mask to 1D boolean array.
    return E, inlier_mask
    


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