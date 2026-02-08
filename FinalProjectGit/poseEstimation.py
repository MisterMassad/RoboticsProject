import sys
import cv2
import math
import numpy as np

"""

Pipeline Overview:

# Step 1: Image Acquisition and Preprocessing
    - Load images in BGR format.
    - Downsample (or keep as is) based on target downscale area.
    - Convert to Grayscale.
    - Scale intrinsics accordingly.
    
# Step 2: Feature Detection using SIFT
    - Detect SIFT keypoints and compute descriptors.
    
# Step 3: Match descriptors between two images
    - Adaptive ratio test based on median distance.
    - Feature matching including symmetry test to ensure mutual best matches.
    
# Step 4: Extract matched keypoint coordinates
    - Use the descriptor matches to get the point coordinates.
    
# Step 5: Find the Essential Matrix
    - Use MAGSAC to robustly estimate the Essential matrix from matched points.

# Step 6: Recover pose from Essential Matrix
    - Split multiple E candidates if necessary.
    - Recover relative camera pose (R, t) from the Essential matrix.
    - Find the best pose based on inlier count.
    
# Step 7: Wrapper functions
    - One geometry wrapper, and one for the whole pipeline.
    
# Helper Functions
    - Load calibration from .npz file.
    - .npz should have the calibration matrix with key "K".

# Main Function
    - Command-line interface to run the pose estimation between two images given calibration.

"""


# Step 1: Image Acquisition and Preprocessing

def preprocess_bgr(image_bgr, target_area=800000):
    """
    Preprocessing of image input:
        1) Calculate area of the image and downsample if larger than target_area.
        2) Convert BGR to Grayscale to allow SIFT to work.
        
    Arguments:
        1) image_bgr (np.ndarray): Input BGR image. (H,W, 3)
        2) target_area (int): Target area (in pixels) to downsample to if image is larger.
        
    Returns:
        image_bgr_resized (np.ndarray): Resized BGR image. (H,W, 3)
        image_gray (np.ndarray): Grayscale image. (H,W)
        s (float): Scale factor applied during resize. 1.0 if no resize.
        
    This function downsamples images based on their total pixel area rather than just width.
    This handles Portrait, Landscape, and Square images identically.
    
    The returned scale factor s can be used to scale the intrinsics matrix K accordingly.
    """
    if image_bgr is None:
        return None, None, 1.0
    
    h, w = image_bgr.shape[:2] # Get height and width
    current_area = h * w # Current image area in pixels
    s = 1.0 # Default scale factor (no resize)
    
    if current_area > target_area:
        s = math.sqrt(target_area / current_area) # Calculate scale factor
        new_size = (int(w * s), int(h * s)) # New size as (width, height)
        # INTER_AREA prevents aliasing during downsampling,
        image_bgr = cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)
    
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    return image_bgr, image_gray, s

# Helper: Scale intrinsics matrix K by a scale factor s
# If we resize the image by s, then we need to also scale the intrinsics accordingly.
# Otherwise, the intrinsics will not match the resized image coordinates, resulting in poor pose estimation.
# If the image wasn't resized, then we can just use K as is, as s = 1.0.

def scale_intrinsics(K, s):
    K2 = K.copy().astype(np.float64)
    K2[0, 0] *= s # fx scales with x-axis pixels scaling
    K2[1, 1] *= s # fy scales with y-axis pixels scaling
    K2[0, 2] *= s # cx (principal point x) scales with x-axis pixels scaling
    K2[1, 2] *= s # cy (principal point y) 
    return K2

# Step 2: Feature Detection using SIFT

def detect_sift_features(img_gray, nfeatures=2000, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6):
    """
        This function detects SIFT keypoints and computes descriptors.
    I chose SIFT because it is scale and rotation invariant, making it robust for various viewpoints.
    It performs much better than ORB, although it's slower.
    
    Parameters:
        img_gray (np.ndarray): Grayscale input image.
        nfeatures (int): Maximum number of features to retain.
        contrastThreshold (float): Contrast threshold for SIFT.
        edgeThreshold (float): Edge threshold for SIFT.
        sigma (float): Sigma for Gaussian filter in SIFT.
        
    Returns:
        keypoints: Detected keypoints.
        descriptors: Corresponding SIFT descriptors.
        
    """
    
    # The following parameters control how many points survive filtering and pyramid levels.
    sift = cv2.SIFT_create(
        nfeatures=int(nfeatures),
        contrastThreshold=float(contrastThreshold),
        edgeThreshold=float(edgeThreshold),
        sigma=float(sigma)
    )
    keypoints, descriptors = sift.detectAndCompute(img_gray, None) # Detect keypoints and compute descriptors
    return keypoints, descriptors


# Step 3: Match descriptors between two images

# Step 3.1: Adaptive ratio test

"""
Adaptive ratio test for descriptor matching. This function chooses the Lowe's ratio threshold adaptively based on the median distance of the matches.
The intuition of this function:
    - When the scene has many strong and distinctive features, the median distance tends to be lower, thus we can use a stricter ratio (lower value) to filter out weak matches.
    - When the matches are weaker and noiser, the median is larger, we relax the ratio to allow more matches to pass through.
"""
def adaptive_ratio_test(knn_matches):
    distances = []
    for pair in knn_matches:
        if len(pair) >= 2:
            distances.append(pair[0].distance)

    if not distances:
        return 0.75

    median_dist = np.median(distances)

    # Smaller median -> more distinctive matches -> stricter ratio
    # Larger median -> weaker matches -> looser ratio
    if median_dist < 100:
        ratio = 0.55
    elif median_dist < 200:
        ratio = 0.70
    elif median_dist < 300:
        ratio = 0.80
    elif median_dist < 400:
        ratio = 0.85
    else:
        ratio = 0.90

    # Standard deviation captures how spread the matches are
    std_dist = np.std(distances)
    
    # Coefficient of Variation
    # High CV -> mixed quality matches -> relax ratio
    # Low CV -> consistent quality matches -> tighten ratio
    cv = std_dist / (median_dist + 1e-6)

    if cv > 0.7:
        ratio += 0.05
    elif cv < 0.2:
        ratio -= 0.05

    # Finally, we clamp the ratio to reasonable bounds
    return float(np.clip(ratio, 0.5, 0.95))

# Step 3.2: Match descriptors with optional symmetry test


def match_descriptors(desc1, desc2,
                      ratio_thresh=0.75,
                      max_matches=500,
                      use_symmetry=True,
                      use_adaptive=True,
                      min_mutual=40):  # fallback threshold

    # If either image has no descriptors, return empty list because we can't match
    if desc1 is None or desc2 is None:
        return []

    # Sift descriptors are float32 by default, so we use NORM_L2 to match them
    # crossCheck=False because we will implement our own symmetry test if needed
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # -------- 1 -> 2 --------
    # First image to second image matching
    # For each descriptor in img1, retrieve the two best matches in img2
    knn12 = bf.knnMatch(desc1, desc2, k=2)

    if use_adaptive and len(knn12) > 0:
        effective_ratio = adaptive_ratio_test(knn12) # Compute the adaptive ratio based on matches stats
    else:
        effective_ratio = float(ratio_thresh)

    good12 = []
    for pair in knn12:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < effective_ratio * n.distance: # Lowe's ratio test with adaptive ratio, accept only if best match is significantly better than second best
            good12.append(m)

    # Sort matches by descriptor distance
    good12 = sorted(good12, key=lambda m: m.distance)
    one_way = good12[:max_matches] # Keep the top matches up to max_matches to reduce the outlier probability

    if not use_symmetry:
        return one_way

    # If we're already low on matches, don't risk mutual starvation and return one-way matches
    if len(one_way) < min_mutual:
        return one_way

    # -------- 2 -> 1 --------
    # Second image to first image matching
    # Reverse of above 1 -> 2
    knn21 = bf.knnMatch(desc2, desc1, k=2)
    good21 = []
    for pair in knn21:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < effective_ratio * n.distance:
            good21.append(m)

    # -------- Mutual --------
    # Enforce symmetry: keep only matches that are mutual best matches
    pairs21 = set((m.trainIdx, m.queryIdx) for m in good21)
    mutual = [m for m in one_way if (m.queryIdx, m.trainIdx) in pairs21]
    mutual = sorted(mutual, key=lambda m: m.distance)

    # Fallback if mutual starves, then we will just return one-way matches
    if len(mutual) < min_mutual:
        return one_way

    return mutual[:max_matches]


# Step 4: Extract matched keypoint coordinates - Use the descriptor matches to get the point coordinates.

def extract_matched_points(kpts1, kpts2, matches):
    """
    Given keypoints and matches, extract matched point coordinates.

    Returns:
        pts1, pts2: Nx2 float32 arrays of matched points.
    """
    # Extract the (x, y) coordinates of the matched keypoints
    pts1 = np.array([kpts1[m.queryIdx].pt for m in matches], dtype=np.float32)
    pts2 = np.array([kpts2[m.trainIdx].pt for m in matches], dtype=np.float32)
    
    return pts1, pts2


# Step 5: Find the Essential Matrix

def estimate_essential_matrix(pts1, pts2, K, prob=0.999, threshold=1.0):
    """
    Estimate the Essential Matrix using RANSAC.
    
    Arguments:
        pts1, pts2: Nx2 arrays of matched pixel coordinates.
        K: 3x3 camera intrinsic matrix.
        prob: RANSAC confidence/success probability.
        threshold: RANSAC reprojection threshold in pixels.
    
    Returns:
        E: 3x3 Essential matrix or **stacked** candidates.
        mask: Inlier mask from RANSAC.
    """
    
    # Essential matrix requires at least 5 points to be solved.
    if pts1.shape[0] < 5:
        print("Not enough points to estimate the Essential matrix. Need at least 5.")
        return None, None
    
    # Robust estimation of the Essential matrix using MAGSAC.
    # Handles outliers in feature matching, and returns an inlier mask indicating which matches are inliers.
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.USAC_MAGSAC, prob=prob, threshold=threshold)
    
    if E is None or mask is None: # We check if E is none and not if E.shape is (3, 3) because it can return multiple candidates stacked vertically.
        print("Failed to compute a valid Essential matrix.")
        return None, None
    
    # Convert mask to boolean array for indexing points arrays
    inlier_mask = mask.ravel().astype(bool)  # 1D boolean mask

    return E, inlier_mask
    

# Step 6: Recover pose from Essential Matrix. Estimate essential matrix can return multiple candidates stacked vertically, so we need to split E candidates if necessary.

# Step 6.1: Split stacked Essential matrix candidates

def split_essential_candidates(E):
    """
    Split stacked Essential matrix candidates into a list of 3x3 matrices.
    Arguments:
        E: Essential matrix or stacked candidates.
    Returns:
        List of 3x3 Essential matrices.
    """
    # Handles all of the possible shapes of E returned by findEssentialMat.
    # None, or a single 3x3 matrix, or multiple stacked 3x3 matrices.
    if E is None:
        return []
    if E.shape == (3, 3):
        return [E]
    if E.ndim == 2 and E.shape[1] == 3 and (E.shape[0] % 3 == 0):
        n = E.shape[0] // 3
        return [E[3*i:3*i+3, :] for i in range(n)]
    return [E[:3, :3]] # Fallback to first 3x3 if shape is unexpected

# Step 6.2: Recover pose from Essential matrix

def recover_pose_from_E(E, pts1, pts2, K):
    """
    Recover the relative camera pose (R, t) from the Essential matrix.
    
    Arguments:
        E: 3x3 Essential matrix.
        pts1, pts2: Nx2 arrays of matched pixel coordinates.
        K: 3x3 camera intrinsic matrix.
    """
    best = None

    # Recover pose for all possible decompositions by trying all candidate E matrices and choose the one with the most cheirality-consistent inliers
    for Ei in split_essential_candidates(E):
        try:
            pose_inliers, R, t, _ = cv2.recoverPose(Ei, pts1, pts2, K)
        except cv2.error:
            continue

        if best is None or int(pose_inliers) > best[0]:
            best = (int(pose_inliers), R, t)

    if best is None:
        return None, None, 0

    # Returns the best R, t, and inlier count
    return best[1], best[2], best[0]


# Step 7: Wrapper functions. One geometry wrapper, and one for the whole pipeline.

# Below is a wrapper function that encapsulates the entire pose estimation pipeline between two images.
def estimate_pose_two_images(img1_bgr, img2_bgr, K,
                             pixels=800000,
                             nfeatures=10000,
                             ratio_thresh=0.75,
                             max_matches=500,
                             use_symmetry=True,
                             prob=0.999,
                             threshold=1.0,
                             min_E_inliers=5,
                             warn_E_inliers=12):

    img1c_, g1, s1 = preprocess_bgr(img1_bgr, target_area=pixels)
    img2c_, g2, s2 = preprocess_bgr(img2_bgr, target_area=pixels)

    if g1 is None or g2 is None:
        return None, None

    K_use = scale_intrinsics(K, s1) if s1 != 1.0 else K

    kpts1, desc1 = detect_sift_features(g1, nfeatures=nfeatures)
    kpts2, desc2 = detect_sift_features(g2, nfeatures=nfeatures)
    
    
    matches = match_descriptors(
        desc1, desc2,
        ratio_thresh=ratio_thresh,
        max_matches=max_matches,
        use_symmetry=use_symmetry,
        use_adaptive=True,
        min_mutual=40    
    )

    if len(matches) < 5:
        print("[ERROR] Too few matches to estimate E (Need >= 5).")
        return None, None

    pts1, pts2 = extract_matched_points(kpts1, kpts2, matches)

    # ---- Estimate E + inliers  ----
    E, inlier_mask = estimate_essential_matrix(
        pts1, pts2, K_use,
        prob=prob,
        threshold=threshold
    )
    
    if E is None:
        print("[ERROR] E estimation failed.")
        return None, None


    pts1_in = pts1[inlier_mask]
    pts2_in = pts2[inlier_mask]

    n_inliers = pts1_in.shape[0]
    if n_inliers < min_E_inliers:
        print(f"[POSE] Too few E inliers: {n_inliers} < {min_E_inliers}")
        return None, None
    
    if n_inliers < warn_E_inliers:
        print(f"[Warning] Low Confidence Pose: Only {n_inliers} inliers found. Results may be inaccurate.\n")

    # ---- Pose recovery ----
    R, t, pose_inliers = recover_pose_from_E(E, pts1_in, pts2_in, K_use)
    if R is None:
        return None, None

    return R, t

# Helper Functions

# Helper 1: Load the calibration from a .npz file

def load_calibration_npz(npz_path):
    """
    Load calibration from a .npz file.
    Expects at least K under key "K".
    If no key "K" is found, attempts to auto-detect a 3x3 matrix.
    
    Arguments:
        npz_path (str): Path to the .npz calibration file.
    Returns:
        K (np.ndarray): 3x3 camera intrinsic matrix.
    Raises:
        ValueError: If no valid 3x3 matrix is found.
    """
    data = np.load(npz_path)
    
    # Search explicitly for "K"
    if "K" in data:
        K = np.asarray(data["K"]).astype(np.float64)
        if K.shape == (3, 3):
            return K
        else:
            raise ValueError(f"'K' in {npz_path} is not 3x3. Found shape: {K.shape}")
        
    # Auto-detect K by looking for a 3x3 matrix
    for key in data.keys():
        arr = np.asarray(data[key])
        if arr.shape == (3, 3):
            print(f"[INFO] Auto-detected 'K' from key '{key}' as intrinsics matrix.")
            return arr.astype(np.float64)
    
    raise ValueError(
        f"No 3x3 calibration matrix found in {npz_path}"
        f"Keys found: {list(data.keys())}"
    )
    

def main():
    """
    Usage:
        python your_script.py img1.jpg img2.jpg calib.npz
    Where calib.npz contains at least 'K'.
    """
    
    if len(sys.argv) != 4:
        print("Usage: python script.py path/to/img1.jpg path/to/img2.jpg path/to/calib.npz")
        sys.exit(1)

    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    calib_path = sys.argv[3]

    img1_bgr = cv2.imread(img1_path)
    img2_bgr = cv2.imread(img2_path)

    if img1_bgr is None:
        print(f"Error: could not load image 1, image is None: {img1_path}")
        sys.exit(1)
    if img2_bgr is None:
        print(f"Error: could not load image 2, image is None: {img2_path}")
        sys.exit(1)

    K = load_calibration_npz(calib_path)


    # Dist is loaded for future undistortion if needed, but not used in this pipeline. / ARUCO
    # print("[INFO - CAMERA INTRINSICS] Loaded K:")
    
    # print(K)
    

    # Core pipeline
    R, t = estimate_pose_two_images(
    img1_bgr, img2_bgr, K,
    pixels=800000,
    nfeatures= 3000,
    ratio_thresh=0.80,
    max_matches=400,
    use_symmetry=True,
    prob=0.999,
    threshold=1.0,
    min_E_inliers=5,
    warn_E_inliers=12
    )

    if R is None or t is None:
        print("[ERROR] Pose estimation failed.")
        sys.exit(2)

    print("\n================ FINAL POSE ================\n")
    print("[R] Rotation matrix (camera1 -> camera2):\n")
    print(R)
    print("\n[t] Translation direction (up to scale):\n")
    print(t.ravel())
    print("\n===========================================\n")
    
    # print(f"[t norm] {float(np.linalg.norm(t)):.6f}  (VO is direction-only, so ~1)")


if __name__ == "__main__":
    main()
