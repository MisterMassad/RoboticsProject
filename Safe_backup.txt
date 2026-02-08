import sys
import cv2
import math
import numpy as np



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
    """
    if image_bgr is None:
        return None, None, 1.0
    
    h, w = image_bgr.shape[:2]
    current_area = h * w
    s = 1.0
    
    if current_area > target_area:
        s = math.sqrt(target_area / current_area)
        new_size = (int(w * s), int(h * s))
        # INTER_AREA is the 'smart' resizer that prevents aliasing
        image_bgr = cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)
    
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return image_bgr, image_gray, s

# Helper: Scale intrinsics matrix K by a scale factor s
# If we resize the image by s, then we need to also scale the intrinsics accordingly.
# Otherwise, the intrinsics will not match the resized image coordinates, resulting in poor pose estimation.
# If the image wasn't resized, then we can just use K as is, as s = 1.0.
def scale_intrinsics(K, s):
    K2 = K.copy().astype(np.float64)
    K2[0, 0] *= s # fx
    K2[1, 1] *= s # fy
    K2[0, 2] *= s # cx
    K2[1, 2] *= s # cy
    return K2

# Step 2: Feature Detection using SIFT

def detect_sift_features(img_gray, nfeatures=2000, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6):
    sift = cv2.SIFT_create(
        nfeatures=int(nfeatures),
        contrastThreshold=float(contrastThreshold),
        edgeThreshold=float(edgeThreshold),
        sigma=float(sigma)
    )
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)
    return keypoints, descriptors


# Step 3: Match descriptors between two images

# Step 3.1: Adaptive ratio test

def adaptive_ratio_test(knn_matches):
    distances = []
    for pair in knn_matches:
        if len(pair) >= 2:
            distances.append(pair[0].distance)

    if not distances:
        return 0.75

    median_dist = np.median(distances)

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

    std_dist = np.std(distances)
    cv = std_dist / (median_dist + 1e-6)

    if cv > 0.7:
        ratio += 0.05
    elif cv < 0.2:
        ratio -= 0.05

    return float(np.clip(ratio, 0.5, 0.95))

# Step 3.2: Match descriptors with optional symmetry test


def match_descriptors(desc1, desc2,
                      ratio_thresh=0.75,
                      max_matches=500,
                      use_symmetry=True,
                      use_adaptive=True,
                      min_mutual=40):  # fallback threshold

    if desc1 is None or desc2 is None:
        return []

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # -------- 1 -> 2 --------
    knn12 = bf.knnMatch(desc1, desc2, k=2)

    if use_adaptive and len(knn12) > 0:
        effective_ratio = adaptive_ratio_test(knn12)
    else:
        effective_ratio = float(ratio_thresh)

    good12 = []
    for pair in knn12:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < effective_ratio * n.distance:
            good12.append(m)

    good12 = sorted(good12, key=lambda m: m.distance)
    one_way = good12[:max_matches]

    if not use_symmetry:
        return one_way

    # If we're already low on matches, don't risk mutual starvation
    if len(one_way) < min_mutual:
        return one_way

    # -------- 2 -> 1 --------
    knn21 = bf.knnMatch(desc2, desc1, k=2)
    good21 = []
    for pair in knn21:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < effective_ratio * n.distance:
            good21.append(m)

    # -------- Mutual --------
    pairs21 = set((m.trainIdx, m.queryIdx) for m in good21)
    mutual = [m for m in one_way if (m.queryIdx, m.trainIdx) in pairs21]
    mutual = sorted(mutual, key=lambda m: m.distance)

    # Fallback if mutual starves
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
    
    pts1 = np.array([kpts1[m.queryIdx].pt for m in matches], dtype=np.float32)
    pts2 = np.array([kpts2[m.trainIdx].pt for m in matches], dtype=np.float32)
    
    return pts1, pts2


# Step 5: Find the Essential Matrix

def estimate_essential_matrix(pts1, pts2, K, prob=0.999, threshold=1.0):
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
    
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.USAC_MAGSAC, prob=prob, threshold=threshold)
    
    if E is None or mask is None: # We check if E is none and not if E.shape is (3, 3) because it can return multiple candidates stacked vertically.
        print("Failed to compute a valid Essential matrix.") # REMOVE THE PRINT -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        return None, None
    
    inlier_mask = mask.ravel().astype(bool)  # 1D boolean mask

    return E, inlier_mask
    

# Step 6: Recover pose from Essential Matrix. Estimate essential matrix can return multiple candidates stacked vertically, so we need to split E candidates if necessary.

# Step 6.1: Split stacked Essential matrix candidates

def split_essential_candidates(E):
    """
    Split stacked Essential matrix candidates into a list of 3x3 matrices.
    """
    if E is None:
        return []
    if E.shape == (3, 3):
        return [E]
    if E.ndim == 2 and E.shape[1] == 3 and (E.shape[0] % 3 == 0):
        n = E.shape[0] // 3
        return [E[3*i:3*i+3, :] for i in range(n)]
    return [E[:3, :3]]

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

    # Recover pose for all possible decompositions
    for Ei in split_essential_candidates(E):
        try:
            pose_inliers, R, t, _ = cv2.recoverPose(Ei, pts1, pts2, K)
        except cv2.error:
            continue

        if best is None or int(pose_inliers) > best[0]:
            best = (int(pose_inliers), R, t)

    if best is None:
        return None, None, 0

    return best[1], best[2], best[0]


# Step 7: Wrapper functions. One geometry wrapper, and one for the whole pipeline.


def estimate_pose_two_images(img1_bgr, img2_bgr, K,
                             pixels=800000,
                             nfeatures=10000,
                             ratio_thresh=0.75,
                             max_matches=500,
                             use_symmetry=True,
                             prob=0.999,
                             threshold=1.0,
                             min_E_inliers=5):

    img1c_, g1, s1 = preprocess_bgr(img1_bgr, target_area=pixels)
    img2c_, g2, s2 = preprocess_bgr(img2_bgr, target_area=pixels)

    if g1 is None or g2 is None:
        return None, None

    # if abs(s1 - s2) > 1e-6:
    #     print(f"[WARN] Different resize scales: s1={s1}, s2={s2}. Results may be inconsistent.")

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

    if len(matches) < 8:
        print("[ERROR] Not enough matches (<8).")
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


    if pts1_in.shape[0] < min_E_inliers:
        print(f"[POSE] Too few E inliers: {pts1_in.shape[0]} < {min_E_inliers}")
        return None, None
    
    if pts1_in.shape[0] >= min_E_inliers and pts1_in.shape[0] < 12:
        print(f"Low Confidence Pose: Only {pts1_in.shape[0]} inliers found. Results may be inaccurate.")

    # ---- Pose recovery ----
    R, t, pose_inliers = recover_pose_from_E(E, pts1_in, pts2_in, K_use)
    if R is None:
        return None, None

    return R, t

# def estimate_pose_two_images(img1_bgr, img2_bgr, K,
#                              max_width=None,
#                              nfeatures=2500,
#                              ratio_thresh=0.75,
#                              max_matches=500,
#                              use_symmetry=True,
#                              prob=0.999,
#                              threshold=1.0,
#                              min_E_inliers=15,
#                              ):

#     img1c, g1, s1 = preprocess_bgr(img1_bgr, max_width=max_width)
#     img2c, g2, s2 = preprocess_bgr(img2_bgr, max_width=max_width)

#     if g1 is None or g2 is None:
#         return None, None

#     if abs(s1 - s2) > 1e-6:
#         print(f"[WARN] Different resize scales: s1={s1}, s2={s2}. Results may be inconsistent.")

#     K_use = scale_intrinsics(K, s1) if s1 != 1.0 else K
                
        
#     kpts1, desc1 = detect_sift_features(g1, nfeatures=nfeatures)
#     kpts2, desc2 = detect_sift_features(g2, nfeatures=nfeatures)

#     matches = match_descriptors(desc1, desc2,
#                                 ratio_thresh=ratio_thresh,
#                                 max_matches=max_matches,
#                                 use_symmetry=use_symmetry)
    
#     if len(matches) < 8:
#         print("[ERROR] Not enough matches (<8).")
#         return None, None

#     pts1, pts2 = extract_matched_points(kpts1, kpts2, matches)

#     # ---- Estimate E + inliers  ----
#     E, inlier_mask = estimate_essential_matrix(
#         pts1, pts2, K_use,
#         prob=prob,
#         threshold=threshold
#     )
#     if E is None:
#         print("[ERROR] E estimation failed.")
#         return None, None

    
#     pts1_in = pts1[inlier_mask]
#     pts2_in = pts2[inlier_mask]

#     if pts1_in.shape[0] < min_E_inliers:
#         print(f"[POSE] Too few E inliers: {pts1_in.shape[0]} < {min_E_inliers}")
#         return None, None

#     # ---- Pose recovery ----
#     R, t, pose_inliers = recover_pose_from_E(E, pts1_in, pts2_in, K_use)
#     if R is None:
#         print("[ERROR] recoverPose failed.")
#         return None, None

#     return R, t


# Helper Functions

# Helper 1: Load the calibration from a .npz file

def load_calibration_npz(npz_path, K_key="K", dist_key_candidates=("dist", "distCoeffs", "D")):
    """
    Load calibration from a .npz file.
    Expects at least K under key K_key (default "K").
    Optionally loads distortion under common keys.
    """
    data = np.load(npz_path)
    if K_key not in data:
        raise KeyError(f"Could not find '{K_key}' in {npz_path}. Keys found: {list(data.keys())}")

    K = data[K_key].astype(np.float64)

    dist = None
    for key in dist_key_candidates:
        if key in data:
            dist = data[key]
            break

    return K, dist

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

    K, dist = load_calibration_npz(calib_path)


    # Dist is loaded for future undistortion if needed, but not used in this pipeline. / ARUCO
    # print("[INFO - CAMERA INTRINSICS] Loaded K:")
    
    # print(K)
    
    print("\n[INFO] Starting pose estimation...")

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
    min_E_inliers=12,
    debug=True
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
    
    print(f"[t norm] {float(np.linalg.norm(t)):.6f}  (VO is direction-only, so ~1)")


if __name__ == "__main__":
    main()
