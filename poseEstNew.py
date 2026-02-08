import sys
import cv2
import numpy as np

import os
import math

"""
Pipeline:

# Step 1: Image Acquisition and Preprocessing
- Load two images.
- Convert to gray-scale.
- Resize if necessary.


# Step 2: Feature Detection and Matching
- Detect keypoints using ORB.
- Extract descriptors.
- Match descriptors using BFMatcher with Lowe's ratio test. Using symmetric matching (time-consuming but more accurate).

# Step 3: Find the Essential Matrix
- Estimate the Essential matrix using RANSAC.
- Validate the Essential matrix. We can get more than one candidate stacked vertically.


# Step 4: Recover Pose from Essential Matrix
- Decompose the Essential matrix to get rotation and translation.
- If multiple candidates, choose the one with the most inliers.

# 5: Output Results
- Print the estimated rotation and translation.
"""

### Visualization and Debugging ###

def ensure_dir(path):
    if path is None:
        return
    os.makedirs(path, exist_ok=True)

def imshow_or_save(win, img, wait=0, out_dir=None, fname=None, enable_imshow=True):
    """
    If out_dir is provided -> saves image. If enable_imshow -> shows window.
    """
    if out_dir is not None and fname is not None:
        ensure_dir(out_dir)
        cv2.imwrite(os.path.join(out_dir, fname), img)

    if enable_imshow:
        cv2.imshow(win, img)
        cv2.waitKey(wait)

def draw_keypoints(img_bgr, kpts, max_kpts=5000):
    """
    Draw keypoints (clipped to max_kpts to keep it fast/clean).
    """
    if len(kpts) > max_kpts:
        kpts = kpts[:max_kpts]
    return cv2.drawKeypoints(img_bgr, kpts, None,
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

def draw_matches(img1, img2, kpts1, kpts2, matches, max_to_draw=200):
    """
    Draw first N matches.
    """
    matches = matches[:max_to_draw]
    return cv2.drawMatches(img1, kpts1, img2, kpts2, matches, None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

def draw_inlier_matches_img(img1, img2, kpts1, kpts2, matches, inlier_mask, max_to_draw=200):
    """
    Draw only matches where inlier_mask is True.
    """
    if inlier_mask is None:
        return None
    inlier_mask = np.asarray(inlier_mask).ravel().astype(bool)
    inlier_matches = [m for m, inl in zip(matches, inlier_mask) if inl]
    inlier_matches = inlier_matches[:max_to_draw]
    if len(inlier_matches) == 0:
        return None

    return cv2.drawMatches(img1, kpts1, img2, kpts2, inlier_matches, None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

def print_k_stats(K, name="K"):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    print(f"[{name}] fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f}")

def compute_match_distance_stats(matches):
    if not matches:
        return None
    d = np.array([m.distance for m in matches], dtype=np.float64)
    return {
        "min": float(np.min(d)),
        "p25": float(np.percentile(d, 25)),
        "median": float(np.median(d)),
        "p75": float(np.percentile(d, 75)),
        "max": float(np.max(d)),
        "mean": float(np.mean(d)),
    }

def draw_epipolar_lines(img1_bgr, img2_bgr, pts1, pts2, F, inlier_mask=None, max_pairs=30):
    """
    Draw epipolar lines for pairs (pts1 <-> pts2) using F.
    If inlier_mask is provided, uses inliers only.
    """
    if F is None:
        return None, None

    h1, w1 = img1_bgr.shape[:2]
    h2, w2 = img2_bgr.shape[:2]

    if inlier_mask is not None:
        inlier_mask = np.asarray(inlier_mask).ravel().astype(bool)
        pts1 = pts1[inlier_mask]
        pts2 = pts2[inlier_mask]

    n = min(len(pts1), max_pairs)
    if n < 5:
        return None, None

    pts1s = pts1[:n].reshape(-1, 1, 2)
    pts2s = pts2[:n].reshape(-1, 1, 2)

    # lines in image1 for points in image2
    lines1 = cv2.computeCorrespondEpilines(pts2s, 2, F).reshape(-1, 3)
    # lines in image2 for points in image1
    lines2 = cv2.computeCorrespondEpilines(pts1s, 1, F).reshape(-1, 3)

    out1 = img1_bgr.copy()
    out2 = img2_bgr.copy()

    rng = np.random.default_rng(0)
    colors = rng.integers(0, 255, size=(n, 3), dtype=np.uint8)

    for i in range(n):
        a, b, c = lines1[i]
        if abs(b) > 1e-9:
            x0, y0 = 0, int(round((-c - a*0)/b))
            x1, y1 = w1, int(round((-c - a*w1)/b))
            col = tuple(int(x) for x in colors[i])
            cv2.line(out1, (x0, y0), (x1, y1), col, 1)
            cv2.circle(out1, tuple(np.int32(pts1[i])), 4, col, -1)

        a, b, c = lines2[i]
        if abs(b) > 1e-9:
            x0, y0 = 0, int(round((-c - a*0)/b))
            x1, y1 = w2, int(round((-c - a*w2)/b))
            col = tuple(int(x) for x in colors[i])
            cv2.line(out2, (x0, y0), (x1, y1), col, 1)
            cv2.circle(out2, tuple(np.int32(pts2[i])), 4, col, -1)

    return out1, out2

###########################################################################


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
        s = max_width / float(width) # Scale factor, obtained from width ratio which is calculated as new_width / old_width which means intuitively that if the image is wider than max_width, we need to scale it down.
        new_size = (int(width * s), int(height * s)) # (width, height)
        image_bgr = cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)
        
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return image_bgr, image_gray, s

def scale_intrinsics(K, s):
    K2 = K.copy().astype(np.float64)
    K2[0, 0] *= s # fx
    K2[1, 1] *= s # fy
    K2[0, 2] *= s # cx
    K2[1, 2] *= s # cy
    return K2

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

def match_descriptors(desc1, desc2, ratio_thresh=0.75, max_matches=500, use_symmetry=True, debug=True):
    if desc1 is None or desc2 is None:
        if debug:
            print("[MATCH] desc1 or desc2 is None -> 0 matches")
        return []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # 1 -> 2
    knn12 = bf.knnMatch(desc1, desc2, k=2)
    good12 = []
    for pair in knn12:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good12.append(m)

    if debug:
        print(f"[MATCH] ratio pass 1->2: {len(good12)}/{len(knn12)}")

    if not use_symmetry:
        good12 = sorted(good12, key=lambda m: m.distance)
        mutual = good12[:max_matches]
        if debug:
            stats = compute_match_distance_stats(mutual)
            print(f"[MATCH] one-way matches: {len(mutual)} stats={stats}")
        return mutual

    # 2 -> 1
    knn21 = bf.knnMatch(desc2, desc1, k=2)
    good21 = []
    for pair in knn21:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good21.append(m)

    if debug:
        print(f"[MATCH] ratio pass 2->1: {len(good21)}/{len(knn21)}")

    # mutual consistency
    pairs21 = set((m.trainIdx, m.queryIdx) for m in good21)  # (idx_in_desc1, idx_in_desc2)

    mutual = [m for m in good12 if (m.queryIdx, m.trainIdx) in pairs21]
    mutual = sorted(mutual, key=lambda m: m.distance)

    if debug:
        print(f"[MATCH] mutual matches: {len(mutual)} (before cap={max_matches})")
        stats = compute_match_distance_stats(mutual)
        if stats:
            print(f"[MATCH] dist stats: {stats}")

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
        return None, None
    
    
    inlier_mask = mask.ravel().astype(bool)  # 1D boolean mask

    #### Debug info  --------------------------------------- ######
    if debug:
        num_inliers = int(inlier_mask.sum())
        print(f"E shape: {None if E is None else E.shape}, mask: {None if mask is None else mask.shape}")
        print(f"[E] Inliers: {num_inliers}/{len(pts1)} "
            f"({num_inliers/len(pts1):.2f})")
        
    ##### --------------------------------------- ######

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

# Step 7.1
def estimate_pose_from_points(pts1, pts2, K, prob=0.999, threshold=1.0, min_E_inliers=15, debug=False):
    E, inlier_mask = estimate_essential_matrix(pts1, pts2, K, prob=prob, threshold=threshold, debug=debug)
    if E is None:
        return None, None, None

    pts1_in = pts1[inlier_mask]
    pts2_in = pts2[inlier_mask]

    if pts1_in.shape[0] < min_E_inliers:
        if debug:
            print(f"[POSE] Too few E inliers: {pts1_in.shape[0]} < {min_E_inliers}")
        return None, None, None

    R, t, pose_inliers = recover_pose_from_E(E, pts1_in, pts2_in, K)
    if R is None:
        return None, None, None

    if debug:
        print(f"[POSE] recoverPose inliers: {pose_inliers}/{len(pts1_in)}")

    return R, t, inlier_mask

# Step 7.2: Full pipeline wrapper
def estimate_pose_two_images(img1_bgr, img2_bgr, K,
                             max_width=None,
                             nfeatures=2500,
                             ratio_thresh=0.75,
                             max_matches=500,
                             use_symmetry=True,
                             prob=0.999,
                             threshold=1.0,
                             min_E_inliers=15,
                             debug=True,
                             viz=True,
                             viz_dir="debug_viz",
                             enable_imshow=True):

    img1c, g1, s1 = preprocess_bgr(img1_bgr, max_width=max_width)
    img2c, g2, s2 = preprocess_bgr(img2_bgr, max_width=max_width)

    if g1 is None or g2 is None:
        return None, None

    if abs(s1 - s2) > 1e-6 and debug:
        print(f"[WARN] Different resize scales: s1={s1}, s2={s2}. Results may be inconsistent.")

    K_use = scale_intrinsics(K, s1) if s1 != 1.0 else K

    if debug:
        print_k_stats(K, "K_orig")
        if s1 != 1.0:
            print(f"[INFO] resize scale s={s1:.4f}")
            print_k_stats(K_use, "K_scaled")

    kpts1, desc1 = detect_orb_features(g1, nfeatures=nfeatures)
    kpts2, desc2 = detect_orb_features(g2, nfeatures=nfeatures)

    if debug:
        print(f"[ORB] nfeatures={nfeatures} -> detected kpts: img1={len(kpts1)} img2={len(kpts2)}")
        if desc1 is None or desc2 is None:
            print(f"[ORB] desc1 None? {desc1 is None}, desc2 None? {desc2 is None}")

    if viz:
        kp1_img = draw_keypoints(img1c, kpts1)
        kp2_img = draw_keypoints(img2c, kpts2)
        imshow_or_save("Keypoints img1", kp1_img, wait=1, out_dir=viz_dir, fname="01_keypoints_img1.png", enable_imshow=enable_imshow)
        imshow_or_save("Keypoints img2", kp2_img, wait=1, out_dir=viz_dir, fname="02_keypoints_img2.png", enable_imshow=enable_imshow)

    matches = match_descriptors(desc1, desc2,
                                ratio_thresh=ratio_thresh,
                                max_matches=max_matches,
                                use_symmetry=use_symmetry,
                                debug=debug)

    if debug:
        print(f"[MATCH] use_symmetry={use_symmetry} ratio={ratio_thresh} max_matches={max_matches}")
        print(f"[MATCH] final matches={len(matches)}")

    if viz and len(matches) > 0:
        mimg = draw_matches(img1c, img2c, kpts1, kpts2, matches, max_to_draw=200)
        imshow_or_save("Raw matches", mimg, wait=1, out_dir=viz_dir, fname="03_raw_matches.png", enable_imshow=enable_imshow)

    if len(matches) < 8:
        if debug:
            print("[ERROR] Not enough matches (<8).")
        return None, None

    pts1, pts2 = extract_matched_points(kpts1, kpts2, matches)

    # ---- Estimate E + inliers (we already have your function) ----
    E, inlier_mask = estimate_essential_matrix(
        pts1, pts2, K_use,
        prob=prob,
        threshold=threshold,
        debug=debug
    )
    if E is None:
        if debug:
            print("[ERROR] E estimation failed.")
        return None, None

    if viz:
        inl_img = draw_inlier_matches_img(img1c, img2c, kpts1, kpts2, matches, inlier_mask, max_to_draw=200)
        if inl_img is not None:
            imshow_or_save("Inlier matches (E-RANSAC)", inl_img, wait=1, out_dir=viz_dir, fname="04_inlier_matches.png", enable_imshow=enable_imshow)
        else:
            if debug:
                print("[VIZ] No inlier matches to draw.")

    pts1_in = pts1[inlier_mask]
    pts2_in = pts2[inlier_mask]

    if debug:
        print(f"[E] inliers={len(pts1_in)}/{len(pts1)} threshold={threshold} prob={prob}")
        print(f"[E] candidates: {len(split_essential_candidates(E))}")

    if pts1_in.shape[0] < min_E_inliers:
        if debug:
            print(f"[POSE] Too few E inliers: {pts1_in.shape[0]} < {min_E_inliers}")
        return None, None

    # Optional epipolar visualization: need F.
    # Relationship: E = K^T F K  => F = K^{-T} E K^{-1}
    if viz:
        try:
            Kinv = np.linalg.inv(K_use)
            # choose the first E candidate for epipolar visualization (visual only!)
            E0 = split_essential_candidates(E)[0]
            F = Kinv.T @ E0 @ Kinv
            epi1, epi2 = draw_epipolar_lines(img1c, img2c, pts1, pts2, F, inlier_mask=inlier_mask, max_pairs=30)
            if epi1 is not None and epi2 is not None:
                imshow_or_save("Epilines img1", epi1, wait=1, out_dir=viz_dir, fname="05_epilines_img1.png", enable_imshow=enable_imshow)
                imshow_or_save("Epilines img2", epi2, wait=1, out_dir=viz_dir, fname="06_epilines_img2.png", enable_imshow=enable_imshow)
        except Exception as e:
            if debug:
                print(f"[VIZ] Epipolar draw failed: {e}")

    # ---- Pose recovery ----
    R, t, pose_inliers = recover_pose_from_E(E, pts1_in, pts2_in, K_use)
    if R is None:
        if debug:
            print("[ERROR] recoverPose failed.")
        return None, None

    if debug:
        print(f"[POSE] recoverPose inliers: {pose_inliers}/{len(pts1_in)}")
        print("[POSE] t is direction only (scale unknown).")

    if enable_imshow and viz:
        # keep windows until keypress at end if you want
        print("[INFO] Press any key on an OpenCV window to close all.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return R, t


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

# Helper 2.1: Calculate the axis-angle from a rotation matrix
def rot_to_axis_angle_deg(R):
    # Rodrigues: rotation vector rvec where ||rvec|| = angle (radians)
    rvec, _ = cv2.Rodrigues(R)
    angle_rad = float(np.linalg.norm(rvec))
    angle_deg = float(np.degrees(angle_rad))
    axis = (rvec.reshape(-1) / (angle_rad + 1e-12)).tolist()
    return angle_deg, axis

# Helper 2.2: Calculate the Euler angles from a rotation matrix
def rotmat_to_euler_deg_zyx(R):
    # ZYX: yaw(Z), pitch(Y), roll(X)
    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6
    if not singular:
        roll  = math.atan2(R[2,1], R[2,2])
        pitch = math.atan2(-R[2,0], sy)
        yaw   = math.atan2(R[1,0], R[0,0])
    else:
        roll  = math.atan2(-R[1,2], R[1,1])
        pitch = math.atan2(-R[2,0], sy)
        yaw   = 0.0
    return np.degrees([roll, pitch, yaw])  # roll, pitch, yaw

# Helper 3: Translation
def t_to_dir_and_angles(t):
    v = t.reshape(-1).astype(np.float64)
    v = v / (np.linalg.norm(v) + 1e-12)  # unit direction
    az = float(np.degrees(np.arctan2(v[0], v[2])))   # left/right in XZ
    el = float(np.degrees(np.arctan2(v[1], math.sqrt(v[0]**2 + v[2]**2))))  # up/down
    return v, az, el


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


def main():
    """
    Usage:
        python your_script.py img1.jpg img2.jpg calib.npz
    Where calib.npz contains at least 'K'.
    """
    
    print("[DEBUG] main started, argv =", sys.argv)

    if len(sys.argv) != 4:
        print("Usage: python your_script.py path/to/img1.jpg path/to/img2.jpg path/to/calib.npz")
        sys.exit(1)

    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    calib_path = sys.argv[3]

    img1_bgr = cv2.imread(img1_path)
    img2_bgr = cv2.imread(img2_path)

    if img1_bgr is None:
        print(f"Error: could not load image 1: {img1_path}")
        sys.exit(1)
    if img2_bgr is None:
        print(f"Error: could not load image 2: {img2_path}")
        sys.exit(1)

    K, dist = load_calibration_npz(calib_path)


    # Dist is loaded for future undistortion if needed, but not used in this pipeline. / ARUCO
    print("[INFO] Loaded K:")
    print(K)
    
    print("\n[INFO] Starting pose estimation...")
    print("[DEBUG] calling estimate_pose_two_images")

    # Core pipeline
    R, t = estimate_pose_two_images(
    img1_bgr, img2_bgr, K,
    max_width=900,
    nfeatures= 4500,
    ratio_thresh=0.75,
    max_matches=500,
    use_symmetry=True,
    prob=0.999,
    threshold=1.0,
    min_E_inliers=15,
    debug=True,
    viz=True,
    viz_dir="debug_viz",
    enable_imshow=True
)


    if R is None or t is None:
        print("[ERROR] Pose estimation failed.")
        sys.exit(2)

    print("\n================ FINAL POSE ================")
    print("[R] Rotation matrix (camera1 -> camera2):")
    print(R)
    print("[t] Translation direction (up to scale):")
    print(t.ravel())
    print("===========================================\n")
    
    ang_deg, axis = rot_to_axis_angle_deg(R)
    print(f"[R] Axis-angle: angle={ang_deg:.3f} deg, axis=({axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f})")
    
    roll, pitch, yaw = rotmat_to_euler_deg_zyx(R)
    print(f"[ROT euler ZYX deg] roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}")
    
    # ---- Translation reporting (direction only) ----
    t_dir, az, el = t_to_dir_and_angles(t)
    print(f"[t_dir unit] {t_dir}")
    print(f"[t_dir angles] azimuth_deg={az:.2f}, elevation_deg={el:.2f}")

    print(f"[t norm] {float(np.linalg.norm(t)):.6f}  (VO is direction-only, so ~1)")


if __name__ == "__main__":
    main()
