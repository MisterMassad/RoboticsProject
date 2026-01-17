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
import csv
from pathlib import Path

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

def preprocess_frame(frame_bgr, max_width=900):
    if frame_bgr is None:
        return None, None
    
    height, width = frame_bgr.shape[:2]
    if width > max_width:
        scale = max_width / float(width)
        new_size = (int(width * scale), int(height * scale))
        frame_bgr = cv2.resize(frame_bgr, new_size, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    return frame_bgr, gray

# Step 2: Feature Detection using ORB

def detect_orb_features(gray_img, n_features=1000):
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(gray_img, None)
    return keypoints, descriptors


# Step 3: Match descriptors between two images
def match_descriptors(desc1, desc2, ratio_thresh=0.75, max_matches=400, use_symmetry=True):
    """
    ORB matching for VO:
    - KNN match + Lowe ratio test
    - optional symmetric (mutual) check to remove many false matches
    - cap matches to keep RANSAC stable and fast
    """
    if desc1 is None or desc2 is None:
        return []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # KNN from 1->2
    knn12 = bf.knnMatch(desc1, desc2, k=2)
    good12 = []
    for m, n in knn12:
        if m.distance < ratio_thresh * n.distance:
            good12.append(m)

    if not use_symmetry:
        # Sort by best distance and cap
        good12 = sorted(good12, key=lambda m: m.distance)
        return good12[:max_matches]

    # KNN from 2->1 (for symmetry)
    knn21 = bf.knnMatch(desc2, desc1, k=2)
    good21 = []
    for m, n in knn21:
        if m.distance < ratio_thresh * n.distance:
            good21.append(m)

    # Build a set of mutual pairs (trainIdx/queryIdx flipped)
    pairs21 = set((m.trainIdx, m.queryIdx) for m in good21)  # (idx_in_desc1, idx_in_desc2)

    mutual = []
    for m in good12:
        if (m.queryIdx, m.trainIdx) in pairs21:
            mutual.append(m)

    mutual = sorted(mutual, key=lambda m: m.distance)
    return mutual[:max_matches]


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



def estimate_relative_pose(gray1, gray2, K, n_features=1000, ratio=0.75, ransac_thresh=0.5):
    """
    Returns:
        ok (bool),
        R (3x3), t (3x1),
        stats dict
    """
    kpts1, desc1 = detect_orb_features(gray1, n_features=n_features)
    kpts2, desc2 = detect_orb_features(gray2, n_features=n_features)

    matches = match_descriptors(desc1, desc2, ratio_thresh=ratio)

    stats = {
        "kpts1": len(kpts1),
        "kpts2": len(kpts2),
        "matches_good": len(matches),
        "inliers_pose": 0
    }

    if len(matches) < 8:
        return False, None, None, None, stats

    pts1, pts2 = extract_matched_points(kpts1, kpts2, matches)

    E, mask_E = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=ransac_thresh)
    if E is None:
        return False, None, None, None, stats

    mask_E = mask_E.ravel().astype(bool)
    pts1_in = pts1[mask_E]
    pts2_in = pts2[mask_E]

    if pts1_in.shape[0] < 8:
        return False, None, None, None, stats

    inliers, R, t, pose_mask = cv2.recoverPose(E, pts1_in, pts2_in, K)
    stats["inliers_pose"] = int(inliers)
    stats["matches_good"] = len(matches)
    stats["matches_E_inliers"] = int(mask_E.sum())


    return True, R, t, pose_mask, stats


def Rt_to_T(R, t, t_scale=1.0):
    """
    Build 4x4 transform from R and t.
    Since monocular t is up-to-scale, we normalize and apply a constant step scale.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R

    t = t.reshape(3)
    norm = np.linalg.norm(t)
    if norm > 1e-9:
        t = (t / norm) * float(t_scale)

    T[:3, 3] = t
    return T


def draw_pose_inliers(frame1_bgr, frame2_bgr, debug_pack, max_draw=40):
    """
    Debug visualization: draw inlier matches used by recoverPose.
    """
    kpts1, kpts2, matches, pose_mask = debug_pack
    inlier_matches = [m for m, inl in zip(matches, pose_mask.ravel()) if inl]
    inlier_matches = inlier_matches[:max_draw]
    vis = cv2.drawMatches(
        frame1_bgr, kpts1,
        frame2_bgr, kpts2,
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return vis


def process_video(video_path,
                  stride=1,
                  max_width=900,
                  n_features=1000,
                  ratio=0.75,
                  ransac_thresh=1.0,
                  t_step=1.0,
                  csv_path="trajectory_v2.csv",
                  debug_show_every=0):
    """
    Version 2: monocular VO with guessed intrinsics.
    - stride: process every Nth frame
    - t_step: constant step size per successful update (arbitrary units)
    - debug_show_every: 0 disables; otherwise show debug window every N successful steps.
    """

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Read first frame
    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame from video.")

    frame0_bgr, prev_gray = preprocess_frame(frame0, max_width=max_width)
    if prev_gray is None:
        raise RuntimeError("Failed preprocessing the first frame.")

    K = guess_intrinsics(prev_gray.shape)
    print("[INFO] Using guessed intrinsics K:")
    print(K)

    T_global = np.eye(4, dtype=np.float64)

    # CSV logging
    csv_path = Path(csv_path)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step_idx", "frame_idx",
            "kpts_prev", "kpts_curr", "good_matches", "pose_inliers",
            "tx", "ty", "tz"
        ])

        frame_idx = 0
        step_idx = 0
        successful_steps = 0

        while True:
            # Skip frames according to stride
            for _ in range(stride):
                ok, frame = cap.read()
                frame_idx += 1
                if not ok:
                    cap.release()
                    print("[INFO] End of video.")
                    return

            curr_bgr, curr_gray = preprocess_frame(frame, max_width=max_width)
            if curr_gray is None:
                continue

            ok_pose, R, t, pose_mask, stats = estimate_relative_pose(
                prev_gray, curr_gray, K,
                n_features=n_features,
                ratio=ratio,
                ransac_thresh=ransac_thresh
            )

            if ok_pose and stats["inliers_pose"] >= 8:
                T_rel = Rt_to_T(R, t, t_scale=t_step)
                T_global = T_global @ T_rel

                tx, ty, tz = T_global[:3, 3].tolist()

                writer.writerow([
                    step_idx, frame_idx,
                    stats["kpts1"], stats["kpts2"], stats["matches_good"], stats["inliers_pose"],
                    tx, ty, tz
                ])

                successful_steps += 1
                if successful_steps % 20 == 0:
                    print(f"[INFO] step={step_idx} frame={frame_idx} inliers={stats['inliers_pose']} "
                          f"pos=({tx:.3f},{ty:.3f},{tz:.3f})")

                # Optional debug visualization (inlier matches)
                if debug_show_every and (successful_steps % debug_show_every == 0):
                    vis = draw_pose_inliers(frame0_bgr, curr_bgr, stats["_debug"], max_draw=40)
                    cv2.imshow("Pose inlier matches (debug)", vis)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC
                        break

                # Slide window
                prev_gray = curr_gray
                frame0_bgr = curr_bgr
                step_idx += 1

            else:
                # Log a failure row (optional). Here we just skip update.
                # Still slide window so we keep moving.
                prev_gray = curr_gray
                frame0_bgr = curr_bgr
                step_idx += 1

        cap.release()
        cv2.destroyAllWindows()

def main():
    if len(sys.argv) < 2:
        print("Usage: python video_vo_v2.py path/to/video.mp4 [csv_out]")
        sys.exit(1)

    video_path = sys.argv[1]
    csv_out = sys.argv[2] if len(sys.argv) >= 3 else "trajectory_v2.csv"

    process_video(
        video_path=video_path,
        stride=2,              # 30fps -> 15fps roughly
        max_width=900,
        n_features=1000,
        ratio=0.75,
        ransac_thresh=0.5,
        t_step=1.0,            # arbitrary units per successful step
        csv_path=csv_out,
        debug_show_every=0     # set to e.g. 10 to show debug every 10 successful steps
    )
    print(f"[INFO] Saved trajectory CSV: {csv_out}")


if __name__ == "__main__":
    main()