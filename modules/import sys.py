import sys
import cv2
import numpy as np
import csv
from pathlib import Path

# ============================================================
# Preprocess
# ============================================================
def preprocess_frame(frame_bgr, max_width=900):
    """
    Resize (if needed) + convert to grayscale.
    Returns (frame_bgr_resized, gray)
    """
    if frame_bgr is None:
        return None, None

    h, w = frame_bgr.shape[:2]
    if max_width is not None and w > max_width:
        scale = max_width / float(w)
        new_size = (int(w * scale), int(h * scale))
        frame_bgr = cv2.resize(frame_bgr, new_size, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return frame_bgr, gray


# ============================================================
# ORB + Matching
# ============================================================
def detect_orb_features(gray_img, n_features=2000):
    # Slightly higher features helps slow forward motion
    orb = cv2.ORB_create(nfeatures=int(n_features))
    keypoints, descriptors = orb.detectAndCompute(gray_img, None)
    return keypoints, descriptors


def match_descriptors(desc1, desc2, ratio_thresh=0.75, max_matches=400, use_symmetry=True):
    """
    ORB matching:
    - KNN match + Lowe ratio test
    - optional symmetric check
    - cap matches for stability/speed
    """
    if desc1 is None or desc2 is None:
        return []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # KNN from 1->2
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

    # KNN from 2->1 for symmetry
    knn21 = bf.knnMatch(desc2, desc1, k=2)
    good21 = []
    for pair in knn21:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good21.append(m)

    pairs21 = set((m.trainIdx, m.queryIdx) for m in good21)  # (idx_in_desc1, idx_in_desc2)

    mutual = []
    for m in good12:
        if (m.queryIdx, m.trainIdx) in pairs21:
            mutual.append(m)

    mutual = sorted(mutual, key=lambda m: m.distance)
    return mutual[:max_matches]


def extract_matched_points(kpts1, kpts2, matches):
    pts1 = np.array([kpts1[m.queryIdx].pt for m in matches], dtype=np.float32)
    pts2 = np.array([kpts2[m.trainIdx].pt for m in matches], dtype=np.float32)
    return pts1, pts2


# ============================================================
# Intrinsics (guessed; replace with real calibration later)
# ============================================================
def guess_intrinsics(image_shape, f_factor=0.9):
    """
    Guess intrinsics from image size.
    image_shape is (H, W).
    """
    h, w = image_shape
    f = f_factor * max(h, w)
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], dtype=np.float64)
    return K


# ============================================================
# Pose estimation (E + recoverPose)
# ============================================================
def estimate_relative_pose(gray1, gray2, K,
                           n_features=2000,
                           ratio=0.75,
                           ransac_thresh=1.0,
                           prob=0.999):
    """
    Returns:
      ok (bool), R (3x3), t (3x1), stats (dict)
    """
    kpts1, desc1 = detect_orb_features(gray1, n_features=n_features)
    kpts2, desc2 = detect_orb_features(gray2, n_features=n_features)

    matches = match_descriptors(desc1, desc2, ratio_thresh=ratio, max_matches=400, use_symmetry=True)

    stats = {
        "kpts1": len(kpts1),
        "kpts2": len(kpts2),
        "matches_good": len(matches),
        "matches_E_inliers": 0,
        "inliers_pose": 0,
    }

    if len(matches) < 8:
        stats["_fail_reason"] = "not_enough_matches"
        return False, None, None, stats

    pts1, pts2 = extract_matched_points(kpts1, kpts2, matches)

    E, mask_E = cv2.findEssentialMat(
        pts1, pts2, K,
        method=cv2.RANSAC,
        prob=prob,
        threshold=ransac_thresh
    )

    if E is None:
        stats["_fail_reason"] = "E_none"
        return False, None, None, stats

    # Sometimes E can be stacked (multiple solutions); keep the first 3x3 for stability.
    if E.ndim == 2 and E.shape[0] > 3:
        E = E[:3, :3]

    mask_E = mask_E.ravel().astype(bool)
    stats["matches_E_inliers"] = int(mask_E.sum())

    pts1_in = pts1[mask_E]
    pts2_in = pts2[mask_E]

    if pts1_in.shape[0] < 8:
        stats["_fail_reason"] = "not_enough_E_inliers"
        return False, None, None, stats

    # recoverPose returns the inlier count directly (this is the TRUE count)
    inliers_pose, R, t, pose_mask = cv2.recoverPose(E, pts1_in, pts2_in, K)
    stats["inliers_pose"] = int(inliers_pose)

    # Store debug pack for visualization (FIXED: now exists)
    # We'll map pose inliers back to match indices via mask_E.
    stats["_debug"] = {
        "kpts1": kpts1,
        "kpts2": kpts2,
        "matches": matches,
        "mask_E": mask_E,                 # length = len(matches)
        "pose_mask": pose_mask.ravel()    # length = sum(mask_E)
    }

    return True, R, t, stats


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


# ============================================================
# Debug visualization: draw pose inliers
# ============================================================
def draw_pose_inliers(frame1_bgr, frame2_bgr, dbg, max_draw=60):
    """
    Draw matches that:
      - passed EssentialMat RANSAC (mask_E)
      - and were inliers in recoverPose (pose_mask)
    """
    kpts1 = dbg["kpts1"]
    kpts2 = dbg["kpts2"]
    matches = dbg["matches"]
    mask_E = dbg["mask_E"]
    pose_mask = dbg["pose_mask"]

    # Indices of matches that are E-inliers
    e_inlier_match_indices = np.flatnonzero(mask_E)

    # pose_mask aligns with the E-inlier points (pts1_in / pts2_in)
    pose_inliers = np.flatnonzero(pose_mask.astype(bool))
    # map pose inlier positions back to match indices
    selected_match_indices = e_inlier_match_indices[pose_inliers]

    inlier_matches = [matches[i] for i in selected_match_indices[:max_draw]]

    vis = cv2.drawMatches(
        frame1_bgr, kpts1,
        frame2_bgr, kpts2,
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return vis


# ============================================================
# Video processing
# ============================================================
def process_video(video_path,
                  stride=4,
                  max_width=900,
                  n_features=2000,
                  ratio=0.75,
                  ransac_thresh=1.0,
                  t_step=1.0,
                  min_inliers=25,
                  csv_path="trajectory_v2.csv",
                  print_every=1,
                  debug_show_every=0):
    """
    Monocular VO (up-to-scale) using Essential matrix + recoverPose.

    - stride: process every Nth frame (increases baseline; very important for slow forward motion)
    - min_inliers: minimum recoverPose inliers to accept an update
    - print_every: 1 prints every processed pair; 20 prints every 20, etc.
    - debug_show_every: 0 disables; otherwise shows debug window every N successful steps.
    """

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame from video.")

    prev_bgr, prev_gray = preprocess_frame(frame0, max_width=max_width)
    if prev_gray is None:
        raise RuntimeError("Failed preprocessing the first frame.")

    K = guess_intrinsics(prev_gray.shape)
    print("[INFO] Using guessed intrinsics K:")
    print(K)
    print(f"[INFO] stride={stride}, n_features={n_features}, ransac_thresh={ransac_thresh}, "
          f"min_inliers={min_inliers}, max_width={max_width}")

    T_global = np.eye(4, dtype=np.float64)

    csv_path = Path(csv_path)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step_idx", "frame_idx",
            "kpts_prev", "kpts_curr",
            "good_matches", "E_inliers", "pose_inliers",
            "tx", "ty", "tz"
        ])

        frame_idx = 0
        step_idx = 0
        successful_steps = 0

        while True:
            # advance by stride frames
            for _ in range(stride):
                ok, frame = cap.read()
                frame_idx += 1
                if not ok:
                    cap.release()
                    cv2.destroyAllWindows()
                    print("[INFO] End of video.")
                    return

            curr_bgr, curr_gray = preprocess_frame(frame, max_width=max_width)
            if curr_gray is None:
                continue

            ok_pose, R, t, stats = estimate_relative_pose(
                prev_gray, curr_gray, K,
                n_features=n_features,
                ratio=ratio,
                ransac_thresh=ransac_thresh
            )

            # Print per pair (or every N pairs)
            if print_every and (step_idx % print_every == 0):
                fail_reason = stats.get("_fail_reason", "")
                print(f"[PAIR] step={step_idx:05d} frame={frame_idx:05d} "
                      f"kpts=({stats['kpts1']},{stats['kpts2']}) "
                      f"matches={stats['matches_good']} "
                      f"E_inl={stats.get('matches_E_inliers', 0)} "
                      f"pose_inl={stats['inliers_pose']} "
                      f"{('FAIL:'+fail_reason) if (not ok_pose) else ''}")

            if ok_pose and stats["inliers_pose"] >= min_inliers:
                T_rel = Rt_to_T(R, t, t_scale=t_step)
                T_global = T_global @ T_rel

                tx, ty, tz = T_global[:3, 3].tolist()

                writer.writerow([
                    step_idx, frame_idx,
                    stats["kpts1"], stats["kpts2"],
                    stats["matches_good"],
                    stats.get("matches_E_inliers", 0),
                    stats["inliers_pose"],
                    tx, ty, tz
                ])

                successful_steps += 1

                # Optional debug visualization
                if debug_show_every and (successful_steps % debug_show_every == 0):
                    vis = draw_pose_inliers(prev_bgr, curr_bgr, stats["_debug"], max_draw=60)
                    cv2.imshow("Pose inlier matches (debug)", vis)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC
                        break

            # Slide window regardless (keeps moving)
            prev_gray = curr_gray
            prev_bgr = curr_bgr
            step_idx += 1

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# CLI
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python video_vo.py path/to/video.mp4 [csv_out]")
        sys.exit(1)

    video_path = sys.argv[1]
    csv_out = sys.argv[2] if len(sys.argv) >= 3 else "trajectory_v2.csv"

    process_video(
        video_path=video_path,
        stride=4,               # increase baseline for slow forward motion
        max_width=900,
        n_features=2000,
        ratio=0.75,
        ransac_thresh=1.0,       # pixels; 0.5 can be too strict after resizing/noise
        t_step=1.0,              # arbitrary units (monocular scale unknown)
        min_inliers=25,          # IMPORTANT: avoid accepting degenerate poses
        csv_path=csv_out,
        print_every=1,           # print every processed pair
        debug_show_every=0       # set to e.g. 10 to visualize every 10 successful steps
    )
    print(f"[INFO] Saved trajectory CSV: {csv_out}")


if __name__ == "__main__":
    main()
