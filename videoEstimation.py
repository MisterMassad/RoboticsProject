import sys
import cv2
import numpy as np
import csv
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# Preprocess
# ============================================================
def preprocess_frame(frame_bgr, max_width=900):
    """
    Resize (if needed) + grayscale.
    Returns: frame_bgr_resized, gray, scale
    """
    if frame_bgr is None:
        return None, None, 1.0

    h, w = frame_bgr.shape[:2]
    scale = 1.0
    if max_width is not None and w > max_width:
        scale = max_width / float(w)
        new_size = (int(w * scale), int(h * scale))
        frame_bgr = cv2.resize(frame_bgr, new_size, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return frame_bgr, gray, scale

# ============================================================
# ORB + Matching
# ============================================================
def detect_orb_features(gray_img, n_features=2500):
    orb = cv2.ORB_create(nfeatures=int(n_features))
    kpts, desc = orb.detectAndCompute(gray_img, None)
    return kpts, desc


def match_descriptors(desc1, desc2, ratio_thresh=0.75, max_matches=600, use_symmetry=True):
    if desc1 is None or desc2 is None:
        return []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # KNN 1->2
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

    # KNN 2->1 for symmetry check
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
# Intrinsics
# ============================================================
def guess_intrinsics(image_shape, f_factor=0.9):
    h, w = image_shape
    f = f_factor * max(h, w)
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], dtype=np.float64)
    return K


def scale_intrinsics(K, s):
    """Scale fx,fy,cx,cy by s after resizing the image."""
    K2 = K.copy().astype(np.float64)
    K2[0, 0] *= s
    K2[1, 1] *= s
    K2[0, 2] *= s
    K2[1, 2] *= s
    return K2


# ============================================================
# Helpers
# ============================================================
def split_essential_candidates(E):
    if E is None:
        return []
    if E.shape == (3, 3):
        return [E]
    if E.ndim == 2 and E.shape[1] == 3 and (E.shape[0] % 3 == 0):
        n = E.shape[0] // 3
        return [E[3*i:3*i+3, :] for i in range(n)]
    return [E[:3, :3]]


def normalize_vec(v):
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    return v / n


def undistort_points(pts, K, dist):
    """
    pts: Nx2 float32 pixel points
    returns Nx2 float32 undistorted pixel points (still pixel coords, because P=K)
    """
    if dist is None:
        return pts
    pts_ = pts.reshape(-1, 1, 2).astype(np.float32)
    und = cv2.undistortPoints(pts_, K, dist, P=K)
    return und.reshape(-1, 2).astype(np.float32)


# ============================================================
# Pose estimation (NO planar rejection in FF mode)
# ============================================================
def estimate_relative_pose(gray1, gray2, K,
                           distCoeffs=None,
                           n_features=2500,
                           ratio=0.75,
                           ransac_thresh_E=1.5,
                           prob=0.999,
                           min_parallax_px=0.0):
    """
    Returns: ok, R, t, stats
    """
    kpts1, desc1 = detect_orb_features(gray1, n_features=n_features)
    kpts2, desc2 = detect_orb_features(gray2, n_features=n_features)

    matches = match_descriptors(desc1, desc2, ratio_thresh=ratio, max_matches=600, use_symmetry=True)

    stats = {
        "kpts1": len(kpts1),
        "kpts2": len(kpts2),
        "matches_good": len(matches),
        "matches_E_inliers": 0,
        "inliers_pose": 0,
        "E_candidates": 0,
        "median_parallax": 0.0,
        "_fail_reason": ""
    }

    if len(matches) < 8:
        stats["_fail_reason"] = "not_enough_matches"
        return False, None, None, stats

    pts1, pts2 = extract_matched_points(kpts1, kpts2, matches)

    # Undistort if provided
    pts1u = undistort_points(pts1, K, distCoeffs)
    pts2u = undistort_points(pts2, K, distCoeffs)

    # Essential matrix
    E, mask_E = cv2.findEssentialMat(
        pts1u, pts2u, K,
        method=cv2.RANSAC,
        prob=prob,
        threshold=float(ransac_thresh_E)
    )

    if E is None or mask_E is None:
        stats["_fail_reason"] = "E_none"
        return False, None, None, stats

    mask_E = mask_E.ravel().astype(bool)
    stats["matches_E_inliers"] = int(mask_E.sum())

    pts1_in = pts1u[mask_E]
    pts2_in = pts2u[mask_E]

    if pts1_in.shape[0] < 8:
        stats["_fail_reason"] = "not_enough_E_inliers"
        return False, None, None, stats

    # Parallax (debug info + optional gating)
    flow = np.linalg.norm(pts2_in - pts1_in, axis=1)
    stats["median_parallax"] = float(np.median(flow)) if flow.size else 0.0
    if stats["median_parallax"] < float(min_parallax_px):
        stats["_fail_reason"] = "low_parallax"
        return False, None, None, stats

    # Choose best E candidate by recoverPose inliers
    E_candidates = split_essential_candidates(E)
    stats["E_candidates"] = len(E_candidates)

    best = None  # (inliers, R, t)
    for Ei in E_candidates:
        try:
            inl, R, t, _ = cv2.recoverPose(Ei, pts1_in, pts2_in, K)
        except cv2.error:
            continue
        if best is None or int(inl) > best[0]:
            best = (int(inl), R, t)

    if best is None:
        stats["_fail_reason"] = "recoverPose_failed"
        return False, None, None, stats

    stats["inliers_pose"] = best[0]
    return True, best[1], best[2], stats


# ============================================================
# Plotting (ALWAYS saves, even if only 1 point)
# ============================================================
def plot_trajectory_xy(traj_xyz, out_path):
    traj = np.array(traj_xyz, dtype=np.float64)
    x = traj[:, 0]
    y = traj[:, 1]

    plt.figure()
    if len(traj_xyz) >= 2:
        plt.plot(x, y, marker="o")
    else:
        plt.scatter(x, y)

    plt.title("VO trajectory (X-Y, arbitrary scale)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200)
    plt.close()
    print(f"[INFO] Saved trajectory PNG: {out_path}")


def save_interactive_3d_trajectory(traj_xyz, out_html):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[WARN] plotly not installed. Run: pip install plotly")
        return

    traj = np.array(traj_xyz, dtype=np.float64)
    x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="lines+markers" if len(traj_xyz) >= 2 else "markers",
        marker=dict(size=3),
        line=dict(width=4),
        name="Estimated trajectory"
    ))

    if len(traj_xyz) >= 1:
        fig.add_trace(go.Scatter3d(
            x=[x[0]], y=[y[0]], z=[z[0]],
            mode="markers+text",
            marker=dict(size=6),
            text=["START"],
            textposition="top center",
            name="Start"
        ))
        fig.add_trace(go.Scatter3d(
            x=[x[-1]], y=[y[-1]], z=[z[-1]],
            mode="markers+text",
            marker=dict(size=6),
            text=["END"],
            textposition="top center",
            name="End"
        ))

    fig.update_layout(
        title="Monocular VO Trajectory (3D, arbitrary scale)",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.write_html(str(out_html))
    print(f"[INFO] Saved interactive 3D HTML: {out_html}")


# ============================================================
# Frame-to-frame VO (stride=1 => every frame)
# ============================================================
def process_video_frame_to_frame(video_path,
                                 max_width=900,
                                 n_features=2500,
                                 ratio=0.75,
                                 ransac_thresh_E=1.5,
                                 min_parallax_px=0.0,
                                 stride=1,
                                 base_step=0.01,
                                 min_pose_inliers=15,
                                 print_every=1,
                                 K_in=None,
                                 distCoeffs=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame from video.")

    _, prev_gray, s0 = preprocess_frame(frame0, max_width=max_width)
    if prev_gray is None:
        raise RuntimeError("Failed preprocessing first frame.")

    # Choose/scale K
    if K_in is None:
        K = guess_intrinsics(prev_gray.shape)
        print("[INFO] Using guessed intrinsics K (based on resized frame):")
    else:
        K = scale_intrinsics(K_in, s0)
        print("[INFO] Using PROVIDED intrinsics K (scaled to resized frame):")

    print(K)
    print(f"[INFO] FF mode | stride={stride}, n_features={n_features}, ratio={ratio}, "
          f"ransac_E={ransac_thresh_E}, min_parallax_px={min_parallax_px}, "
          f"min_pose_inliers={min_pose_inliers}, base_step={base_step}, max_width={max_width}")

    # Global pose
    R_global = np.eye(3, dtype=np.float64)
    t_global = np.zeros(3, dtype=np.float64)
    traj = [t_global.copy()]

    # CSV
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f_pairs:
        w = csv.writer(f_pairs)
        w.writerow([
            "frame_idx",
            "kpts_prev", "kpts_curr",
            "matches", "E_inliers", "pose_inliers",
            "median_parallax",
            "pos_x", "pos_y", "pos_z",
            "accepted", "fail_reason"
        ])

        frame_idx = 0

        while True:
            # advance by stride frames (stride=1 means every frame)
            curr_gray = None
            for _ in range(stride):
                ok, frame = cap.read()
                if not ok:
                    curr_gray = None
                    break
                frame_idx += 1
                _, curr_gray, _ = preprocess_frame(frame, max_width=max_width)

            if curr_gray is None:
                break

            ok_pose, R_rel, t_rel, stats = estimate_relative_pose(
                prev_gray, curr_gray, K,
                distCoeffs=distCoeffs,
                n_features=n_features,
                ratio=ratio,
                ransac_thresh_E=ransac_thresh_E,
                min_parallax_px=min_parallax_px
            )

            E_inl = int(stats.get("matches_E_inliers", 0))
            pose_inl = int(stats.get("inliers_pose", 0))
            med_px = float(stats.get("median_parallax", 0.0))
            fail_reason = stats.get("_fail_reason", "")

            accepted = int(ok_pose and pose_inl >= int(min_pose_inliers))

            if accepted:
                # direction only (monocular scale unknown)
                t_cam = normalize_vec(t_rel.reshape(3))
                t_world_dir = R_global @ t_cam

                t_global = t_global + float(base_step) * t_world_dir
                R_global = R_global @ R_rel
                traj.append(t_global.copy())

            if print_every and (frame_idx % print_every == 0):
                print(f"[FF] frame={frame_idx:05d} "
                      f"kpts=({stats.get('kpts1',0)},{stats.get('kpts2',0)}) "
                      f"m={stats.get('matches_good',0)} "
                      f"E_inl={E_inl} pose_inl={pose_inl} "
                      f"parallax={med_px:.2f} accepted={accepted} "
                      f"{('FAIL:'+fail_reason) if (not ok_pose) else ''}")

            w.writerow([
                frame_idx,
                stats.get("kpts1", 0), stats.get("kpts2", 0),
                stats.get("matches_good", 0), E_inl, pose_inl,
                f"{med_px:.4f}",
                f"{t_global[0]:.6f}", f"{t_global[1]:.6f}", f"{t_global[2]:.6f}",
                accepted, fail_reason
            ])

            prev_gray = curr_gray  # ALWAYS slide

    cap.release()
    cv2.destroyAllWindows()

    # Save plots
    plot_trajectory_xy(traj, OUT_PNG)
    save_interactive_3d_trajectory(traj, OUT_HTML)

    print("\n================ FINAL POSE (arbitrary scale) ================")
    print("[FINAL] R_global =")
    print(R_global)
    print("[FINAL] t_global =", t_global)
    print("==============================================================\n")

    print(f"[INFO] Saved CSV : {OUT_CSV}")
    print(f"[INFO] Saved PNG : {OUT_PNG}")
    print(f"[INFO] Saved HTML: {OUT_HTML}")


# ============================================================
# CLI
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python vo_ff_fixed.py path/to/video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]

    # When your instructor gives you intrinsics:
    # K_in must match ORIGINAL video resolution before resizing.
    K_in = None
    distCoeffs = None

    process_video_frame_to_frame(
        video_path=video_path,
        max_width=900,
        n_features=2500,
        ratio=0.75,
        ransac_thresh_E=1.5,      # a bit looser helps iPhone + resize
        min_parallax_px=0.0,      # keep 0 for debugging
        stride=1,                 # every frame (smooth)
        base_step=0.01,           # visualization only
        min_pose_inliers=15,      # raise to 30 later for stability
        print_every=1,
        K_in=K_in,
        distCoeffs=distCoeffs
    )


if __name__ == "__main__":
    main()