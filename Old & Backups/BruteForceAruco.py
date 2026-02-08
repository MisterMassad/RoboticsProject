"""
aruco_pose_from_video.py

- Loads camera intrinsics + distortion from calibration_filtered.npz
- Auto-detects the ArUco dictionary used in the video (tries many)
- Detects markers in every frame
- Estimates camera pose (R,t) per frame using solvePnP (with known marker size)
- Saves:
    1) poses_aruco.csv   (frame_idx, rvec, tvec, num_markers, ids)
    2) trajectory_aruco.png (XY plot)
    3) trajectory_aruco_3d.html (interactive 3D plot)
    4) aruco_debug.png   (a debug frame with detected markers drawn)
"""

import sys
import cv2
import numpy as np
import csv
from pathlib import Path
import matplotlib.pyplot as plt


# -------------------------
# Output paths (script dir)
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
OUT_CSV  = BASE_DIR / "poses_aruco.csv"
OUT_PNG  = BASE_DIR / "trajectory_aruco.png"
OUT_HTML = BASE_DIR / "trajectory_aruco_3d.html"
OUT_DBG  = BASE_DIR / "aruco_debug.png"


# -------------------------
# ArUco dicts to try
# -------------------------
ARUCO_DICTS_TO_TRY = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
]


# ============================================================
# Utils
# ============================================================
def ensure_aruco_available():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco not found. Install opencv-contrib-python:\n"
            "  pip install -U opencv-contrib-python"
        )


def load_calib_npz(npz_path: str):
    data = np.load(npz_path)
    print("[INFO] NPZ keys:", list(data.keys()))

    K = None
    dist = None

    for k in ["K", "camera_matrix", "mtx", "camMatrix", "intrinsics"]:
        if k in data:
            K = data[k]
            break

    for k in ["dist", "distCoeffs", "dist_coeffs", "d", "distortion"]:
        if k in data:
            dist = data[k]
            break

    if K is None:
        raise ValueError("Could not find camera matrix in NPZ. Check printed keys above.")
    if dist is None:
        print("[WARN] Could not find dist coeffs in NPZ. Using zeros.")
        dist = np.zeros((5, 1), dtype=np.float64)

    K = np.array(K, dtype=np.float64)
    dist = np.array(dist, dtype=np.float64).reshape(-1, 1)

    print("[INFO] K:\n", K)
    print("[INFO] dist shape:", dist.shape)
    return K, dist


def resize_keep_aspect(frame, max_width=None):
    if frame is None:
        return frame, 1.0
    if max_width is None:
        return frame, 1.0
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame, 1.0
    s = max_width / float(w)
    new_size = (int(w * s), int(h * s))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA), s


def scale_intrinsics(K, s):
    """Scale fx, fy, cx, cy by s after resizing."""
    K2 = K.copy().astype(np.float64)
    K2[0, 0] *= s
    K2[1, 1] *= s
    K2[0, 2] *= s
    K2[1, 2] *= s
    return K2


def make_detector_params():
    """
    Robust params for small / blurry markers.
    Works with both old and new OpenCV APIs.
    """
    try:
        params = cv2.aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 53
        params.adaptiveThreshWinSizeStep = 10
        params.adaptiveThreshConstant = 7
        params.minMarkerPerimeterRate = 0.02
        params.maxMarkerPerimeterRate = 4.0
        params.polygonalApproxAccuracyRate = 0.05
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 30
        params.cornerRefinementMinAccuracy = 0.1
        return params
    except Exception:
        # Old API
        params = cv2.aruco.DetectorParameters_create()
        return params


def detect_markers(gray, dict_name, params):
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    try:
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, rejected = detector.detectMarkers(gray)
    except Exception:
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    return corners, ids


def find_best_dictionary(video_path, K, dist, max_frames=150, stride=2, max_width=None):
    """
    Scores each dictionary by total detected markers over sample frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    params = make_detector_params()
    scores = {name: 0 for name in ARUCO_DICTS_TO_TRY}

    fidx = 0
    used = 0
    while used < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        fidx += 1
        if stride > 1 and (fidx % stride != 0):
            continue
        used += 1

        frame_r, s = resize_keep_aspect(frame, max_width=max_width)
        gray = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        for name in ARUCO_DICTS_TO_TRY:
            corners, ids = detect_markers(gray, name, params)
            if ids is not None:
                scores[name] += int(len(ids))

    cap.release()

    top5 = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
    print("[INFO] Dictionary scores (top 5):", top5)
    best_name, best_score = top5[0]
    if best_score == 0:
        print("[WARN] No dictionary produced any detections in sampled frames.")
    else:
        print("[INFO] Best dict:", best_name, "score=", best_score)
    return best_name


def marker_object_points(marker_length_m):
    """
    Marker corners in marker coordinate frame (Z=0 plane):
    Order must match OpenCV corner order.
    """
    L = float(marker_length_m)
    half = L / 2.0
    # OpenCV returns corners in order: top-left, top-right, bottom-right, bottom-left
    obj = np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float64)
    return obj


def solve_pose_from_markers(corners, ids, K, dist, marker_length_m):
    """
    Uses all detected markers in a frame to solve a single pose via solvePnP.
    Returns ok, rvec, tvec, used_marker_count
    """
    if ids is None or len(ids) == 0:
        return False, None, None, 0

    obj_corners = marker_object_points(marker_length_m)

    obj_pts = []
    img_pts = []

    for c in corners:
        c = c.reshape(4, 2).astype(np.float64)
        for i in range(4):
            obj_pts.append(obj_corners[i])
            img_pts.append(c[i])

    obj_pts = np.array(obj_pts, dtype=np.float64).reshape(-1, 3)
    img_pts = np.array(img_pts, dtype=np.float64).reshape(-1, 2)

    if len(obj_pts) < 4:
        return False, None, None, 0

    ok, rvec, tvec = cv2.solvePnP(
        objectPoints=obj_pts,
        imagePoints=img_pts,
        cameraMatrix=K,
        distCoeffs=dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    return bool(ok), rvec, tvec, int(len(ids))


def plot_xy(traj_xyz, out_path):
    if len(traj_xyz) < 2:
        print("[WARN] Not enough points to plot XY trajectory.")
        return

    traj = np.array(traj_xyz, dtype=np.float64)
    x = traj[:, 0]
    y = traj[:, 1]

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.title("ArUco camera trajectory (X-Y)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200)
    plt.close()
    print("[INFO] Saved trajectory PNG:", out_path)


def save_interactive_3d(traj_xyz, out_html):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[WARN] plotly not installed. Run: pip install plotly")
        return

    if len(traj_xyz) < 2:
        print("[WARN] Not enough points to plot 3D trajectory.")
        return

    traj = np.array(traj_xyz, dtype=np.float64)
    x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="lines+markers",
        marker=dict(size=3),
        line=dict(width=4),
        name="Camera path"
    ))
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
        title="ArUco camera trajectory (3D)",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.write_html(str(out_html))
    print("[INFO] Saved interactive 3D HTML:", out_html)


# ============================================================
# Main processing
# ============================================================
def process_video_aruco(
    video_path,
    calib_npz_path,
    marker_length_m,
    max_width=None,
    sample_max_frames=150,
    sample_stride=2,
    save_debug_frame_number=30,
    print_every=10
):
    ensure_aruco_available()

    print("[INFO] Script directory:", BASE_DIR)
    print("[INFO] Will save CSV :", OUT_CSV)
    print("[INFO] Will save PNG :", OUT_PNG)
    print("[INFO] Will save HTML:", OUT_HTML)
    print("[INFO] Will save DBG :", OUT_DBG)

    K_full, dist_full = load_calib_npz(calib_npz_path)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Read first frame to determine resize scale
    ok, frame0 = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Could not read first frame.")

    frame0_r, s0 = resize_keep_aspect(frame0, max_width=max_width)
    gray0 = cv2.cvtColor(frame0_r, cv2.COLOR_BGR2GRAY)

    # Scale intrinsics to resized resolution (if resizing)
    K = scale_intrinsics(K_full, s0)
    dist = dist_full.copy()

    print("[INFO] Resize scale s0 =", s0)
    print("[INFO] Using K for resized frames:\n", K)

    # Auto-find dictionary (run on sampled frames)
    best_dict = find_best_dictionary(
        video_path=video_path,
        K=K,
        dist=dist,
        max_frames=sample_max_frames,
        stride=sample_stride,
        max_width=max_width
    )

    if best_dict is None:
        cap.release()
        raise RuntimeError("Could not choose a dictionary (None).")

    params = make_detector_params()

    # Reset video to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # CSV
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "frame_idx",
            "num_markers",
            "ids",
            "rvec_x", "rvec_y", "rvec_z",
            "tvec_x", "tvec_y", "tvec_z"
        ])

        traj = []  # camera positions in world (approx) per frame
        saved_dbg = False

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            frame_r, _ = resize_keep_aspect(frame, max_width=max_width)
            gray = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

            corners, ids = detect_markers(gray, best_dict, params)

            ok_pose, rvec, tvec, nmk = solve_pose_from_markers(
                corners, ids, K, dist, marker_length_m
            )

            if ok_pose:
                # For a planar marker, tvec is camera position in marker frame (up to coordinate convention).
                # We'll store it directly; for plotting, this is still a consistent 3D path.
                t = tvec.reshape(3).astype(np.float64)
                traj.append(t.copy())

                ids_list = ids.flatten().tolist() if ids is not None else []
                w.writerow([
                    frame_idx,
                    nmk,
                    " ".join(map(str, ids_list)),
                    float(rvec[0]), float(rvec[1]), float(rvec[2]),
                    float(tvec[0]), float(tvec[1]), float(tvec[2]),
                ])
            else:
                ids_list = ids.flatten().tolist() if ids is not None else []
                w.writerow([
                    frame_idx,
                    nmk,
                    " ".join(map(str, ids_list)),
                    "", "", "",
                    "", "", "",
                ])

            # Save a debug frame once (draw detected markers)
            if (not saved_dbg) and (frame_idx >= save_debug_frame_number):
                vis = frame_r.copy()
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(vis, corners, ids)
                cv2.imwrite(str(OUT_DBG), vis)
                print("[INFO] Saved debug frame with detections:", OUT_DBG)
                saved_dbg = True

            if print_every and (frame_idx % print_every == 0):
                det = 0 if ids is None else len(ids)
                print(f"[INFO] frame={frame_idx:05d} detected={det} pose_ok={int(ok_pose)}")

    cap.release()
    cv2.destroyAllWindows()

    # Plot trajectory
    if len(traj) < 2:
        print("[WARN] Not enough valid poses to plot trajectory. Check aruco_debug.png and dictionary.")
        return

    plot_xy(traj, OUT_PNG)
    save_interactive_3d(traj, OUT_HTML)

    print("\n================ FINAL (last pose) ================")
    print("[FINAL] last tvec =", traj[-1])
    print("==================================================\n")
    print("[DONE] Outputs:")
    print(" -", OUT_CSV)
    print(" -", OUT_PNG)
    print(" -", OUT_HTML)
    print(" -", OUT_DBG)
    print("[INFO] Best dictionary used:", best_dict)


# ============================================================
# CLI
# ============================================================
def main():
    """
    Usage:
      python aruco_pose_from_video.py video.mp4 calibration_filtered.npz 0.05

    Where:
      0.05 is marker size in meters (5 cm). Use the REAL printed marker size.
    """
    if len(sys.argv) < 4:
        print("Usage: python aruco_pose_from_video.py video.mp4 calibration_filtered.npz marker_size_m")
        print("Example: python aruco_pose_from_video.py friend.mp4 calibration_filtered.npz 0.05")
        sys.exit(1)

    video_path = sys.argv[1]
    calib_npz = sys.argv[2]
    marker_size_m = float(sys.argv[3])

    # IMPORTANT: For detection, avoid too much resizing if markers are small.
    process_video_aruco(
        video_path=video_path,
        calib_npz_path=calib_npz,
        marker_length_m=marker_size_m,
        max_width=None,            # set e.g. 900 if you MUST; None is best for detection
        sample_max_frames=150,
        sample_stride=2,
        save_debug_frame_number=30,
        print_every=10
    )


if __name__ == "__main__":
    main()
