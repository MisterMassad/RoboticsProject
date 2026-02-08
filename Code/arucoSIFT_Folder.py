# pairs_eval.py  (DROP-IN REPLACEMENT)
#
# What this fixes vs your current evaluator:
# 1) Correct SO(3) rotation error: geodesic angle of R_ref^T R_est
# 2) Correctly handles the **direction ambiguity** between pipelines:
#    compares CORE forward (cam2<-cam1) vs CORE inverse (cam1<-cam2),
#    and picks the one with LOWER rotation error (optionally uses translation
#    direction as a tiebreaker).
# 3) When choosing the inverse direction, it uses the **proper inverse transform**
#    t_inv = -R^T t (not just -t).
# 4) Euler values are for reporting only; errors are computed on SO(3) and t-dir.
#
# CSV columns kept exactly as you requested.

import argparse
import os
import glob
import csv
import random
import math
import numpy as np
import cv2

from SIFT_POSE import estimate_pose_two_images


# ----------------------------
# Rotations / angles helpers
# ----------------------------
def rotmat_to_euler_deg_zyx(R: np.ndarray):
    """
    Returns (roll, pitch, yaw) in degrees using ZYX convention.
    ZYX means: yaw around Z, pitch around Y, roll around X.
    """
    R = np.asarray(R, dtype=np.float64)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-9
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])       # X
        pitch = math.atan2(-R[2, 0], sy)          # Y
        yaw = math.atan2(R[1, 0], R[0, 0])        # Z
    else:
        # Gimbal lock
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    roll, pitch, yaw = np.degrees([roll, pitch, yaw])
    return float(roll), float(pitch), float(yaw)


def rotation_geodesic_error_deg(R_ref: np.ndarray, R_est: np.ndarray) -> float:
    """
    Geodesic distance on SO(3): angle( R_ref^T R_est ).
    Returns angle in degrees in [0, 180].
    """
    R_ref = np.asarray(R_ref, dtype=np.float64)
    R_est = np.asarray(R_est, dtype=np.float64)

    R_delta = R_ref.T @ R_est
    tr = float(np.trace(R_delta))
    val = (tr - 1.0) / 2.0
    val = float(np.clip(val, -1.0, 1.0))
    return float(np.degrees(np.arccos(val)))


def wrap_deg(a: float) -> float:
    """Wrap angle to [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


def angle_diff_deg(a: float, b: float) -> float:
    """Smallest signed diff a-b, wrapped."""
    return wrap_deg(a - b)


def euler_rms_error_deg(yaw_roll_pitch_a, yaw_roll_pitch_b) -> float:
    """
    yaw_roll_pitch_* are tuples (yaw, roll, pitch).
    RMS over the 3 wrapped differences.
    NOTE: Euler error is NOT a proper rotation metric; it's only for display/debug.
    """
    dy = angle_diff_deg(yaw_roll_pitch_a[0], yaw_roll_pitch_b[0])
    dr = angle_diff_deg(yaw_roll_pitch_a[1], yaw_roll_pitch_b[1])
    dp = angle_diff_deg(yaw_roll_pitch_a[2], yaw_roll_pitch_b[2])
    return float(math.sqrt((dy * dy + dr * dr + dp * dp) / 3.0))


def translation_dir_error_deg(t_ref, t_est, eps=1e-12) -> float | None:
    """
    Angle between translation directions (degrees).
    Two-view monocular has sign ambiguity for t (t and -t equivalent),
    so we take min(angle(t_ref,t_est), angle(t_ref,-t_est)).
    """
    tr = np.asarray(t_ref, dtype=np.float64).reshape(3)
    te = np.asarray(t_est, dtype=np.float64).reshape(3)
    nr = float(np.linalg.norm(tr))
    ne = float(np.linalg.norm(te))
    if nr < eps or ne < eps:
        return None

    ur = tr / nr
    ue = te / ne

    dot1 = float(np.clip(np.dot(ur, ue), -1.0, 1.0))
    dot2 = float(np.clip(np.dot(ur, -ue), -1.0, 1.0))
    ang1 = float(np.degrees(np.arccos(dot1)))
    ang2 = float(np.degrees(np.arccos(dot2)))
    return min(ang1, ang2)


def rodrigues_to_R(rvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R


def Rt_to_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_T(T):
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    Tinv = np.eye(4, dtype=np.float64)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv


# ----------------------------
# ArUco detection + pose
# ----------------------------
def get_aruco_dict(dict_name: str):
    if not hasattr(cv2.aruco, dict_name):
        raise ValueError(f"Unknown aruco dict: {dict_name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))


def detect_and_estimate_pose(img_bgr, K, dist, marker_length_m, aruco_dict, corner_refine=True):
    """
    Returns:
      ok (bool),
      rvec (3,),
      tvec (3,)
    Pose is marker pose w.r.t camera:
      X_cam = R * X_marker + t
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # OpenCV 4.7+ style
    params = cv2.aruco.DetectorParameters()
    if corner_refine:
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _rejected = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return False, None, None

    rvecs, tvecs, _obj = cv2.aruco.estimatePoseSingleMarkers(
        corners, marker_length_m, K, dist
    )

    # pick largest area marker (robust if multiple detected)
    areas = []
    for c in corners:
        pts = c.reshape(-1, 2).astype(np.float32)
        areas.append(abs(cv2.contourArea(pts)))
    best_i = int(np.argmax(areas))

    rvec = rvecs[best_i, 0, :].astype(np.float64)
    tvec = tvecs[best_i, 0, :].astype(np.float64)
    return True, rvec, tvec


def relative_pose_from_aruco(img1, img2, K, dist, marker_length_m, aruco_dict):
    """
    Computes relative camera transform cam2 <- cam1 using the marker as a bridge.
    """
    ok1, rvec1, tvec1 = detect_and_estimate_pose(img1, K, dist, marker_length_m, aruco_dict)
    ok2, rvec2, tvec2 = detect_and_estimate_pose(img2, K, dist, marker_length_m, aruco_dict)
    if not ok1 or not ok2:
        return None, None  # fail

    R1 = rodrigues_to_R(rvec1)
    R2 = rodrigues_to_R(rvec2)
    T_cam_marker_1 = Rt_to_T(R1, tvec1)
    T_cam_marker_2 = Rt_to_T(R2, tvec2)

    # cam2 <- cam1
    T_marker_cam1 = invert_T(T_cam_marker_1)
    T_cam2_cam1 = T_cam_marker_2 @ T_marker_cam1

    R_rel = T_cam2_cam1[:3, :3]
    t_rel = T_cam2_cam1[:3, 3]
    return R_rel, t_rel


# ----------------------------
# Load calibration .npz
# ----------------------------
def load_calib_npz(npz_path):
    data = np.load(npz_path)

    for kkey in ["K", "cameraMatrix", "mtx"]:
        if kkey in data:
            K = data[kkey]
            break
    else:
        raise KeyError("Could not find K in npz (tried: K, cameraMatrix, mtx)")

    for dkey in ["dist", "distCoeffs", "distortion", "dist_coeffs", "D"]:
        if dkey in data:
            dist = data[dkey]
            break
    else:
        dist = np.zeros((1, 5), dtype=np.float64)

    K = np.asarray(K, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)
    return K, dist


# ----------------------------
# CORE pipeline hook
# ----------------------------
def run_core_two_view(img1_bgr, img2_bgr, K, core_params):
    R, t = estimate_pose_two_images(
        img1_bgr, img2_bgr, K,
        pixels=core_params["max_width"],
        nfeatures=core_params["nfeatures"],
        ratio_thresh=core_params["ratio_thresh"],
        max_matches=core_params["max_matches"],
        use_symmetry=core_params["use_symmetry"],
        prob=core_params["prob"],
        threshold=core_params["threshold"],
        min_E_inliers=core_params["min_E_inliers"],
        debug=True,
        viz=False,
        enable_imshow=False
    )
    if R is None or t is None:
        return None, None
    return np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64).reshape(3)


# ----------------------------
# Pair sampling
# ----------------------------
def collect_images(paths, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
    all_files = []
    for p in paths:
        if os.path.isdir(p):
            for e in exts:
                all_files.extend(glob.glob(os.path.join(p, f"*{e}")))
                all_files.extend(glob.glob(os.path.join(p, f"*{e.upper()}")))
        elif os.path.isfile(p):
            all_files.append(p)
    # dedupe + sort for reproducibility
    all_files = sorted(list(dict.fromkeys(all_files)))
    return all_files


def sample_pairs(n, k, rng):
    """
    Sample up to k unique unordered pairs (i<j) from n items.
    If n is small, enumerate all pairs then sample.
    """
    total = n * (n - 1) // 2
    if total == 0:
        return []

    k = min(k, total)

    if total <= 200000:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(pairs)
        return pairs[:k]

    seen = set()
    out = []
    while len(out) < k:
        i = rng.randrange(0, n)
        j = rng.randrange(0, n)
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append((a, b))
    return out


# ----------------------------
# Correct direction-invariant comparison
# ----------------------------
def invert_rt(R, t):
    """
    Inverse of transform (R,t) where x2 = R x1 + t.
    Returns (R_inv, t_inv) such that x1 = R_inv x2 + t_inv.
    """
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv


def choose_best_direction(R_ref, t_ref, R_core, t_core):
    """
    Compare CORE forward vs CORE inverse vs ArUco reference.
    Returns:
      R_best, t_best, rot_err_best, t_dir_err_best, used_inverse_bool
    """
    # forward hypothesis
    rot_f = rotation_geodesic_error_deg(R_ref, R_core)
    t_f = translation_dir_error_deg(t_ref, t_core)

    # inverse hypothesis
    R_inv, t_inv = invert_rt(R_core, t_core)
    rot_i = rotation_geodesic_error_deg(R_ref, R_inv)
    t_i = translation_dir_error_deg(t_ref, t_inv)

    # pick by rotation first (primary metric)
    if rot_i + 1e-12 < rot_f:
        return R_inv, t_inv, rot_i, t_i, True
    if rot_f + 1e-12 < rot_i:
        return R_core, t_core, rot_f, t_f, False

    # tie-breaker: translation direction if both defined
    if (t_f is not None) and (t_i is not None):
        if t_i < t_f:
            return R_inv, t_inv, rot_i, t_i, True
        return R_core, t_core, rot_f, t_f, False

    # otherwise keep forward
    return R_core, t_core, rot_f, t_f, False


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="Folders and/or image paths. Example: --inputs folderA folderB")
    ap.add_argument("--calib_npz", required=True)
    ap.add_argument("--marker_length", type=float, required=True,
                    help="Marker side length in METERS, e.g. 0.04 for 4cm")
    ap.add_argument("--dict", default="DICT_4X4_50")
    ap.add_argument("--pairs", type=int, default=500,
                    help="How many random pairs to evaluate (max is N choose 2)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv_out", default="pairs_eval.csv")

    # CORE params
    ap.add_argument("--max_width", type=int, default=800000)
    ap.add_argument("--nfeatures", type=int, default=3500)
    ap.add_argument("--ratio_thresh", type=float, default=0.85)
    ap.add_argument("--max_matches", type=int, default=500)
    ap.add_argument("--use_symmetry", action="store_true")
    ap.add_argument("--prob", type=float, default=0.999)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--min_E_inliers", type=int, default=8)

    # behavior
    ap.add_argument("--skip_if_no_aruco", action="store_true",
                    help="Skip pairs where either image has no ArUco detection (recommended).")

    args = ap.parse_args()
    rng = random.Random(args.seed)

    images = collect_images(args.inputs)
    if len(images) < 2:
        raise RuntimeError(f"Need at least 2 images. Found {len(images)}.")

    K, dist = load_calib_npz(args.calib_npz)
    dist = np.zeros((1, 5), dtype=np.float64)
    aruco_dict = get_aruco_dict(args.dict)

    core_params = dict(
        max_width=args.max_width,
        nfeatures=args.nfeatures,
        ratio_thresh=args.ratio_thresh,
        max_matches=args.max_matches,
        use_symmetry=args.use_symmetry,
        prob=args.prob,
        threshold=args.threshold,
        min_E_inliers=args.min_E_inliers,
    )

    pairs = sample_pairs(len(images), args.pairs, rng)

    header = [
        "img1", "img2",
        "aruco_yaw_deg", "aruco_roll_deg", "aruco_pitch_deg",
        "core_yaw_deg",  "core_roll_deg",  "core_pitch_deg",
        "rot_err_deg", "euler_rms_err_deg",
        "t_dir_err_deg", "used_core_inverse"
    ]

    rows_written = 0
    rows_skipped = 0
    rows_failed_core = 0

    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        for (i, j) in pairs:
            p1, p2 = images[i], images[j]
            img1 = cv2.imread(p1)
            img2 = cv2.imread(p2)
            if img1 is None or img2 is None:
                rows_skipped += 1
                continue

            # ArUco relative (reference): cam2 <- cam1
            R_aruco, t_aruco = relative_pose_from_aruco(
                img1, img2, K, dist, args.marker_length, aruco_dict
            )
            if R_aruco is None:
                rows_skipped += 1
                continue

            # CORE relative
            R_core_raw, t_core_raw = run_core_two_view(img1, img2, K, core_params)
            if R_core_raw is None:
                rows_failed_core += 1
                continue

            # Choose direction (forward vs inverse) based on correct transform inverse
            R_core, t_core, rot_err_deg, t_dir_err, used_inv = choose_best_direction(
                R_aruco, t_aruco, R_core_raw, t_core_raw
            )

            # Reporting eulers (yaw, roll, pitch)
            ar_roll, ar_pitch, ar_yaw = rotmat_to_euler_deg_zyx(R_aruco)
            co_roll, co_pitch, co_yaw = rotmat_to_euler_deg_zyx(R_core)

            ar_yaw_roll_pitch = (ar_yaw, ar_roll, ar_pitch)
            co_yaw_roll_pitch = (co_yaw, co_roll, co_pitch)

            # Euler RMS (display metric)
            euler_err = euler_rms_error_deg(ar_yaw_roll_pitch, co_yaw_roll_pitch)

            w.writerow([
                os.path.basename(p1), os.path.basename(p2),
                f"{ar_yaw:.6f}", f"{ar_roll:.6f}", f"{ar_pitch:.6f}",
                f"{co_yaw:.6f}", f"{co_roll:.6f}", f"{co_pitch:.6f}",
                f"{rot_err_deg:.6f}", f"{euler_err:.6f}",
                "" if t_dir_err is None else f"{t_dir_err:.6f}",
                int(used_inv),
            ])

            rows_written += 1

    print(f"[DONE] wrote: {args.csv_out}")
    print(f"  images: {len(images)}")
    print(f"  requested pairs: {len(pairs)}")
    print(f"  rows written: {rows_written}")
    print(f"  skipped (no aruco / bad read): {rows_skipped}")
    print(f"  failed core: {rows_failed_core}")


if __name__ == "__main__":
    main()
