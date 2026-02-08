import argparse
import numpy as np
import cv2
import math

from poseDebug import estimate_pose_two_images


# Angle and Translation calculators
def rot_to_axis_angle_deg(R):
    rvec, _ = cv2.Rodrigues(R)
    angle_rad = float(np.linalg.norm(rvec))
    angle_deg = float(np.degrees(angle_rad))
    axis = (rvec.reshape(-1) / (angle_rad + 1e-12)).tolist()
    return angle_deg, axis

def rotmat_to_euler_deg_zyx(R):
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



# ----------------------------
# Helpers: transforms + errors
# ----------------------------
def rodrigues_to_R(rvec):
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return R

def Rt_to_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T

def invert_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Tinv = np.eye(4, dtype=np.float64)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv

def relative_T(T_a, T_b):
    """Return T_ab that maps coordinates from frame a to frame b: X_b = T_ab * X_a"""
    return invert_T(T_a) @ T_b

def rotation_angle_deg(R):
    # angle = acos((trace(R)-1)/2)
    tr = np.trace(R)
    val = (tr - 1.0) / 2.0
    val = np.clip(val, -1.0, 1.0)
    ang = np.degrees(np.arccos(val))
    return float(ang)

def angle_between_deg(a, b, eps=1e-12):
    a = a.reshape(-1)
    b = b.reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return float("nan")
    cosv = np.dot(a, b) / (na * nb)
    cosv = np.clip(cosv, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))

def unit(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n


# ----------------------------
# ArUco detection + pose
# ----------------------------
def get_aruco_dict(dict_name: str):
    """
    Common dict names:
      DICT_4X4_50, DICT_4X4_100, DICT_4X4_250, DICT_4X4_1000,
      DICT_5X5_100, ...,
      DICT_6X6_250, ...
      DICT_ARUCO_ORIGINAL
    """
    if not hasattr(cv2.aruco, dict_name):
        raise ValueError(f"Unknown aruco dict: {dict_name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))

def detect_and_estimate_pose(
    img_bgr,
    K,
    dist,
    marker_length_m,
    aruco_dict,
    corner_refine=True,
):
    """
    Returns:
      ok (bool),
      rvec_cam_marker (3,),
      tvec_cam_marker (3,),
      ids (Nx1),
      corners
    Pose is the marker pose w.r.t camera: X_cam = R * X_marker + t
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    params = cv2.aruco.DetectorParameters()
    if corner_refine:
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return False, None, None, ids, corners

    # Estimate pose for each detected marker
    # rvecs/tvecs shape: (N, 1, 3)
    rvecs, tvecs, _obj = cv2.aruco.estimatePoseSingleMarkers(
        corners, marker_length_m, K, dist
    )

    # Choose ONE marker (best = closest marker / largest projected area)
    # We'll select the marker with largest corner area in pixels.
    areas = []
    areas = []
    for c in corners:
        pts = c.reshape(-1, 2).astype(np.float32)
        areas.append(abs(cv2.contourArea(pts)))
    best_i = int(np.argmax(areas))


    rvec = rvecs[best_i, 0, :].astype(np.float64)
    tvec = tvecs[best_i, 0, :].astype(np.float64)

    return True, rvec, tvec, ids, corners

def draw_debug(img_bgr, corners, ids, rvec, tvec, K, dist, marker_length_m):
    out = img_bgr.copy()
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(out, corners, ids)
    if rvec is not None and tvec is not None:
        cv2.drawFrameAxes(out, K, dist, rvec, tvec, marker_length_m * 0.5)
    return out


# ----------------------------
# Load calibration .npz
# ----------------------------
def load_calib_npz(npz_path):
    data = np.load(npz_path)
    # Try common key names
    for kkey in ["K", "cameraMatrix", "mtx"]:
        if kkey in data:
            K = data[kkey]
            break
    else:
        raise KeyError("Could not find K in npz (tried: K, cameraMatrix, mtx)")

    for dkey in ["dist", "distCoeffs", "distortion", "dist_coeffs"]:
        if dkey in data:
            dist = data[dkey]
            break
    else:
        # If missing, assume zero distortion (still works, but less accurate)
        dist = np.zeros((1, 5), dtype=np.float64)

    K = np.asarray(K, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)  # OpenCV tolerates shapes
    return K, dist


# ----------------------------
# Plug your core pipeline here
# ----------------------------
def run_core_two_view(img1_bgr, img2_bgr, K):
    """
    Replace this with your actual call.

    Option A (recommended): import your function, e.g.
        from core_two_view import estimate_relative_pose
        return estimate_relative_pose(img1_bgr, img2_bgr, K)

    Expected return:
        R_core (3x3), t_core (3,), and optionally extra stuff.
    """
def run_core_two_view(img1_bgr, img2_bgr, K, feature_type):
    R, t = estimate_pose_two_images(
        img1_bgr, img2_bgr, K,
        max_width=2500,
        nfeatures=1500,
        ratio_thresh=0.75,
        max_matches=500,
        use_symmetry=True,
        prob=0.999,
        threshold=1.0,
        min_E_inliers=8,
        debug=True,     # keep comparison script clean
        viz=True,       # no extra windows
        enable_imshow=False,
        feature_type=feature_type
    )
    if R is None or t is None:
        raise RuntimeError("CORE pipeline failed to estimate pose.")
    return R, np.asarray(t, dtype=np.float64).reshape(3)


# ----------------------------
# Main evaluation
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img1", required=True)
    ap.add_argument("--img2", required=True)
    ap.add_argument("--calib_npz", required=True, help="npz containing K and dist")
    ap.add_argument("--marker_length", type=float, required=True,
                    help="Marker side length in METERS, e.g. 0.04 for 4cm")
    ap.add_argument("--dict", default="DICT_4X4_50")
    ap.add_argument("--det", default="ORB", help="detector type for CORE (ORB, SIFT, etc.)")
    ap.add_argument("--show", action="store_true", help="show debug windows")
    
    args = ap.parse_args()

    img1 = cv2.imread(args.img1)
    img2 = cv2.imread(args.img2)
    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not read one of the images.")

    K, dist = load_calib_npz(args.calib_npz)
    aruco_dict = get_aruco_dict(args.dict)

    ok1, rvec1, tvec1, ids1, corners1 = detect_and_estimate_pose(
        img1, K, dist, args.marker_length, aruco_dict
    )
    ok2, rvec2, tvec2, ids2, corners2 = detect_and_estimate_pose(
        img2, K, dist, args.marker_length, aruco_dict
    )

    if not ok1 or not ok2:
        print("[ARUCO] Failed to detect marker in one/both images.")
        print(f"  img1 detected: {ok1}, img2 detected: {ok2}")
        return

    # Build camera<-marker transforms for each image
    R1 = rodrigues_to_R(rvec1)
    R2 = rodrigues_to_R(rvec2)
    T_cam_marker_1 = Rt_to_T(R1, tvec1)
    T_cam_marker_2 = Rt_to_T(R2, tvec2)

    # cam2 <- cam1 using marker as reference
    T_marker_cam1 = invert_T(T_cam_marker_1)
    T_cam2_cam1_aruco = T_cam_marker_2 @ T_marker_cam1

    R_aruco = T_cam2_cam1_aruco[:3, :3]
    t_aruco = T_cam2_cam1_aruco[:3, 3]

    print("========== ARUCO RELATIVE (cam2 <- cam1) ==========")
    print("R_aruco:\n", R_aruco)
    print("t_aruco (meters):", t_aruco, "  ||t||=", float(np.linalg.norm(t_aruco)))
    print("===================================================")

    ang_deg, axis = rot_to_axis_angle_deg(R_aruco)
    roll, pitch, yaw = rotmat_to_euler_deg_zyx(R_aruco)

    print(f"[ARUCO] axis-angle: {ang_deg:.2f} deg axis={axis}")
    print(f"[ARUCO] euler (roll,pitch,yaw): {roll:.2f}, {pitch:.2f}, {yaw:.2f} deg")
    print(f"[ARUCO] t (m): {t_aruco}  ||t||={np.linalg.norm(t_aruco):.6f} m")
    print(f"[ARUCO] t (cm): {t_aruco*100.0}  ||t||={np.linalg.norm(t_aruco)*100.0:.3f} cm")

    # --- Core two-view ---
    try:
        R_core, t_core = run_core_two_view(img1, img2, K, args.det)
    except NotImplementedError:
        print("\n[CORE] Not connected yet.")
        if args.show:
            dbg1 = draw_debug(img1, corners1, ids1, rvec1, tvec1, K, dist, args.marker_length)
            dbg2 = draw_debug(img2, corners2, ids2, rvec2, tvec2, K, dist, args.marker_length)

            cv2.namedWindow("img1 aruco", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.namedWindow("img2 aruco", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.imshow("img1 aruco", dbg1)
            cv2.imshow("img2 aruco", dbg2)
            cv2.resizeWindow("img1 aruco", 900, 600)
            cv2.resizeWindow("img2 aruco", 900, 600)
            cv2.moveWindow("img1 aruco", 50, 50)
            cv2.moveWindow("img2 aruco", 1000, 50)
            cv2.waitKey(0)
        return

    t_core = np.asarray(t_core, dtype=np.float64).reshape(3)

    # --- Optional: print CORE in same "human" format ---
    ang_c, axis_c = rot_to_axis_angle_deg(R_core)
    r_c, p_c, y_c = rotmat_to_euler_deg_zyx(R_core)
    print("\n========== CORE RELATIVE (cam2 <- cam1) ==========")
    print("R_core:\n", R_core)
    print(f"[CORE] axis-angle: {ang_c:.2f} deg axis={axis_c}")
    print(f"[CORE] euler (roll,pitch,yaw): {r_c:.2f}, {p_c:.2f}, {y_c:.2f} deg")
    print(f"[CORE] t (arb): {t_core}  ||t||={np.linalg.norm(t_core):.6f}")
    print("==================================================")

    # --- Compare ---
    # Rotation error between relative rotations
    R_err = R_aruco.T @ R_core
    rot_err_deg = rotation_angle_deg(R_err)

    # Translation direction error (scale-free), handling sign ambiguity
    err_pos = angle_between_deg(unit(t_aruco), unit(t_core))
    err_neg = angle_between_deg(unit(t_aruco), unit(-t_core))
    use_neg = (err_neg < err_pos)
    t_core_used = (-t_core) if use_neg else t_core
    tdir_err_deg = min(err_pos, err_neg)

    # Scale estimate (ONLY meaningful for this pair, not a universal truth)
    core_norm = float(np.linalg.norm(t_core_used))
    ar_norm = float(np.linalg.norm(t_aruco))
    scale = (ar_norm / core_norm) if core_norm > 1e-12 else float("nan")

    print("\n============== CORE vs ARUCO COMPARISON ==============")
    print(f"Rotation error (deg): {rot_err_deg:.4f}")
    print(f"Translation dir error (deg): {tdir_err_deg:.4f}")
    print(f"[t sign] err(t)={err_pos:.4f} deg, err(-t)={err_neg:.4f} deg -> using {'-t' if use_neg else '+t'}")
    print(f"ArUco ||t|| (m): {ar_norm:.6f}  ({ar_norm*100.0:.2f} cm)")
    print(f"Core  ||t|| (arb): {core_norm:.6f}")
    print(f"Estimated scale (m per arb-unit): {scale:.6f}")
    print("======================================================")

    if args.show:
        dbg1 = draw_debug(img1, corners1, ids1, rvec1, tvec1, K, dist, args.marker_length)
        dbg2 = draw_debug(img2, corners2, ids2, rvec2, tvec2, K, dist, args.marker_length)

        cv2.namedWindow("img1 aruco", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow("img2 aruco", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        cv2.imshow("img1 aruco", dbg1)
        cv2.imshow("img2 aruco", dbg2)

        cv2.resizeWindow("img1 aruco", 900, 600)
        cv2.resizeWindow("img2 aruco", 900, 600)

        cv2.moveWindow("img1 aruco", 50, 50)
        cv2.moveWindow("img2 aruco", 1000, 50)

        cv2.waitKey(0)


if __name__ == "__main__":
    main()
