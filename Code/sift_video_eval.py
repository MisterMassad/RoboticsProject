import cv2
import numpy as np
import math
import argparse
import matplotlib.pyplot as plt
import os
import csv

# ----------------------------
# IMPORT YOUR CORE TWO-VIEW
# ----------------------------
from SIFT_POSE import estimate_pose_two_images


# ----------------------------
# Basic SE(3) helpers
# ----------------------------
def Rt_to_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T

def invert_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def rotation_angle_deg(R):
    tr = np.trace(R)
    val = (tr - 1.0) / 2.0
    val = np.clip(val, -1.0, 1.0)
    return float(np.degrees(np.arccos(val)))

def unit(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.linalg.norm(v)
    return v / (n + eps)

def angle_between_deg(a, b):
    a = unit(a); b = unit(b)
    cosv = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosv)))

def rodrigues_to_R(rvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R


# ----------------------------
# Load calibration
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
# ArUco
# ----------------------------
def get_aruco_dict(dict_name: str):
    if not hasattr(cv2.aruco, dict_name):
        raise ValueError(f"Unknown aruco dict: {dict_name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))

def detect_and_estimate_pose(img_bgr, K, dist, marker_length_m, aruco_dict, corner_refine=True):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    params = cv2.aruco.DetectorParameters()
    if corner_refine:
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _rejected = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return False, None, None

    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, marker_length_m, K, dist
    )

    # choose marker with largest pixel area
    areas = []
    for c in corners:
        pts = c.reshape(-1, 2).astype(np.float32)
        areas.append(abs(cv2.contourArea(pts)))
    best_i = int(np.argmax(areas))

    rvec = rvecs[best_i, 0, :].astype(np.float64)
    tvec = tvecs[best_i, 0, :].astype(np.float64)
    return True, rvec, tvec

def aruco_relative_T(frameA, frameB, K, dist, marker_length_m, aruco_dict):
    ok1, rvec1, tvec1 = detect_and_estimate_pose(frameA, K, dist, marker_length_m, aruco_dict)
    ok2, rvec2, tvec2 = detect_and_estimate_pose(frameB, K, dist, marker_length_m, aruco_dict)
    if not ok1 or not ok2:
        return None

    R1 = rodrigues_to_R(rvec1)
    R2 = rodrigues_to_R(rvec2)
    T_cam_marker_1 = Rt_to_T(R1, tvec1)
    T_cam_marker_2 = Rt_to_T(R2, tvec2)

    # camB <- camA
    T_marker_camA = invert_T(T_cam_marker_1)
    T_camB_camA = T_cam_marker_2 @ T_marker_camA
    return T_camB_camA


# ----------------------------
# CORE relative (SIFT two-view)
# ----------------------------
def core_relative_T(frameA, frameB, K,
                    max_width=900,
                    nfeatures=2000,
                    ratio_thresh=0.80,
                    max_matches=400,
                    prob=0.999,
                    threshold=1.0,
                    min_E_inliers=12):
    R, t = estimate_pose_two_images(
        frameA, frameB, K,
        max_width=max_width,
        nfeatures=nfeatures,
        ratio_thresh=ratio_thresh,
        max_matches=max_matches,
        use_symmetry=True,
        prob=prob,
        threshold=threshold,
        min_E_inliers=min_E_inliers,
        debug=False,
        viz=False,
        enable_imshow=False
    )
    if R is None or t is None:
        return None
    t = np.asarray(t, dtype=np.float64).reshape(3)
    return Rt_to_T(R, t)


# ----------------------------
# Video runner: adaptive stride + gating + logging
# ----------------------------
def run_on_video(video_path, calib_npz, marker_length_m,
                 dict_name="DICT_4X4_50",
                 stride=5,
                 max_steps=2000,
                 show_preview=False,
                 undistort=True,
                 max_core_step_rot_deg=60.0,        # gate to kill 180 disasters
                 save_bad_frames=True,
                 bad_frame_rot_err_deg=60.0,        # when ARUCO exists and rot error is huge
                 out_dir="video_eval_out"):

    os.makedirs(out_dir, exist_ok=True)
    bad_dir = os.path.join(out_dir, "bad_frames")
    if save_bad_frames:
        os.makedirs(bad_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "steps.csv")

    K, dist = load_calib_npz(calib_npz)
    aruco_dict = get_aruco_dict(dict_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame from video.")

    def undist_f(f):
        return f

    frame0 = undist_f(frame0)

    # Accumulated transforms: T_i<-0
    T_core_acc = np.eye(4, dtype=np.float64)
    T_aruco_acc = np.eye(4, dtype=np.float64)

    core_positions = [T_core_acc[:3, 3].copy()]
    aruco_positions = [T_aruco_acc[:3, 3].copy()]

    rot_errs = []
    tdir_errs = []
    scale_estimates = []

    # Logging
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "step_idx", "frame_idx_a", "frame_idx_b", "used_stride",
            "core_ok", "aruco_ok",
            "core_step_rot_deg",
            "aruco_step_rot_deg",
            "rot_err_deg",
            "tdir_err_deg",
            "aruco_t_norm_m",
            "core_t_norm_arb",
            "scale_m_per_core"
        ])

        prev = frame0
        frame_idx = 0
        step_idx = 0

        # we keep the "frame index of prev"
        prev_frame_idx = 0

        while step_idx < max_steps:
            # Adaptive stride candidates: stride, 2*stride, 3*stride
            stride_candidates = [stride, stride * 2, stride * 3]

            chosen = None
            chosen_frame = None
            chosen_frame_idx = None

            # We will attempt reading ahead; easiest safe way: read sequentially and buffer last candidate.
            # We'll read up to max candidate, and keep frames at those points.
            max_jump = stride_candidates[-1]
            buffered = {}

            ok_read = True
            for j in range(1, max_jump + 1):
                ok, fr = cap.read()
                frame_idx += 1
                if not ok:
                    ok_read = False
                    break
                if j in stride_candidates:
                    buffered[j] = undist_f(fr)

                if show_preview:
                    vis = fr.copy()
                    cv2.putText(vis, f"reading frame {frame_idx}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.imshow("preview", vis)
                    if cv2.waitKey(1) & 0xFF == 27:
                        ok_read = False
                        break

            if not ok_read:
                break

            # Try candidates in order until we get a "good" CORE step (passes gate)
            for cand in stride_candidates:
                fr = buffered.get(cand, None)
                if fr is None:
                    continue

                T_core_step = core_relative_T(prev, fr, K)
                if T_core_step is None:
                    continue

                core_step_rot = rotation_angle_deg(T_core_step[:3, :3])
                if core_step_rot > max_core_step_rot_deg:
                    # reject catastrophic step
                    continue

                # good enough
                chosen = cand
                chosen_frame = fr
                chosen_frame_idx = prev_frame_idx + cand
                break

            # If no candidate worked, move prev forward to the farthest frame to avoid freezing, and continue
            if chosen is None:
                # take the farthest buffered frame as new prev (best chance to recover)
                far = stride_candidates[-1]
                if far in buffered:
                    prev = buffered[far]
                    prev_frame_idx += far
                continue

            frame = chosen_frame

            # CORE step re-compute (or keep from above if you want to store it)
            T_core_step = core_relative_T(prev, frame, K)
            if T_core_step is None:
                prev = frame
                prev_frame_idx = chosen_frame_idx
                continue

            core_step_rot = rotation_angle_deg(T_core_step[:3, :3])
            if core_step_rot > max_core_step_rot_deg:
                prev = frame
                prev_frame_idx = chosen_frame_idx
                continue

            # ArUco step
            T_aruco_step = aruco_relative_T(prev, frame, K, dist, marker_length_m, aruco_dict)

            # accumulate core
            T_core_acc = T_core_step @ T_core_acc

            # defaults for logging
            aruco_ok = (T_aruco_step is not None)
            rot_err = float("nan")
            tdir_err = float("nan")
            aruco_step_rot = float("nan")
            aruco_t_norm = float("nan")
            core_t_norm = float(np.linalg.norm(T_core_step[:3, 3]))
            scale = float("nan")

            if aruco_ok:
                T_aruco_acc = T_aruco_step @ T_aruco_acc

                R_core = T_core_step[:3, :3]
                t_core = T_core_step[:3, 3]
                R_aru = T_aruco_step[:3, :3]
                t_aru = T_aruco_step[:3, 3]

                aruco_step_rot = rotation_angle_deg(R_aru)
                aruco_t_norm = float(np.linalg.norm(t_aru))

                # rotation error
                R_err = R_aru.T @ R_core
                rot_err = rotation_angle_deg(R_err)
                rot_errs.append(rot_err)

                # translation direction error with sign ambiguity
                epos = angle_between_deg(t_aru, t_core)
                eneg = angle_between_deg(t_aru, -t_core)
                tdir_err = min(epos, eneg)
                tdir_errs.append(tdir_err)

                # scale estimate for this step
                n_core = float(np.linalg.norm(t_core))
                n_aru = float(np.linalg.norm(t_aru))
                if n_core > 1e-12 and n_aru > 1e-12:
                    scale = (n_aru / n_core)
                    scale_estimates.append(scale)

                # Save bad frames when error is huge
                if save_bad_frames and np.isfinite(rot_err) and rot_err >= bad_frame_rot_err_deg:
                    out_path = os.path.join(
                        bad_dir, f"bad_step_{step_idx:05d}_frame_{chosen_frame_idx}_rotErr_{rot_err:.1f}.jpg"
                    )
                    cv2.imwrite(out_path, frame)

            # record positions
            core_positions.append(T_core_acc[:3, 3].copy())
            aruco_positions.append(T_aruco_acc[:3, 3].copy())

            # log row
            writer.writerow([
                step_idx, prev_frame_idx, chosen_frame_idx, chosen,
                1, 1 if aruco_ok else 0,
                float(core_step_rot),
                float(aruco_step_rot),
                float(rot_err),
                float(tdir_err),
                float(aruco_t_norm),
                float(core_t_norm),
                float(scale)
            ])

            # advance
            prev = frame
            prev_frame_idx = chosen_frame_idx
            step_idx += 1

        if show_preview:
            cv2.destroyAllWindows()

    cap.release()

    core_positions = np.array(core_positions, dtype=np.float64)
    aruco_positions = np.array(aruco_positions, dtype=np.float64)

    # Apply global scale to core trajectory if we have scale samples
    core_positions_scaled = core_positions.copy()
    if len(scale_estimates) > 0:
        s = float(np.median(scale_estimates))
        core_positions_scaled = core_positions * s
        print(f"[SCALE] median scale = {s:.6f} meters per core-unit (from {len(scale_estimates)} samples)")
    else:
        print("[SCALE] No ArUco scale samples found; core trajectory stays arbitrary scale.")

    # Print summary
    if len(rot_errs) > 0:
        rot_errs_np = np.array(rot_errs, dtype=np.float64)
        tdir_errs_np = np.array(tdir_errs, dtype=np.float64)
        print(f"[ERR] rot mean={rot_errs_np.mean():.2f} deg, p95={np.percentile(rot_errs_np,95):.2f} deg")
        print(f"[ERR] tdir mean={tdir_errs_np.mean():.2f} deg, p95={np.percentile(tdir_errs_np,95):.2f} deg")
    else:
        print("[ERR] No per-step errors computed (ArUco not visible on any processed step).")

    print(f"[OUT] wrote: {csv_path}")
    if save_bad_frames:
        print(f"[OUT] bad frames (if any): {bad_dir}")

    return core_positions_scaled, aruco_positions, rot_errs, tdir_errs


def plot_traj(core_xyz, aruco_xyz, title="Trajectory (meters if scaled)"):
    # XZ plot
    plt.figure()
    plt.plot(core_xyz[:, 0], core_xyz[:, 2], label="CORE (scaled)")
    plt.plot(aruco_xyz[:, 0], aruco_xyz[:, 2], label="ARUCO")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title(title + " - XZ")
    plt.legend()
    plt.grid(True)

    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(core_xyz[:, 0], core_xyz[:, 1], core_xyz[:, 2], label="CORE (scaled)")
    ax.plot(aruco_xyz[:, 0], aruco_xyz[:, 1], aruco_xyz[:, 2], label="ARUCO")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title + " - 3D")
    ax.legend()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--calib_npz", required=True)
    ap.add_argument("--marker_length", type=float, required=True)
    ap.add_argument("--dict", default="DICT_4X4_50")
    ap.add_argument("--stride", type=int, default=5, help="base stride; adaptive tries x1, x2, x3")
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--out_dir", default="video_eval_out")
    ap.add_argument("--max_core_rot", type=float, default=60.0, help="gate: reject core steps with rot > this")
    ap.add_argument("--save_bad_frames", action="store_true")
    ap.add_argument("--bad_rot_err", type=float, default=60.0, help="save frame if rot error >= this (needs ArUco)")
    args = ap.parse_args()

    core_xyz, aruco_xyz, rot_errs, tdir_errs = run_on_video(
        args.video, args.calib_npz, args.marker_length,
        dict_name=args.dict,
        stride=args.stride,
        max_steps=args.max_steps,
        show_preview=args.preview,
        max_core_step_rot_deg=args.max_core_rot,
        save_bad_frames=args.save_bad_frames,
        bad_frame_rot_err_deg=args.bad_rot_err,
        out_dir=args.out_dir
    )

    plot_traj(core_xyz, aruco_xyz, title=f"Video Trajectory (adaptive stride base={args.stride})")


if __name__ == "__main__":
    main()
