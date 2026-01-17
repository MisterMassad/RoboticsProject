import os
import glob
import cv2
import numpy as np

# ---------------------------
# Helpers
# ---------------------------

def _make_object_points(pattern_size, square_size):
    """
    pattern_size = (cols, rows) of INNER corners, e.g. (8, 5)
    square_size in meters (e.g. 0.03 for 3cm)
    """
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size)
    return objp

def _imshow_fit(win, img, max_w=1200, max_h=800, wait=1):
    """Show an image resized to fit screen-ish bounds (for huge iPhone/GoPro images)."""
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imshow(win, img)
    cv2.waitKey(wait)

# ---------------------------
# Corner collection
# ---------------------------

def collect_from_images(images_glob, pattern_size, square_size,
                        show=False, max_images=None, use_sb=True):
    objp = _make_object_points(pattern_size, square_size)

    objpoints = []
    imgpoints = []

    image_paths = sorted(glob.glob(images_glob))
    if max_images is not None:
        image_paths = image_paths[:max_images]

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images matched glob: {images_glob}")

    img_size = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    if show:
        cv2.namedWindow("corners", cv2.WINDOW_NORMAL)

    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            print(f"[WARN] Could not read: {p}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])  # (w, h)

        # Stronger detector (recommended)
        if use_sb:
            ret, corners = cv2.findChessboardCornersSB(gray, pattern_size)
        else:
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

        if not ret:
            print(f"[MISS] {os.path.basename(p)}")
            if show:
                _imshow_fit("corners", img, wait=30)
            continue

        # Subpixel refinement still helps
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners_refined)

        print(f"[OK]   {os.path.basename(p)} corners={len(corners_refined)}")

        if show:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners_refined, ret)
            _imshow_fit("corners", vis, wait=80)

    if show:
        cv2.destroyAllWindows()

    if len(objpoints) < 10:
        raise RuntimeError(f"Not enough valid detections ({len(objpoints)}). Need ~10+.")

    return objpoints, imgpoints, img_size

def collect_from_video(video_path, pattern_size, square_size,
                       every_n_frames=10, show=False, max_frames=400, use_sb=True):
    objp = _make_object_points(pattern_size, square_size)

    objpoints = []
    imgpoints = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    img_size = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    if show:
        cv2.namedWindow("video corners", cv2.WINDOW_NORMAL)

    frame_idx = 0
    used = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % every_n_frames != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])

        if use_sb:
            found, corners = cv2.findChessboardCornersSB(gray, pattern_size)
        else:
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

        if not found:
            if show:
                _imshow_fit("video corners", frame, wait=1)
            continue

        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners_refined)
        used += 1
        print(f"[OK] frame={frame_idx} used={used}")

        if show:
            vis = frame.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners_refined, found)
            _imshow_fit("video corners", vis, wait=10)

        if used >= max_frames:
            break

    cap.release()
    if show:
        cv2.destroyAllWindows()

    if len(objpoints) < 10:
        raise RuntimeError(f"Not enough valid detections ({len(objpoints)}). Need ~10+.")

    return objpoints, imgpoints, img_size

# ---------------------------
# Calibration + filtering
# ---------------------------

def calibrate(objpoints, imgpoints, img_size, model="pinhole"):
    """
    model:
      - "iphone"  : stabilize by fixing K3 (often helps iPhone lens switching/noise)
      - "gopro"   : wide lens, use RATIONAL_MODEL (k4-k6)
      - "pinhole" : default OpenCV model (k1-k5)
    """
    if img_size is None:
        raise ValueError("img_size is None")

    flags = 0
    if model == "iphone":
        flags |= cv2.CALIB_FIX_K3
    elif model == "gopro":
        flags |= cv2.CALIB_RATIONAL_MODEL

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None, flags=flags
    )

    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, img_size, alpha=0.0)
    return rms, K, dist, rvecs, tvecs, newK, roi

def mean_reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, dist):
    total_err = 0.0
    total_points = 0

    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2)
        n = len(objpoints[i])
        total_err += err * err
        total_points += n

    rmse = np.sqrt(total_err / total_points)
    return float(rmse)

def per_view_errors(objpoints, imgpoints, rvecs, tvecs, K, dist):
    """Mean pixel error per corner for each view."""
    errs = []
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(objpoints[i])
        errs.append(float(err))
    return errs

def calibrate_with_filter(objpoints, imgpoints, img_size, model="gopro", keep_ratio=0.7):
    """
    1) Calibrate
    2) Compute per-view error
    3) Keep best keep_ratio views
    4) Recalibrate
    """
    rms1, K1, dist1, rvecs1, tvecs1, newK1, roi1 = calibrate(objpoints, imgpoints, img_size, model=model)
    errs = per_view_errors(objpoints, imgpoints, rvecs1, tvecs1, K1, dist1)

    idx = np.argsort(errs)
    keep_n = max(10, int(keep_ratio * len(idx)))
    keep = idx[:keep_n]

    obj_f = [objpoints[i] for i in keep]
    img_f = [imgpoints[i] for i in keep]

    rms2, K2, dist2, rvecs2, tvecs2, newK2, roi2 = calibrate(obj_f, img_f, img_size, model=model)
    reproj2 = mean_reprojection_error(obj_f, img_f, rvecs2, tvecs2, K2, dist2)

    print(f"[FILTER] kept {keep_n}/{len(idx)} views (keep_ratio={keep_ratio})")
    print(f"[FILTER] initial rms={rms1:.6f}, filtered reproj_RMSE={reproj2:.6f}")

    return rms2, K2, dist2, rvecs2, tvecs2, newK2, roi2, reproj2

# ---------------------------
# Save
# ---------------------------

def save_npz(out_path, K, dist, img_size, rms, reproj_rmse, newK=None, roi=None,
             pattern_size=None, square_size=None, source=None, model=None):
    payload = {
        "K": K,
        "distCoeffs": dist,
        "image_size": np.array(img_size, dtype=np.int32),
        "rms": float(rms),
        "reproj_rmse": float(reproj_rmse),
    }
    if newK is not None:
        payload["newK"] = newK
    if roi is not None:
        payload["roi"] = np.array(roi, dtype=np.int32)
    if pattern_size is not None:
        payload["pattern_size"] = np.array(pattern_size, dtype=np.int32)
    if square_size is not None:
        payload["square_size"] = float(square_size)
    if source is not None:
        payload["source"] = str(source)
    if model is not None:
        payload["model"] = str(model)

    np.savez(out_path, **payload)
    print(f"[SAVED] {out_path}")
    print(f"  K=\n{K}")
    print(f"  dist={dist.ravel()}")
    print(f"  rms={rms:.6f}, reproj_RMSE={reproj_rmse:.6f}")

# ---------------------------
# Main
# ---------------------------

def main():
    MODE = "images"             # "images" or "video"
    PATTERN_SIZE = (8, 5)       # inner corners (cols, rows)
    SQUARE_SIZE = 0.03          # meters (3.0 cm)
    OUT_NPZ = "calibration_filtered.npz"

    SHOW = True
    USE_SB = True               # use findChessboardCornersSB (recommended)
    MODEL = "iphone"             # "gopro" or "iphone" or "pinhole"
    KEEP_RATIO = 0.6            # keep best 70% views after first calibration

    if MODE == "images":
        IMAGES_GLOB = r"C:\Users\Michel Massad\Desktop\Robotics\Calibration\iPhone16Pro_2\*.JPG"
        objpoints, imgpoints, img_size = collect_from_images(
            IMAGES_GLOB, PATTERN_SIZE, SQUARE_SIZE, show=SHOW, use_sb=USE_SB
        )
        src = IMAGES_GLOB
    else:
        VIDEO_PATH = "calib_video.mp4"
        objpoints, imgpoints, img_size = collect_from_video(
            VIDEO_PATH, PATTERN_SIZE, SQUARE_SIZE, every_n_frames=10, show=SHOW, use_sb=USE_SB
        )
        src = VIDEO_PATH

    # Calibrate with filtering (recommended)
    rms, K, dist, rvecs, tvecs, newK, roi, reproj_rmse = calibrate_with_filter(
        objpoints, imgpoints, img_size, model=MODEL, keep_ratio=KEEP_RATIO
    )

    save_npz(
        OUT_NPZ, K, dist, img_size, rms, reproj_rmse,
        newK=newK, roi=roi, pattern_size=PATTERN_SIZE,
        square_size=SQUARE_SIZE, source=src, model=MODEL
    )

if __name__ == "__main__":
    main()
