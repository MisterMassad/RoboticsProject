# Michel Massad
# Monocular Two-View Pose Estimation

This repository contains a **monocular two-view pose estimation pipeline** built using classical computer vision methods.  
Given **two RGB images** and **camera intrinsics**, the system estimates the **relative rotation and translation (up to scale)** between the camera views.

This project was developed as part of a **Navigation / Robotics course**.
A lot of time was spent on robustness, evaluation, debugging tools, and understanding failure cases.

---

## What This Project Does

- Takes **two images from the same camera**
- Uses **calibrated intrinsics** (or estimated ones if needed)
- Estimates:
  - Relative **rotation matrix**
  - Relative **translation direction** (monocular → no absolute scale)
- Includes **evaluation tools**, **calibration support**, **video testing**, and **ArUco-based ground truth comparison**

This is **not full SLAM** and **not VO over time** — it’s a clean, controlled two-view geometry pipeline.

---

## Core Pipeline

1. Load images (image mode or video frames)
2. **area-based downscaling** (camera-agnostic)
3. Convert to grayscale
4. Extract features (ORB and SIFT were both tested)
5. Descriptor matching:
   - KNN matching
   - Lowe’s ratio test
   - **Adaptive ratio test**
   - **symmetric (mutual) matching**
6. Estimate Essential Matrix using **USAC_MAGSAC**
7. Recover pose using `cv2.recoverPose`
8. Sanity checks:
   - Minimum inliers
   - Low-confidence warnings
   - Candidate selection when multiple E matrices are returned
9. Output final `(R, t)` pose

---

## Feature Choices

- **ORB**
  - Fast
  - Lightweight
  - Very sensitive to image resolution
- **SIFT**
  - Much more stable across scales
  - Better rotation accuracy
  - Slower (~5× compared to ORB)

Both were implemented and **extensively evaluated**, especially under resolution changes.

---

## Matching Improvements

Several improvements were added over the basic Lowe’s ratio test:

- **Adaptive ratio threshold**
  - Based on median descriptor distance
  - Adjusted using distance dispersion
- **Adaptive symmetry selection**
  - Uses symmetric matching when possible
  - Falls back to one-way matching if mutual matches are too few
- Minimum match and inlier logic to avoid unstable poses

These changes significantly reduced catastrophic failures.

---

## Camera Calibration

- Supports **multiple calibrated cameras**
- Calibration stored as `.npz` files
- Intrinsics are:
  - Automatically scaled when images are resized
  - Loaded using key `"K"` (or auto-detected if missing)
- Distortion parameters are intentionally **not used** (not required by the project)

---

## Evaluation & Debugging Tools

This repo includes **a lot of evaluation infrastructure**, not just the final pipeline:

- ArUco-based **ground truth pose estimation**
- Image-pair batch evaluation
- CSV export of:
  - Rotation error
  - Euler angle error
  - Translation direction error
- Video frame-to-frame testing (debugging only)
- Visual debugging:
  - Keypoints
  - Raw matches
  - Inliers
  - Epipolar lines
  - 3D pose visualization

ArUco is **only used for evaluation**, not for pose estimation.

---

## Repository Structure (High Level)

- `poseEstimation.py` – Core two-view pose estimation pipeline  
- `calibration/` – Camera calibration files (`.npz`)  
- `aruco/` – ArUco pose estimation and comparison tools  
- `evaluation/` – Batch evaluation, CSV export, analysis scripts  
- `video_tools/` – Frame-by-frame video testing utilities  

(Some folders may be optional depending on usage.)

---

## How to Run (Two Images)

```bash
python poseEstimation.py path/to/img1.jpg path/to/img2.jpg path/to/calibration.npz
