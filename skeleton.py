#!/usr/bin/env python3
"""
Live skeleton with constrained 3D joint angles.

Pipeline:
    MediaPipe   -> 2D keypoints (accurate) + world landmarks (weak depth)
    calibration -> your bone lengths and which way is down, over a few seconds
    solver      -> 3D pose that reprojects onto the 2D keypoints under real
                   camera intrinsics, with your bone lengths, a rigid torso,
                   knees that bend the way knees bend, and constant-velocity
                   continuity between frames
    One Euro    -> jitter removal without the lag of a fixed EMA, followed by
                   a projection back onto the bone lengths

Against synthetic ground truth (test_pose3d.py) this cuts mean joint-angle
error 62% versus reading pose_world_landmarks directly -- 13.6 deg to 5.2 --
and p95 from 42 to 17.

How much each part is worth, in mean degrees (test_pose3d.py --ablate):

    gravity reference       0.77     knees, tilts and trunk lean
    knee hinge constraint   0.75
    constant-velocity       0.33

Accuracy is not uniform across joints, and the spread matters more than the
average. Knees land near 2.3 deg and the tilts near 1 deg; shoulders, hips and
trunk sit around 3-4; elbows around 5. Wrists are 12-20 deg and should not be
trusted: they are measured from the index-finger landmark across a segment a
few dozen pixels long, whose depth is close to unobservable, and they sit near
180 degrees where noise can only bend the reading away from straight. Ankles
have a milder version of the same problem. Run with --per-joint to see this.

There is no setup step, nothing to print, and nothing to measure or pass in.
Open the laptop, run it, stand back far enough that your head and feet are
both in frame, and follow the on-screen prompt for a few seconds.

The two things a system like this normally makes you configure are both worked
out during that prompt:

    Screen tilt. Your laptop screen leans back, so the camera in its bezel
    points up at you, and trunk lean and both tilts are measured against
    vertical -- a screen leaning back 15 degrees puts 15 degrees of bias on all
    three. Standing up straight during calibration makes you a plumb line, so
    the tilt is read off your own trunk in the solved 3D poses. Roughly 12
    degrees of tilt comes back with about 1.5 degrees of residual error.

    (The lid angle sensor would seem like the obvious source here, and this
    machine has one -- an HID service on usage page 0x20. It reports nothing
    to an unprivileged process, and it would answer a slightly wrong question
    anyway: it gives the lid's angle to the *base*, which only equals the
    camera's angle to gravity when the laptop is on a level surface, and not
    on a lap or a stack of books. Reading it off the body measures the thing
    that actually matters.)

    Bone lengths, from the world landmarks over the same window.

Field of view is assumed rather than measured, and that is a deliberate trade
rather than a gap: joint angles move less than a degree across a focal length
0.7x to 1.5x of truth. Only the distance readout depends on it, so distance is
shown marked approximate.

Install:
    pip install opencv-python mediapipe numpy scipy
    pip install pyobjc-framework-AVFoundation   # optional, for camera names

Run:
    python skeleton.py
    python skeleton.py --list
    python skeleton.py --model heavy --raw      # A/B against unconstrained

Keys:
    q / ESC   quit
    s         toggle skeleton-only view
    m         toggle mirror
    h         toggle HUD
    a         toggle angle labels at each joint
    p         toggle the angle panel
    d         toggle solver diagnostics
    c         recalibrate bone lengths
    r         start/stop CSV logging
"""

import argparse
import csv
import os
import sys
import time
import urllib.request

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

import pose3d
from pose3d import (ALL_LABELS, BLAZE_IDS, JOINT_SPECS, N_JOINTS,
                    BoneCalibrator, OneEuroFilter, PoseSolver,
                    compute_angles, enforce_bone_lengths)

import reps
import sonify

IS_MAC = sys.platform == "darwin"
BACKEND = cv2.CAP_AVFOUNDATION if IS_MAC else cv2.CAP_ANY

# Joints that must be visible for a frame to count toward calibration. The
# torso is what anchors both the bone medians and the vertical estimate.
CALIB_CORE = (pose3d.L_SHOULDER, pose3d.R_SHOULDER,
              pose3d.L_HIP, pose3d.R_HIP,
              pose3d.L_KNEE, pose3d.R_KNEE)

BUILTIN_HINTS = ("facetime", "built-in", "builtin", "isight", "macbook")
# Desk View is a warped downward crop of the ultra-wide, and a phone used as a
# Continuity Camera hands control of the session to another device. Neither is
# auto-selected; pass --camera to use one deliberately.
AVOID_HINTS = ("desk view", "iphone", "ipad", "phone", "continuity")

# Assumed horizontal field of view for a laptop webcam, degrees. It stands in
# for a real calibration and does not need to be exact: see
# pose3d.intrinsics_from_fov for why joint angles hardly depend on it and the
# reported distance does.
FOV_WEBCAM = 60.0

WARMUP_FRAMES = 20
DARK_THRESHOLD = 12.0
MAX_PROBE_INDEX = 4

FULL_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}

# BGR
BONE_COLOR = (170, 255, 0)
JOINT_COLOR = (0, 200, 255)
WEAK_COLOR = (90, 90, 90)
HUD_COLOR = (240, 240, 240)
PANEL_BG = (24, 24, 24)
DIM_COLOR = (130, 130, 130)
REC_COLOR = (60, 60, 255)
WARN_COLOR = (60, 200, 255)


def parse_args():
    p = argparse.ArgumentParser(description="Live skeleton with 3D joint angles.")
    p.add_argument("--camera", type=int, default=None,
                   help="camera index; omit to auto-select the built-in camera")
    p.add_argument("--list", action="store_true", help="list cameras and exit")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--model", choices=["lite", "full", "heavy"], default="full")
    p.add_argument("--model-path", default=None)
    p.add_argument("--calib-frames", type=int, default=90,
                   help="frames used to measure your bone lengths")
    p.add_argument("--vis", type=float, default=0.5)
    p.add_argument("--no-hinge", action="store_true",
                   help="drop the anatomical knee constraint")
    p.add_argument("--raw", action="store_true",
                   help="skip the solver and report MediaPipe world landmarks "
                        "directly, for A/B comparison")
    p.add_argument("--no-filter", action="store_true",
                   help="disable the One Euro filter")
    p.add_argument("--min-cutoff", type=float, default=1.0,
                   help="One Euro min cutoff; lower = smoother when still")
    p.add_argument("--beta", type=float, default=0.7,
                   help="One Euro speed coefficient; higher = less lag")

    a = p.add_argument_group("audio spike")
    a.add_argument("--audio", action="store_true",
                   help="sonify knee angle; pitch falls as you descend")
    a.add_argument("--target", type=float, default=90.0,
                   help="target knee angle, degrees; press t to capture the "
                        "angle you are currently holding instead")
    a.add_argument("--curve", choices=sonify.CURVES, default="log",
                   help="pitch mapping; 'near' concentrates resolution at "
                        "the target")
    a.add_argument("--signal", choices=sonify.SIGNALS, default="pitch",
                   help="target-lock layer over the pitch channel")
    a.add_argument("--smoothing", type=float, default=8.0,
                   help="audio parameter smoothing, Hz; lower is smoother "
                        "and laggier")
    a.add_argument("--f-low", type=float, default=200.0)
    a.add_argument("--f-high", type=float, default=800.0)

    b = p.add_argument_group("phase B measurement")
    b.add_argument("--blind", type=int, default=0, metavar="N",
                   help="blank the screen, tone on, stop after N reps")
    b.add_argument("--silent", type=int, default=0, metavar="N",
                   help="blank the screen, NO tone, stop after N reps. This "
                        "is the control condition and must be run first: "
                        "once the tone has taught you the depth you cannot "
                        "un-learn it")
    b.add_argument("--log", default=None, metavar="PATH",
                   help="write one CSV row per completed rep")
    return p.parse_args()


def summarize(rep_log, target_deg, label):
    """Print the depth-accuracy table for one condition."""
    if not rep_log:
        return
    err = np.array([e.bottom_deg - target_deg for e in rep_log])
    sd = err.std(ddof=1) if len(err) > 1 else float("nan")
    print(f"\n=== {label} ===")
    print(f"  reps                 {len(rep_log)}")
    print(f"  target               {target_deg:6.1f} deg")
    print(f"  mean absolute error  {np.abs(err).mean():6.2f} deg")
    print(f"  mean signed error    {err.mean():+6.2f} deg   "
          f"({'shallow' if err.mean() > 0 else 'deep'} on average)")
    print(f"  standard deviation   {sd:6.2f} deg")
    print(f"  worst rep            {np.abs(err).max():6.2f} deg off")
    print("  Tracker knee error is ~2.3 deg; differences below that are "
          "not measurable here.")


# --------------------------------------------------------------------------
# Camera selection
# --------------------------------------------------------------------------

def _allow_continuity_camera():
    """
    Make an iPhone used as a webcam visible to device discovery.

    AVFoundation hides Continuity Cameras unless the running bundle declares
    NSCameraUseContinuityCameraDeviceType, and a plain Python process has no
    Info.plist to declare it in. Without this the phone is missing from our
    list while OpenCV still counts it, so the names we print end up describing
    different cameras than the indices we open.
    """
    try:
        from Foundation import NSBundle
        bundle = NSBundle.mainBundle()
        info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
        info["NSCameraUseContinuityCameraDeviceType"] = True
    except Exception:
        pass


def avfoundation_names():
    if not IS_MAC:
        return None
    try:
        import AVFoundation as AVF
    except ImportError:
        return None
    _allow_continuity_camera()
    try:
        type_names = [
            "AVCaptureDeviceTypeBuiltInWideAngleCamera",
            "AVCaptureDeviceTypeExternal",
            "AVCaptureDeviceTypeExternalUnknown",
            "AVCaptureDeviceTypeContinuityCamera",
            "AVCaptureDeviceTypeDeskViewCamera",
        ]
        types = [getattr(AVF, n) for n in type_names if hasattr(AVF, n)]
        session = (AVF.AVCaptureDeviceDiscoverySession
                   .discoverySessionWithDeviceTypes_mediaType_position_(
                       types, AVF.AVMediaTypeVideo, 0))
        devices = list(session.devices())
    except Exception:
        try:
            import AVFoundation as AVF
            devices = list(
                AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeVideo))
        except Exception:
            return None
    try:
        return [str(d.localizedName()) for d in devices]
    except Exception:
        return None


def probe_brightness(index):
    cap = cv2.VideoCapture(index, BACKEND)
    if not cap.isOpened():
        cap.release()
        return None
    frame = None
    for _ in range(WARMUP_FRAMES):
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    return None if frame is None else float(frame.mean())


def list_cameras():
    names = avfoundation_names()
    if names:
        print("Detected cameras:")
        for i, name in enumerate(names):
            low = name.lower()
            if any(h in low for h in AVOID_HINTS):
                tag = "  <- skipped by default"
            elif any(h in low for h in BUILTIN_HINTS):
                tag = "  <- built-in (used by default)"
            else:
                tag = ""
            print(f"  {i}: {name}{tag}")
        return
    print("Camera names unavailable (install pyobjc-framework-AVFoundation). "
          "Probing indices:")
    for i in range(MAX_PROBE_INDEX):
        b = probe_brightness(i)
        print(f"  {i}: " + ("not available" if b is None else
                            f"mean brightness {b:6.2f}"
                            f"{'  (looks black)' if b < DARK_THRESHOLD else ''}"))


def resolve_camera(requested):
    names = avfoundation_names()
    if requested is not None:
        label = names[requested] if names and requested < len(names) else "unknown"
        return requested, label
    if names:
        for i, name in enumerate(names):
            if any(h in name.lower() for h in BUILTIN_HINTS):
                return i, name
        for i, name in enumerate(names):
            if not any(h in name.lower() for h in AVOID_HINTS):
                return i, name
    print("Probing cameras (install pyobjc-framework-AVFoundation to skip)...")
    for i in range(MAX_PROBE_INDEX):
        b = probe_brightness(i)
        if b is not None and b >= DARK_THRESHOLD:
            return i, f"index {i}"
    return 0, "index 0 (fallback)"


# --------------------------------------------------------------------------

def ensure_model(kind, explicit_path):
    if explicit_path:
        if not os.path.exists(explicit_path):
            raise SystemExit(f"No model file at {explicit_path}")
        return explicit_path
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "pose_landmarker")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"pose_landmarker_{kind}.task")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    url = MODEL_URLS[kind]
    print(f"Downloading {kind} model -> {path}")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as exc:
        if os.path.exists(path):
            os.remove(path)
        raise SystemExit(
            f"Could not download the model ({exc}).\n"
            f"Grab it manually:\n  curl -L -o {path} {url}\n"
            "Or pass --model-path.")
    return path


def resolve_intrinsics(width, height):
    """
    Camera matrix with no setup step and nothing to configure.

    macOS will not tell us the field of view. AVCaptureDeviceFormat has a
    videoFieldOfView property and it reads 0.0 for every camera on this
    machine, the built-in and an iPhone alike, so there is nothing to query
    and a fixed figure is what is left.

    That is a much better trade than it sounds. Sweeping the assumed focal
    length from 0.7x to 1.5x of truth moves mean joint-angle error by under a
    degree, because the bone-length and reprojection constraints between them
    pin the *shape* of the skeleton nearly independently of how the viewing
    rays are scaled. What a wrong field of view does move is the reported
    distance, roughly proportionally, so that number is shown marked as
    approximate and nothing else changes.
    """
    return pose3d.intrinsics_from_fov(width, height, FOV_WEBCAM), FOV_WEBCAM


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

def text_out(img, text, org, scale=0.6, color=HUD_COLOR, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thick, cv2.LINE_AA)


def draw_skeleton(canvas, px, vis, vis_thresh):
    for a, b in FULL_CONNECTIONS:
        solid = vis[a] > vis_thresh and vis[b] > vis_thresh
        cv2.line(canvas, px[a], px[b], BONE_COLOR if solid else WEAK_COLOR,
                 3 if solid else 1, cv2.LINE_AA)
    for i, (x, y) in enumerate(px):
        if vis[i] <= vis_thresh:
            continue
        cv2.circle(canvas, (x, y), 6, JOINT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 6, (20, 20, 20), 1, cv2.LINE_AA)


def draw_joint_labels(canvas, px_full, angles):
    for label, _, vertex, _ in JOINT_SPECS:
        value = angles.get(label)
        if value is None:
            continue
        x, y = px_full[BLAZE_IDS[vertex]]
        text_out(canvas, f"{value:.0f}", (x + 10, y - 8), 0.55)


def draw_panel(canvas, angles, depth_m, recording, rec_rows,
               depth_exact=True):
    h, w = canvas.shape[:2]
    pad, row_h, panel_w = 12, 22, 215
    panel_h = pad * 2 + row_h * (len(ALL_LABELS) + 2)
    x0, y0 = w - panel_w - 16, 16

    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.62, canvas, 0.38, 0, canvas)
    cv2.rectangle(canvas, (x0, y0), (x0 + panel_w, y0 + panel_h), (70, 70, 70), 1)

    y = y0 + pad + 16
    text_out(canvas, "joint angles (deg)", (x0 + pad, y), 0.52)
    y += row_h
    for label in ALL_LABELS:
        value = angles.get(label)
        if value is None:
            text_out(canvas, f"{label:<14}   --", (x0 + pad, y), 0.5, DIM_COLOR)
        else:
            text_out(canvas, f"{label:<14} {value:5.1f}", (x0 + pad, y), 0.5)
        y += row_h
    if depth_m is not None:
        # Distance rides directly on the assumed field of view in a way the
        # angles above do not, so it is marked when that figure was a guess.
        suffix = "" if depth_exact else " ~"
        text_out(canvas, f"{'distance':<14} {depth_m:4.2f}m{suffix}",
                 (x0 + pad, y), 0.5,
                 HUD_COLOR if depth_exact else DIM_COLOR)

    if recording:
        cv2.circle(canvas, (x0 + panel_w - 18, y0 + 18), 6, REC_COLOR, -1,
                   cv2.LINE_AA)
        text_out(canvas, f"REC {rec_rows}", (x0 + panel_w - 100, y0 + 23),
                 0.45, REC_COLOR)


def up_tilt_deg(up):
    """Camera pitch implied by a world-up vector, in degrees."""
    return float(np.degrees(np.arctan2(-up[2], -up[1])))


def draw_calib_overlay(canvas, done, target):
    # Both instructions here are load-bearing, and neither is obvious.
    #
    # Stand up straight: the trunk doubles as the plumb line that tells the
    # solver which way is down, so a camera tilt gets measured out of a
    # subject who is actually vertical. Calibrate mid-slouch and every trunk
    # and tilt reading afterwards inherits the slouch.
    #
    # Move your arms and legs: bone lengths are read out of foreshortening,
    # and a limb held at one angle for the whole window gives one measurement
    # repeated, not many. Sweeping through a range is what separates a
    # segment's length from its depth.
    h, w = canvas.shape[:2]
    msg = "MEASURING YOUR BONE LENGTHS"
    sub = "stand up straight, whole body in frame"
    sub2 = "sweep your arms and bend a knee a few times"
    frac = done / max(1, target)

    text_out(canvas, msg, (w // 2 - 250, h // 2 - 44), 0.9, WARN_COLOR, 2)
    text_out(canvas, sub, (w // 2 - 300, h // 2 - 10), 0.6, HUD_COLOR)
    text_out(canvas, sub2, (w // 2 - 300, h // 2 + 14), 0.6, HUD_COLOR)

    bx, by, bw = w // 2 - 200, h // 2 + 30, 400
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + 18), (60, 60, 60), -1)
    cv2.rectangle(canvas, (bx, by), (bx + int(bw * frac), by + 18),
                  WARN_COLOR, -1)
    text_out(canvas, f"{done}/{target}", (bx + bw + 12, by + 15), 0.55)


# --------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.list:
        list_cameras()
        return

    cam_index, cam_label = resolve_camera(args.camera)
    print(f"Using camera {cam_index}: {cam_label}")

    model_path = ensure_model(args.model, args.model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    cap = cv2.VideoCapture(cam_index, BACKEND)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {cam_index}. Try --list.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    for _ in range(WARMUP_FRAMES):
        cap.read()

    ok, probe = cap.read()
    if not ok:
        raise SystemExit("Camera opened but returned no frames.")
    H, W = probe.shape[:2]

    K, fov_deg = resolve_intrinsics(W, H)
    print(f"fx {K[0,0]:.1f}  ({fov_deg:.0f} deg FOV, assumed). Joint angles "
          "barely depend on this; the distance readout does.")

    calibrator = BoneCalibrator(args.calib_frames)
    solver = None
    lengths = None
    euro = OneEuroFilter(min_cutoff=args.min_cutoff, beta=args.beta)

    # --silent is the control condition, so it blanks the screen exactly like
    # --blind but opens no stream. Anything visual left on screen in either
    # mode is depth feedback by another route and would contaminate the run.
    blind_mode = args.blind > 0 or args.silent > 0
    target_reps = args.blind or args.silent
    target_deg = args.target
    detector = reps.BottomDetector()
    rep_log = []
    knee = None

    sonifier = None
    if args.audio or args.blind > 0:
        acfg = sonify.Config(
            target_deg=target_deg, f_low=args.f_low, f_high=args.f_high,
            curve=args.curve, signal=args.signal, smoothing_hz=args.smoothing)
        sonifier = sonify.Sonifier(acfg).start()
        print(f"Audio: {acfg.curve}/{acfg.signal}, target {target_deg:.0f} deg"
              f" -> {sonify.angle_to_hz(acfg, target_deg):.0f} Hz, "
              f"output latency {sonifier.output_latency_ms:.0f} ms")

    rep_file = rep_csv = None
    if args.log:
        rep_file = open(args.log, "w", newline="")
        rep_csv = csv.writer(rep_file)
        rep_csv.writerow(["rep", "bottom_deg", "error_deg", "t_sec",
                          "condition"])
    condition = ("blind+tone" if args.blind else
                 "silent control" if args.silent else "open")

    # World vertical in camera coordinates. Trunk lean and both tilts are
    # measured against it, so a screen tilted back biases all three by its own
    # angle. Worked out from your standing posture during calibration, below.
    up = pose3d.up_from_pitch(0.0)

    skeleton_only = False
    mirrored = True
    show_hud = True
    labels_on = True
    panel_on = True
    diag_on = False

    csv_file = csv_writer = None
    rec_rows = 0

    fps = 0.0
    solve_ms = 0.0
    start = time.perf_counter()
    last_t = start
    last_stamp = -1
    info = {}

    vis_thresh = args.vis
    vis_checked = False

    window = "skeleton"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    landmarker = vision.PoseLandmarker.create_from_options(options)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Dropped frame from camera, stopping.")
                break

            # Geometry runs on the true image. Mirroring is display-only --
            # solving on a flipped frame would silently swap handedness.
            H, W = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            stamp = int((time.perf_counter() - start) * 1000)
            if stamp <= last_stamp:
                stamp = last_stamp + 1
            last_stamp = stamp

            result = landmarker.detect_for_video(mp_image, stamp)

            canvas = frame if not skeleton_only else np.zeros_like(frame)
            if mirrored:
                canvas = cv2.flip(canvas, 1)

            angles = {label: None for label in ALL_LABELS}
            depth_m = None
            tracking = bool(result.pose_landmarks)

            if tracking:
                lms = result.pose_landmarks[0]
                vis_full = np.array([lm.visibility for lm in lms])
                uv_full = np.array([[lm.x * W, lm.y * H] for lm in lms])

                if not vis_checked:
                    vis_checked = True
                    if vis_full.max() < 1e-6:
                        print("Model reports no visibility scores; "
                              "treating all joints as visible.")
                        vis_thresh = -1.0
                        vis_full = np.ones_like(vis_full)

                # Pixel coordinates for drawing, mirrored if the view is.
                px_full = [(int(W - 1 - u) if mirrored else int(u), int(v))
                           for u, v in uv_full]
                draw_skeleton(canvas, px_full, vis_full, vis_thresh)

                uv = uv_full[BLAZE_IDS]
                vis = vis_full[BLAZE_IDS]
                if vis_thresh < 0:
                    vis = np.ones_like(vis)

                world = None
                if result.pose_world_landmarks:
                    wl = result.pose_world_landmarks[0]
                    world = np.array([[l.x, l.y, l.z] for l in wl])[BLAZE_IDS]

                if world is not None:
                    # Undistort once, up front. The calibrator stores these
                    # points and later re-solves them to find vertical, so it
                    # needs the same ideal-pinhole frame the solver works in.
                    pts = uv

                    if args.raw:
                        angles = compute_angles(world, vis, up=up)
                    elif solver is None:
                        # Still measuring bone lengths. Gate on the torso
                        # rather than on every joint: add() already drops
                        # individual bones whose endpoints are not visible,
                        # and a hand or foot dipping below threshold is normal
                        # and near-continuous. Requiring all nineteen at once
                        # lets calibration stall forever on a subject who is
                        # standing there perfectly well.
                        core_vis = min(vis[j] for j in CALIB_CORE)
                        if core_vis >= 0.4 or vis_thresh < 0:
                            calibrator.add(world, vis, vis_thresh=0.4, uv=pts)
                        if calibrator.done:
                            lengths = calibrator.result()
                            solver = PoseSolver(K, lengths,
                                                use_hinges=not args.no_hinge)
                            print("Bone lengths (cm): " + "  ".join(
                                f"{l*100:.1f}" for l in lengths[:8]) + " ...")
                            est = calibrator.estimate_up(K, lengths)
                            if est is not None:
                                up = est
                                print(f"Screen tilt: {up_tilt_deg(up):+.1f} deg")
                            else:
                                print("Could not read vertical off the "
                                      "calibration (were you standing up "
                                      "straight?). Assuming a level screen.")
                        draw_calib_overlay(canvas, calibrator.count,
                                           args.calib_frames)
                    else:
                        t0 = time.perf_counter()
                        X, info = solver.solve(pts, vis, world,
                                               vis_thresh=max(vis_thresh, 0.0))
                        solve_ms = 0.8 * solve_ms + 0.2 * (
                            (time.perf_counter() - t0) * 1000.0)
                        if not args.no_filter:
                            X = euro(X.ravel()).reshape(N_JOINTS, 3)
                            # Smoothing each coordinate on its own knows
                            # nothing about the skeleton and quietly shortens
                            # limbs during fast motion. Put them back.
                            X = enforce_bone_lengths(X, lengths, weights=vis)
                        angles = compute_angles(X, vis, up=up)
                        depth_m = float(np.median(X[:, 2]))

                if labels_on and any(v is not None for v in angles.values()):
                    draw_joint_labels(canvas, px_full, angles)

                if csv_writer is not None and any(
                        v is not None for v in angles.values()):
                    row = [f"{stamp/1000.0:.3f}"]
                    row += ["" if angles[l] is None else f"{angles[l]:.2f}"
                            for l in ALL_LABELS]
                    row.append("" if depth_m is None else f"{depth_m:.3f}")
                    csv_writer.writerow(row)
                    rec_rows += 1
            else:
                euro.reset()
                if solver is not None:
                    solver.reset()

            knee = reps.knee_signal(angles)
            if sonifier is not None:
                sonifier.update(knee)

            event = detector.update(knee, stamp / 1000.0)
            if event is not None:
                rep_log.append(event)
                err = event.bottom_deg - target_deg
                print(f"rep {event.index + 1:3d}   bottom "
                      f"{event.bottom_deg:6.1f} deg   error {err:+6.1f}")
                if rep_csv is not None:
                    rep_csv.writerow([event.index, f"{event.bottom_deg:.2f}",
                                      f"{err:.2f}", f"{event.t:.3f}",
                                      condition])
                    rep_file.flush()
                if target_reps and len(rep_log) >= target_reps:
                    break

            if panel_on:
                draw_panel(canvas, angles, depth_m,
                           csv_writer is not None, rec_rows,
                           depth_exact=False)

            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            if show_hud:
                mode = "raw (no solver)" if args.raw else (
                    "solved" if solver else "calibrating")
                lines = [
                    f"{fps:5.1f} fps",
                    "tracking" if tracking else "no pose",
                    f"mode: {mode}",
                    f"cam: {cam_label}",
                    f"fov: {fov_deg:.0f} deg (assumed)",
                    f"screen tilt: {up_tilt_deg(up):+.1f} deg",
                ]
                if sonifier is not None:
                    hz = sonifier.current_hz
                    acfg = sonifier.cfg
                    ear = (sonifier.output_latency_ms or 0.0) + (
                        1000.0 / fps if fps else 0.0)
                    lines += [
                        f"audio: {acfg.curve}/{acfg.signal}"
                        + ("  muted" if sonifier.muted else ""),
                        f"tone: {'--' if hz is None else f'{hz:6.1f} Hz'}"
                        f"   smooth {acfg.smoothing_hz:4.1f} Hz",
                        f"target: {target_deg:5.1f} deg   err: "
                        + ("--" if knee is None
                           else f"{knee - target_deg:+5.1f}"),
                        f"reps: {len(rep_log)}",
                        f"glass-to-ear: {ear:5.1f} ms (loop+audio, est)",
                    ]
                lines += [
                    "s skeleton  m mirror  a labels  p panel",
                    "d diagnostics  c recalibrate  r log  q quit",
                ]
                if sonifier is not None:
                    lines.append(
                        "1/2 curve  [ ] signal  t target  -/= smooth  SPC mute")
                y = 28
                for text in lines:
                    color = (WARN_COLOR if text.endswith("GUESSED")
                             else HUD_COLOR)
                    text_out(canvas, text, (16, y), color=color)
                    y += 26

                if diag_on and info:
                    y += 8
                    for k in ("reproj_px", "bone_err_mm", "depth_m", "nfev"):
                        if k in info:
                            text_out(canvas, f"{k}: {info[k]:.3f}"
                                     if isinstance(info[k], float)
                                     else f"{k}: {info[k]}", (16, y), 0.55)
                            y += 24
                    text_out(canvas, f"solve: {solve_ms:.1f} ms", (16, y), 0.55)

            if blind_mode:
                # Rep count only. The subject needs to know how far through
                # the set they are; anything about depth would be the very
                # feedback the control condition exists to withhold.
                canvas[:] = 0
                text_out(canvas, f"{len(rep_log)} / {target_reps}",
                         (40, 90), scale=2.0, thick=3)

            cv2.imshow(window, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                skeleton_only = not skeleton_only
            elif key == ord("m"):
                mirrored = not mirrored
            elif key == ord("h"):
                show_hud = not show_hud
            elif key == ord("a"):
                labels_on = not labels_on
            elif key == ord("p"):
                panel_on = not panel_on
            elif key == ord("d"):
                diag_on = not diag_on
            elif key in (ord("1"), ord("2")) and sonifier is not None:
                sonifier.set_curve("log" if key == ord("1") else "near")
                print(f"curve: {sonifier.cfg.curve}")
            elif key in (ord("["), ord("]")) and sonifier is not None:
                step = 1 if key == ord("]") else -1
                i = sonify.SIGNALS.index(sonifier.cfg.signal)
                sonifier.set_signal(
                    sonify.SIGNALS[(i + step) % len(sonify.SIGNALS)])
                print(f"signal: {sonifier.cfg.signal}")
            elif key == ord("t"):
                if knee is None:
                    print("No knee angle right now; hold the position.")
                else:
                    target_deg = knee
                    if sonifier is not None:
                        sonifier.set_target(target_deg)
                    print(f"Target set to {target_deg:.1f} deg "
                          "(previous reps in this session now measure "
                          "against it)")
            elif key in (ord("-"), ord("=")) and sonifier is not None:
                sonifier.set_smoothing(sonifier.cfg.smoothing_hz
                                       * (0.7 if key == ord("-") else 1.4))
                print(f"smoothing: {sonifier.cfg.smoothing_hz:.1f} Hz")
            elif key == ord(" ") and sonifier is not None:
                sonifier.muted = not sonifier.muted
            elif key == ord("c"):
                calibrator = BoneCalibrator(args.calib_frames)
                solver = None
                lengths = None
                euro.reset()
                up = pose3d.up_from_pitch(0.0)
                print("Recalibrating bone lengths...")
            elif key == ord("r"):
                if csv_writer is None:
                    path = time.strftime("angles_%Y%m%d_%H%M%S.csv")
                    csv_file = open(path, "w", newline="")
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(["t_sec"] + ALL_LABELS + ["distance_m"])
                    rec_rows = 0
                    print(f"Logging -> {path}")
                else:
                    csv_writer = None
                    csv_file.close()
                    csv_file = None
                    print(f"Stopped logging ({rec_rows} rows)")
    finally:
        if csv_file is not None:
            csv_file.close()
        if rep_file is not None:
            rep_file.close()
        if sonifier is not None:
            sonifier.stop()
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()
        summarize(rep_log, target_deg, condition)


if __name__ == "__main__":
    main()
