"""
Synthetic ground truth for measuring the solver.

The point of this module is to be pessimistic in the same places reality is.
An easy benchmark is worse than none: it will happily certify a change that
does nothing, because the thing that actually costs you degrees was never
modeled.

What is modeled here that a naive harness leaves out:

  * Bone lengths are NOT handed to the solver. The pipeline has to calibrate
    them the same way it does live, from the same corrupted data. Measuring a
    solver on ground-truth limb lengths measures a solver you do not have.
  * The camera is tilted. A laptop webcam sits below eye level and points up
    ~10-15 deg, which biases every angle measured against a camera axis by
    exactly that much.
  * Keypoints have heavy tails, not clean Gaussian noise. Detectors mostly do
    well and occasionally place a joint somewhere absurd.
  * Joints get occluded and visibility drops, in runs rather than per-frame.
  * Motions include the genuinely ambiguous ones -- a limb swinging through
    the camera axis, where the 2D projection is nearly identical whether the
    limb is coming toward you or going away.

Coordinate frames:
    body    y-down, origin between the hips, +z is the direction the
            subject faces
    world   gravity-aligned, y-down
    camera  OpenCV: +x right, +y down, +z forward. Tilted relative to world.
"""

import numpy as np

import pose3d
from pose3d import BONES, N_JOINTS, _project

# MediaPipe's world landmarks compress the depth axis and are noisy in all
# three. These numbers are in the range people measure on BlazePose; the exact
# values matter less than the fact that z is much worse than xy.
WORLD_Z_GAIN = 0.65
WORLD_XY_NOISE = 0.015
WORLD_Z_NOISE = 0.055

# 2D keypoint error: mostly a few pixels, occasionally badly wrong.
PIXEL_NOISE = 2.5
OUTLIER_RATE = 0.01
OUTLIER_PIXELS = 35.0

# Fraction of frames a given joint spends occluded, and the per-frame chance
# of coming back once it is. Together these set how long a dropout lasts:
# an exit rate of 0.08 means a typical occlusion runs about a dozen frames.
OCCLUSION_DUTY = 0.08
OCCLUSION_EXIT = 0.08


def build_body():
    """A plausible standing skeleton in body coordinates, meters, y-down."""
    X = np.zeros((N_JOINTS, 3))
    X[pose3d.L_HIP] = [-0.09, 0.00, 0.0]
    X[pose3d.R_HIP] = [0.09, 0.00, 0.0]
    X[pose3d.L_SHOULDER] = [-0.18, -0.50, 0.0]
    X[pose3d.R_SHOULDER] = [0.18, -0.50, 0.0]
    X[pose3d.NOSE] = [0.00, -0.75, 0.06]
    X[pose3d.L_ELBOW] = [-0.24, -0.22, 0.02]
    X[pose3d.R_ELBOW] = [0.24, -0.22, 0.02]
    X[pose3d.L_WRIST] = [-0.28, 0.06, 0.04]
    X[pose3d.R_WRIST] = [0.28, 0.06, 0.04]
    X[pose3d.L_INDEX] = [-0.30, 0.22, 0.05]
    X[pose3d.R_INDEX] = [0.30, 0.22, 0.05]
    X[pose3d.L_KNEE] = [-0.10, 0.46, 0.01]
    X[pose3d.R_KNEE] = [0.10, 0.46, 0.01]
    X[pose3d.L_ANKLE] = [-0.11, 0.90, 0.00]
    X[pose3d.R_ANKLE] = [0.11, 0.90, 0.00]
    X[pose3d.L_HEEL] = [-0.11, 0.97, -0.04]
    X[pose3d.R_HEEL] = [0.11, 0.97, -0.04]
    X[pose3d.L_FOOT] = [-0.11, 0.97, 0.14]
    X[pose3d.R_FOOT] = [0.11, 0.97, 0.14]
    return X


# The foot is one rigid triangle hanging off the ankle. Offsets, not
# independent joints, so the ankle-heel-toe lengths cannot drift.
FOOT_OFFSETS = {
    pose3d.L_HEEL: np.array([0.0, 0.07, -0.04]),
    pose3d.L_FOOT: np.array([0.0, 0.07, 0.14]),
    pose3d.R_HEEL: np.array([0.0, 0.07, -0.04]),
    pose3d.R_FOOT: np.array([0.0, 0.07, 0.14]),
}


def bone_lengths_of(X):
    return np.array([np.linalg.norm(X[a] - X[b]) for a, b in BONES])


def symmetrize(lengths):
    out = np.asarray(lengths, dtype=np.float64).copy()
    for i, j in pose3d.SYMMETRIC_PAIRS:
        m = 0.5 * (out[i] + out[j])
        out[i] = out[j] = m
    return out


def rot_x(deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# --------------------------------------------------------------------------
# Motions
# --------------------------------------------------------------------------
#
# Each returns (body_pose, world_rotation). Bone lengths are held exactly
# constant by construction -- limbs are rebuilt from rotations rather than by
# translating endpoints, so any length drift in the output is a bug here and
# not something the solver should have to absorb.


def _place_limb(X, root, mid, tip, l1, l2, dir1, dir2):
    X[mid] = X[root] + l1 * dir1 / np.linalg.norm(dir1)
    X[tip] = X[mid] + l2 * dir2 / np.linalg.norm(dir2)


def _limb_lengths(base, a, b, c):
    return (np.linalg.norm(base[b] - base[a]), np.linalg.norm(base[c] - base[b]))


def motion_squat(base, i, n):
    """Sagittal squat: knees and hips flex, trunk pitches forward."""
    X = base.copy()
    phase = 0.5 * (1 - np.cos(2 * np.pi * i / n))    # 0 -> 1 -> 0
    knee = np.radians(80.0 * phase)                  # flexion at the knee
    hip = np.radians(70.0 * phase)

    for hip_i, knee_i, ankle_i, heel_i, foot_i in (
            (pose3d.L_HIP, pose3d.L_KNEE, pose3d.L_ANKLE, pose3d.L_HEEL,
             pose3d.L_FOOT),
            (pose3d.R_HIP, pose3d.R_KNEE, pose3d.R_ANKLE, pose3d.R_HEEL,
             pose3d.R_FOOT)):
        l_thigh, l_shank = _limb_lengths(base, hip_i, knee_i, ankle_i)
        # Thigh swings forward (+z), shank swings back relative to the thigh.
        thigh_dir = np.array([0.0, np.cos(hip), np.sin(hip)])
        shank_dir = np.array([0.0, np.cos(hip - knee), np.sin(hip - knee)])
        _place_limb(X, hip_i, knee_i, ankle_i, l_thigh, l_shank,
                    thigh_dir, shank_dir)
        # Foot stays flat on the floor.
        X[heel_i] = X[ankle_i] + FOOT_OFFSETS[heel_i]
        X[foot_i] = X[ankle_i] + FOOT_OFFSETS[foot_i]

    # Trunk counterbalances forward. Rotate the upper body about the hips.
    R = rot_x(-np.degrees(hip) * 0.55)
    for j in (pose3d.L_SHOULDER, pose3d.R_SHOULDER, pose3d.NOSE,
              pose3d.L_ELBOW, pose3d.R_ELBOW, pose3d.L_WRIST, pose3d.R_WRIST,
              pose3d.L_INDEX, pose3d.R_INDEX):
        X[j] = R @ base[j]
    return X


def motion_reach(base, i, n):
    """
    Right arm reaches straight at the camera and back.

    This is the worst case for a single camera: the arm projects to almost the
    same handful of pixels whether it is extended toward the lens or folded
    back, so 2D reprojection barely distinguishes them. Nearly all of the
    remaining error in any monocular system lives in motions like this.
    """
    X = base.copy()
    phase = np.sin(2 * np.pi * i / n)
    l_upper, l_fore = _limb_lengths(base, pose3d.R_SHOULDER, pose3d.R_ELBOW,
                                    pose3d.R_WRIST)
    l_hand = np.linalg.norm(base[pose3d.R_INDEX] - base[pose3d.R_WRIST])

    # Upper arm sweeps from hanging down to pointing at the camera.
    a = np.radians(90.0 * (0.5 + 0.5 * phase))
    upper = np.array([0.15, np.cos(a), np.sin(a)])
    # Elbow flexes; forearm stays in the sagittal-ish plane of the upper arm.
    flex = np.radians(30.0 + 60.0 * (0.5 - 0.5 * phase))
    fore = rot_x(-np.degrees(flex)) @ upper
    _place_limb(X, pose3d.R_SHOULDER, pose3d.R_ELBOW, pose3d.R_WRIST,
                l_upper, l_fore, upper, fore)
    X[pose3d.R_INDEX] = X[pose3d.R_WRIST] + l_hand * fore / np.linalg.norm(fore)
    return X


def motion_rotate(base, i, n):
    """Arms swinging while the subject turns (yaw comes from the Scene)."""
    X = base.copy()
    swing = np.radians(20.0 * np.sin(2 * np.pi * i / n))
    # Arms swing out of phase, as in walking.
    for sh, el, wr, ix, sign in (
            (pose3d.L_SHOULDER, pose3d.L_ELBOW, pose3d.L_WRIST,
             pose3d.L_INDEX, 1.0),
            (pose3d.R_SHOULDER, pose3d.R_ELBOW, pose3d.R_WRIST,
             pose3d.R_INDEX, -1.0)):
        l_upper, l_fore = _limb_lengths(base, sh, el, wr)
        l_hand = np.linalg.norm(base[ix] - base[wr])
        a = sign * swing
        upper = np.array([0.25 * np.sign(base[sh][0]), np.cos(a), np.sin(a)])
        fore = rot_x(-25.0) @ upper
        _place_limb(X, sh, el, wr, l_upper, l_fore, upper, fore)
        X[ix] = X[wr] + l_hand * fore / np.linalg.norm(fore)
    return X


MOTIONS = {
    "squat": motion_squat,
    "reach": motion_reach,
    "rotate": motion_rotate,
}


# --------------------------------------------------------------------------
# Camera and observation model
# --------------------------------------------------------------------------

class Scene:
    """
    Generates a sequence of (uv, vis, world_landmarks) observations plus the
    camera-frame ground truth, for one motion.
    """

    def __init__(self, motion="squat", n_frames=150, width=1280, height=720,
                 focal_scale=0.87, camera_pitch_deg=12.0, distance=2.6,
                 yaw_amplitude=25.0, occlusion=True, seed=0):
        self.rng = np.random.default_rng(seed)
        self.n_frames = n_frames
        self.width, self.height = width, height
        self.motion = MOTIONS[motion]
        self.camera_pitch = camera_pitch_deg
        self.distance = distance
        self.yaw_amplitude = yaw_amplitude
        self.occlusion = occlusion

        f = focal_scale * width
        self.K = np.array([[f, 0, width / 2.0],
                           [0, f, height / 2.0],
                           [0, 0, 1.0]])
        self.base = build_body()
        self.true_lengths = symmetrize(bone_lengths_of(self.base))

        # World -> camera. The camera sits below the subject looking up, so
        # world "up" is not camera "up".
        self.R_wc = rot_x(self.camera_pitch)
        self._occ_state = np.ones(N_JOINTS)

    @property
    def gravity_up_cam(self):
        """World up (0,-1,0) expressed in camera coordinates."""
        return self.R_wc @ np.array([0.0, -1.0, 0.0])

    def truth(self, i):
        """Ground-truth joints in the camera frame, meters."""
        X_body = self.motion(self.base, i, self.n_frames)
        yaw = self.yaw_amplitude * np.sin(2 * np.pi * i / self.n_frames + 0.7)
        X_world = (rot_y(yaw) @ X_body.T).T
        X_cam = (self.R_wc @ X_world.T).T
        return X_cam + np.array([0.05, -0.10, self.distance])

    def _visibility(self, i):
        vis = np.full(N_JOINTS, 0.95)
        if not self.occlusion:
            return vis

        # Occlusions come in runs: a joint stays hidden for a stretch of
        # frames. Sampling independently per frame would let a median filter
        # trivially recover, which is not how real dropouts behave.
        #
        # The entry and exit rates are deliberately asymmetric. Toggling with
        # one shared probability drives every joint to a 50% duty cycle, which
        # would mean the shoulders of a subject standing squarely in frame are
        # hidden half the time. That is not a hard benchmark, it is a broken
        # one: it starves calibration, and it gates so many angles out of the
        # comparison that what survives is mostly noise.
        p_out = OCCLUSION_EXIT
        p_in = OCCLUSION_DUTY * p_out / (1.0 - OCCLUSION_DUTY)
        r = self.rng.random(N_JOINTS)
        hidden = self._occ_state < 0.5
        self._occ_state = np.where(
            np.where(hidden, r < p_out, r < p_in),
            1.0 - self._occ_state, self._occ_state)

        vis = np.where(self._occ_state > 0.5, vis,
                       self.rng.uniform(0.05, 0.35, N_JOINTS))
        # Hands and feet are the least reliable landmarks even unoccluded.
        for j in (pose3d.L_INDEX, pose3d.R_INDEX, pose3d.L_FOOT, pose3d.R_FOOT,
                  pose3d.L_HEEL, pose3d.R_HEEL):
            vis[j] = min(vis[j], 0.8)
        return vis

    def observe(self, i):
        """Returns (uv, vis, world_landmarks, X_true)."""
        X_true = self.truth(i)
        uv = _project(X_true, self.K[0, 0], self.K[1, 1],
                      self.K[0, 2], self.K[1, 2])
        uv = uv + self.rng.normal(0, PIXEL_NOISE, uv.shape)
        outliers = self.rng.random(N_JOINTS) < OUTLIER_RATE
        if outliers.any():
            uv[outliers] += self.rng.normal(0, OUTLIER_PIXELS,
                                            (outliers.sum(), 2))

        vis = self._visibility(i)

        # Fake pose_world_landmarks: hip-centered, camera-aligned, with the
        # depth axis compressed and noisy. Low-visibility joints are worse.
        mid_hip = 0.5 * (X_true[pose3d.L_HIP] + X_true[pose3d.R_HIP])
        world = X_true - mid_hip
        scale = 1.0 + 0.5 * (1.0 - np.clip(vis, 0, 1))[:, None]
        world[:, :2] += self.rng.normal(0, WORLD_XY_NOISE,
                                        world[:, :2].shape) * scale
        world[:, 2] = (world[:, 2] * WORLD_Z_GAIN
                       + self.rng.normal(0, WORLD_Z_NOISE, N_JOINTS)
                       * scale[:, 0])
        return uv, vis, world, X_true

    def frames(self):
        for i in range(self.n_frames):
            yield self.observe(i)


def motion_calib(base, i, n):
    """
    The pose sequence a good calibration asks for.

    Bone lengths are recovered from foreshortening, so a limb has to be seen
    at several angles before its length is separable from its depth. Someone
    standing still gives one measurement repeated sixty times. This sweeps the
    arms through a wide arc and cycles a knee, which is what the on-screen
    prompt is asking for when it says to move around.
    """
    X = base.copy()
    phase = 2 * np.pi * i / n

    for sh, el, wr, ix, off in (
            (pose3d.L_SHOULDER, pose3d.L_ELBOW, pose3d.L_WRIST,
             pose3d.L_INDEX, 0.0),
            (pose3d.R_SHOULDER, pose3d.R_ELBOW, pose3d.R_WRIST,
             pose3d.R_INDEX, np.pi * 0.6)):
        l_upper, l_fore = _limb_lengths(base, sh, el, wr)
        l_hand = np.linalg.norm(base[ix] - base[wr])
        a = np.radians(75.0) * np.sin(phase + off)
        lift = np.radians(50.0) * (0.5 + 0.5 * np.cos(phase + off))
        upper = np.array([0.3 * np.sign(base[sh][0]) + np.sin(lift),
                          np.cos(a), np.sin(a)])
        fore = rot_x(-40.0 - 30.0 * np.sin(phase + off)) @ upper
        _place_limb(X, sh, el, wr, l_upper, l_fore, upper, fore)
        X[ix] = X[wr] + l_hand * fore / np.linalg.norm(fore)

    # One knee cycles through flexion so the thigh and shank are not stuck at
    # a single foreshortening for the whole window.
    knee = np.radians(55.0) * (0.5 + 0.5 * np.sin(phase * 1.3))
    hip_f = knee * 0.35
    l_thigh, l_shank = _limb_lengths(base, pose3d.L_HIP, pose3d.L_KNEE,
                                     pose3d.L_ANKLE)
    thigh_dir = np.array([0.0, np.cos(hip_f), np.sin(hip_f)])
    shank_dir = np.array([0.0, np.cos(hip_f - knee), np.sin(hip_f - knee)])
    _place_limb(X, pose3d.L_HIP, pose3d.L_KNEE, pose3d.L_ANKLE,
                l_thigh, l_shank, thigh_dir, shank_dir)
    X[pose3d.L_HEEL] = X[pose3d.L_ANKLE] + FOOT_OFFSETS[pose3d.L_HEEL]
    X[pose3d.L_FOOT] = X[pose3d.L_ANKLE] + FOOT_OFFSETS[pose3d.L_FOOT]
    return X


MOTIONS["calib"] = motion_calib
