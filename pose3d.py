"""
Constrained monocular 3D pose estimation.

MediaPipe's `pose_world_landmarks` are regressed from a learned body prior and
are only loosely consistent frame to frame -- in particular the depth axis is
compressed and noisy. This module uses them only as an initializer, a bone
ruler and a weak branch tiebreaker, and instead solves for the 3D skeleton
that:

  1. reprojects onto the observed 2D keypoints under real camera intrinsics,
  2. has the subject's actual bone lengths,
  3. keeps a rigid torso via cross-brace constraints,
  4. bends its knees the way a knee can bend,
  5. moves continuously from where the previous frames were heading.

With metric bone lengths and known intrinsics, absolute depth is observable
through perspective foreshortening, so this recovers scale as well as shape.

What is deliberately NOT here, having been built and measured:

  * A bundle adjustment over bone lengths. Fitting shared lengths to many
    frames of 2D at once is the textbook answer, and it does help when
    calibration happens mid-movement with the limbs foreshortened. Calibrating
    standing -- which is what the prompt asks for -- puts every limb broadside
    to the camera, which is the one orientation the world landmarks measure
    well, and the plain median is then already within a few percent. The
    adjustment cost six seconds and made angles worse.
  * A search over mirrored limb branches. It cannot be made to work, for a
    reason worth stating: the two branches are geometrically near-identical,
    which is what makes the ambiguity an ambiguity. Score them on geometry
    alone and the test fires on noise and wrecks limbs that were correct;
    include the temporal term and it can never fire, because that term is
    exactly what is holding the limb in place. Anatomy and continuity are the
    only things that separate the branches, so they are applied continuously,
    as constraints, rather than as an after-the-fact search.

See test_pose3d.py --ablate for the numbers behind both.

Coordinate frame: OpenCV camera convention. +x right, +y down, +z forward
(away from the camera, into the scene). Units are meters.
"""

import math
import time

import numpy as np
from scipy.optimize import least_squares

# --------------------------------------------------------------------------
# Skeleton definition
# --------------------------------------------------------------------------

# BlazePose indices we actually solve for. Solving all 33 wastes time on face
# points that contribute nothing to joint angles.
BLAZE_IDS = [
    0,              # nose
    11, 12,         # shoulders
    13, 14,         # elbows
    15, 16,         # wrists
    19, 20,         # index fingers (needed for wrist angle)
    23, 24,         # hips
    25, 26,         # knees
    27, 28,         # ankles
    29, 30,         # heels (brace the ankle against foot-index jitter)
    31, 32,         # foot index (needed for ankle angle)
]
N_JOINTS = len(BLAZE_IDS)
BLAZE_TO_LOCAL = {b: i for i, b in enumerate(BLAZE_IDS)}

# Local indices
NOSE = BLAZE_TO_LOCAL[0]
L_SHOULDER, R_SHOULDER = BLAZE_TO_LOCAL[11], BLAZE_TO_LOCAL[12]
L_ELBOW, R_ELBOW = BLAZE_TO_LOCAL[13], BLAZE_TO_LOCAL[14]
L_WRIST, R_WRIST = BLAZE_TO_LOCAL[15], BLAZE_TO_LOCAL[16]
L_INDEX, R_INDEX = BLAZE_TO_LOCAL[19], BLAZE_TO_LOCAL[20]
L_HIP, R_HIP = BLAZE_TO_LOCAL[23], BLAZE_TO_LOCAL[24]
L_KNEE, R_KNEE = BLAZE_TO_LOCAL[25], BLAZE_TO_LOCAL[26]
L_ANKLE, R_ANKLE = BLAZE_TO_LOCAL[27], BLAZE_TO_LOCAL[28]
L_HEEL, R_HEEL = BLAZE_TO_LOCAL[29], BLAZE_TO_LOCAL[30]
L_FOOT, R_FOOT = BLAZE_TO_LOCAL[31], BLAZE_TO_LOCAL[32]

# Rigid segments. The torso cross-braces are not anatomical bones, but the
# torso is close enough to rigid that constraining the diagonals is what stops
# the shoulder girdle from folding through itself in ambiguous views. The foot
# triangle (ankle-heel-toe) does the same job for the ankle, which otherwise
# pivots on a single noisy toe landmark.
BONES = [
    (L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
    (L_SHOULDER, R_HIP), (R_SHOULDER, L_HIP),      # cross-braces
    (NOSE, L_SHOULDER), (NOSE, R_SHOULDER),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST), (L_WRIST, L_INDEX),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST), (R_WRIST, R_INDEX),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE), (L_ANKLE, L_FOOT),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE), (R_ANKLE, R_FOOT),
    (L_ANKLE, L_HEEL), (L_HEEL, L_FOOT),
    (R_ANKLE, R_HEEL), (R_HEEL, R_FOOT),
]
N_BONES = len(BONES)

# Bones whose lengths should be averaged across the body midline.
SYMMETRIC_PAIRS = [
    (2, 3), (4, 5), (6, 7),
    (8, 11), (9, 12), (10, 13),
    (14, 17), (15, 18), (16, 19),
    (20, 22), (21, 23),
]


def _length_groups():
    """Bone indices grouped by shared (left/right symmetric) length."""
    parent = list(range(N_BONES))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in SYMMETRIC_PAIRS:
        ra, rb = find(a), find(b)
        parent[max(ra, rb)] = min(ra, rb)

    groups = {}
    for k in range(N_BONES):
        groups.setdefault(find(k), []).append(k)
    return [groups[r] for r in sorted(groups)]


LENGTH_GROUPS = _length_groups()

# (label, proximal, vertex, distal) in local indices.
JOINT_SPECS = [
    ("L shoulder", L_ELBOW, L_SHOULDER, L_HIP),
    ("R shoulder", R_ELBOW, R_SHOULDER, R_HIP),
    ("L elbow", L_SHOULDER, L_ELBOW, L_WRIST),
    ("R elbow", R_SHOULDER, R_ELBOW, R_WRIST),
    ("L wrist", L_ELBOW, L_WRIST, L_INDEX),
    ("R wrist", R_ELBOW, R_WRIST, R_INDEX),
    ("L hip", L_SHOULDER, L_HIP, L_KNEE),
    ("R hip", R_SHOULDER, R_HIP, R_KNEE),
    ("L knee", L_HIP, L_KNEE, L_ANKLE),
    ("R knee", R_HIP, R_KNEE, R_ANKLE),
    ("L ankle", L_KNEE, L_ANKLE, L_FOOT),
    ("R ankle", R_KNEE, R_ANKLE, R_FOOT),
]
DERIVED_LABELS = ["trunk lean", "shoulder tilt", "hip tilt"]
ALL_LABELS = [s[0] for s in JOINT_SPECS] + DERIVED_LABELS

# Hinge joints: (proximal root, hinge, distal, sign).
# sign=+1 means the distal segment must fall on the *posterior* side of the
# proximal segment (knees), -1 means anterior (elbows).
KNEE_HINGES = [
    (L_HIP, L_KNEE, L_ANKLE, +1.0),
    (R_HIP, R_KNEE, R_ANKLE, +1.0),
]
ELBOW_HINGES = [
    (L_SHOULDER, L_ELBOW, L_WRIST, -1.0),
    (R_SHOULDER, R_ELBOW, R_WRIST, -1.0),
]


# --------------------------------------------------------------------------
# Residual weighting (all residuals are normalized to unitless sigmas)
# --------------------------------------------------------------------------

SIGMA_PIXEL = 2.5       # expected 2D keypoint error, pixels
SIGMA_BONE = 0.006      # how hard bone lengths are enforced, meters
SIGMA_TEMPORAL = 0.045  # allowed motion per frame before it costs, meters
SIGMA_PRIOR = 0.045     # trust in the MediaPipe shape as a branch tiebreaker
SIGMA_HINGE = 0.010     # how hard anatomical bend direction is enforced, m
HINGE_DEADZONE = 0.010  # meters of wrong-way bend tolerated before it costs

# Constant-velocity extrapolation. Damped because a keypoint glitch on one
# frame should not be projected forward at full strength into the next.
VELOCITY_DAMPING = 0.6

# Fingers, heels and toes hang off segments a few dozen pixels long, so their
# depth is the least observable thing in the skeleton -- the viewing ray runs
# almost parallel to the segment, and a pixel of keypoint noise slides the
# joint centimetres along it.
#
# The obvious response is to lean harder on continuity and the learned prior
# for exactly these joints. Measured, it does not pay: MediaPipe's depth is
# biased in the same direction the geometry is weak, so a stiffer pull toward
# it buys smoothness and keeps the bias. Left at 1.0, and kept as a knob
# because it is the natural thing to reach for and the ablation is the
# argument against reaching for it.
STIFF_JOINTS = ("L_INDEX", "R_INDEX", "L_HEEL", "R_HEEL", "L_FOOT", "R_FOOT")
STIFFNESS = 1.0


# --------------------------------------------------------------------------
# Small geometry helpers
# --------------------------------------------------------------------------

def _unit(v, eps=1e-9):
    n = np.linalg.norm(v)
    return v / n if n > eps else np.zeros_like(v)


def _project(X, fx, fy, cx, cy):
    """(N,3) camera-frame points -> (N,2) pixels. Assumes z > 0."""
    z = np.maximum(X[:, 2], 1e-4)
    u = fx * X[:, 0] / z + cx
    v = fy * X[:, 1] / z + cy
    return np.stack([u, v], axis=1)


def enforce_bone_lengths(X, lengths, weights=None, iterations=6):
    """
    Project a point cloud back onto the bone-length constraints.

    Gauss-Seidel over the bones, as in position-based dynamics. Intended for
    use after temporal filtering, which smooths each coordinate independently
    and so has no idea the skeleton exists.

    Worth being clear about what this buys, since it is not accuracy: it costs
    about 0.1 degrees of mean angle error, which is inside the noise, and in
    exchange the emitted skeleton holds its bone lengths to well under a
    millimetre instead of breathing by three. Angles barely notice; anything
    that treats the output as a body does.
    """
    X = X.copy()
    w = np.ones(len(X)) if weights is None else np.clip(weights, 0.05, 1.0)

    for _ in range(iterations):
        for k, (a, b) in enumerate(BONES):
            d = X[a] - X[b]
            dist = np.linalg.norm(d)
            if dist < 1e-9:
                continue
            corr = (dist - lengths[k]) / dist * d
            # Split the correction so a confident joint moves less than an
            # uncertain one.
            wa, wb = w[a], w[b]
            total = wa + wb
            X[a] -= corr * (wb / total)
            X[b] += corr * (wa / total)
    return X


def similarity_align(P, X, w):
    """
    Weighted Procrustes with uniform scale: find s, R, t minimizing
    ||s*R*P + t - X||. Used to bring the MediaPipe shape into the solver's
    frame so it can act as a prior without imposing its (wrong) scale.
    """
    ww = np.maximum(w, 1e-6)
    ws = ww.sum()
    mu_p = (P * ww[:, None]).sum(0) / ws
    mu_x = (X * ww[:, None]).sum(0) / ws
    Pc = P - mu_p
    Xc = X - mu_x

    H = (Pc * ww[:, None]).T @ Xc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    var_p = (ww[:, None] * Pc ** 2).sum()
    s = float(S @ D.diagonal()) / var_p if var_p > 1e-12 else 1.0
    s = float(np.clip(s, 0.2, 5.0))

    return (s * (R @ P.T).T) + (mu_x - s * (R @ mu_p))


# --------------------------------------------------------------------------
# Anatomical hinge constraints
# --------------------------------------------------------------------------
#
# A knee is a hinge: the shank swings one way only. That fact is the strongest
# single-camera cue for resolving the forward/backward depth ambiguity,
# because the mirrored solution -- the one that reprojects just as well -- is
# precisely the one that bends the joint backwards. Reprojection error cannot
# see the difference; this can.
#
# The reference direction is built from the pelvis together with the thigh,
# never from the trunk as a whole. Bending forward at the waist pitches the
# trunk without changing which way a knee bends, so a trunk-relative
# constraint would fire on every squat and every hip hinge.
#
# The same construction is available for elbows and is switched off by
# default, on evidence: the humerus rotates freely, which swings the elbow's
# hinge axis away from the shoulder line the constraint assumes, and the
# constraint then fires on perfectly good poses. It costs 2.7 degrees of mean
# error. Anything overhead or behind the back breaks it.

def _hinge_terms(X, hinges, lateral):
    """
    Signed wrong-way bend, in meters, for each hinge. Positive means the joint
    is bent the way it cannot bend.

    Magnitude scales with how far out of plane the segment is, so a nearly
    straight limb -- where the bend direction is undefined and the 2D evidence
    is weak -- contributes nothing, and the constraint stays silent exactly
    where it would otherwise be guessing.

    `hinges` is an index bundle from `hinge_arrays`. This sits inside the
    optimizer's inner loop, so it is written over all hinges at once; the same
    arithmetic spelled as a Python loop over three-element vectors costs more
    in numpy call overhead than the rest of the Jacobian put together.
    """
    roots, mids, tips, signs = hinges
    lat = _unit(lateral)
    if not np.any(lat):
        return np.zeros(len(roots))

    prox = X[mids] - X[roots]
    n_prox = np.linalg.norm(prox, axis=1)
    t = prox / np.maximum(n_prox, 1e-9)[:, None]

    # Hinge axis: perpendicular to both the body's lateral axis and the
    # proximal segment. This is the direction the distal segment may swing.
    allowed = np.cross(lat, t) * signs[:, None]
    n_allowed = np.linalg.norm(allowed, axis=1)
    allowed = allowed / np.maximum(n_allowed, 1e-9)[:, None]

    dist = X[tips] - X[mids]
    perp = dist - (dist * t).sum(1)[:, None] * t
    out = -(perp * allowed).sum(1)
    return np.where((n_prox < 1e-6) | (n_allowed < 1e-6), 0.0, out)


def hinge_arrays(hinges):
    """Pack a list of hinge tuples into index arrays for `_hinge_terms`."""
    if not hinges:
        z = np.zeros(0, dtype=int)
        return z, z, z, np.zeros(0)
    return (np.array([h[0] for h in hinges]),
            np.array([h[1] for h in hinges]),
            np.array([h[2] for h in hinges]),
            np.array([h[3] for h in hinges], dtype=float))


def _lateral_axes(X):
    """(pelvis lateral, shoulder lateral), both pointing to the subject's left."""
    return X[L_HIP] - X[R_HIP], X[L_SHOULDER] - X[R_SHOULDER]


# --------------------------------------------------------------------------
# Bone length calibration
# --------------------------------------------------------------------------

class BoneCalibrator:
    """
    Accumulates bone lengths from MediaPipe world landmarks, then medians.

    Simple, and good enough for a reason worth knowing when deciding how to
    calibrate: the world landmarks' error is concentrated in depth, so a
    segment lying broadside to the camera is measured well and one pointing
    at the camera is measured short. Standing upright puts nearly every
    segment broadside, which is why the calibration prompt asks for it.
    Calibrating mid-movement, with the limbs foreshortened, is what makes
    these numbers bad -- and no estimator downstream fully recovers from it.
    """

    def __init__(self, target_frames=90):
        self.target_frames = target_frames
        self.samples = []
        self.frames = []

    @property
    def count(self):
        return len(self.samples)

    @property
    def done(self):
        return len(self.samples) >= self.target_frames

    def add(self, world_pts, vis, vis_thresh=0.5, uv=None):
        """world_pts: (N_JOINTS, 3) metric points from pose_world_landmarks."""
        lengths = np.full(N_BONES, np.nan)
        for k, (a, b) in enumerate(BONES):
            if min(vis[a], vis[b]) < vis_thresh:
                continue
            lengths[k] = np.linalg.norm(world_pts[a] - world_pts[b])
        self.samples.append(lengths)
        if uv is not None:
            self.frames.append((np.asarray(uv, dtype=np.float64).copy(),
                                np.asarray(vis, dtype=np.float64).copy(),
                                np.asarray(world_pts, dtype=np.float64).copy()))

    def result(self):
        """Median bone lengths with left/right symmetry enforced."""
        arr = np.array(self.samples)
        with np.errstate(all="ignore"):
            lengths = np.nanmedian(arr, axis=0)

        # Any bone never seen falls back to the global median so the solver
        # still has something to work with.
        fallback = np.nanmedian(lengths)
        lengths = np.where(np.isnan(lengths), fallback, lengths)

        for group in LENGTH_GROUPS:
            lengths[group] = lengths[group].mean()
        return lengths

    def estimate_up(self, K, lengths, n_frames=60, min_frames=8):
        """
        World "up" in camera coordinates, from the subject's own trunk axis.

        A laptop webcam sits below the subject and points upward, often by
        10-15 degrees. Every angle measured against a camera axis -- trunk
        lean, shoulder tilt, hip tilt -- then carries that tilt as a straight
        bias. The subject standing during calibration is a plumb line, so the
        median trunk direction recovers it with no extra hardware.

        The trunk axis is read off *solved* poses rather than off the world
        landmarks, and that distinction is most of the value here. The trunk
        runs mostly vertical with a small depth component, and the depth
        component is exactly what the tilt is made of -- so measuring it in
        landmarks whose depth axis is compressed recovers only about two
        thirds of the tilt and quietly leaves the rest as bias.

        Roughly three quarters of the tilt comes back, and the shortfall is a
        bias rather than noise: the shape prior pulls the solved trunk toward
        MediaPipe's compressed depth, which is the very axis the tilt lives
        on. Averaging more frames does not remove that part. Measuring the
        camera's pitch once with a phone level and passing it in beats this,
        and `skeleton.py --camera-pitch` exists for exactly that; this is the
        default that needs nothing from the user.

        Returns None when there is not enough evidence, in which case the
        caller should fall back to the camera's own vertical.
        """
        if len(self.frames) < min_frames:
            return None

        # A contiguous run from the end of the window: the subject has settled
        # by then, and consecutive frames let the solver warm-start.
        window = self.frames[-n_frames:]
        solver = PoseSolver(K, lengths)
        axes = []
        for uv, vis, world in window:
            core = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
            X, _ = solver.solve(uv, vis, world)
            if min(vis[j] for j in core) < 0.5:
                continue
            axis = (0.5 * (X[L_SHOULDER] + X[R_SHOULDER])
                    - 0.5 * (X[L_HIP] + X[R_HIP]))
            if np.linalg.norm(axis) > 1e-3:
                axes.append(_unit(axis))
        if len(axes) < min_frames:
            return None

        # Two passes, so a few frames of the subject leaning over to adjust
        # something do not drag the estimate with them.
        axes = np.array(axes)
        up = _unit(np.median(axes, axis=0))
        keep = axes @ up > math.cos(math.radians(15.0))
        if keep.sum() >= min_frames:
            up = _unit(np.median(axes[keep], axis=0))

        # Only trust a plausible correction. A wild estimate means the subject
        # was not standing, and the camera axis is the safer default.
        if np.degrees(math.acos(np.clip(-up[1], -1.0, 1.0))) > 25.0:
            return None
        return up


# --------------------------------------------------------------------------
# One Euro filter
# --------------------------------------------------------------------------

class OneEuroFilter:
    """
    Jitter/lag tradeoff that adapts to speed: heavy smoothing when still,
    light smoothing when moving fast. Strictly better than a fixed EMA here.
    """

    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.8, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, freq):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def __call__(self, x, t=None):
        x = np.asarray(x, dtype=np.float64)
        t = time.perf_counter() if t is None else t

        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x

        dt = t - self.t_prev
        if dt > 1e-6:
            self.freq = 1.0 / dt
        self.t_prev = t

        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff, self.freq)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        self.dx_prev = dx_hat

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = 1.0 / (1.0 + (1.0 / (2.0 * math.pi * cutoff)) * self.freq)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        return x_hat


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------

class PoseSolver:
    """
    Per-frame constrained fit, warm-started from a constant-velocity
    prediction.

    Geometry (reprojection + bone lengths) fixes metric shape. The hinge
    constraints and the aligned MediaPipe prior pick which of the
    geometrically-valid branches is the anatomically plausible one -- geometry
    alone cannot, because the mirrored pose reprojects just as well. Neither
    half is enough on its own.
    """

    F_SCALE = 3.0

    def __init__(self, K, bone_lengths, max_nfev=40, use_hinges=True,
                 use_elbow_hinges=False):
        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])
        self.cx = float(K[0, 2])
        self.cy = float(K[1, 2])
        self.bone_lengths = np.asarray(bone_lengths, dtype=np.float64)
        self.max_nfev = max_nfev
        self.X_prev = None
        self.X_prev2 = None
        self.bone_a = np.array([b[0] for b in BONES])
        self.bone_b = np.array([b[1] for b in BONES])

        knees = KNEE_HINGES if use_hinges else []
        elbows = ELBOW_HINGES if use_hinges and use_elbow_hinges else []
        self.hinges = list(knees) + list(elbows)
        self.n_hinge = len(self.hinges)
        self._knee_arr = hinge_arrays(knees)
        self._elbow_arr = hinge_arrays(elbows)

        # Coordinates the hinge residuals actually depend on, so the numeric
        # part of the Jacobian stays small.
        joints = set()
        for root, mid, tip, _ in self.hinges:
            joints.update((root, mid, tip))
        if knees:
            joints.update((L_HIP, R_HIP))
        if elbows:
            joints.update((L_SHOULDER, R_SHOULDER))
        self._hinge_cols = np.array(
            sorted(3 * j + c for j in joints for c in range(3)), dtype=int)

        self._stiffness = np.ones(N_JOINTS)
        for name in STIFF_JOINTS:
            self._stiffness[globals()[name]] = STIFFNESS

        # Index arrays so the Jacobian can be filled with slice assignments
        # instead of a Python loop per joint and per bone on every iteration.
        j = np.arange(N_JOINTS)
        self._ju, self._jv = 2 * j, 2 * j + 1
        self._cx_i, self._cy_i, self._cz_i = 3 * j, 3 * j + 1, 3 * j + 2
        self._bone_cols_a = 3 * self.bone_a[:, None] + np.arange(3)[None, :]
        self._bone_cols_b = 3 * self.bone_b[:, None] + np.arange(3)[None, :]
        self._bone_rows = np.repeat(np.arange(N_BONES)[:, None], 3, axis=1)
        self._diag = np.arange(3 * N_JOINTS)

    # -- hinge block ------------------------------------------------------

    def _hinge_raw(self, X):
        """Signed wrong-way bend per hinge, in meters, before the deadzone."""
        lat_hip, lat_sh = _lateral_axes(X)
        vals = []
        if len(self._knee_arr[0]):
            vals.append(_hinge_terms(X, self._knee_arr, lat_hip))
        if len(self._elbow_arr[0]):
            vals.append(_hinge_terms(X, self._elbow_arr, lat_sh))
        return np.concatenate(vals)

    def _hinge_residual(self, X):
        if not self.hinges:
            return np.zeros(0)
        return np.maximum(0.0, self._hinge_raw(X) - HINGE_DEADZONE) / SIGMA_HINGE

    # -- residuals and analytic Jacobian ---------------------------------

    def _residuals(self, params, uv, w_rep, X_ref, w_temp, prior):
        X = params.reshape(N_JOINTS, 3)

        z = np.maximum(X[:, 2], 1e-4)
        proj = np.stack([self.fx * X[:, 0] / z + self.cx,
                         self.fy * X[:, 1] / z + self.cy], axis=1)
        r_rep = ((proj - uv) * w_rep[:, None] / SIGMA_PIXEL).ravel()

        d = X[self.bone_a] - X[self.bone_b]
        lengths = np.linalg.norm(d, axis=1)
        r_bone = (lengths - self.bone_lengths) / SIGMA_BONE

        if X_ref is None:
            r_temp = np.zeros(3 * N_JOINTS)
        else:
            r_temp = ((X - X_ref) * w_temp / SIGMA_TEMPORAL).ravel()

        if prior is None:
            r_prior = np.zeros(3 * N_JOINTS)
        else:
            r_prior = ((X - prior) * w_temp / SIGMA_PRIOR).ravel()

        return np.concatenate([r_rep, r_bone, r_temp, r_prior,
                               self._hinge_residual(X)])

    def _jacobian(self, params, uv, w_rep, X_ref, w_temp, prior):
        X = params.reshape(N_JOINTS, 3)
        n_var = 3 * N_JOINTS
        n_res = 2 * N_JOINTS + N_BONES + 2 * n_var + self.n_hinge
        J = np.zeros((n_res, n_var))

        # Reprojection: d(u,v)/dX for a pinhole camera.
        z = np.maximum(X[:, 2], 1e-4)
        k = w_rep / SIGMA_PIXEL
        J[self._ju, self._cx_i] = k * self.fx / z
        J[self._ju, self._cz_i] = -k * self.fx * X[:, 0] / z ** 2
        J[self._jv, self._cy_i] = k * self.fy / z
        J[self._jv, self._cz_i] = -k * self.fy * X[:, 1] / z ** 2

        # Bone length: d|d|/dX = +/- unit vector.
        row0 = 2 * N_JOINTS
        d = X[self.bone_a] - X[self.bone_b]
        norms = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        u = d / norms[:, None] / SIGMA_BONE
        J[row0 + self._bone_rows, self._bone_cols_a] = u
        J[row0 + self._bone_rows, self._bone_cols_b] = -u

        # Temporal and prior are both diagonal.
        row0 += N_BONES
        if X_ref is not None:
            J[row0 + self._diag, self._diag] = (w_temp / SIGMA_TEMPORAL).ravel()
        row0 += n_var
        if prior is not None:
            J[row0 + self._diag, self._diag] = (w_temp / SIGMA_PRIOR).ravel()
        row0 += n_var

        # Hinge terms are cheap but awkward to differentiate by hand, and they
        # touch only a handful of joints, so finite-difference just those
        # columns. The residual is clamped at zero, so when every joint bends
        # the right way these rows are not merely small but exactly flat and
        # the block can be skipped -- which is the usual case, and keeps the
        # constraint from costing anything while it is not binding. The margin
        # covers the difference step itself.
        if self.n_hinge:
            raw = self._hinge_raw(X)
            if raw.max() > HINGE_DEADZONE - 1e-4:
                f0 = np.maximum(0.0, raw - HINGE_DEADZONE) / SIGMA_HINGE
                eps = 1e-6
                flat = params.copy()
                for c in self._hinge_cols:
                    flat[c] += eps
                    f1 = self._hinge_residual(flat.reshape(N_JOINTS, 3))
                    flat[c] -= eps
                    J[row0:row0 + self.n_hinge, c] = (f1 - f0) / eps

        return J

    # -- initialization ---------------------------------------------------

    def _initial_guess(self, uv, w, world_pts, z_sign):
        """
        Place the MediaPipe shape at the depth implied by bone foreshortening.

        For a bone of known metric length L spanning l pixels near the optical
        axis, z is about f*L/l. Median over well-observed bones is robust to
        any single bad keypoint.
        """
        P = world_pts.copy()
        P[:, 2] *= z_sign

        z_estimates = []
        for k, (a, b) in enumerate(BONES):
            if w[a] < 0.3 or w[b] < 0.3:
                continue
            pix = np.linalg.norm(uv[a] - uv[b])
            if pix < 8.0:
                continue
            z_estimates.append(self.fx * self.bone_lengths[k] / pix)
        t_z = float(np.median(z_estimates)) if z_estimates else 2.5
        t_z = float(np.clip(t_z, 0.4, 8.0))

        ww = np.maximum(w, 1e-3)
        cen_uv = (uv * ww[:, None]).sum(0) / ww.sum()
        cen_P = (P * ww[:, None]).sum(0) / ww.sum()
        t_x = (cen_uv[0] - self.cx) / self.fx * t_z - cen_P[0]
        t_y = (cen_uv[1] - self.cy) / self.fy * t_z - cen_P[1]

        X0 = P + np.array([t_x, t_y, t_z - cen_P[2]])
        X0[:, 2] = np.maximum(X0[:, 2], 0.2)
        return X0

    # -- solve ------------------------------------------------------------

    def _run(self, X0, uv, w, X_ref, w_temp, world_pts, max_nfev):
        prior = similarity_align(world_pts, X0, w)
        return least_squares(
            self._residuals, X0.ravel(), jac=self._jacobian,
            args=(uv, w, X_ref, w_temp, prior),
            method="trf", loss="soft_l1", f_scale=self.F_SCALE,
            max_nfev=max_nfev, xtol=1e-8, ftol=1e-8, gtol=1e-10, verbose=0)

    def solve(self, uv, vis, world_pts, vis_thresh=0.5):
        """
        uv:        (N_JOINTS, 2) observed pixel coordinates
        vis:       (N_JOINTS,) visibility in [0, 1]
        world_pts: (N_JOINTS, 3) MediaPipe world landmarks

        Returns (X, info) with X as (N_JOINTS, 3) in meters, camera frame.
        """
        uv = np.asarray(uv, dtype=np.float64)
        w = np.clip(vis, 0.0, 1.0)
        w = np.where(w < vis_thresh, 0.05 * w, w)  # downweight, never drop
        w_temp = np.repeat(
            (np.clip(w, 0.2, 1.0) * self._stiffness)[:, None], 3, axis=1)

        # Constant-velocity prediction. Anchoring the temporal term to where
        # the body is heading rather than to where it was lets the term stay
        # strong -- and so keep suppressing depth jitter -- without dragging
        # behind fast motion the way a zero-velocity prior does.
        if self.X_prev is not None and self.X_prev2 is not None:
            X_ref = self.X_prev + VELOCITY_DAMPING * (self.X_prev - self.X_prev2)
        else:
            X_ref = self.X_prev

        if self.X_prev is not None:
            starts = [X_ref]
        else:
            # Cold start only: try both depth branches and keep the better.
            # Once tracking, continuity picks the branch, which is the only
            # thing that reliably can.
            starts = [self._initial_guess(uv, w, world_pts, +1.0),
                      self._initial_guess(uv, w, world_pts, -1.0)]

        best = None
        for X0 in starts:
            res = self._run(X0, uv, w, X_ref, w_temp, world_pts, self.max_nfev)
            if best is None or res.cost < best.cost:
                best = res

        X = best.x.reshape(N_JOINTS, 3)
        self.X_prev2 = self.X_prev
        self.X_prev = X.copy()

        proj = _project(X, self.fx, self.fy, self.cx, self.cy)
        px_err = np.linalg.norm(proj - uv, axis=1)
        bone_d = X[self.bone_a] - X[self.bone_b]
        bone_err = np.abs(np.linalg.norm(bone_d, axis=1) - self.bone_lengths)

        info = {
            "cost": float(best.cost),
            "reproj_px": float(np.average(px_err, weights=np.maximum(w, 1e-3))),
            "bone_err_mm": float(bone_err.mean() * 1000.0),
            "depth_m": float(np.median(X[:, 2])),
            "nfev": int(best.nfev),
        }
        return X, info

    def reset(self):
        self.X_prev = None
        self.X_prev2 = None


# --------------------------------------------------------------------------
# Angles
# --------------------------------------------------------------------------

def _angle(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def compute_angles(X, vis, vis_thresh=0.35, up=None):
    """
    Interior joint angles in degrees from solved 3D points.

    `up` is world vertical expressed in camera coordinates. The trunk and tilt
    angles are close to meaningless without it: measured against the camera's
    own axes they inherit the camera's tilt, which for a laptop webcam is a
    10-15 degree bias sitting on top of every reading. Defaults to the
    camera's vertical, which is right only for a perfectly level camera.

    A caution about the numbers this returns: an angle near 180 degrees is
    measured from three nearly collinear points, where position noise can only
    bend the result away from straight and never past it. Those readings --
    typically wrist and ankle, whose short terminal segments are also the
    least observable in depth -- are biased low, and no amount of averaging
    removes it. Treat them as much softer than the knee and elbow figures.
    """
    up = np.array([0.0, -1.0, 0.0]) if up is None else _unit(np.asarray(up))

    angles = {}
    for label, a, b, c in JOINT_SPECS:
        if min(vis[a], vis[b], vis[c]) < vis_thresh:
            angles[label] = None
        else:
            angles[label] = _angle(X[a] - X[b], X[c] - X[b])

    core = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
    if min(vis[i] for i in core) >= vis_thresh:
        mid_sh = 0.5 * (X[L_SHOULDER] + X[R_SHOULDER])
        mid_hip = 0.5 * (X[L_HIP] + X[R_HIP])
        angles["trunk lean"] = _angle(mid_sh - mid_hip, up)

        # Tilt is measured in the plane perpendicular to gravity, so a
        # shoulder line pointing away from the camera does not read as tilted.
        for label, a, b in (("shoulder tilt", L_SHOULDER, R_SHOULDER),
                            ("hip tilt", L_HIP, R_HIP)):
            axis = X[a] - X[b]
            horiz = axis - np.dot(axis, up) * up
            if np.linalg.norm(horiz) < 1e-6:
                angles[label] = None
                continue
            deg = _angle(axis, horiz)
            angles[label] = None if deg is None else math.copysign(
                deg, -np.dot(axis, up))
    else:
        for label in DERIVED_LABELS:
            angles[label] = None

    return angles


def up_from_pitch(degrees):
    """
    World vertical in camera coordinates for a camera pitched up by `degrees`.

    Positive means the lens is tilted upward, which is where a laptop webcam
    below eye level always ends up.
    """
    t = math.radians(degrees)
    return np.array([0.0, -math.cos(t), -math.sin(t)])


def intrinsics_from_fov(width, height, fov_deg):
    """
    Camera matrix from a horizontal field of view, assuming square pixels and
    the principal point at the image centre.

    How much the field of view needs to be right depends entirely on what you
    are reading off the screen, and the two answers are very different:

      Joint angles barely move. Across a focal length from 0.7x to 1.5x of
      truth -- a field of view anywhere between about 45 and 85 degrees --
      mean angle error changes by well under a degree, because the bone-length
      and reprojection constraints between them pin the *shape* of the
      skeleton almost independently of how the rays are scaled.

      Distance moves a lot, close to proportionally. A focal length 30% off
      puts the reported distance off by roughly 0.7 m at 2.6 m.

    So the default is fine for angles and approximate for distance, and that
    is the trade this program makes rather than asking anyone to print a
    checkerboard. Pass --fov if you know your camera's real figure.
    """
    f = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0],
                     [0.0, f, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def default_intrinsics(width, height, fov_deg=60.0):
    """Backwards-compatible wrapper around `intrinsics_from_fov`."""
    return intrinsics_from_fov(width, height, fov_deg)
