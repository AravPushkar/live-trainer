"""
Measure the pipeline against synthetic ground truth.

This runs the *whole* pipeline, calibration included, on data from
simulate.py. Nothing is handed the right answer: bone lengths get measured the
same way they are measured live, from the same corrupted world landmarks, and
the camera is tilted the way a real one is. A benchmark that skips those steps
reports the accuracy of a system you do not have.

    python test_pose3d.py                # headline numbers
    python test_pose3d.py --ablate       # what each piece is worth
    python test_pose3d.py --per-joint    # where the error actually lives
"""

import argparse
import time

import numpy as np

import pose3d
from pose3d import (ALL_LABELS, N_JOINTS, BoneCalibrator, OneEuroFilter,
                    PoseSolver, compute_angles, enforce_bone_lengths)
from simulate import Scene

CALIB_FRAMES = 60
CALIB_MOTION = "calib"
MOTIONS = ["squat", "reach", "rotate"]


class Config:
    """Which improvements are switched on."""

    def __init__(self, hinges=True, elbow_hinges=False, velocity=True,
                 gravity=True, constrained_filter=True, stiffness=1.0):
        self.hinges = hinges
        self.elbow_hinges = elbow_hinges
        self.velocity = velocity
        self.gravity = gravity
        self.constrained_filter = constrained_filter
        self.stiffness = stiffness


def angle_errors(a_true, a_est):
    errs = {}
    for label in ALL_LABELS:
        if a_true[label] is None or a_est[label] is None:
            continue
        errs[label] = abs(a_est[label] - a_true[label])
    return errs


def run_motion(motion, cfg, seed=0, n_frames=150, collect=None):
    """Calibrate then track. Returns (solver errors, baseline errors, timings)."""
    scene = Scene(motion=motion, n_frames=n_frames, seed=seed)

    # ---- calibration phase: exactly what skeleton.py does live ----
    # Calibration always happens standing and moving the limbs through a
    # range, whatever is measured afterwards. That is what the on-screen
    # prompt asks for, and both halves of it matter: standing is what the
    # vertical estimate needs (calibrating mid-squat hands it a trunk that is
    # pitched forward and calls the result gravity), and the limb movement is
    # what stops every segment from being measured at a single foreshortening.
    calib = BoneCalibrator(CALIB_FRAMES)
    calib_scene = Scene(motion=CALIB_MOTION, n_frames=CALIB_FRAMES,
                        seed=seed + 100)
    for i in range(CALIB_FRAMES):
        uv, vis, world, _ = calib_scene.observe(i)
        calib.add(world, vis, vis_thresh=0.4, uv=uv)

    t0 = time.perf_counter()
    lengths, calib_info = calib.result(), {"method": "median"}
    up = calib.estimate_up(scene.K, lengths) if cfg.gravity else None
    if up is None:
        up = np.array([0.0, -1.0, 0.0])
    calib_info["seconds"] = time.perf_counter() - t0

    # ---- tracking ----
    pose3d.STIFFNESS = cfg.stiffness
    solver = PoseSolver(scene.K, lengths,
                        use_hinges=cfg.hinges,
                        use_elbow_hinges=cfg.elbow_hinges)
    pose3d.STIFFNESS = 1.0
    if not cfg.velocity:
        pose3d.VELOCITY_DAMPING = 0.0
    euro = OneEuroFilter(min_cutoff=1.0, beta=0.7)

    solved, baseline, times, pos_err = [], [], [], []
    t_sim = 0.0
    for i in range(n_frames):
        uv, vis, world, X_true = scene.observe(i)

        t0 = time.perf_counter()
        X, info = solver.solve(uv, vis, world)
        X = euro(X.ravel(), t=t_sim).reshape(N_JOINTS, 3)
        if cfg.constrained_filter:
            X = enforce_bone_lengths(X, lengths, weights=vis)
        times.append((time.perf_counter() - t0) * 1000.0)
        t_sim += 1.0 / 30.0

        # Ground truth angles use true gravity; the estimate uses whatever the
        # pipeline worked out for itself.
        a_true = compute_angles(X_true, vis, up=scene.gravity_up_cam)
        a_est = compute_angles(X, vis, up=up)
        a_base = compute_angles(world, vis, up=up)

        solved.append(angle_errors(a_true, a_est))
        baseline.append(angle_errors(a_true, a_base))

        c_t = X_true - X_true.mean(0)
        c_s = X - X.mean(0)
        pos_err.append(np.linalg.norm(c_s - c_t, axis=1).mean() * 1000.0)

    pose3d.VELOCITY_DAMPING = 0.6
    if collect is not None:
        collect.update(calib_info=calib_info, lengths=lengths, scene=scene,
                       up=up, pos_err=pos_err)
    return solved, baseline, times


def flatten(dicts):
    return np.array([v for d in dicts for v in d.values()])


def summarize(errs):
    a = flatten(errs)
    if a.size == 0:
        return "no data"
    return (f"mean {a.mean():6.2f}   median {np.median(a):6.2f}   "
            f"p95 {np.percentile(a, 95):6.2f}   max {a.max():6.2f}")


def run_all(cfg, seeds=(0, 1, 2), n_frames=150, per_joint=False):
    all_solved, all_base, all_times = [], [], []
    per_motion = {}
    joint_errs = {label: [] for label in ALL_LABELS}
    extras = {}

    for motion in MOTIONS:
        m_solved, m_base = [], []
        for seed in seeds:
            info = {}
            s, b, t = run_motion(motion, cfg, seed=seed, n_frames=n_frames,
                                 collect=info)
            m_solved += s
            m_base += b
            all_times += t
            if motion not in extras:
                extras[motion] = info
        all_solved += m_solved
        all_base += m_base
        per_motion[motion] = (flatten(m_solved), flatten(m_base))
        if per_joint:
            for d in m_solved:
                for k, v in d.items():
                    joint_errs[k].append(v)

    return {
        "solved": flatten(all_solved),
        "baseline": flatten(all_base),
        "times": np.array(all_times),
        "per_motion": per_motion,
        "joint_errs": joint_errs,
        "extras": extras,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablate", action="store_true",
                    help="turn each improvement off in turn and report the cost")
    ap.add_argument("--per-joint", action="store_true")
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    seeds = tuple(range(args.seeds))
    cfg = Config()
    r = run_all(cfg, seeds=seeds, n_frames=args.frames, per_joint=args.per_joint)

    print("=== angle error vs ground truth (degrees) ===")
    print(f"  MediaPipe world landmarks : {summarize_arr(r['baseline'])}")
    print(f"  this pipeline             : {summarize_arr(r['solved'])}")
    imp = 100.0 * (1 - r["solved"].mean() / r["baseline"].mean())
    print(f"  -> {imp:.0f}% lower mean error")

    print("\n=== by motion (mean deg: baseline -> solved) ===")
    for motion, (s, b) in r["per_motion"].items():
        print(f"  {motion:<8} {b.mean():6.2f} -> {s.mean():6.2f}   "
              f"(p95 {np.percentile(b,95):5.1f} -> {np.percentile(s,95):5.1f})")

    print("\n=== bone length calibration ===")
    for motion, info in r["extras"].items():
        ci = info["calib_info"]
        scene = info["scene"]
        got = info["lengths"]
        err = 100.0 * np.abs(got - scene.true_lengths) / scene.true_lengths
        print(f"  {motion:<8} {ci.get('method','?'):<7} "
              f"mean |err| {err.mean():4.1f}%  max {err.max():4.1f}%  "
              f"({ci.get('seconds', 0):.1f}s)")

    print("\n=== camera tilt recovery ===")
    for motion, info in r["extras"].items():
        scene = info["scene"]
        got, want = info["up"], scene.gravity_up_cam
        deg = np.degrees(np.arccos(np.clip(np.dot(got, want), -1, 1)))
        print(f"  {motion:<8} true tilt {scene.camera_pitch:.0f} deg, "
              f"residual error after correction {deg:.1f} deg")

    print("\n=== timing (ms/frame) ===")
    t = r["times"]
    print(f"  mean {t.mean():.1f}   median {np.median(t):.1f}   "
          f"p99 {np.percentile(t,99):.1f}   -> {1000.0/t.mean():.0f} fps budget")

    if args.per_joint:
        print("\n=== per joint (mean deg) ===")
        for label in ALL_LABELS:
            v = np.array(r["joint_errs"][label])
            if v.size:
                print(f"  {label:<14} {v.mean():6.2f}   p95 {np.percentile(v,95):6.2f}")

    if args.ablate:
        print("\n=== ablation: mean angle error with each piece removed ===")
        base_mean = r["solved"].mean()
        print(f"  {'everything on':<26} {base_mean:6.2f}")
        variants = [
            ("no hinge constraints", dict(hinges=False)),
            ("no constant-velocity", dict(velocity=False)),
            ("no gravity reference", dict(gravity=False)),
            ("no constrained filter", dict(constrained_filter=False)),
            ("+ elbow hinges", dict(elbow_hinges=True)),
            ("+ stiff terminal joints", dict(stiffness=2.5)),
        ]
        for name, kw in variants:
            rr = run_all(Config(**kw), seeds=seeds, n_frames=args.frames)
            m = rr["solved"].mean()
            delta = m - base_mean
            print(f"  {name:<26} {m:6.2f}   {delta:+6.2f}")


def summarize_arr(a):
    if a.size == 0:
        return "no data"
    return (f"mean {a.mean():6.2f}   median {np.median(a):6.2f}   "
            f"p95 {np.percentile(a, 95):6.2f}")


if __name__ == "__main__":
    main()
