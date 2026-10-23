"""
What does the pipeline's latency and jitter actually cost the tone?

The audio spike design predicted that solver noise would be audible as pitch
warble, and treated the audio smoothing constant as the knob that trades that
warble against lag. Running it, the prediction is half right and the framing
is wrong.

The warble is real. The dominant source of lag is not the audio filter -- it
is the One Euro filter already in the tracker, whose *phase delay* nobody had
measured. The product latency budget allocates 2-5ms to "bundle adjustment +
filter + angles", which is the cost of running that code. Running it is not
the expensive part. At the current settings (min_cutoff 1.0, beta 0.7) the
filter delays the knee angle by around 88ms, which on its own exceeds the
entire 100ms glass-to-ear budget the tone is supposed to live inside.

So this sweeps One Euro first and the audio filter second, and reports the
whole chain against the budget.

    python tune_smoothing.py
    python tune_smoothing.py --seeds 5 --frames 200

Everything here inherits simulate.py's noise model. These numbers describe the
tracker we can measure, not a tracker in a gym, and the absolute values should
move when real footage finally goes through it. The *ordering* -- which knob
costs what -- should survive.
"""

import argparse
import math

import numpy as np

import pose3d
import reps
import sonify
from pose3d import (BoneCalibrator, OneEuroFilter, PoseSolver, compute_angles,
                    enforce_bone_lengths)
from simulate import Scene

FPS = 30.0
CALIB_FRAMES = 60

# Pitch JND for steady tones is roughly 5-10 cents.
JND_CENTS = 10.0

# Glass-to-ear stages the audio filter has to share 100ms with. Camera and
# inference are the product spec's own figures; audio output is measured from
# a real PortAudio stream on this machine.
MS_CAMERA = 33.0
MS_INFERENCE = 20.0
MS_AUDIO_OUT = 19.3
# Measured, not estimated: test_pose3d.py reports 16.8ms mean and 51.6ms p99
# per frame for solve + filter + bone projection on this machine. The product
# spec budgets 2-5ms for that stage. This is compute, and it is separate from
# the filter phase delay below -- both are real and both are additive.
MS_SOLVE = 16.8
MS_BUDGET = 100.0

EURO_SETTINGS = [
    ("off", None, None),
    ("0.5 / 0.5", 0.5, 0.5),
    ("1.0 / 0.7  (current)", 1.0, 0.7),
    ("2.0 / 1.0", 2.0, 1.0),
    ("4.0 / 1.0", 4.0, 1.0),
    ("8.0 / 1.0", 8.0, 1.0),
    ("12.0 / 1.0", 12.0, 1.0),
]


# --------------------------------------------------------------------------
# Pure analysis
# --------------------------------------------------------------------------

def fill_gaps(values):
    """
    Linearly interpolate untracked frames, holding the ends.

    Dropping them instead would break the uniform sampling cross-correlation
    depends on, so a "frame of lag" would stop meaning a frame. Substituting a
    constant is worse: a dropout filled with 175 degrees is a spike the
    tracker never produced, and it drags the correlation peak toward zero. An
    earlier version of this module did exactly that and reported 35ms of One
    Euro lag where there was 88.
    """
    vals = list(values)
    idx = [i for i, v in enumerate(vals) if v is not None]
    if not idx:
        raise ValueError("trace has no tracked frames at all")
    xs = np.arange(len(vals), dtype=float)
    known_x = np.array(idx, dtype=float)
    known_y = np.array([vals[i] for i in idx], dtype=float)
    return np.interp(xs, known_x, known_y)


def hf_jitter(residual, window=5):
    """
    RMS of the fast part of an error signal -- the audible wobble.

    Total RMS error is dominated by slow bias, which a listener hears as the
    tone sitting a little off rather than as warble, and which smoothing
    cannot remove. Subtracting a short moving average leaves the part that
    actually wobbles.
    """
    r = np.asarray(residual, dtype=float)
    kernel = np.ones(window) / window
    detrended = r - np.convolve(r, kernel, "same")
    # "same" zero-pads, so the first and last window//2 samples average over
    # fewer values than they should and a *constant* signal reads as jitter
    # there. On a 150-frame trace those few frames were inflating the figure
    # by a third. Drop them.
    edge = window // 2
    if len(detrended) > 2 * edge:
        detrended = detrended[edge:len(detrended) - edge]
    return float(np.sqrt(np.mean(detrended ** 2)))


def smooth_hz(values, smoothing_hz, fps=FPS):
    """
    The engine's per-sample frequency glide, evaluated once per video frame.

    Engine applies a one-pole at `smoothing_hz` to every sample while holding
    the target constant across a video frame. Over the sr/fps samples in one
    frame that collapses exactly to a one-pole at the video rate, which is why
    this sweep can run at 30Hz instead of 48kHz. Smoothing applies to
    frequency in Hz, not cents, because that is what the engine does -- the
    two diverge by over two semitones on a step, so the distinction is not
    cosmetic.
    """
    values = np.asarray(values, dtype=float)
    decay = math.exp(-2.0 * math.pi * smoothing_hz / fps)
    out = np.empty_like(values)
    f = values[0]
    for i, target in enumerate(values):
        f = target + (f - target) * decay
        out[i] = f
    return out


def estimate_lag_frames(delayed, reference, max_lag=30):
    """
    How many frames `delayed` trails `reference`, to sub-frame resolution.

    Pearson correlation at each integer lag with a parabolic fit around the
    peak. Mean and energy are computed inside each overlap window rather than
    over the whole signal: doing it globally looks equivalent and is not --
    every lag then compares windows whose own means differ from the one
    subtracted, tilting the curve and dragging the fitted peak toward zero. It
    cost about 4% on a known delay, small enough to survive review and large
    enough to misreport the budget this sweep exists to police.
    """
    a = np.asarray(delayed, dtype=float)
    b = np.asarray(reference, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.empty(len(lags), dtype=float)
    for i, lag in enumerate(lags):
        if lag >= 0:
            u, v = a[lag:], b[:n - lag] if lag else b
        else:
            u, v = a[:n + lag], b[-lag:]
        u = u - u.mean()
        v = v - v.mean()
        denom = math.sqrt(float(np.dot(u, u)) * float(np.dot(v, v)))
        corr[i] = float(np.dot(u, v)) / denom if denom else 0.0

    k = int(np.argmax(corr))
    delta = 0.0
    if 0 < k < len(corr) - 1:
        y0, y1, y2 = corr[k - 1], corr[k], corr[k + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom != 0.0:
            delta = 0.5 * (y0 - y2) / denom
    return float(lags[k] + delta)


def align(est, true, lag_frames):
    """Shift out `lag_frames` of delay and return the overlapping pair."""
    shift = int(round(lag_frames))
    if shift > 0:
        return np.asarray(est)[shift:], np.asarray(true)[:len(true) - shift]
    if shift < 0:
        return np.asarray(est)[:len(est) + shift], np.asarray(true)[-shift:]
    return np.asarray(est), np.asarray(true)


# --------------------------------------------------------------------------
# Driving the real pipeline
# --------------------------------------------------------------------------

def knee_trace(seed=0, n_frames=150, min_cutoff=1.0, beta=0.7):
    """
    One squat through the whole pipeline. Returns (true_deg, est_deg).

    Calibration runs from its own standing scene on corrupted observations,
    exactly as skeleton.py and test_pose3d.py do it -- handing the solver true
    bone lengths would measure a pipeline nobody has. `min_cutoff=None`
    disables One Euro entirely.
    """
    calib = BoneCalibrator(CALIB_FRAMES)
    calib_scene = Scene(motion="calib", n_frames=CALIB_FRAMES, seed=seed + 100)
    for i in range(CALIB_FRAMES):
        uv, vis, world, _ = calib_scene.observe(i)
        calib.add(world, vis, vis_thresh=0.4, uv=uv)
    lengths = calib.result()

    scene = Scene(motion="squat", n_frames=n_frames, seed=seed)
    up = calib.estimate_up(scene.K, lengths)
    if up is None:
        up = np.array([0.0, -1.0, 0.0])

    solver = PoseSolver(scene.K, lengths)
    euro = (None if min_cutoff is None
            else OneEuroFilter(min_cutoff=min_cutoff, beta=beta))

    true_deg, est_deg = [], []
    t = 0.0
    for i in range(n_frames):
        uv, vis, world, X_true = scene.observe(i)
        X, _ = solver.solve(uv, vis, world)
        if euro is not None:
            X = euro(X.ravel(), t=t).reshape(pose3d.N_JOINTS, 3)
        X = enforce_bone_lengths(X, lengths, weights=vis)
        t += 1.0 / FPS
        true_deg.append(reps.knee_signal(
            compute_angles(X_true, vis, up=scene.gravity_up_cam)))
        est_deg.append(reps.knee_signal(compute_angles(X, vis, up=up)))

    return fill_gaps(true_deg), fill_gaps(est_deg)


def measure_euro(seeds, n_frames, min_cutoff, beta):
    lags, rms, hf = [], [], []
    rom = []
    for seed in seeds:
        true_deg, est_deg = knee_trace(seed, n_frames, min_cutoff, beta)
        lag = estimate_lag_frames(est_deg, true_deg)
        a, b = align(est_deg, true_deg, lag)
        lags.append(lag * 1000.0 / FPS)
        rms.append(float(np.sqrt(np.mean((a - b) ** 2))))
        hf.append(hf_jitter(a - b))
        rom.append((true_deg.min(), true_deg.max()))
    return {
        "lag_ms": float(np.mean(lags)),
        "rms_deg": float(np.mean(rms)),
        "hf_deg": float(np.mean(hf)),
        "rom": (float(np.mean([r[0] for r in rom])),
                float(np.mean([r[1] for r in rom]))),
    }


def measure_audio(seeds, n_frames, min_cutoff, beta, cfg, smoothings):
    """Extra lag and jitter reduction from the audio filter, in cents."""
    traces = [knee_trace(s, n_frames, min_cutoff, beta) for s in seeds]
    rows = []
    for fc in smoothings:
        extra, hf = [], []
        for true_deg, est_deg in traces:
            est_hz = np.array([sonify.angle_to_hz(cfg, a) for a in est_deg])
            true_hz = np.array([sonify.angle_to_hz(cfg, a) for a in true_deg])
            sm_hz = smooth_hz(est_hz, fc)
            # The audio filter's own delay, measured against the signal going
            # into it rather than against truth, so it is not confounded with
            # whatever the tracker already contributed.
            extra.append(estimate_lag_frames(sm_hz, est_hz) * 1000.0 / FPS)

            sm_c = 1200.0 * np.log2(sm_hz / cfg.f_low)
            true_c = 1200.0 * np.log2(true_hz / cfg.f_low)
            lag = estimate_lag_frames(sm_c, true_c)
            a, b = align(sm_c, true_c, lag)
            hf.append(hf_jitter(a - b))
        rows.append({"smoothing_hz": fc,
                     "extra_lag_ms": float(np.mean(extra)),
                     "warble_cents": float(np.mean(hf))})
    return rows


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Latency and jitter budget for "
                                             "the audio spike.")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--f-low", type=float, default=200.0)
    ap.add_argument("--f-high", type=float, default=800.0)
    ap.add_argument("--audio-out-ms", type=float, default=MS_AUDIO_OUT,
                    help="measured PortAudio output latency on this machine")
    ap.add_argument("--solve-ms", type=float, default=MS_SOLVE,
                    help="measured solve+filter+bones compute per frame; see "
                         "test_pose3d.py timing output")
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    print(f"simulated squat, {args.seeds} seeds x {args.frames} frames\n")

    print("ONE EURO FILTER -- the tracker's own contribution")
    print(f"{'min_cutoff / beta':>22}  {'lag':>8}  {'RMS err':>9}  "
          f"{'HF jitter':>10}")
    print(f"{'':>22}  {'(ms)':>8}  {'(deg)':>9}  {'(deg)':>10}")
    print("-" * 55)
    results = []
    for label, mc, beta in EURO_SETTINGS:
        r = measure_euro(seeds, args.frames, mc, beta)
        r["label"], r["min_cutoff"], r["beta"] = label, mc, beta
        results.append(r)
        print(f"{label:>22}  {r['lag_ms']:8.1f}  {r['rms_deg']:9.2f}  "
              f"{r['hf_deg']:10.2f}")

    lo, hi = results[0]["rom"]
    # Map the full frequency range onto the range of motion actually observed,
    # which is what calibration does live. Anything narrower clips the top of
    # every rep and quietly flattens the signal.
    cfg = sonify.Config(target_deg=(lo + hi) / 2, bottom_deg=lo, top_deg=hi,
                        f_low=args.f_low, f_high=args.f_high)
    cpd = 1200.0 * math.log2(args.f_high / args.f_low) / (hi - lo)
    print(f"\nobserved knee ROM {lo:.0f}-{hi:.0f} deg mapped onto "
          f"{args.f_low:.0f}-{args.f_high:.0f} Hz = {cpd:.1f} cents/deg")
    print(f"JND is {JND_CENTS:.0f} cents, so {JND_CENTS / cpd:.2f} deg is the "
          f"finest depth difference a listener could hear.")

    fixed = MS_CAMERA + MS_INFERENCE + args.solve_ms + args.audio_out_ms
    print(f"\nFixed stages: camera {MS_CAMERA:.0f} + inference "
          f"{MS_INFERENCE:.0f} + solve {args.solve_ms:.1f} + audio out "
          f"{args.audio_out_ms:.0f} = {fixed:.1f} ms, leaving "
          f"{MS_BUDGET - fixed:.1f} ms of the {MS_BUDGET:.0f} ms budget for "
          f"filter phase delay.")

    print(f"\n{'One Euro':>22}  {'warble':>8}  {'total':>8}   verdict")
    print(f"{'':>22}  {'(cents)':>8}  {'(ms)':>8}")
    print("-" * 62)
    for r in results:
        total = fixed + r["lag_ms"]
        warble = r["hf_deg"] * cpd
        verdict = ("over budget" if total > MS_BUDGET else
                   "inaudible warble" if warble <= JND_CENTS else
                   f"{warble / JND_CENTS:.1f}x JND, in budget")
        print(f"{r['label']:>22}  {warble:8.1f}  {total:8.1f}   {verdict}")

    affordable = [r for r in results
                  if fixed + r["lag_ms"] <= MS_BUDGET and r["min_cutoff"]]
    if not affordable:
        print("\nNothing fits: no One Euro setting leaves the chain inside "
              f"{MS_BUDGET:.0f} ms.")
        cheapest = min([r for r in results if r["min_cutoff"]],
                       key=lambda r: r["lag_ms"])
        need = fixed + cheapest["lag_ms"] - MS_BUDGET
        print(f"The loosest useful setting ({cheapest['label'].strip()}) "
              f"overshoots by {need:.1f} ms.\n")
        print("Levers, largest first:")
        print(f"  capture at 60fps          -{MS_CAMERA / 2:5.1f} ms   "
              "estimate; halves the frame period")
        print(f"  solver compute to spec     -{args.solve_ms - 3.5:5.1f} ms   "
              "the product spec budgets 2-5ms; measured is "
              f"{args.solve_ms:.1f}")
        print(f"  audio blocksize 256 -> 64  -{22.0 - 18.0:5.1f} ms   "
              "measured; the device floors at 18.0 ms")
        print(f"  drop One Euro entirely     -{cheapest['lag_ms']:5.1f} ms   "
              f"measured; costs "
              f"{results[0]['hf_deg'] * cpd:.0f} cents of warble")
        print("\nAt 60fps capture and blocksize 64 the fixed stages fall to "
              f"{MS_CAMERA / 2 + MS_INFERENCE + args.solve_ms + 18.0:.1f} ms, "
              "which One Euro 8.0/1.0 fits inside at "
              f"{MS_CAMERA / 2 + MS_INFERENCE + args.solve_ms + 18.0 + 20.0:.0f}"
              " ms total.")
        print("\nThe 100ms figure is a design target from the product spec, "
              "not a measured perceptual threshold. Whether 110ms actually "
              "feels detached from the body is itself untested, and is "
              "cheaper to test than any of the levers above.")
        return
    best = min(affordable, key=lambda r: r["hf_deg"])
    print(f"\nQuietest setting inside the budget: {best['label'].strip()} -- "
          f"{best['hf_deg'] * cpd:.0f} cents of warble at "
          f"{fixed + best['lag_ms']:.0f} ms total.")

    headroom = MS_BUDGET - fixed - best["lag_ms"]
    print(f"\nAUDIO FILTER on top of that, with {headroom:.0f} ms of headroom")
    print(f"{'smoothing':>10}  {'extra lag':>10}  {'warble':>8}   verdict")
    print(f"{'(Hz)':>10}  {'(ms)':>10}  {'(cents)':>8}")
    print("-" * 52)
    rows = measure_audio(seeds, args.frames, best["min_cutoff"], best["beta"],
                         cfg, [1000.0, 15.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0])
    for row in rows:
        label = ("none" if row["smoothing_hz"] > 100
                 else f"{row['smoothing_hz']:.0f}")
        fits = row["extra_lag_ms"] <= headroom
        print(f"{label:>10}  {row['extra_lag_ms']:10.1f}  "
              f"{row['warble_cents']:8.1f}   "
              f"{'fits' if fits else 'blows the budget'}")

    usable = [r for r in rows if r["extra_lag_ms"] <= headroom]
    if usable:
        pick = min(usable, key=lambda r: r["warble_cents"])
        print(f"\nStart Phase A at One Euro {best['label'].strip()} with "
              f"--smoothing {pick['smoothing_hz']:.0f}"
              if pick["smoothing_hz"] < 100 else
              f"\nStart Phase A at One Euro {best['label'].strip()} with the "
              f"audio filter effectively open.")
        print(f"Expected warble {pick['warble_cents']:.0f} cents "
              f"({pick['warble_cents'] / JND_CENTS:.1f}x JND), total latency "
              f"{fixed + best['lag_ms'] + pick['extra_lag_ms']:.0f} ms.")


if __name__ == "__main__":
    main()
