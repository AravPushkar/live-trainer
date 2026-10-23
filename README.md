# Constrained monocular 3D pose, with audio form feedback

A single-camera 3D skeleton tracker aimed at live strength-training form
feedback, plus an experimental sonification layer that maps joint angle to a
continuous tone instead of to speech.

The premise: by the time a spoken cue like "go a little deeper" has been
synthesised and said, the rep is over. A tone whose pitch tracks your knee
angle arrives inside the movement, and pitch is processed pre-attentively, with
no language step.

## What works

**3D pose from one camera.** MediaPipe's `pose_world_landmarks` are regressed
from a learned body prior, so their depth axis is compressed and noisy. They
are used here only as an initialiser, a bone ruler and a weak tiebreaker. The
solver instead looks for the pose that reprojects onto the observed 2D
keypoints under real camera intrinsics, carries the subject's own bone lengths,
keeps a rigid torso, bends its knees the way a knee bends, and moves
continuously from where the previous frames were heading.

Against synthetic ground truth this cuts mean joint-angle error 62% versus
reading the world landmarks directly — 13.6° to 5.2° — and p95 from 42° to 17°.

Accuracy is not uniform, and the spread matters more than the average: knees
land near 2.3° and the tilts near 1°; shoulders, hips and trunk sit at 3–4°;
elbows near 5°. **Wrists are 12–20° and should not be trusted** — they are
measured from the index-finger landmark across a segment a few dozen pixels
long, whose depth is close to unobservable.

**No setup step.** Camera tilt is read off your own trunk during calibration
(a laptop screen leaning back 15° otherwise puts 15° of bias on every angle
measured against vertical). Bone lengths come from the same window.

**Audio.** A phase-continuous oscillator with per-sample frequency glide, two
mapping curves (linear in cents, or resolution concentrated at the target) and
three target signals (pitch alone, or a drone whose beating stops at target, or
a tone landing on a perfect fifth).

## What is unproven

- Ground truth is **synthetic**. No real footage has been through the pipeline.
  The simulator models tilted cameras, heavy-tailed keypoint noise, occlusion
  runs and limbs swinging through the camera axis, but it is still a simulator,
  and its squat only reaches 100° of knee flexion — shallower than parallel,
  and short of the deep flexion where monocular depth is hardest.
- Whether anyone can hit a target depth by ear has never been tested on a
  person. See [docs/audio-spike-design.md](docs/audio-spike-design.md) §7.
- The glass-to-ear latency chain does not currently fit inside 100ms. See
  §11 of the same document; the short version is that a low-pass filter's
  phase delay is invisible to a profiler, and the One Euro filter was costing
  ~98ms to buy 0.7° of jitter.

## Install

```
python -m venv .venv
.venv/bin/pip install opencv-python mediapipe numpy scipy sounddevice
.venv/bin/pip install pyobjc-framework-AVFoundation   # optional, camera names
```

## Run

Live tracker. Stand back far enough that your head and feet are both in frame,
then follow the on-screen prompt for a few seconds.

```
.venv/bin/python skeleton.py
.venv/bin/python skeleton.py --list
.venv/bin/python skeleton.py --model heavy --raw      # A/B vs unconstrained
```

With the tone:

```
.venv/bin/python skeleton.py --audio --min-cutoff 8.0 --beta 1.0 --smoothing 8
```

Keys: `1`/`2` mapping curve, `[`/`]` target signal, `t` capture target from the
angle you are holding, `-`/`=` smoothing, space to mute. `q` quits.

Blind measurement — run the silent control **first**, since the tone teaches
the depth and you cannot un-learn it:

```
.venv/bin/python skeleton.py --silent 10 --target 88 --log control.csv
.venv/bin/python skeleton.py --blind 10  --target 88 --log tone.csv
```

## Tests

```
.venv/bin/python test_sonify.py            # 64 unit tests, no camera or audio device
.venv/bin/python test_pose3d.py            # solver vs synthetic ground truth
.venv/bin/python test_pose3d.py --ablate   # what each constraint is worth
.venv/bin/python test_pose3d.py --per-joint
.venv/bin/python tune_smoothing.py         # latency and jitter budget
```

## Layout

| File | |
|---|---|
| `pose3d.py` | solver, One Euro filter, bone calibration, joint angles |
| `skeleton.py` | live webcam app, HUD, measurement modes |
| `simulate.py` | synthetic ground truth |
| `sonify.py` | angle to tone: mapping and synthesis |
| `reps.py` | bottom-of-rep detection |
| `tune_smoothing.py` | latency and jitter sweep over the real pipeline |
