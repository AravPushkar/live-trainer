# Phase 0.5 — Audio Spike Design

**Date:** 2026-08-28
**Status:** Design approved, not yet implemented
**Scope:** macOS only. Python only. No mobile work.

---

## 1. Why this, now

The product thesis is that non-verbal audio beats speech for mid-rep form
correction. Nothing in the repository tests that thesis. What exists is a
constrained monocular 3D pose tracker (`pose3d.py`, `skeleton.py`) measured
against synthetic ground truth (`simulate.py`, `test_pose3d.py`) at 5.2 deg
mean joint-angle error, 2.3 deg at the knees.

The tracker is the expensive half and it is built. The differentiator is the
cheap half and it does not exist in any form. This spike closes that gap.

Two facts from the existing code make the squat the right first target:

- Knees are the best-measured joint in the skeleton (~2.3 deg mean error).
  Squat depth is a knee angle. The strongest signal drives the primary channel.
- The gravity reference recovered during calibration removes camera tilt from
  trunk lean, so a second channel exists later if wanted. Not in this spike.

## 2. Goal

Produce a number, not an impression: **does a continuous tone let a person hit
a target squat depth more accurately than proprioception alone?**

Secondary: a first end-to-end glass-to-ear latency measurement, which §6.10 of
the product spec flags as one of the highest-value unbuilt tests.

## 3. Non-goals

Explicitly out of scope, and a reader must not assume any of these are handled:

- Any mobile work. No Swift, no Kotlin, no C++ core.
- The Stage 6 form-rules engine. No valgus, no trunk-lean faults, no
  asymmetry, no tempo. A minimal bottom-of-rep detector only, and only
  because Phase B cannot measure without one.
- Any exercise other than the back/air squat.
- Real-video validation (Phase 0). The tracker's accuracy remains measured
  against a simulator, and that limitation carries forward into this spike's
  results unchanged.
- Music ducking, exercise library, UI, subscription plumbing.
- Multi-channel audio for non-depth faults. §4.3 of the product spec calls
  multiplexing more than two channels an unsolved design problem; this spike
  uses one channel plus an optional target-lock layer and does not attempt to
  solve it.

## 4. Success criteria

The spike succeeds if it produces a completed Phase B result table, whatever
the result says. A negative result is a success — it retires the thesis
cheaply, which is the entire point of doing this before mobile work.

The interesting thresholds, stated in advance so they cannot be moved
afterward:

- **Strong positive:** tone condition mean absolute depth error is at least
  50% lower than silent control, and tone-condition standard deviation is
  lower too (consistency, not just accuracy).
- **Weak positive:** any improvement in mean absolute error that exceeds the
  tracker's own 2.3 deg knee uncertainty.
- **Negative:** no improvement, or the tone is unusable for a reason found
  during Phase A (irritating, ambiguous, warbling beyond tolerance).

n = 1 subject. This is a spike, not a study. It answers "is this worth
pursuing", not "does this work for humans in general".

---

## 5. Architecture

Three new modules, following the existing split where `pose3d.py` is pure math
and `skeleton.py` is the I/O shell.

```
skeleton.py  (existing shell: camera, MediaPipe, solver, HUD)
     |
     |  angles["L knee"], angles["R knee"]  -- once per video frame, ~30fps
     |
     +--> reps.BottomDetector.update()  --> RepEvent(index, bottom_deg, t)
     |
     +--> sonify.Sonifier.update()      --> shared target frequency (float)
                    |
                    |  audio callback thread, ~5ms buffers @ 48kHz
                    v
              phase-continuous oscillator --> speaker / headphones
```

### 5.1 `sonify.py` — angle in, audio out

Pure with respect to the vision stack: imports `numpy` and `sounddevice` and
nothing else. No cv2, no mediapipe. This is what makes it testable offline.

**Public interface:**

```python
class Sonifier:
    def __init__(self, target_deg, top_deg=175.0, bottom_deg=70.0,
                 f_low=200.0, f_high=800.0,
                 curve="log", signal="pitch",
                 smoothing_hz=8.0,
                 samplerate=48000, blocksize=256): ...

    def start(self): ...          # opens the OutputStream
    def stop(self): ...           # closes it; also __enter__/__exit__
    def update(self, angle_deg):  # angle_deg may be None -> gate to silence
    def set_target(self, angle_deg): ...
    def set_curve(self, name): ...    # "log" | "near"
    def set_signal(self, name): ...   # "pitch" | "beat" | "fifth"
    def set_smoothing(self, hz): ...

    @property
    def output_latency_ms(self): ...  # from the PortAudio stream
```

**Pure functions, separately testable with no audio device:**

```python
def angle_to_cents(angle_deg, target_deg, top_deg, bottom_deg, curve) -> float
def cents_to_hz(cents, f_ref) -> float
def render(config, angles, frames_per_angle) -> np.ndarray
```

`render` drives the same synthesis code the callback uses, offline, into a
buffer. Every audio test goes through it. No test requires a speaker.

### 5.2 `reps.py` — bottom-of-rep detection

One job, ~40 lines. A three-state machine with hysteresis over the knee angle.

```python
class BottomDetector:
    def __init__(self, complete_deg=160.0, descend_deg=140.0,
                 min_frames=5): ...
    def update(self, angle_deg, t) -> RepEvent | None
```

```
TOP        --(angle < descend_deg)--> DESCENDING
DESCENDING --(tracks running minimum)--> DESCENDING
DESCENDING --(angle > complete_deg)--> TOP  [emit RepEvent(min_angle)]
```

Separate enter/leave thresholds (140 descend / 160 complete) so a signal
hovering at the boundary cannot rattle and emit phantom reps. The name differs
from `Sonifier.top_deg` deliberately: that one is the top of the *pitch
mapping range*, this one is the *rep-completion* threshold, and they have no
reason to hold the same value. `min_frames` rejects reps
shorter than a plausible squat.

`angle_deg is None` (tracking lost) suspends the state machine without
emitting; a rep interrupted by lost tracking is discarded rather than logged
with a wrong bottom.

This is deliberately not the Stage 6 state machine from the product spec. It
detects one event on one signal for one exercise, to serve Phase B.

### 5.3 `skeleton.py` integration

Small, additive changes to the existing loop:

- New flags: `--audio`, `--target`, `--curve`, `--signal`, `--blind N`,
  `--silent N`, `--log PATH`
- Stream lifecycle around the existing `try/finally`
- One `sonifier.update(knee)` call per frame in the tracking branch
- One `detector.update(knee, t)` call per frame
- New keys, alongside the existing set:
  - `1` / `2`  — curve: log / compressed-near-target
  - `[` / `]`  — cycle target signal
  - `t`        — capture target from current angle
  - `-` / `=`  — audio smoothing down / up
  - `space`    — mute toggle
- HUD additions: current frequency, target angle, live depth error, rep count,
  and total latency

---

## 6. Detailed design

### 6.1 The rate mismatch

This is the part that decides whether the spike sounds like a product or a
broken toy, and it is the single most likely place to get it wrong.

The video loop delivers a new angle every ~33ms. The audio callback needs
samples every ~5ms. Recomputing frequency once per video frame and assigning
it to the oscillator produces two artifacts: a 30Hz zipper on the frequency
parameter, and a click on every phase discontinuity.

The design:

- **Phase accumulator, never reset.** Frequency changes advance phase at a new
  rate; phase itself stays continuous. Resetting phase on a frequency change
  is the click.
- **Per-sample frequency glide inside the callback.** The callback interpolates
  from its current frequency toward the target across the buffer, as a
  one-pole lowpass at `smoothing_hz`. Parameter updates at 30Hz become smooth
  at 48kHz.
- **One shared float, no locks.** The video thread writes the target frequency;
  the callback reads it. A lock in an audio callback means dropouts. A torn
  read of a Python float is not possible under the GIL, and a one-frame-stale
  value is inaudible.
- **Gate amplitude, not the stream.** Muting ramps amplitude to zero over a few
  milliseconds rather than stopping the stream. Stopping and restarting a
  PortAudio stream mid-set costs tens of milliseconds and clicks.

### 6.2 Mapping — two orthogonal axes

The product spec treats mapping curve and target signalling as separate open
questions (§4.3). They are exposed as two independent controls rather than
hardcoded presets, which is both more informative and less code.

**Axis 1 — curve** (`--curve`, keys `1`/`2`):

- `log` — linear in cents between `top_deg` and `bottom_deg`. Pitch perception
  is logarithmic, so "linear in Hz" sounds non-uniform; linear here means
  linear in log-frequency.
- `near` — resolution concentrated near the target. The last 10 deg of depth
  spans as much pitch as the first 60 deg. Expected to win, since precision
  is only wanted in one place.

**Axis 2 — target signal** (`--signal`, keys `[`/`]`):

- `pitch` — pitch alone. The baseline. Hitting target means recognizing an
  absolute pitch, which requires learned pitch memory.
- `beat` — pitch, **plus** a fixed drone at the target frequency. The
  interference between drone and moving tone beats at their frequency
  difference; at target the difference is zero and the beating stops.
- `fifth` — pitch, plus a drone a perfect fifth (3:2) below target frequency,
  so hitting depth lands on a consonant interval.

Note that `beat` and `fifth` are **layers on top of** `pitch`, not
replacements. This matters: beating is only perceptible below roughly 15-20Hz
of difference, so beat alone carries no coarse information — at 60 deg from
target the difference is well past beating into roughness. Pitch gives coarse
position, the layer gives fine target lock. Building them as replacements
would produce a mode with no guidance until you were already nearly correct.

`beat` is the one worth betting on. It is the most precise discrimination
human hearing performs, and unlike pitch memory it needs no learned reference
— zero beat rate is self-evident.

### 6.3 A prediction, stated before building — now measured, see §11

Mapping 175 deg to 70 deg of knee travel across 200-800Hz is 2400 cents over
105 deg, or ~23 cents per degree. Pitch JND is roughly 5-10 cents for steady
tones.

The tracker's 2.3 deg knee error therefore lands at roughly **50 cents — half
a semitone, loudly audible.**

**You will hear the tracking noise.** The ear is not the bottleneck; the
solver is. That is the correct place for the bottleneck, but it has a
consequence: the audio parameter probably needs smoothing beyond what the One
Euro filter already provides, and that trades directly against the sub-100ms
premise the whole product rests on.

One Euro smooths hard at low speed, so the bottom of the squat — where depth
is judged — should already be clean. The warble will be worst mid-descent,
where the filter opens up for responsiveness. This may turn out to be fine:
mid-descent is where coarse guidance is wanted and fine precision is not.

`smoothing_hz` is live-tunable by keypress specifically so this can be found
by ear rather than guessed. **Where that knob settles is the most valuable
single output of Phase A**, because it is the first empirical read on the
jitter-versus-lag tradeoff that the mobile product will inherit.

### 6.4 Gating

The tone gates to silence when:

- MediaPipe reports no pose
- knee landmark visibility falls below the solver's threshold
- the solver fails to converge

Silence beats garbage. A spike that plays noise when tracking drops gets
judged on the noise, and the failure mode is indistinguishable from the
concept being bad.

**Signal source:** mean of `L knee` and `R knee`, falling back to whichever is
visible, gating if neither is. Hip angle is arguably the truer definition of
parallel — the product spec's §6.7 makes this point — but hips measure at 3-4
deg against knees at 2.3 deg, so the better-measured joint wins for the spike.
Revisit when the hip signal is needed for anything else.

### 6.5 Target source

`--target 90` with a default, plus the `t` key to capture the target from the
current angle: squat to what you consider correct depth, press `t`, done.

This sidesteps the absolute-versus-personalized product decision (spec §6.7)
without pretending to answer it. The spike needs *a* target that is real for
one body; it does not need the general answer.

---

## 7. Protocols

### 7.1 Phase A — tune by ear

Screen on, HUD visible, all controls live. Squat, switch curves and signals,
move the smoothing knob, form an opinion.

Exit criteria — Phase A ends when all of these are recorded:

1. One curve/signal combination chosen, with a stated reason.
2. A `smoothing_hz` value that removes objectionable warble without
   perceptible lag, and a note on whether such a value exists at all.
3. A comfortable frequency range for a 45-minute session, which may not be
   200-800Hz.
4. Measured end-to-end latency from the HUD.

If no combination is tolerable, Phase A is where the project learns that, and
Phase B does not run.

### 7.2 Phase B — blind measurement

Two conditions, screen blanked in both:

- `--silent N` — no tone. Pure proprioception. **Run this first.**
- `--blind N` — tone on, chosen configuration.

Order is not negotiable. Once the tone teaches the depth, it cannot be
un-learned, and a tone-then-silent order would flatter the result by measuring
a trained subject against an untrained one.

Both log per-rep bottom angle to CSV. `skeleton.py` already has CSV logging
behind the `r` key; `--log PATH` extends that machinery to open a file at
startup and write one row per `RepEvent` rather than one row per frame.

Each mode runs until N reps are detected, then closes the stream, restores the
screen and prints a summary table: per-rep bottom angle, mean absolute error
from target, and standard deviation.

The control condition measures exactly the claim in product spec §2.1 — that
lifters believe they are hitting depth when they are 15-20 deg short. If the
silent control reproduces that, the control is trustworthy. If it does not,
that itself is worth knowing before building a product premised on it.

**Confound to record, not to fix:** n=1, unblinded to condition (the subject
obviously knows whether a tone is playing), and no washout between conditions.
All three are acceptable for a go/no-go spike and all three must be written
into the result so no downstream reader mistakes it for a study.

---

## 8. Testing

Following `test_pose3d.py`'s approach: measure offline against synthetic
input, no hardware in the loop.

**`sonify.py`:**
- `angle_to_cents` is monotonic in the right direction for both curves
- both curves map `target_deg` to the same cents value (target is a fixed
  point regardless of curve, or switching curves mid-set moves the target)
- `near` has strictly higher cents-per-degree at the target than `log`
- rendered output has no sample-to-sample discontinuity above a threshold,
  across an angle sequence containing step changes
- rendered output is exactly zero after a `None` angle plus the ramp length
- `beat` mode: FFT of the rendered buffer at target angle shows a single peak;
  off-target shows two peaks separated by the expected difference

**`reps.py`:**
- clean synthetic squat trace produces exactly N reps with correct bottoms
- a trace that hovers at the descend threshold produces zero phantom reps
- a `None` gap mid-rep discards that rep and does not emit
- reps shorter than `min_frames` are rejected

**Integration:** angle traces recorded from a real session via the existing
CSV logging, replayed through `sonify.render` offline, checked for artifacts.
This reuses the tracker's real noise characteristics without needing the
camera in the test loop.

---

## 9. Risks

1. **The warble makes it unusable.** §6.3 predicts the tracker's noise is
   audible at roughly half a semitone. If no smoothing value resolves this
   without introducing perceptible lag, the spike fails on the tracker's
   accuracy rather than on the audio concept — and those two failures must not
   be confused when reporting the result.
2. **n=1 and unblinded.** The subject knows which condition they are in and
   wants the tone to work. Real, unfixable at this scale, must be stated.
3. **Learning confound.** Depth accuracy improves with practice regardless of
   feedback. Running silent-first mitigates the direction of this bias but
   does not remove it — it makes the tone condition the *advantaged* one.
   Report accordingly.
4. **A good result proves less than it appears to.** Success means the tone
   works on macOS, at a fixed distance, on a laptop webcam, on one body, on
   synthetic-validated tracking. Product spec risks 1, 2 and 3 — accuracy
   floor on real video, gym filming, camera placement in the wild — are all
   untouched by this spike.

---

## 10. Deliberately unanswered

Carried forward from the product spec, not resolved here:

- Whether form targets should be absolute or personalized to calibrated
  anatomy (§6.7). The spike uses a captured personal target because it is
  convenient, not because the question is settled.
- How non-depth faults are encoded in audio without becoming noise (§4.3).
  One channel plus one target-lock layer is the ceiling here.
- Whether rep confirmation needs a discrete sound at lockout. Not built. If
  Phase A shows the tone's return to the top pitch is insufficient
  confirmation, that is a finding to record.
- Everything about real video, gates, and thresholds. Phase 0 is still
  unstarted and this spike does not advance it.


---

## 11. Measured, 2026-08-28 — before Phase A

`tune_smoothing.py` runs simulate.py's squat through the real solver, One Euro
and bone-length projection, and reports what each filter costs. Four seeds,
150 frames. Reproduce with `python tune_smoothing.py`.

### 11.1 The prediction in §6.3 was half right, and the framing was wrong

Warble is real and audible. But the dominant source of lag is not the audio
filter this spike was built to tune — it is **the One Euro filter already in
the tracker**, whose phase delay nobody had measured.

| One Euro (min_cutoff / beta) | lag | RMS error | HF jitter |
|---|---|---|---|
| off | 2.0 ms | 4.47° | 2.28° |
| 0.5 / 0.5 | 144.3 ms | 3.89° | 0.62° |
| **1.0 / 0.7 (current default)** | **97.7 ms** | 3.95° | 0.71° |
| 2.0 / 1.0 | 60.3 ms | 4.01° | 0.89° |
| 4.0 / 1.0 | 35.8 ms | 4.07° | 1.13° |
| 8.0 / 1.0 | 20.0 ms | 4.20° | 1.47° |
| 12.0 / 1.0 | 14.1 ms | 4.27° | 1.66° |

**The current settings spend 98ms of delay** — on their own, more than the
entire 100ms glass-to-ear budget the tone is supposed to live inside, before
camera, inference or audio output are counted.

### 11.2 The product spec's latency budget has a category error

Product spec §6.10 allocates "~2–5ms" to *"Bundle adjustment + filter +
angles"*. That is the cost of **running** that code, and it is correct: the
solve measures a few milliseconds per frame.

Running it is not the expensive part. A low-pass filter delays the signal
passing through it, and that delay is invisible to a profiler. At current
settings One Euro contributes 98ms of *phase delay* against a budgeted 2–5ms
of *compute*. The budget is not merely optimistic, it is measuring a different
quantity, and the same error will reappear in the C++ core if it is copied
forward. **Every filter in the chain needs a phase-delay line in that table,
separate from its compute line.**

### 11.3 Nothing fits inside 100ms

Two compute figures belong in the budget that were not in my first pass, and
both are measured rather than estimated:

- **Solver compute: 16.8ms mean, 51.6ms p99** per frame for solve + One Euro +
  bone projection (`test_pose3d.py` timing output). Product spec §6.10 budgets
  2–5ms for this stage.
- **Audio output: 19.3ms** at blocksize 128. Swept across blocksizes on this
  machine: 1024→38.0ms, 512→27.3, 256→22.0, 128→19.3, 64→18.0, 32→18.0. The
  device floors at 18.0ms, so there is ~4ms available here and no more.
  `Config.blocksize` now defaults to 128 rather than 256: it takes most of the
  saving at half the callback rate of 64, which matters because MediaPipe is
  competing for the same cores.

Fixed stages: camera 33 + inference 20 + solve 16.8 + audio out 19.3 =
**89.1ms**, leaving 10.9ms for filter phase delay.

| One Euro | warble | total latency | verdict |
|---|---|---|---|
| off | 68.5¢ | 91.1 ms | 6.8× JND, fits |
| 1.0 / 0.7 (current) | 21.4¢ | 186.8 ms | over |
| 4.0 / 1.0 | 33.9¢ | 124.9 ms | over |
| 8.0 / 1.0 | 44.4¢ | 109.1 ms | over |
| 12.0 / 1.0 | 50.1¢ | 103.2 ms | over by 3.2 ms |

**No setting with any filtering at all fits.** The only configuration inside
100ms is One Euro off, at 68 cents of warble — nearly seven times JND, which
is most of a semitone of wobble on a body holding still.

The audio filter, which this spike was built to tune, turns out to be the
smallest term in the problem.

### 11.4 What would have to change

| Lever | Saving | Basis |
|---|---|---|
| Capture at 60fps | −16.5 ms | estimate; halves the frame period |
| Solver compute to spec | −13.3 ms | spec budgets 2–5ms; measured 16.8 |
| Blocksize 256 → 64 | −4.0 ms | measured; floors at 18.0ms |
| Drop One Euro | −14.1 ms | measured; costs 68¢ of warble |

At 60fps capture and blocksize 64, fixed stages fall to 71.3ms and One Euro
8.0/1.0 fits at **91ms total, 44 cents of warble**. That is the only
combination found that is both filtered and inside budget, and it depends on a
60fps capture path that does not exist yet.

**The cheapest test is not on that table.** The 100ms figure is a design target
from product spec §6.10 — "roughly 100ms", asserted, never measured against a
person. Whether 110ms actually feels detached from the body is a question about
perception that a single afternoon answers, and every lever above is more
expensive than asking it. Phase A should measure it directly: run the tone at
several deliberately inflated latencies and find where it stops feeling
attached.

### 11.5 What this does and does not settle

**Settled:** the audio filter was never the interesting knob. One Euro was, and
it is mistuned for this use by roughly 5× in cutoff — 98ms of delay to buy
0.71° of jitter, where 8.0/1.0 buys 1.47° for 20ms. The §6.3 plan to find the
smoothing value by ear during Phase A would have spent a session tuning a 4ms
effect with a 98ms one sitting upstream, unexamined.

**Settled:** the product spec's latency budget cannot be repaired by adjusting
its numbers, because two of its rows measure the wrong quantity. Every filter
needs a phase-delay line separate from its compute line, and the solve row is
off by 3–10×.

**Not settled, and it is now the main question:** every configuration is either
over budget or audibly wobbly. Whether 3–7× JND of warble is disqualifying or
merely unlovely is a question about human perception under load that no amount
of simulation answers. Phase A still has to happen — but it should now also
measure the latency threshold, which was previously assumed.

**Caveats that must travel with these numbers:**

- The observed knee ROM is **100–180°**. That is a shallow squat; parallel is
  ~90° and below-parallel ~70°. Deep flexion is the hardest geometry for
  monocular depth — limbs foreshortened, joints occluding each other — so real
  error at the bottom of a real squat is likely **worse** than this. The
  simulator is not pessimistic enough in exactly the place the product cares
  most about, and fixing that is a prerequisite to trusting any of this.
- Everything inherits `simulate.py`'s noise model. Absolute values should move
  when real footage goes through the pipeline. The *ordering* — which knob
  costs what — should survive.
- Camera (33ms) and inference (20ms) are the product spec's estimates, never
  measured. Solver compute and audio output are measured on this machine.
