"""
Sonification of a joint angle: angle in, continuous tone out.

Imports numpy and (lazily) sounddevice, and nothing from the vision stack. The
synthesis core is a plain function over an angle sequence, so every test in
test_sonify.py runs offline against a rendered buffer with no audio device and
no camera -- the same arrangement that lets test_pose3d.py measure the solver
without a webcam.

Two things here are less obvious than they look.

THE RATE MISMATCH. The video loop hands over a new angle every ~33ms; the
audio callback needs samples every ~5ms. Assigning a new frequency to an
oscillator once per video frame produces a 30Hz zipper on the parameter and a
click on every phase discontinuity. So phase is accumulated and never reset --
a frequency change alters the rate phase advances, not phase itself -- and the
frequency glides toward its target *per sample* inside the callback, as a
one-pole at `smoothing_hz`. Where that knob wants to sit is an open question
the spike exists to answer: the tracker's ~2.3 degree knee error lands at
roughly 50 cents under the default mapping, which is half a semitone and
plainly audible, so some smoothing is needed, and every millisecond of it is
spent against the sub-100ms premise the product rests on.

WHY THE BEAT PAIR IS UNBALANCED. Two equal-amplitude tones at the same
frequency sum to 2A*cos(dphi/2), which is silence when they land out of phase.
Beat mode's whole cue is the beating slowing to a stop at the target, and if
it happened to stop at a null the user would hear the app die at exactly the
moment they got it right. Unequal amplitudes (0.6 and 0.35) bound the sum away
from zero while leaving the modulation obvious.
"""

import math
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

# Beat mode's carrier sits below the pitch range rather than inside it. Put it
# at, say, 300Hz and it collides with the sweeping pitch voice whenever the
# subject passes through that angle, and the two channels become one
# unreadable cluster. Below the range they stay separable by ear and in a
# spectrum.
BEAT_CARRIER_HZ = 120.0

# Beating is perceived as beating up to roughly 20Hz of difference; past that
# the ear resolves two separate pitches and the fine channel stops carrying
# anything. 1Hz per degree puts the useful +/-20 degrees inside that window.
BEAT_HZ_PER_DEG = 1.0
BEAT_MAX_HZ = 20.0

# How hard the "near" curve concentrates resolution at the target. This is the
# ratio of its cents-per-degree at the target to the linear curve's.
NEAR_CONCENTRATION = 2.5

# Voice amplitudes before normalization. See the module docstring for why the
# beat pair is deliberately unequal.
AMP_PITCH = 1.0
AMP_BEAT_DRONE = 0.6
AMP_BEAT_MOVING = 0.35
AMP_FIFTH_DRONE = 0.5

GATE_RAMP_MS = 5.0

CURVES = ("log", "near")
SIGNALS = ("pitch", "beat", "fifth")


class Config(NamedTuple):
    """
    Everything the mapping and synthesis need. A NamedTuple so tests and
    keybindings can say `cfg._replace(curve="near")` without mutating shared
    state under a running audio callback.
    """

    target_deg: float
    top_deg: float = 175.0
    bottom_deg: float = 70.0
    f_low: float = 200.0
    f_high: float = 800.0
    curve: str = "log"
    signal: str = "pitch"
    smoothing_hz: float = 8.0
    samplerate: int = 48000
    # 128 measures 19.3ms of output latency against 22.0 at 256, and the
    # device floors at 18.0 by 64 -- there is nothing left below that but
    # callback rate. 128 takes most of the saving at half the wakeups of 64,
    # which matters because MediaPipe is competing for the same cores.
    blocksize: int = 128
    gain: float = 0.4


# --------------------------------------------------------------------------
# Mapping
# --------------------------------------------------------------------------

def span_cents(cfg: Config) -> float:
    """Total pitch travel between bottom_deg and top_deg, in cents."""
    return 1200.0 * math.log2(cfg.f_high / cfg.f_low)


def cents_to_hz(cents: float, f_ref: float) -> float:
    return f_ref * (2.0 ** (cents / 1200.0))


def angle_to_cents(cfg: Config, angle_deg: float) -> float:
    """
    Cents above `f_low`, zero at bottom_deg and span_cents at top_deg.

    Pitch is perceived logarithmically, so the "linear" curve is linear in
    cents rather than in Hz -- mapped linearly in Hz, the same number of
    degrees sounds like a much smaller interval at the top of the range than
    at the bottom.

    The "near" curve warps that mapping to concentrate resolution at the
    target. It is built as two power curves meeting at the target, which fixes
    both endpoints *and* the target itself: switching curve mid-set must not
    move the note you just spent five minutes learning by ear.
    """
    lo, hi = cfg.bottom_deg, cfg.top_deg
    u = (angle_deg - lo) / (hi - lo)
    u = min(1.0, max(0.0, u))

    if cfg.curve == "log":
        w = u
    elif cfg.curve == "near":
        ut = (cfg.target_deg - lo) / (hi - lo)
        ut = min(1.0 - 1e-6, max(1e-6, ut))
        p = NEAR_CONCENTRATION
        if u <= ut:
            w = ut * (u / ut) ** p
        else:
            s = (u - ut) / (1.0 - ut)
            w = ut + (1.0 - ut) * (1.0 - (1.0 - s) ** p)
    else:
        raise ValueError(f"unknown curve {cfg.curve!r}, expected one of "
                         f"{CURVES}")

    return w * span_cents(cfg)


def angle_to_hz(cfg: Config, angle_deg: float) -> float:
    return cents_to_hz(angle_to_cents(cfg, angle_deg), cfg.f_low)


def beat_delta_hz(cfg: Config, angle_deg: float) -> float:
    """Frequency offset of the moving carrier from the drone. Zero at target."""
    delta = (angle_deg - cfg.target_deg) * BEAT_HZ_PER_DEG
    return max(-BEAT_MAX_HZ, min(BEAT_MAX_HZ, delta))


def voice_frequencies(cfg: Config, angle_deg: float
                      ) -> List[Tuple[float, float]]:
    """
    The oscillators to sound for this angle, as (hz, amplitude).

    The pitch voice is always present and always first. `beat` and `fifth` add
    a target-lock layer *on top of* it rather than replacing it: beating is
    only audible within about 20Hz, so a beat-only mode would say nothing at
    all about depth until the subject was already nearly correct, which is
    precisely when they least need telling.
    """
    voices = [(angle_to_hz(cfg, angle_deg), AMP_PITCH)]

    if cfg.signal == "pitch":
        pass
    elif cfg.signal == "beat":
        delta = beat_delta_hz(cfg, angle_deg)
        voices.append((BEAT_CARRIER_HZ, AMP_BEAT_DRONE))
        voices.append((BEAT_CARRIER_HZ + delta, AMP_BEAT_MOVING))
    elif cfg.signal == "fifth":
        # A perfect fifth below the target note, so arriving at target lands a
        # 3:2 against the drone.
        voices.append((angle_to_hz(cfg, cfg.target_deg) * 2.0 / 3.0,
                       AMP_FIFTH_DRONE))
    else:
        raise ValueError(f"unknown signal {cfg.signal!r}, expected one of "
                         f"{SIGNALS}")

    return voices


# --------------------------------------------------------------------------
# Synthesis
# --------------------------------------------------------------------------

MAX_VOICES = 3


class Engine:
    """
    Phase-continuous additive synthesis with per-sample frequency glide.

    Holds one phase accumulator per voice slot, kept across blocks and across
    frequency changes. Slot order is fixed by voice_frequencies, so the pitch
    voice always keeps its phase even when the target-lock layer is switched
    underneath it.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._phase = np.zeros(MAX_VOICES)
        self._freq = np.zeros(MAX_VOICES)
        self._amp = np.zeros(MAX_VOICES)
        self._gate = 0.0
        self._primed = False

    def _glide_alpha(self) -> float:
        return 1.0 - math.exp(
            -2.0 * math.pi * self.cfg.smoothing_hz / self.cfg.samplerate)

    def render(self, angle_deg: Optional[float], n: int) -> np.ndarray:
        """
        Produce `n` samples for a single held angle. `angle_deg` is None when
        the joint is not tracked, which gates the output to silence.
        """
        cfg = self.cfg
        sr = cfg.samplerate

        tracked = angle_deg is not None
        if tracked:
            voices = voice_frequencies(cfg, angle_deg)
            target_f = np.zeros(MAX_VOICES)
            target_a = np.zeros(MAX_VOICES)
            total = sum(a for _, a in voices)
            for i, (f, a) in enumerate(voices):
                target_f[i] = f
                target_a[i] = a / total
            if not self._primed:
                # First tracked block: start at pitch rather than gliding up
                # from zero, which would sound like a siren on every set.
                self._freq = target_f.copy()
                self._primed = True
        else:
            target_f = self._freq.copy()
            target_a = self._amp.copy()

        # Gate as a linear ramp, not a one-pole: a one-pole approaches zero
        # without reaching it, and "almost silent" during a tracking dropout
        # is a faint whine that the user will notice and mistrust.
        # Vectorized rather than looped: this runs inside the audio callback,
        # and a per-sample Python loop there is ~48k interpreter iterations a
        # second competing with a hard 5ms deadline. Clipping to [0, 1] is
        # exactly the clamp, since the gate only ever targets 0 or 1.
        ramp_len = max(1.0, GATE_RAMP_MS * 1e-3 * sr)
        direction = 1.0 if tracked else -1.0
        gate = np.clip(
            self._gate + direction * np.arange(1, n + 1) / ramp_len, 0.0, 1.0)
        self._gate = float(gate[-1])

        alpha = self._glide_alpha()
        out = np.zeros(n)
        two_pi = 2.0 * math.pi

        for i in range(MAX_VOICES):
            if target_a[i] == 0.0 and self._amp[i] == 0.0:
                continue

            # Per-sample one-pole glide on frequency. Vectorized as a closed
            # form rather than a loop: after k samples the frequency has
            # covered 1-(1-alpha)^k of the distance to its target.
            k = np.arange(1, n + 1)
            decay = (1.0 - alpha) ** k
            freq = target_f[i] + (self._freq[i] - target_f[i]) * decay
            amp = target_a[i] + (self._amp[i] - target_a[i]) * decay

            phase = self._phase[i] + two_pi * np.cumsum(freq) / sr
            out += amp * np.sin(phase)

            self._phase[i] = math.fmod(phase[-1], two_pi)
            self._freq[i] = freq[-1]
            self._amp[i] = amp[-1]

        return out * gate * cfg.gain


def render(cfg: Config, angles, frames_per_angle: int) -> np.ndarray:
    """Render a whole angle sequence offline. Used by the tests."""
    engine = Engine(cfg)
    return np.concatenate(
        [engine.render(a, frames_per_angle) for a in angles])


class Sonifier:
    """
    Live audio output driven by one angle per video frame.

    The video thread calls `update()` at ~30fps; PortAudio calls `_callback()`
    at ~190Hz from its own thread. They share exactly two pieces of mutable
    state -- the latest angle and the config -- and neither is locked. A lock
    held in an audio callback is a dropout waiting for a slow frame, and
    neither shared value needs one: the angle is a single float rebind, and
    the config is a NamedTuple replaced wholesale, so the callback sees either
    the old config or the new one and never a half-applied mixture. A
    one-frame-stale angle is 33ms old and inaudible.
    """

    def __init__(self, cfg: Config, device=None):
        self.cfg = cfg
        self.device = device
        self.muted = False
        self._engine = Engine(cfg)
        self._angle: Optional[float] = None
        self._stream = None

    # -- parameters, all safe to call from the video thread ----------------

    def update(self, angle_deg: Optional[float]) -> None:
        """One angle per video frame. None means the joint is not tracked."""
        self._angle = angle_deg

    def _apply(self, **kw) -> None:
        cfg = self.cfg._replace(**kw)
        self.cfg = cfg
        self._engine.cfg = cfg

    def set_target(self, angle_deg: float) -> None:
        self._apply(target_deg=float(angle_deg))

    def set_curve(self, name: str) -> None:
        if name not in CURVES:
            raise ValueError(f"unknown curve {name!r}, expected one of "
                             f"{CURVES}")
        self._apply(curve=name)

    def set_signal(self, name: str) -> None:
        if name not in SIGNALS:
            raise ValueError(f"unknown signal {name!r}, expected one of "
                             f"{SIGNALS}")
        self._apply(signal=name)

    def set_smoothing(self, hz: float) -> None:
        self._apply(smoothing_hz=max(0.5, float(hz)))

    # -- readouts for the HUD ----------------------------------------------

    @property
    def current_hz(self) -> Optional[float]:
        if self._angle is None:
            return None
        return angle_to_hz(self.cfg, self._angle)

    @property
    def output_latency_ms(self) -> Optional[float]:
        if self._stream is None:
            return None
        return self._stream.latency * 1000.0

    # -- stream ------------------------------------------------------------

    def _callback(self, outdata, frames, time_info, status) -> None:
        angle = None if self.muted else self._angle
        outdata[:, 0] = self._engine.render(angle, frames)

    def start(self) -> "Sonifier":
        import sounddevice as sd
        self._stream = sd.OutputStream(
            samplerate=self.cfg.samplerate,
            blocksize=self.cfg.blocksize,
            channels=1,
            dtype="float32",
            device=self.device,
            latency="low",
            callback=self._callback,
        )
        self._stream.start()
        return self

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self) -> "Sonifier":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
