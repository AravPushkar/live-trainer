"""
Offline tests for the audio spike: reps.py and sonify.py.

Nothing here opens an audio device or a camera. Synthesis is rendered into a
buffer and inspected, the same way test_pose3d.py measures the solver against
simulated input rather than a webcam.

    python test_sonify.py
    python -m unittest test_sonify -v
"""

import unittest

import numpy as np

import reps
import sonify
import tune_smoothing as tune
from reps import BottomDetector
from sonify import Config


def squat_trace(bottom_deg, frames_down=15, frames_up=15, top_deg=175.0,
                hold=0):
    """One squat, top -> bottom -> top, as a list of knee angles."""
    down = np.linspace(top_deg, bottom_deg, frames_down)
    flat = np.full(hold, bottom_deg)
    up = np.linspace(bottom_deg, top_deg, frames_up)
    return list(down) + list(flat) + list(up)


def run(detector, angles, dt=1 / 30):
    """Feed a trace through a detector, collect the events it emits."""
    events = []
    for i, a in enumerate(angles):
        ev = detector.update(a, i * dt)
        if ev is not None:
            events.append(ev)
    return events


class TestBottomDetector(unittest.TestCase):

    def test_counts_each_rep_in_a_clean_trace(self):
        d = BottomDetector()
        angles = squat_trace(85.0) * 3
        events = run(d, angles)
        self.assertEqual(len(events), 3)
        self.assertEqual([e.index for e in events], [0, 1, 2])

    def test_reports_the_lowest_angle_reached(self):
        d = BottomDetector()
        events = run(d, squat_trace(85.0))
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].bottom_deg, 85.0, places=6)

    def test_hovering_at_the_threshold_emits_nothing(self):
        # Sits astride descend_deg=140 without ever standing back up past
        # complete_deg=160. Without hysteresis this rattles and emits reps.
        d = BottomDetector()
        angles = [175.0] + [139.0, 141.0] * 40
        self.assertEqual(run(d, angles), [])

    def test_lost_tracking_mid_rep_discards_that_rep(self):
        d = BottomDetector()
        angles = squat_trace(85.0)
        angles[len(angles) // 2] = None
        self.assertEqual(run(d, angles), [])

    def test_recovers_after_lost_tracking(self):
        d = BottomDetector()
        broken = squat_trace(85.0)
        broken[len(broken) // 2] = None
        events = run(d, broken + squat_trace(85.0))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].index, 0)

    def test_rejects_reps_shorter_than_min_frames(self):
        d = BottomDetector(min_frames=5)
        # Down and up in two frames: a keypoint glitch, not a squat.
        self.assertEqual(run(d, [175.0, 100.0, 175.0]), [])

    def test_does_not_count_a_squat_already_in_progress_at_startup(self):
        # Detector comes up mid-descent. Counting the ascent as a rep would
        # log a bottom angle that was never actually the bottom.
        d = BottomDetector()
        partial = list(np.linspace(120.0, 85.0, 8)) + list(
            np.linspace(85.0, 175.0, 15))
        events = run(d, partial)
        self.assertEqual(events, [])

    def test_arms_after_the_first_full_standing_position(self):
        d = BottomDetector()
        partial = list(np.linspace(120.0, 85.0, 8)) + list(
            np.linspace(85.0, 175.0, 15))
        events = run(d, partial + squat_trace(90.0))
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].bottom_deg, 90.0, places=6)


CFG = Config(target_deg=90.0)


def peaks(signal, samplerate, floor_frac=0.15):
    """Frequencies of spectral peaks above a fraction of the tallest peak."""
    mag = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
    freqs = np.fft.rfftfreq(len(signal), 1 / samplerate)
    keep = mag > floor_frac * mag.max()
    # Collapse adjacent bins into one peak each.
    out, prev = [], -10
    for i in np.nonzero(keep)[0]:
        if i != prev + 1:
            out.append(freqs[i])
        prev = i
    return out


class TestMapping(unittest.TestCase):

    def test_endpoints_span_the_full_frequency_range(self):
        for curve in ("log", "near"):
            cfg = CFG._replace(curve=curve)
            self.assertAlmostEqual(sonify.angle_to_hz(cfg, cfg.bottom_deg),
                                   cfg.f_low, places=6)
            self.assertAlmostEqual(sonify.angle_to_hz(cfg, cfg.top_deg),
                                   cfg.f_high, places=6)

    def test_pitch_rises_with_angle(self):
        for curve in ("log", "near"):
            cfg = CFG._replace(curve=curve)
            hz = [sonify.angle_to_hz(cfg, a)
                  for a in np.linspace(cfg.bottom_deg, cfg.top_deg, 200)]
            self.assertTrue(all(b > a for a, b in zip(hz, hz[1:])),
                            f"{curve} curve is not monotonic")

    def test_both_curves_agree_at_the_target(self):
        # If they disagreed, switching curve mid-set would move the target and
        # silently invalidate the depth you had just learned by ear.
        a = sonify.angle_to_hz(CFG._replace(curve="log"), CFG.target_deg)
        b = sonify.angle_to_hz(CFG._replace(curve="near"), CFG.target_deg)
        self.assertAlmostEqual(a, b, places=6)

    def test_near_curve_has_finer_resolution_at_the_target(self):
        def slope(curve):
            cfg = CFG._replace(curve=curve)
            d = 0.05
            lo = sonify.angle_to_cents(cfg, cfg.target_deg - d)
            hi = sonify.angle_to_cents(cfg, cfg.target_deg + d)
            return (hi - lo) / (2 * d)
        self.assertGreater(slope("near"), 1.5 * slope("log"))

    def test_near_curve_is_coarser_far_from_the_target(self):
        # Concentrating resolution somewhere means removing it elsewhere.
        def slope(curve, angle):
            cfg = CFG._replace(curve=curve)
            d = 0.05
            return (sonify.angle_to_cents(cfg, angle + d)
                    - sonify.angle_to_cents(cfg, angle - d)) / (2 * d)
        self.assertLess(slope("near", 170.0), slope("log", 170.0))

    def test_cents_to_hz_is_an_octave_per_1200_cents(self):
        self.assertAlmostEqual(sonify.cents_to_hz(0, 200.0), 200.0, places=9)
        self.assertAlmostEqual(sonify.cents_to_hz(1200, 200.0), 400.0,
                               places=9)


class TestBeatLayer(unittest.TestCase):

    def test_beat_delta_is_zero_at_the_target(self):
        cfg = CFG._replace(signal="beat")
        self.assertAlmostEqual(sonify.beat_delta_hz(cfg, cfg.target_deg), 0.0,
                               places=9)

    def test_beat_delta_is_signed_by_which_side_of_target(self):
        cfg = CFG._replace(signal="beat")
        self.assertGreater(sonify.beat_delta_hz(cfg, cfg.target_deg + 10), 0)
        self.assertLess(sonify.beat_delta_hz(cfg, cfg.target_deg - 10), 0)

    def test_beat_delta_is_clamped_into_the_audible_beating_range(self):
        # Past roughly 20Hz the two tones stop beating and are heard as two
        # separate pitches, so the fine channel carries nothing beyond that.
        cfg = CFG._replace(signal="beat")
        far = sonify.beat_delta_hz(cfg, cfg.top_deg)
        self.assertLessEqual(abs(far), sonify.BEAT_MAX_HZ + 1e-9)

    def test_beat_pair_is_co_frequent_at_the_target(self):
        cfg = CFG._replace(signal="beat")
        voices = sonify.voice_frequencies(cfg, cfg.target_deg)
        carriers = sorted(f for f, a in voices if f != sonify.angle_to_hz(
            cfg, cfg.target_deg))
        self.assertEqual(len(carriers), 2)
        self.assertAlmostEqual(carriers[0], carriers[1], places=9)

    def test_beat_pair_has_unequal_amplitudes_so_it_cannot_null(self):
        # Two equal tones at the same frequency and opposing phase cancel to
        # silence. The beat freezing at a null is indistinguishable from the
        # app crashing, so the pair is deliberately unbalanced.
        cfg = CFG._replace(signal="beat")
        voices = sonify.voice_frequencies(cfg, cfg.target_deg)
        pitch_hz = sonify.angle_to_hz(cfg, cfg.target_deg)
        amps = [a for f, a in voices if f != pitch_hz]
        self.assertNotAlmostEqual(amps[0], amps[1], places=3)

    def test_fifth_signal_lands_on_a_three_to_two_ratio_at_target(self):
        cfg = CFG._replace(signal="fifth")
        voices = sonify.voice_frequencies(cfg, cfg.target_deg)
        self.assertEqual(len(voices), 2)
        hi, lo = sorted((f for f, a in voices), reverse=True)
        self.assertAlmostEqual(hi / lo, 1.5, places=6)


class TestSynthesis(unittest.TestCase):

    def test_a_step_change_in_angle_produces_no_click(self):
        # A click is a phase discontinuity. At 800Hz and 48kHz the largest
        # legitimate sample-to-sample step is about 0.105 of full scale.
        cfg = CFG._replace(signal="pitch")
        out = sonify.render(cfg, [175.0, 85.0, 175.0], frames_per_angle=4800)
        self.assertLess(np.abs(np.diff(out)).max(), 0.15)

    def test_untracked_frames_fade_to_exact_silence(self):
        cfg = CFG._replace(signal="pitch")
        out = sonify.render(cfg, [120.0, None, None], frames_per_angle=4800)
        tail = out[-2400:]
        self.assertEqual(np.abs(tail).max(), 0.0)

    def test_the_gate_ramps_rather_than_cutting(self):
        cfg = CFG._replace(signal="pitch")
        out = sonify.render(cfg, [120.0, None], frames_per_angle=4800)
        self.assertLess(np.abs(np.diff(out)).max(), 0.15)

    def test_rendered_spectrum_matches_the_declared_voices(self):
        cfg = CFG._replace(signal="pitch", smoothing_hz=1000.0)
        out = sonify.render(cfg, [120.0] * 4, frames_per_angle=12000)
        found = peaks(out[-32768:], cfg.samplerate)
        expected = sonify.angle_to_hz(cfg, 120.0)
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0], expected, delta=5.0)

    def test_beating_at_the_target_is_not_silent(self):
        cfg = CFG._replace(signal="beat", smoothing_hz=1000.0)
        out = sonify.render(cfg, [cfg.target_deg] * 4, frames_per_angle=12000)
        self.assertGreater(np.sqrt(np.mean(out[-24000:] ** 2)), 0.1)

    def test_off_target_renders_two_carriers_a_beat_apart(self):
        cfg = CFG._replace(signal="beat", smoothing_hz=1000.0)
        angle = cfg.target_deg + 15.0
        out = sonify.render(cfg, [angle] * 4, frames_per_angle=24000)
        found = peaks(out[-65536:], cfg.samplerate)
        carriers = sorted(f for f in found
                          if abs(f - sonify.BEAT_CARRIER_HZ) < 40.0)
        self.assertEqual(len(carriers), 2)
        self.assertAlmostEqual(carriers[1] - carriers[0],
                               abs(sonify.beat_delta_hz(cfg, angle)),
                               delta=2.0)

    def test_output_never_clips(self):
        cfg = CFG._replace(signal="beat")
        angles = list(np.linspace(175.0, 70.0, 60))
        out = sonify.render(cfg, angles, frames_per_angle=1600)
        self.assertLessEqual(np.abs(out).max(), 1.0)


class TestKneeSignal(unittest.TestCase):

    def test_averages_both_knees_when_both_are_tracked(self):
        self.assertAlmostEqual(
            reps.knee_signal({"L knee": 100.0, "R knee": 110.0}), 105.0)

    def test_falls_back_to_whichever_knee_is_tracked(self):
        self.assertAlmostEqual(
            reps.knee_signal({"L knee": None, "R knee": 110.0}), 110.0)
        self.assertAlmostEqual(
            reps.knee_signal({"L knee": 100.0, "R knee": None}), 100.0)

    def test_gates_when_neither_knee_is_tracked(self):
        self.assertIsNone(reps.knee_signal({"L knee": None, "R knee": None}))

    def test_gates_when_the_keys_are_absent(self):
        self.assertIsNone(reps.knee_signal({}))

    def test_a_single_knee_does_not_silently_halve_the_angle(self):
        # Summing and dividing by two regardless of how many knees were
        # present would report a 55 degree knee for a subject standing at 110
        # with one leg occluded -- deep squat, from standing still.
        one = reps.knee_signal({"L knee": None, "R knee": 110.0})
        self.assertGreater(one, 100.0)


class TestSonifier(unittest.TestCase):
    """
    Drives the audio callback directly with a buffer. Constructing a Sonifier
    must not open a device, or none of this runs on a headless machine.
    """

    def drive(self, son, blocks=8, frames=256):
        out = np.empty((frames, 1), dtype=np.float32)
        chunks = []
        for _ in range(blocks):
            son._callback(out, frames, None, None)
            chunks.append(out[:, 0].copy())
        return np.concatenate(chunks)

    def test_constructing_does_not_open_a_device(self):
        son = sonify.Sonifier(CFG)
        self.assertIsNone(son._stream)

    def test_callback_is_silent_before_any_angle_arrives(self):
        son = sonify.Sonifier(CFG)
        self.assertEqual(np.abs(self.drive(son)).max(), 0.0)

    def test_callback_sounds_once_an_angle_arrives(self):
        son = sonify.Sonifier(CFG)
        son.update(120.0)
        self.assertGreater(np.abs(self.drive(son)).max(), 0.05)

    def test_losing_tracking_returns_to_silence(self):
        son = sonify.Sonifier(CFG)
        son.update(120.0)
        self.drive(son)
        son.update(None)
        self.assertEqual(np.abs(self.drive(son)[-256:]).max(), 0.0)

    def test_muting_silences_without_discarding_the_angle(self):
        son = sonify.Sonifier(CFG)
        son.update(120.0)
        self.drive(son)
        son.muted = True
        self.assertEqual(np.abs(self.drive(son)[-256:]).max(), 0.0)
        son.muted = False
        self.assertGreater(np.abs(self.drive(son)).max(), 0.05)

    def test_setting_the_target_moves_the_note_it_lands_on(self):
        son = sonify.Sonifier(CFG._replace(curve="near"))
        before = sonify.angle_to_hz(son.cfg, 110.0)
        son.set_target(110.0)
        after = sonify.angle_to_hz(son.cfg, 110.0)
        self.assertNotAlmostEqual(before, after, places=3)

    def test_config_changes_reach_the_engine(self):
        # The engine holds its own reference to cfg; if set_signal updated
        # only the Sonifier's copy the audio would silently ignore every
        # keypress during Phase A.
        son = sonify.Sonifier(CFG)
        son.set_signal("beat")
        self.assertEqual(son._engine.cfg.signal, "beat")

    def test_rejects_an_unknown_curve(self):
        son = sonify.Sonifier(CFG)
        with self.assertRaises(ValueError):
            son.set_curve("bezier")
        self.assertEqual(son.cfg.curve, "log")

    def test_rejects_an_unknown_signal(self):
        son = sonify.Sonifier(CFG)
        with self.assertRaises(ValueError):
            son.set_signal("kazoo")
        self.assertEqual(son.cfg.signal, "pitch")

    def test_reports_the_frequency_currently_sounding(self):
        son = sonify.Sonifier(CFG)
        self.assertIsNone(son.current_hz)
        son.update(120.0)
        self.assertAlmostEqual(son.current_hz,
                               sonify.angle_to_hz(CFG, 120.0), places=6)


class TestFullChain(unittest.TestCase):
    """
    The path skeleton.py actually runs: an angles dict per frame -> knee
    signal -> rep detection and synthesis. Each module is correct on its own
    above; this is about the contract between them.
    """

    def chain(self, angle_stream, cfg):
        detector = BottomDetector()
        engine = sonify.Engine(cfg)
        events, audio = [], []
        for i, angles in enumerate(angle_stream):
            knee = reps.knee_signal(angles)
            ev = detector.update(knee, i / 30.0)
            if ev is not None:
                events.append(ev)
            audio.append(engine.render(knee, 1600))   # one video frame @30fps
        return events, np.concatenate(audio)

    def both_knees(self, trace, jitter=0.0, seed=0):
        rng = np.random.default_rng(seed)
        out = []
        for a in trace:
            if a is None:
                out.append({"L knee": None, "R knee": None})
            else:
                out.append({"L knee": a + rng.normal(0, jitter),
                            "R knee": a + rng.normal(0, jitter)})
        return out

    def test_a_clean_set_counts_reps_and_makes_continuous_audio(self):
        cfg = CFG._replace(signal="beat", curve="near")
        stream = self.both_knees(squat_trace(88.0, hold=5) * 5)
        events, audio = self.chain(stream, cfg)
        self.assertEqual(len(events), 5)
        self.assertLess(np.abs(np.diff(audio)).max(), 0.15)
        self.assertLessEqual(np.abs(audio).max(), 1.0)

    def test_occlusion_dropouts_gate_the_tone_and_drop_the_rep(self):
        trace = squat_trace(88.0, hold=5)
        trace[len(trace) // 2] = None
        stream = self.both_knees(trace + squat_trace(88.0, hold=5))
        events, audio = self.chain(stream, CFG)
        self.assertEqual(len(events), 1)
        self.assertLess(np.abs(np.diff(audio)).max(), 0.15)

    def test_tracker_grade_jitter_does_not_manufacture_reps(self):
        # 2.3 deg is the solver's measured knee error against synthetic
        # ground truth. If that alone walks the state machine, every number
        # Phase B produces is noise.
        cfg = CFG
        stream = self.both_knees(squat_trace(88.0, hold=5) * 5, jitter=2.3,
                                 seed=7)
        events, _ = self.chain(stream, cfg)
        self.assertEqual(len(events), 5)

    def test_standing_still_under_jitter_emits_no_reps(self):
        stream = self.both_knees([174.0] * 600, jitter=2.3, seed=3)
        events, _ = self.chain(stream, CFG)
        self.assertEqual(events, [])


class TestSmoothingModel(unittest.TestCase):
    """
    The sweep in tune_smoothing.py models the audio glide once per video
    frame instead of once per sample. If that model drifts from what Engine
    actually does, every number the sweep prints is about a filter that does
    not exist.
    """

    def test_frame_rate_model_matches_the_real_engine(self):
        cfg = CFG._replace(smoothing_hz=6.0, signal="pitch")
        angles = [175.0] * 3 + [88.0] * 20 + [175.0] * 10
        target_hz = [sonify.angle_to_hz(cfg, a) for a in angles]

        engine = sonify.Engine(cfg)
        actual = []
        for a in angles:
            engine.render(a, 1600)          # one 30fps video frame at 48kHz
            actual.append(engine._freq[0])

        model = tune.smooth_hz(target_hz, cfg.smoothing_hz, fps=30.0)
        np.testing.assert_allclose(actual, model, rtol=1e-6)

    def test_smoothing_is_applied_in_hz_not_cents(self):
        # Engine glides frequency in Hz. Gliding cents instead follows a
        # different curve through the same step, so the sweep has to model
        # the one the engine actually implements.
        hz = np.array([200.0] * 2 + [800.0] * 10)
        in_hz = tune.smooth_hz(hz, 6.0, fps=30.0)
        cents = 1200.0 * np.log2(hz / 200.0)
        in_cents = np.array([sonify.cents_to_hz(c, 200.0)
                             for c in tune.smooth_hz(cents, 6.0, fps=30.0)])
        divergence = np.abs(1200.0 * np.log2(in_hz / in_cents)).max()
        self.assertGreater(divergence, 100.0)

    def test_no_smoothing_passes_the_signal_through(self):
        hz = [200.0, 400.0, 300.0]
        np.testing.assert_allclose(
            tune.smooth_hz(hz, 1e6, fps=30.0), hz, rtol=1e-6)


class TestGapFilling(unittest.TestCase):
    """
    Untracked frames cannot simply be dropped -- cross-correlation needs
    uniformly spaced samples, and deleting interior frames silently changes
    what a "frame of lag" means. They also cannot be replaced with a constant:
    substituting 175 degrees for a dropout injects a spike the tracker never
    saw, and measured lag collapsed from 88ms to 35ms purely from that.
    """

    def test_leaves_a_complete_trace_untouched(self):
        x = [100.0, 110.0, 120.0]
        np.testing.assert_allclose(tune.fill_gaps(x), x)

    def test_interpolates_across_an_interior_gap(self):
        got = tune.fill_gaps([100.0, None, None, 130.0])
        np.testing.assert_allclose(got, [100.0, 110.0, 120.0, 130.0])

    def test_holds_the_first_valid_value_back_across_a_leading_gap(self):
        np.testing.assert_allclose(
            tune.fill_gaps([None, None, 120.0, 130.0]),
            [120.0, 120.0, 120.0, 130.0])

    def test_holds_the_last_valid_value_across_a_trailing_gap(self):
        np.testing.assert_allclose(
            tune.fill_gaps([100.0, 110.0, None]), [100.0, 110.0, 110.0])

    def test_refuses_a_trace_with_nothing_in_it(self):
        with self.assertRaises(ValueError):
            tune.fill_gaps([None, None])

    def test_interpolation_does_not_shift_measured_lag(self):
        # The property that matters: punching holes in both signals and
        # filling them must not move the lag estimate.
        t = np.arange(600) / 30.0
        x = 130.0 - 45.0 * np.cos(2 * np.pi * t / 3.0)
        y = 130.0 - 45.0 * np.cos(2 * np.pi * (t - 3.0 / 30.0) / 3.0)
        holed_x = [None if 100 <= i < 108 else v for i, v in enumerate(x)]
        holed_y = [None if 100 <= i < 108 else v for i, v in enumerate(y)]
        lag = tune.estimate_lag_frames(tune.fill_gaps(holed_y),
                                       tune.fill_gaps(holed_x))
        self.assertAlmostEqual(lag, 3.0, delta=0.1)

    def test_constant_substitution_would_corrupt_the_lag(self):
        # Guards the reasoning above: this is the approach fill_gaps replaced.
        t = np.arange(600) / 30.0
        x = 130.0 - 45.0 * np.cos(2 * np.pi * t / 3.0)
        y = 130.0 - 45.0 * np.cos(2 * np.pi * (t - 3.0 / 30.0) / 3.0)
        bad_x = np.array([175.0 if 100 <= i < 140 else v
                          for i, v in enumerate(x)])
        bad_y = np.array([175.0 if 100 <= i < 140 else v
                          for i, v in enumerate(y)])
        self.assertLess(tune.estimate_lag_frames(bad_y, bad_x), 2.5)


class TestHighFrequencyJitter(unittest.TestCase):
    """
    Total RMS error is the wrong measure for warble: it is dominated by slow
    bias, which a listener hears as the tone sitting slightly off rather than
    as wobble. This isolates the fast part.
    """

    def test_a_constant_offset_is_not_jitter(self):
        self.assertAlmostEqual(tune.hf_jitter(np.full(200, 3.0)), 0.0,
                               places=6)

    def test_a_slow_drift_is_not_jitter(self):
        self.assertLess(tune.hf_jitter(np.linspace(0, 10, 400)), 0.05)

    def test_frame_to_frame_wobble_is_jitter(self):
        alternating = np.array([1.0, -1.0] * 200)
        self.assertGreater(tune.hf_jitter(alternating), 0.5)

    def test_more_wobble_reads_as_more_jitter(self):
        rng = np.random.default_rng(0)
        n = 2000
        small = tune.hf_jitter(rng.normal(0, 1.0, n))
        large = tune.hf_jitter(rng.normal(0, 3.0, n))
        self.assertGreater(large, 2.0 * small)


class TestLagEstimator(unittest.TestCase):

    def signal(self, n=600):
        t = np.arange(n) / 30.0
        return 130.0 - 45.0 * np.cos(2 * np.pi * t / 3.0)

    def test_recovers_a_known_whole_frame_delay(self):
        t = np.arange(600) / 30.0
        x = 130.0 - 45.0 * np.cos(2 * np.pi * t / 3.0)
        for delay in (1, 3, 7):
            y = 130.0 - 45.0 * np.cos(2 * np.pi * (t - delay / 30.0) / 3.0)
            self.assertAlmostEqual(tune.estimate_lag_frames(y, x), delay,
                                   delta=0.05, msg=f"delay {delay}")

    def test_recovers_a_fractional_delay(self):
        n = 600
        t = np.arange(n) / 30.0
        x = 130.0 - 45.0 * np.cos(2 * np.pi * t / 3.0)
        y = 130.0 - 45.0 * np.cos(2 * np.pi * (t - 2.5 / 30.0) / 3.0)
        self.assertAlmostEqual(tune.estimate_lag_frames(y, x), 2.5, delta=0.05)

    def test_reports_zero_lag_for_an_undelayed_signal(self):
        x = self.signal()
        self.assertAlmostEqual(tune.estimate_lag_frames(x, x), 0.0, delta=0.1)

    def test_noise_alone_does_not_masquerade_as_lag(self):
        rng = np.random.default_rng(0)
        x = self.signal()
        y = x + rng.normal(0, 2.0, x.shape)
        self.assertAlmostEqual(tune.estimate_lag_frames(y, x), 0.0, delta=0.4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
