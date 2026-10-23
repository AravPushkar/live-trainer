"""
Bottom-of-rep detection from a single joint-angle stream.

This exists to serve the audio spike's measurement phase, which has to answer
"how close to target was the bottom of rep 3" and cannot do that without
knowing where rep 3 was. It is deliberately not the form-evaluation state
machine the product needs: one signal, one exercise, one event.

The whole design problem is a noisy signal near a threshold. A single
threshold at 150 degrees, on a knee angle that jitters by a couple of degrees,
emits a rep every time the noise crosses it -- so the trigger to start
descending (140) is separate from the trigger to complete (160), and the
signal has to travel the full 20 degrees between them to advance. That gap is
far wider than the tracker's ~2.3 degree knee error, so noise alone cannot
walk a rep through the machine.

Lost tracking is treated as poisoning the rep in flight rather than pausing
it. A rep whose bottom fell inside a tracking gap has an unknown bottom, and
logging the lowest angle that happened to be *seen* would quietly report a
number that was never the bottom. Those reps are dropped.

The same reasoning applies at startup, which is why the machine begins in
LOST rather than TOP: if it comes up while the subject is already halfway
down, the first thing it sees is an ascent, and counting that as a rep logs a
bottom that was never observed. It arms only after seeing a full standing
position.
"""

from typing import NamedTuple, Optional


class RepEvent(NamedTuple):
    """One completed repetition."""

    index: int          # 0-based, counting only completed reps
    bottom_deg: float   # lowest angle observed during the descent
    t: float            # timestamp of the frame that completed the rep


# States. LOST means "waiting for a clean standing position before trusting
# anything", and is both the startup state and where tracking failures land.
_LOST = "lost"
_TOP = "top"
_DESCENDING = "descending"


class BottomDetector:
    """
    Emits one RepEvent per completed repetition.

    `complete_deg` is the rep-completion threshold, not the top of any pitch
    mapping range -- sonify.Sonifier has its own `top_deg` meaning that, and
    the two have no reason to hold the same value.
    """

    def __init__(self, complete_deg=160.0, descend_deg=140.0, min_frames=5):
        if descend_deg >= complete_deg:
            raise ValueError(
                "descend_deg must sit below complete_deg, otherwise there is "
                "no hysteresis gap and noise walks reps through the machine")
        self.complete_deg = complete_deg
        self.descend_deg = descend_deg
        self.min_frames = min_frames
        self.count = 0
        self._state = _LOST
        self._min_deg = None
        self._frames = 0

    def reset(self):
        """Forget the rep in flight and disarm. Does not reset the count."""
        self._state = _LOST
        self._min_deg = None
        self._frames = 0

    def update(self, angle_deg: Optional[float], t: float
               ) -> Optional[RepEvent]:
        """
        Feed one frame. Returns a RepEvent on the frame that completes a rep,
        otherwise None. `angle_deg` is None when the joint is not tracked.
        """
        if angle_deg is None:
            self.reset()
            return None

        if self._state == _LOST:
            if angle_deg > self.complete_deg:
                self._state = _TOP
            return None

        if self._state == _TOP:
            if angle_deg < self.descend_deg:
                self._state = _DESCENDING
                self._min_deg = angle_deg
                self._frames = 1
            return None

        # _DESCENDING
        self._min_deg = min(self._min_deg, angle_deg)
        self._frames += 1
        if angle_deg <= self.complete_deg:
            return None

        bottom = self._min_deg
        frames = self._frames
        self._state = _TOP
        self._min_deg = None
        self._frames = 0

        if frames < self.min_frames:
            return None

        event = RepEvent(index=self.count, bottom_deg=bottom, t=t)
        self.count += 1
        return event


def knee_signal(angles) -> Optional[float]:
    """
    The one number that drives both the tone and the rep counter.

    Knees carry this rather than hips because they are the best-measured joint
    in the skeleton -- around 2.3 degrees against synthetic ground truth,
    against 3-4 for hips -- even though hip crease relative to knee is the
    truer definition of squat depth. That substitution is a deliberate trade
    for the spike and not a settled answer.

    Averaging both knees halves the tracker's noise when both are visible, and
    the fallback to a single knee keeps the tone alive through the partial
    occlusion that a phone on the floor produces constantly. It averages over
    the knees actually present, never over a fixed two: a subject standing at
    110 degrees with one leg occluded must not be reported at 55.
    """
    seen = [angles.get(k) for k in ("L knee", "R knee")]
    seen = [a for a in seen if a is not None]
    if not seen:
        return None
    return sum(seen) / len(seen)
