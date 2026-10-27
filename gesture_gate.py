"""Rule gate for rock/paper/scissors, computed from the same palm-frame landmarks
that extract_local_landmarks() produces (wrist-centred, palm-scale normalised).

Idea: after the GRU picks a label, check that the finger geometry actually matches it.
  rock     : 4 fingers curled AND thumb close to the palm
  paper    : 4 fingers extended (thumb unconstrained)
  scissors : index+middle extended, ring+pinky curled, thumb close to the palm
Anything else -> reject (caller shows TRY AGAIN).

Thresholds were read off gru_dataset.npz (2 people, 120 FPS video). They have NOT been
validated on negatives (thumbs-up, pointing, ...) because the dataset has none:
re-check THUMB_MAX with live negatives before trusting it.
"""
import numpy as np

CURL_MAX = 1.35    # tip-to-wrist / MCP-to-wrist below this -> curled   (rock p95 ~1.2-1.34)
EXT_MIN = 1.50     # above this -> extended                              (paper p25 ~1.7-1.8)
THUMB_MAX = 0.90   # thumb tip to index MCP, in palm-scale units         (rock p95 0.69, scissors p95 0.85)
GATE_FRAMES = 3    # median over the last N buffered frames

_FINGERS = ((5, 8), (9, 12), (13, 16), (17, 20))  # index, middle, ring, pinky: (MCP, tip)
_NAMES = ("idx", "mid", "ring", "pky")


def gate_stats(frames):
    """frames: sequence of flattened (63,) features. Returns (ext[4], thumb)."""
    pts = np.median(
        np.stack([
            np.asarray(f, dtype=np.float32).reshape(21, 3)
            for f in frames
        ]),
        axis=0,
    )

    ext = np.array(
        [
            np.linalg.norm(pts[tip]) /
            (np.linalg.norm(pts[mcp]) + 1e-6)
            for mcp, tip in _FINGERS
        ],
        dtype=np.float32,
    )

    thumb = float(np.linalg.norm(pts[4] - pts[5]))

    return ext, thumb


def _states(ext):
    return np.where(ext < CURL_MAX, 0, np.where(ext > EXT_MIN, 2, 1))  # 0 curled, 1 ambiguous, 2 extended


def matches(label, ext, thumb):
    st = _states(ext)
    label = label.lower()
    if label == "rock":
        return bool((st == 0).all() and thumb <= THUMB_MAX)
    if label == "paper":
        return bool((st == 2).all())
    if label == "scissors":
        return bool(st[0] == 2 and st[1] == 2 and st[2] == 0 and st[3] == 0 and thumb <= THUMB_MAX)
    return False


def describe(ext, thumb):
    parts = [f"{n} {v:.2f}" for n, v in zip(_NAMES, ext)]
    return " | ".join(parts) + f" | thumb {thumb:.2f}"


def gate_check(frames, label):
    """Returns (ok, info). info is a printable string for logging/calibration."""
    frames = list(frames)[-GATE_FRAMES:]
    if not frames:
        return False, "no frames"
    ext, thumb = gate_stats(frames)
    return matches(label, ext, thumb), describe(ext, thumb)


if __name__ == "__main__":  # offline sanity check on the training windows
    d = np.load("gru_dataset.npz")
    X, y = d["X"], d["y"]
    names = ["rock", "paper", "scissors"]
    for k, name in enumerate(names):
        idx = np.where(y == k)[0]
        agree = np.mean([gate_check(X[i][-GATE_FRAMES:], name)[0] for i in idx])
        print(f"{name:9s} n={len(idx):4d} gate pass rate {agree:.3f}")