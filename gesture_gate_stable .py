"""Geometry gate for rock/paper/scissors in normalized palm-local coordinates.

This version preserves the uploaded gate's decision thresholds and pass/fail rules,
but adds clearer rejection reasons for debugging the live demo. It does not change
model predictions or the GRU checkpoint.

Note: the gate uses Euclidean distances/norms, which are invariant to a sign flip of
local Z. Therefore, the left-hand Z-flip should not by itself change gate statistics.

Thresholds originated from gru_dataset.npz (2 people, 120 FPS video). They have not
been validated comprehensively on negative gestures (thumbs-up, pointing, etc.).
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


def rejection_reason(label, ext, thumb):
    """Explain which unchanged geometric rule(s) failed."""
    label = str(label).lower()
    states = _states(ext)
    reasons = []

    if label == "rock":
        for i, name in enumerate(_NAMES):
            if states[i] != 0:
                reasons.append(f"{name} not curled (needs < {CURL_MAX:.2f})")
        if thumb > THUMB_MAX:
            reasons.append(f"thumb too far ({thumb:.2f} > {THUMB_MAX:.2f})")
    elif label == "paper":
        for i, name in enumerate(_NAMES):
            if states[i] != 2:
                reasons.append(f"{name} not extended (needs > {EXT_MIN:.2f})")
    elif label == "scissors":
        expected = (2, 2, 0, 0)
        for i, name in enumerate(_NAMES):
            if states[i] != expected[i]:
                state_name = "extended" if expected[i] == 2 else "curled"
                bound = f"> {EXT_MIN:.2f}" if expected[i] == 2 else f"< {CURL_MAX:.2f}"
                reasons.append(f"{name} not {state_name} (needs {bound})")
        if thumb > THUMB_MAX:
            reasons.append(f"thumb too far ({thumb:.2f} > {THUMB_MAX:.2f})")
    else:
        reasons.append(f"unknown class '{label}'")

    return "; ".join(reasons) if reasons else "rule mismatch"


def gate_check(frames, label):
    """Return (accepted, info); info includes measurements and failure reason."""
    frames = list(frames)[-GATE_FRAMES:]
    if not frames:
        return False, "no frames"

    try:
        ext, thumb = gate_stats(frames)
    except (ValueError, TypeError, IndexError) as exc:
        return False, f"invalid landmark frames ({type(exc).__name__})"

    ok = matches(label, ext, thumb)
    info = describe(ext, thumb)
    if not ok:
        info += " | reason: " + rejection_reason(label, ext, thumb)
    return ok, info


if __name__ == "__main__":  # offline sanity check on the training windows
    d = np.load("gru_dataset.npz")
    X, y = d["X"], d["y"]
    names = ["rock", "paper", "scissors"]
    for k, name in enumerate(names):
        idx = np.where(y == k)[0]
        agree = np.mean([gate_check(X[i][-GATE_FRAMES:], name)[0] for i in idx])
        print(f"{name:9s} n={len(idx):4d} gate pass rate {agree:.3f}")