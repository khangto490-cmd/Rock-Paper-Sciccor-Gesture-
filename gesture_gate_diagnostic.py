"""Diagnostic variant of the rock/paper/scissors rule gate. Decision rules are unchanged.

Computed from the same palm-frame landmarks
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


def _median_points(frames):
    """Return the median 21x3 landmark array for the selected frames."""
    return np.median(
        np.stack([np.asarray(f, dtype=np.float32).reshape(21, 3) for f in frames]),
        axis=0,
    )


def gate_stats(frames):
    """frames: sequence of flattened (63,) features. Returns (ext[4], thumb)."""
    pts = _median_points(frames)
    ext = np.array(
        [np.linalg.norm(pts[tip]) / (np.linalg.norm(pts[mcp]) + 1e-6) for mcp, tip in _FINGERS],
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


def thumb_diagnostics(frames):
    """Return diagnostic thumb geometry; these values do NOT affect gate decisions.

    thumb_dist: existing distance from thumb tip (4) to index MCP (5).
    thumb_angle_deg: angle between thumb direction (2 -> 4) and palm axis (0 -> 9).
    thumb_axis_proj: thumb-tip projection along the wrist-to-middle-MCP palm axis.
    thumb_lateral: magnitude of the thumb tip's component perpendicular to that axis.
    """
    pts = _median_points(frames)
    thumb_vec = pts[4] - pts[2]
    palm_axis = pts[9] - pts[0]
    axis_norm = float(np.linalg.norm(palm_axis))
    thumb_norm = float(np.linalg.norm(thumb_vec))

    if axis_norm < 1e-6 or thumb_norm < 1e-6:
        angle_deg = float("nan")
        axis_proj = float("nan")
        lateral = float("nan")
    else:
        axis_unit = palm_axis / axis_norm
        cosine = float(np.dot(thumb_vec, axis_unit) / thumb_norm)
        angle_deg = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        tip_from_wrist = pts[4] - pts[0]
        axis_proj = float(np.dot(tip_from_wrist, axis_unit))
        perpendicular = tip_from_wrist - axis_proj * axis_unit
        lateral = float(np.linalg.norm(perpendicular))

    thumb_dist = float(np.linalg.norm(pts[4] - pts[5]))
    return thumb_dist, angle_deg, axis_proj, lateral


def describe(ext, thumb, frames=None):
    parts = [f"{n} {v:.2f}" for n, v in zip(_NAMES, ext)]
    text = " | ".join(parts) + f" | thumb_dist {thumb:.2f}"
    if frames is not None:
        _, angle_deg, axis_proj, lateral = thumb_diagnostics(frames)
        text += (
            f" | thumb_angle {angle_deg:.1f}deg"
            f" | thumb_axis_proj {axis_proj:.2f}"
            f" | thumb_lateral {lateral:.2f}"
        )
    return text


def gate_check(frames, label):
    """Returns (ok, info). Extra thumb metrics are logging-only, not decision rules."""
    frames = list(frames)[-GATE_FRAMES:]
    if not frames:
        return False, "no frames"
    ext, thumb = gate_stats(frames)
    # Keep matches() and all acceptance thresholds unchanged for a fair comparison.
    return matches(label, ext, thumb), describe(ext, thumb, frames)


if __name__ == "__main__":  # offline sanity check on the training windows
    d = np.load("gru_dataset.npz")
    X, y = d["X"], d["y"]
    names = ["rock", "paper", "scissors"]
    for k, name in enumerate(names):
        idx = np.where(y == k)[0]
        agree = np.mean([gate_check(X[i][-GATE_FRAMES:], name)[0] for i in idx])
        print(f"{name:9s} n={len(idx):4d} gate pass rate {agree:.3f}")