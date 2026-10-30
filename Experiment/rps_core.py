"""
rps_core.py
-----------
Các hàm dùng chung cho train_model.py và play_realtime.py.
PHẢI khớp với analyze_approach.py (cùng cách chuẩn hóa, cùng đặc trưng, cùng luật hình học).
Chỉ cần numpy.
"""

import glob
import json
import os

import numpy as np

GESTURES = ["keo", "bua", "bao"]          # chỉ số 0, 1, 2
G_IDX = {g: i for i, g in enumerate(GESTURES)}
PALM_IDS = [0, 5, 9, 13, 17]
TIPS5, TIPS4, MCPS4 = [4, 8, 12, 16, 20], [8, 12, 16, 20], [5, 9, 13, 17]

EXT_HI, EXT_LO = 1.15, 1.00                        # luật hình học: ngón duỗi khi tỉ lệ >= HI, co khi <= LO
FINGERS = [(6, 8), (10, 12), (14, 16), (18, 20)]   # (khớp giữa, đầu ngón): trỏ, giữa, áp út, út

# Luật chơi: BEATS[m] = cử chỉ mà nước m thắng (kéo thắng bao, búa thắng kéo, bao thắng búa)
BEATS = {0: 2, 1: 0, 2: 1}
PAY = np.array([[0 if m == g else (1 if BEATS[m] == g else -1) for g in range(3)] for m in range(3)], float)
COUNTER = {g: m for m in range(3) for g in range(3) if BEATS[m] == g}   # nước khắc chế cử chỉ g


def normalize(lm):
    """(n,21,3) toạ độ gốc -> chuẩn hóa theo cổ tay và chiều dài cổ tay->gốc ngón giữa."""
    scale = np.linalg.norm(lm[:, 9] - lm[:, 0], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (lm - lm[:, 0:1]) / scale[:, None, None]


def eng_features(P):
    ext = np.linalg.norm(P[:, TIPS5], axis=2)
    curl = np.linalg.norm(P[:, TIPS4] - P[:, MCPS4], axis=2)
    spread = np.stack([np.linalg.norm(P[:, 8] - P[:, 12], axis=1),
                       np.linalg.norm(P[:, 12] - P[:, 16], axis=1)], axis=1)
    return np.concatenate([ext, curl, spread], axis=1)


def visible_shape(P, hi=EXT_HI, lo=EXT_LO):
    """P (n,21,3) đã chuẩn hóa -> (n,) 0 kéo, 1 búa, 2 bao, -1 không rõ (luật hình học, bỏ qua ngón cái)."""
    st = np.zeros((len(P), 4), int)
    for k, (pip, tip) in enumerate(FINGERS):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.linalg.norm(P[:, tip], axis=1) / np.linalg.norm(P[:, pip], axis=1)
        st[:, k] = np.where(r >= hi, 1, np.where(r <= lo, -1, 0))
    shp = np.full(len(P), -1)
    shp[(st == -1).all(axis=1)] = 1
    shp[(st == 1).all(axis=1)] = 2
    shp[(st[:, 0] == 1) & (st[:, 1] == 1) & (st[:, 2] == -1) & (st[:, 3] == -1)] = 0
    return shp


def square_bounds(w, h, frac):
    """Ô vuông ở giữa, trả về (x0, x1, y0, y1) theo toạ độ chuẩn hóa 0..1."""
    side = frac * min(w, h)
    return (0.5 - side / (2 * w), 0.5 + side / (2 * w),
            0.5 - side / (2 * h), 0.5 + side / (2 * h))


def in_square(cx, cy, w, h, frac):
    x0, x1, y0, y1 = square_bounds(w, h, frac)
    return x0 <= cx <= x1 and y0 <= cy <= y1


def palm_center(lm):
    """lm: (21,3) toạ độ chuẩn hóa của ảnh gốc (không lật)."""
    return float(np.mean(lm[PALM_IDS, 0])), float(np.mean(lm[PALM_IDS, 1]))


def proba3(rf, X):
    """predict_proba luôn trả về đủ 3 cột (kéo, búa, bao)."""
    out = np.zeros((len(X), 3))
    out[:, rf.classes_.astype(int)] = rf.predict_proba(X)
    return out


def load_training_frames(outdir, person=None):
    """Đọc mọi ván hợp lệ trong outdir; trả về (X đặc trưng từng khung, y nhãn ý định, số ván)."""
    X, y, n_rounds = [], [], 0
    for path in sorted(glob.glob(os.path.join(outdir, "person_*", "session_*", "r*.npz"))):
        d = np.load(path)
        meta = json.loads(str(d["meta_json"]))
        if not meta.get("valid") or (person and meta["person"] != person):
            continue
        det = np.where(d["detected"])[0]
        if len(det) == 0:
            continue
        Pn = normalize(d["landmarks"][det])
        Pn = Pn[np.isfinite(Pn).all(axis=(1, 2))]
        if len(Pn) == 0:
            continue
        X.append(eng_features(Pn))
        y.append(np.full(len(Pn), G_IDX[meta["gesture"]]))
        n_rounds += 1
    if not X:
        return np.zeros((0, 11)), np.zeros(0, int), 0
    return np.concatenate(X), np.concatenate(y), n_rounds
