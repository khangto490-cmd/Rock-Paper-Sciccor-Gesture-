"""
compare_methods.py
-------------------
So sánh nhiều phương pháp dự đoán TỪNG FRAME trên cùng một bộ dữ liệu, bằng
cùng một bộ số đo, để có số liệu đưa vào báo cáo.

Điểm khác với train_static.py:
  - Chia train/test theo CHUỖI (mỗi lần vung tay nằm trọn ở train hoặc test),
    kiểm tra chéo 5 lần, nên mỗi chuỗi được test đúng 1 lần.
    (Chia ngẫu nhiên theo frame như train_static.py bị rò rỉ: các frame liền
    kề của cùng 1 chuỗi gần như giống hệt nhau, nên accuracy bị thổi phồng.)
  - Đo trực tiếp việc "đổi nhãn sớm hơn bao nhiêu frame", không chỉ accuracy.

Chạy: python compare_methods.py    (không cần webcam, khoảng vài phút)
Kết quả in ra màn hình và lưu vào ket_qua_so_sanh.csv
"""

import csv
import os
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

DATA_DIR = "sequences"
CLASSES = ["bua", "keo", "bao"]          # chỉ số 0, 1, 2
CLS_IDX = {c: i for i, c in enumerate(CLASSES)}

# Ngưỡng dùng để GÁN NHÃN lúc train. Sửa cho khớp với số Hi đang dùng.
T_TRAIN_KEO = 1.30
T_TRAIN_BAO = 1.15

# Ngưỡng THAM CHIẾU, cố định cho mọi phương pháp: mốc "tay đã rõ là kéo/bao".
# Dùng để đo "đổi nhãn sớm hơn mốc này bao nhiêu frame".
REF_KEO = 1.47
REF_BAO = 1.34

RUN = 3            # số frame liên tiếp dự đoán giống nhau mới tính là "đã đổi"
FRAME_MS = 1000 / 30
N_TREES = 100
N_FOLDS = 5

# (tên, loại đặc trưng, số frame "nhãn báo trước")
METHODS = [
    ("M1 goc: 63 toa do",                     "raw",        0),
    ("M2 dac trung tung ngon (11)",           "eng",        0),
    ("M3 toa do + dac trung ngon (74)",       "raw+eng",    0),
    ("M4 dac trung ngon + chuyen dong (33)",  "eng+motion", 0),
    ("M5 = M4 + nhan bao truoc 3 frame",      "eng+motion", 3),
    ("M6 = M4 + nhan bao truoc 6 frame",      "eng+motion", 6),
    # Đối chứng: nhãn báo trước nhưng KHÔNG có đặc trưng chuyển động.
    # So M7 với M5 (và M8 với M6) để biết đặc trưng chuyển động có đóng góp thật không.
    ("M7 = M2 + nhan bao truoc 3 (khong c.dong)", "eng",    3),
    ("M8 = M2 + nhan bao truoc 6 (khong c.dong)", "eng",    6),
]

TIPS5 = [4, 8, 12, 16, 20]       # đầu 5 ngón
TIPS4 = [8, 12, 16, 20]          # đầu 4 ngón (không tính ngón cái)
MCPS4 = [5, 9, 13, 17]           # khớp gốc của 4 ngón đó


# ---------------------------------------------------------------- dữ liệu
def load_sequences():
    seqs, finals = [], []
    for lab in ["keo", "bua", "bao"]:
        d = os.path.join(DATA_DIR, lab)
        for f in sorted(os.listdir(d)):
            if f.endswith(".npy"):
                seqs.append(np.load(os.path.join(d, f)).astype(np.float32))
                finals.append(lab)
    return seqs, finals


def points(seq):
    return seq.reshape(len(seq), 21, 3)


def ext_idx_mid(P):
    return (np.linalg.norm(P[:, 8], axis=1) + np.linalg.norm(P[:, 12], axis=1)) / 2


def ext_all4(P):
    return np.mean([np.linalg.norm(P[:, i], axis=1) for i in TIPS4], axis=0)


# ---------------------------------------------------------------- đặc trưng
def eng_features(P):
    """11 đặc trưng: độ duỗi 5 ngón, độ co 4 ngón, độ xòe giữa các ngón."""
    ext = np.linalg.norm(P[:, TIPS5], axis=2)                     # (n, 5)
    curl = np.linalg.norm(P[:, TIPS4] - P[:, MCPS4], axis=2)      # (n, 4)
    spread = np.stack([np.linalg.norm(P[:, 8] - P[:, 12], axis=1),
                       np.linalg.norm(P[:, 12] - P[:, 16], axis=1)], axis=1)
    return np.concatenate([ext, curl, spread], axis=1)            # (n, 11)


def motion_features(E, lags=(2, 4)):
    """Hiệu số so với 2 và 4 frame trước (chỉ dùng frame QUÁ KHỨ, không nhìn tương lai)."""
    idx = np.arange(len(E))
    return np.concatenate([E - E[np.maximum(idx - lag, 0)] for lag in lags], axis=1)


def build_features(seq, kind):
    if kind == "raw":
        return seq
    E = eng_features(points(seq))
    if kind == "eng":
        return E
    if kind == "raw+eng":
        return np.concatenate([seq, E], axis=1)
    if kind == "eng+motion":
        return np.concatenate([E, motion_features(E)], axis=1)
    raise ValueError(kind)


# ---------------------------------------------------------------- nhãn
def frame_labels(seq, final, lead):
    """Gán nhãn từng frame theo ngưỡng, giống build_frame_dataset.py.
    lead > 0: thêm 'lead' frame ngay TRƯỚC lúc vượt ngưỡng cũng mang nhãn kéo/bao."""
    P = points(seq)
    lab = np.zeros(len(seq), dtype=int)
    if final == "keo":
        mask = ext_idx_mid(P) >= T_TRAIN_KEO
    elif final == "bao":
        mask = ext_all4(P) >= T_TRAIN_BAO
    else:
        return lab
    lab[mask] = CLS_IDX[final]
    if lead > 0 and mask.any():
        c = int(np.argmax(mask))
        lab[max(0, c - lead):c] = CLS_IDX[final]
    return lab


# ---------------------------------------------------------------- số đo
def first_true(mask):
    idx = np.where(mask)[0]
    return int(idx[0]) if len(idx) else None


def first_run_idx(mask, run=RUN):
    """Vị trí đầu tiên có 'run' frame liên tiếp cùng True; None nếu không có."""
    if len(mask) < run:
        return None
    c = np.convolve(mask.astype(int), np.ones(run, dtype=int), mode="valid")
    return first_true(c == run)


def evaluate(stats, seq, final, pred):
    n = len(seq)
    P = points(seq)
    stats["final_acc"].append(float((pred[int(0.8 * n):] == CLS_IDX[final]).mean()))

    if final == "bua":
        stats["bua_false"].append(first_run_idx(pred != 0) is not None)
        stats["bua_frame_false"].append(float((pred != 0).mean()))
        return

    if final == "keo":
        key, tgt, other = "keo", 1, 2
        ref = first_true(ext_idx_mid(P) >= REF_KEO)
    else:
        key, tgt, other = "bao", 2, 1
        ref = first_true(ext_all4(P) >= REF_BAO)

    stats[key + "_wrong"].append(first_run_idx(pred == other) is not None)
    t_pred = first_run_idx(pred == tgt)
    stats[key + "_hit"].append(t_pred is not None)
    if t_pred is not None and ref is not None:
        stats[key + "_lead"].append(ref - t_pred)   # dương = đổi nhãn SỚM hơn mốc tham chiếu


def pct(x):
    return 100 * float(np.mean(x)) if len(x) else float("nan")


def med(x):
    return float(np.median(x)) if len(x) else float("nan")


# ---------------------------------------------------------------- chạy
def main():
    seqs, finals = load_sequences()
    y_seq = np.array([CLS_IDX[f] for f in finals])
    print(f"{len(seqs)} chuoi, kiem tra cheo {N_FOLDS} lan theo chuoi\n")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    rows = []

    for name, kind, lead in METHODS:
        t0 = time.time()
        stats = {k: [] for k in
                 ["keo_lead", "keo_hit", "keo_wrong", "bao_lead", "bao_hit",
                  "bao_wrong", "bua_false", "bua_frame_false", "final_acc"]}
        model, x_probe = None, None

        for tr, te in skf.split(np.zeros(len(seqs)), y_seq):
            Xtr = np.concatenate([build_features(seqs[i], kind) for i in tr])
            ytr = np.concatenate([frame_labels(seqs[i], finals[i], lead) for i in tr])
            model = RandomForestClassifier(
                n_estimators=N_TREES, random_state=42, n_jobs=-1).fit(Xtr, ytr)
            for i in te:
                X = build_features(seqs[i], kind)
                evaluate(stats, seqs[i], finals[i], model.predict(X))
                x_probe = X[len(X) // 2: len(X) // 2 + 1]

        # thời gian dự đoán 1 mẫu (đặt n_jobs=1 giống lúc chạy realtime)
        model.set_params(n_jobs=1)
        t1 = time.perf_counter()
        for _ in range(200):
            model.predict_proba(x_probe)
        ms_predict = (time.perf_counter() - t1) / 200 * 1000

        row = {
            "phuong_phap": name,
            "keo_som_trung_vi_frame": med(stats["keo_lead"]),
            "keo_doi_nhan_duoc_%": pct(stats["keo_hit"]),
            "bao_som_trung_vi_frame": med(stats["bao_lead"]),
            "bao_doi_nhan_duoc_%": pct(stats["bao_hit"]),
            "bua_kich_hoat_nham_%": pct(stats["bua_false"]),
            "nham_kéo_bao_%": pct(stats["keo_wrong"] + stats["bao_wrong"]),
            "chinh_xac_dang_cuoi_%": pct(stats["final_acc"]),
            "ms_du_doan": ms_predict,
        }
        rows.append(row)
        print(f"xong {name}  ({time.time() - t0:.0f}s)")

    # ------------------------------------------------------------ in bảng
    print("\n" + "=" * 118)
    print(f"{'Phuong phap':<40}{'KEO som':>9}{'KEO doi':>9}{'BAO som':>9}{'BAO doi':>9}"
          f"{'BUA nham':>10}{'Nham K/B':>10}{'Dang cuoi':>10}{'ms':>7}")
    print(f"{'':<40}{'(frame)':>9}{'(%)':>9}{'(frame)':>9}{'(%)':>9}"
          f"{'(%)':>10}{'(%)':>10}{'(%)':>10}{'':>7}")
    print("-" * 118)
    for r in rows:
        print(f"{r['phuong_phap']:<40}"
              f"{r['keo_som_trung_vi_frame']:>9.1f}{r['keo_doi_nhan_duoc_%']:>9.1f}"
              f"{r['bao_som_trung_vi_frame']:>9.1f}{r['bao_doi_nhan_duoc_%']:>9.1f}"
              f"{r['bua_kich_hoat_nham_%']:>10.1f}{r['nham_kéo_bao_%']:>10.1f}"
              f"{r['chinh_xac_dang_cuoi_%']:>10.1f}{r['ms_du_doan']:>7.1f}")
    print("=" * 118)
    print(f"\n'som' = so frame model doi nhan TRUOC moc tham chieu (kéo {REF_KEO}, bao {REF_BAO});")
    print(f"        duong = som hon, am = tre hon. 1 frame ~ {FRAME_MS:.0f} ms.")
    print(f"'doi' = % chuoi ma model co doi sang dung kéo/bao (>= {RUN} frame lien tiep).")
    print("'BUA nham' = % chuoi BUA bi model kich hoat sang kéo/bao nham (cang thap cang tot).")
    print("'Nham K/B' = % chuoi kéo/bao ma model tung doan sang cu chi con lai.")
    print("'Dang cuoi' = do chinh xac o 20% frame cuoi cua chuoi (dam bao khong hong dang cuoi).")

    with open("ket_qua_so_sanh.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nDa luu ket_qua_so_sanh.csv")


if __name__ == "__main__":
    main()
