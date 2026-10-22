"""
measure_transition.py
----------------------
Đo trong CHÍNH dữ liệu của Hi: từ lúc ngón bắt đầu rời khỏi dáng nắm tới lúc tay
đã "định hình rõ" là kéo/bao mất bao nhiêu frame.

Vì sao cần số này: nếu cả quá trình chuyển chỉ kéo dài 2-3 frame thì máy
chỉ có 2-3 frame để "đoán sớm" -- đó là trần vật lý ở tốc độ khung hình hiện tại,
không phải lỗi của model.

Chạy: python measure_transition.py   (chỉ đọc file .npy, không cần webcam)
"""

import os

import numpy as np

DATA_DIR = "sequences"

# "Bắt đầu rời dáng nắm": vượt đuôi phân phối của búa (p90 đo ở analyze_extension.py).
START_KEO, START_BAO = 1.18, 1.02
# "Đã định hình rõ": mức thấp nhất của 20% frame cuối chuỗi (p10 ở analyze_extension.py).
FORMED_KEO, FORMED_BAO = 1.76, 1.66

# Tốc độ khung hình lúc Hi THU dữ liệu (chưa biết chắc -> in cả hai mốc).
FPS_LIST = (21, 30)


def ext_idx_mid(P):
    return (np.linalg.norm(P[:, 8], axis=1) + np.linalg.norm(P[:, 12], axis=1)) / 2


def ext_all4(P):
    return np.mean([np.linalg.norm(P[:, i], axis=1) for i in (8, 12, 16, 20)], axis=0)


def first_true(mask):
    idx = np.where(mask)[0]
    return int(idx[0]) if len(idx) else None


def analyze(name, ext_fn, start_thr, formed_thr):
    d = os.path.join(DATA_DIR, name)
    lengths, skipped = [], 0
    for f in sorted(os.listdir(d)):
        if not f.endswith(".npy"):
            continue
        seq = np.load(os.path.join(d, f)).astype(np.float32)
        ext = ext_fn(seq.reshape(len(seq), 21, 3))
        s = first_true(ext >= start_thr)
        e = first_true(ext >= formed_thr)
        if s is None or e is None or e < s:
            skipped += 1
            continue
        lengths.append(e - s)

    lengths = np.array(lengths)
    print(f"\n=== {name.upper()}: tu luc roi dang nam (>= {start_thr}) "
          f"den luc dinh hinh ro (>= {formed_thr}) ===")
    print(f"  so chuoi dung duoc: {len(lengths)}  (bo qua {skipped})")
    for p in (10, 50, 90):
        frames = np.percentile(lengths, p)
        ms = ", ".join(f"{frames * 1000 / fps:.0f} ms o {fps} FPS" for fps in FPS_LIST)
        print(f"  p{p:<2}: {frames:5.1f} frame  ({ms})")
    print(f"  chuyen tiep <= 2 frame: {100 * np.mean(lengths <= 2):.0f}% chuoi")
    print(f"  chuyen tiep <= 4 frame: {100 * np.mean(lengths <= 4):.0f}% chuoi")


def main():
    analyze("keo", ext_idx_mid, START_KEO, FORMED_KEO)
    analyze("bao", ext_all4, START_BAO, FORMED_BAO)
    print("\nCach doc: trung vi (p50) la so frame may co de 'doan som'.")
    print("Neu chi 2-3 frame -> tran vat ly o toc do khung hinh nay.")
    print("Neu 8-10 frame   -> con nhieu du dia, van de nam o model hoac do tre.")


if __name__ == "__main__":
    main()
