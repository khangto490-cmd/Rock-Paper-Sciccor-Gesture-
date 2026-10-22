"""
build_frame_dataset.py
------------------------
Đọc lại các chuỗi .npy đã thu, GÁN LẠI NHÃN THEO TỪNG FRAME (không dùng
nhãn cuối cùng của cả chuỗi nữa) dựa trên độ duỗi ngón tay đo được:

    - Frame trong chuỗi "keo": còn nắm (chưa đủ duỗi ngón trỏ+giữa) -> "bua"
                                đã đủ duỗi -> "keo"
    - Frame trong chuỗi "bao": còn nắm (chưa đủ duỗi cả 4 ngón) -> "bua"
                                đã đủ duỗi -> "bao"
    - Frame trong chuỗi "bua": luôn luôn -> "bua"

Ngưỡng T_KEO, T_BAO lấy từ kết quả phân tích thực tế (analyze_extension.py),
không phải số đoán mò.

Output: frame_dataset.csv (mỗi dòng = 1 frame, không còn khái niệm "chuỗi"
nữa -- đúng với việc dự đoán tĩnh theo từng frame độc lập).
"""

import csv
import os

import numpy as np

DATA_DIR = "sequences"
OUT_CSV = "frame_dataset.csv"

INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 8, 12, 16, 20

# Lấy từ analyze_extension.py -- nằm giữa 2 vùng tách biệt rõ ràng.
T_KEO = 1.30
T_BAO = 1.15


def dist_from_wrist(frame_features, landmark_idx):
    start = landmark_idx * 3
    x, y, z = frame_features[start:start + 3]
    return float(np.sqrt(x**2 + y**2 + z**2))


def ext_index_middle(frame):
    return (dist_from_wrist(frame, INDEX_TIP) +
            dist_from_wrist(frame, MIDDLE_TIP)) / 2


def ext_all_fingers(frame):
    return np.mean([dist_from_wrist(frame, i)
                    for i in (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)])


def relabel_sequence(seq, final_label):
    """Trả về list nhãn, 1 nhãn cho mỗi frame trong seq."""
    labels = []
    for frame in seq:
        if final_label == "keo":
            labels.append("keo" if ext_index_middle(frame) >= T_KEO else "bua")
        elif final_label == "bao":
            labels.append("bao" if ext_all_fingers(frame) >= T_BAO else "bua")
        else:  # "bua" -- luôn là búa, không cần đo
            labels.append("bua")
    return labels


def main():
    rows = []
    counts = {"keo": 0, "bua": 0, "bao": 0}

    for final_label in ["keo", "bua", "bao"]:
        label_dir = os.path.join(DATA_DIR, final_label)
        for fname in sorted(os.listdir(label_dir)):
            if not fname.endswith(".npy"):
                continue
            seq = np.load(os.path.join(label_dir, fname))
            frame_labels = relabel_sequence(seq, final_label)
            for frame, lbl in zip(seq, frame_labels):
                rows.append(list(frame) + [lbl])
                counts[lbl] += 1

    n_features = len(rows[0]) - 1
    header = [f"f{i}" for i in range(n_features)] + ["label"]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Da ghi {len(rows)} frame vao {OUT_CSV}")
    print("Phan bo nhan SAU KHI gan lai:")
    for label, n in counts.items():
        print(f"  {label}: {n} frame")
    print("\n(So sanh: neu gan theo nhan CU (theo ca chuoi), moi lop se")
    print("gan bang nhau. Lech nhieu ve 'bua' la BINH THUONG va DUNG Y DO,")
    print("vi phan lon frame o giai doan dau (con nam tay) deu duoc tinh")
    print("la bua, bat ke chuoi do cuoi cung ra keo/bua/bao.")


if __name__ == "__main__":
    main()