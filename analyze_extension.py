"""
analyze_extension.py
---------------------
Phân tích dữ liệu ĐÃ CÓ (sequences/keo, sequences/bua, sequences/bao) để
tìm ngưỡng "ngón tay đã duỗi ra đủ để tính là kéo/bao", dựa trên khoảng
cách đầu ngón tay tới cổ tay (đã có sẵn trong vector đặc trưng 63 chiều).

Không cần chạy webcam -- chỉ đọc lại các file .npy đã thu.
"""

import os

import numpy as np

DATA_DIR = "sequences"
LABELS = ["keo", "bua", "bao"]

# Vị trí landmark trong vector 63 chiều: mỗi landmark chiếm 3 cột (x,y,z)
# liên tiếp, theo đúng thứ tự 0..20 của MediaPipe Hands.
INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 8, 12, 16, 20


def landmark_distance_from_wrist(frame_features, landmark_idx):
    """
    frame_features: vector 63 chiều của 1 frame (đã chuẩn hóa theo cổ tay
    trong extract_features -- nghĩa là (x,y,z) ở đây CHÍNH LÀ độ lệch so
    với cổ tay rồi, nên chỉ cần tính độ dài vector là ra khoảng cách).
    """
    start = landmark_idx * 3
    x, y, z = frame_features[start:start + 3]
    return float(np.sqrt(x**2 + y**2 + z**2))


def load_all_sequences(label):
    label_dir = os.path.join(DATA_DIR, label)
    seqs = []
    for fname in sorted(os.listdir(label_dir)):
        if fname.endswith(".npy"):
            seqs.append(np.load(os.path.join(label_dir, fname)))
    return seqs


def summarize(name, values):
    values = np.array(values)
    print(f"  {name}: n={len(values)}  "
          f"p10={np.percentile(values,10):.3f}  "
          f"median={np.median(values):.3f}  "
          f"p90={np.percentile(values,90):.3f}")


def main():
    keo_seqs = load_all_sequences("keo")
    bua_seqs = load_all_sequences("bua")
    bao_seqs = load_all_sequences("bao")

    # --- độ duỗi ngón trỏ+giữa (liên quan tới "keo") ---
    print("Do duoi NGON TRO + GIUA (dist toi co tay, trung binh 2 ngon):")

    def ext_index_middle(frame):
        return (landmark_distance_from_wrist(frame, INDEX_TIP) +
                landmark_distance_from_wrist(frame, MIDDLE_TIP)) / 2

    # búa: tay luôn nắm -> nên đo trên TOÀN BỘ frame của búa
    bua_all = [ext_index_middle(f) for seq in bua_seqs for f in seq]
    summarize("BUA (toan bo frame, ky vong THAP)", bua_all)

    # kéo: 20% frame ĐẦU (còn trung tính, kỳ vọng THẤP giống búa)
    keo_first20 = [ext_index_middle(seq[i])
                   for seq in keo_seqs
                   for i in range(max(1, int(len(seq) * 0.2)))]
    summarize("KEO 20% dau (ky vong THAP, giong bua)", keo_first20)

    # kéo: 20% frame CUỐI (đã định hình xong, kỳ vọng CAO)
    keo_last20 = [ext_index_middle(seq[i])
                  for seq in keo_seqs
                  for i in range(int(len(seq) * 0.8), len(seq))]
    summarize("KEO 20% cuoi (ky vong CAO, da ro kéo)", keo_last20)

    # --- độ duỗi cả 4 ngón (liên quan tới "bao") ---
    print("\nDo duoi CA 4 NGON (dist toi co tay, trung binh 4 ngon):")

    def ext_all_fingers(frame):
        return np.mean([landmark_distance_from_wrist(frame, i)
                        for i in (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)])

    bua_all4 = [ext_all_fingers(f) for seq in bua_seqs for f in seq]
    summarize("BUA (toan bo frame, ky vong THAP)", bua_all4)

    bao_first20 = [ext_all_fingers(seq[i])
                   for seq in bao_seqs
                   for i in range(max(1, int(len(seq) * 0.2)))]
    summarize("BAO 20% dau (ky vong THAP, giong bua)", bao_first20)

    bao_last20 = [ext_all_fingers(seq[i])
                  for seq in bao_seqs
                  for i in range(int(len(seq) * 0.8), len(seq))]
    summarize("BAO 20% cuoi (ky vong CAO, da ro bao)", bao_last20)

    print("\nGoi y: chon nguong nam GIUA khoang 'ky vong THAP' va 'ky vong CAO'")
    print("cho tung cap so sanh -- xem p90 cua nhom THAP va p10 cua nhom CAO,")
    print("neu 2 khoang khong de len nhau thi de dat nguong o giua.")


if __name__ == "__main__":
    main()