"""
collect_sequences.py
---------------------
Thu thập dữ liệu CHUỖI (không phải từng frame rời) cho bài toán dự đoán
sớm: mỗi lần vung tay được ghi lại thành 1 chuỗi nhiều frame, lưu thành
1 file .npy riêng.

Cách dùng:
    1. Chạy file này
    2. Bấm 1 / 2 / 3 để chọn nhãn đang muốn thu (keo / bua / bao)
    3. Bấm SPACE để BẮT ĐẦU ghi 1 lần vung tay
    4. Bấm SPACE LẦN NỮA để KẾT THÚC lần vung tay đó (chuỗi được lưu ra file)
    5. Lặp lại cho nhiều lần, đổi nhãn khi cần
    6. Bấm Q để thoát

Mẹo thu dữ liệu:
    - Mỗi lần bấm SPACE bắt đầu, nên bắt đầu từ tay đang ở vị trí nghỉ/hạ
      xuống, rồi vung lên và định hình cử chỉ, giống động tác chơi thật.
    - Thu ít nhất 30-50 lần mỗi nhãn để có đủ dữ liệu train.
    - Đổi tốc độ vung tay (nhanh/chậm), đổi góc tay một chút giữa các lần,
      để model không học chết theo một kiểu chuyển động duy nhất.
"""

import os
import time

import cv2
import mediapipe as mp
import numpy as np

from features import LABELS, extract_features

DATA_DIR = "sequences"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def next_seq_index(label):
    """Tìm số thứ tự tiếp theo cho 1 nhãn, để không ghi đè file cũ."""
    label_dir = os.path.join(DATA_DIR, label)
    os.makedirs(label_dir, exist_ok=True)
    existing = [f for f in os.listdir(label_dir) if f.endswith(".npy")]
    return len(existing) + 1


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    counts = {label: next_seq_index(label) - 1 for label in LABELS}

    cap = cv2.VideoCapture(0)
    current_label_idx = 0
    recording = False
    buffer = []  # các vector đặc trưng của lần vung tay đang ghi

    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            current_label = LABELS[current_label_idx]

            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                if recording:
                    feats = extract_features(hand_lms.landmark)
                    if feats is not None:
                        buffer.append(feats)
                    # feats = None (hiếm khi xảy ra với 1 tay) -> bỏ frame đó,
                    # không làm gián đoạn chuỗi đang ghi.
            elif recording:
                # Mất dấu tay giữa chừng khi đang ghi — vẫn tiếp tục chờ,
                # không tự hủy, để Hi tự quyết định bấm SPACE dừng khi xong.
                cv2.putText(frame, "MAT DAU TAY", (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # --- hiển thị trạng thái ---
            status_color = (0, 0, 255) if recording else (200, 200, 200)
            cv2.putText(frame, f"Nhan: {current_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame,
                        f"DANG GHI ({len(buffer)} frame)" if recording
                        else "Dung (SPACE de bat dau)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            y = 95
            for i, label in enumerate(LABELS):
                cv2.putText(frame, f"[{i+1}] {label}: {counts[label]} chuoi",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 0), 1)
                y += 20

            cv2.imshow("Collect Sequences", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            elif key == ord(' '):
                if not recording:
                    # bắt đầu ghi
                    recording = True
                    buffer = []
                else:
                    # kết thúc ghi -> lưu chuỗi ra file
                    recording = False
                    if len(buffer) >= 5:  # bỏ qua chuỗi quá ngắn (bấm nhầm)
                        counts[current_label] += 1
                        idx = counts[current_label]
                        path = os.path.join(
                            DATA_DIR, current_label, f"seq_{idx:04d}.npy")
                        np.save(path, np.array(buffer, dtype=np.float32))
                        print(f"Da luu {path} ({len(buffer)} frame)")
                    else:
                        print("Chuoi qua ngan, bo qua (co the do bam nham).")
                    buffer = []

            elif key in (ord('1'), ord('2'), ord('3')):
                if not recording:  # không cho đổi nhãn giữa chừng khi đang ghi
                    idx = key - ord('1')
                    if idx < len(LABELS):
                        current_label_idx = idx

    cap.release()
    cv2.destroyAllWindows()

    print("\nTong so chuoi da thu:")
    for label in LABELS:
        print(f"  {label}: {counts[label]}")


if __name__ == "__main__":
    main()