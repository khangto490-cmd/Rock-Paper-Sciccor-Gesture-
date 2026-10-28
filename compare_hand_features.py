"""Compare normalized landmark features for left-hand vs right-hand Rock gestures.

Uses the same local-frame normalization as realtime_gru_game_v7_2_margin_diagnostic.py.
Press R to record a right-hand Rock trial, L for a left-hand Rock trial,
repeat each at least 5 times, and press Q to save and quit.
"""
from pathlib import Path
import csv
import time

import cv2
import mediapipe as mp
import numpy as np

OUTPUT_NPZ = Path("hand_feature_comparison.npz")
OUTPUT_CSV = Path("hand_feature_comparison_summary.csv")
PROCESS_WIDTH = 640
FRAMES_PER_TRIAL = 60  # around 2 seconds at 30 FPS


def safe_normalize(v, eps=1e-6):
    norm = np.linalg.norm(v)
    if norm < eps:
        return None
    return v / norm


def extract_local_landmarks(hand_landmarks):
    """Exact normalization logic used in the current real-time demo."""
    landmarks = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )
    wrist = landmarks[0].copy()
    centered = landmarks - wrist
    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    pinky_mcp = landmarks[17]

    x_axis = safe_normalize(index_mcp - pinky_mcp)
    if x_axis is None:
        return None

    y_temp = middle_mcp - wrist
    y_temp = y_temp - np.dot(y_temp, x_axis) * x_axis
    y_axis = safe_normalize(y_temp)
    if y_axis is None:
        return None

    z_axis = safe_normalize(np.cross(x_axis, y_axis))
    if z_axis is None:
        return None

    y_axis = safe_normalize(np.cross(z_axis, x_axis))
    if y_axis is None:
        return None

    basis = np.stack([x_axis, y_axis, z_axis], axis=1)
    local_landmarks = centered @ basis
    palm_scale = np.linalg.norm(middle_mcp - wrist)
    if palm_scale > 1e-6:
        local_landmarks /= palm_scale
    return local_landmarks.astype(np.float32)


def resize_frame(frame):
    h, w = frame.shape[:2]
    if w <= PROCESS_WIDTH:
        return frame
    scale = PROCESS_WIDTH / w
    return cv2.resize(
        frame,
        (PROCESS_WIDTH, int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )


def save_results(trials, labels):
    if not trials:
        print("Chưa thu được mẫu nào; không tạo file kết quả.")
        return

    features = np.stack(trials).astype(np.float32)  # (trials, frames, 63)
    labels_arr = np.asarray(labels, dtype="U8")
    trial_medians = np.median(features, axis=1)

    np.savez_compressed(
        OUTPUT_NPZ,
        features=features,
        hand_labels=labels_arr,
        trial_medians=trial_medians,
        frames_per_trial=np.int32(FRAMES_PER_TRIAL),
    )

    right = trial_medians[labels_arr == "right"]
    left = trial_medians[labels_arr == "left"]

    axis_names = ("x", "y", "z")
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "feature_index", "landmark_index", "axis",
            "right_group_median", "left_group_median", "absolute_difference"
        ])
        for j in range(63):
            right_val = float(np.median(right[:, j])) if len(right) else float("nan")
            left_val = float(np.median(left[:, j])) if len(left) else float("nan")
            writer.writerow([j, j // 3, axis_names[j % 3], right_val, left_val,
                             abs(right_val - left_val) if len(right) and len(left) else float("nan")])

    print("\n=== KẾT QUẢ THU ĐẶC TRƯNG ===")
    print(f"Số lần thu tay phải: {int(np.sum(labels_arr == 'right'))}")
    print(f"Số lần thu tay trái : {int(np.sum(labels_arr == 'left'))}")
    if len(right) and len(left):
        right_center = np.median(right, axis=0)
        left_center = np.median(left, axis=0)
        abs_diff = np.abs(right_center - left_center)
        landmark_diff = np.linalg.norm(
            (right_center - left_center).reshape(21, 3), axis=1
        )
        print(f"Mean absolute difference across 63 features: {abs_diff.mean():.4f}")
        print(f"Largest single-feature difference: {abs_diff.max():.4f} (feature {int(abs_diff.argmax())})")
        print(f"Largest landmark-vector difference: {landmark_diff.max():.4f} (landmark {int(landmark_diff.argmax())})")
        print("Lưu ý: khác biệt cũng phụ thuộc vào việc hai tư thế Rock có giống nhau đến đâu.")
    else:
        print("Cần thu ít nhất 1 trial cho mỗi tay để so sánh hai nhóm.")
    print(f"Đã lưu: {OUTPUT_NPZ.resolve()}")
    print(f"Đã lưu: {OUTPUT_CSV.resolve()}")


def main():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    trials = []
    labels = []
    active_label = None
    active_frames = []
    previous_status = "Press R = record RIGHT Rock | L = record LEFT Rock | Q = save/quit"

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError("Không mở được webcam. Hãy kiểm tra camera đang dùng.")

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.60,
            min_tracking_confidence=0.60,
        ) as hands:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Không đọc được frame từ webcam.")
                    break

                display = frame.copy()
                small = resize_frame(frame)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = hands.process(rgb)

                local = None
                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    mp_drawing.draw_landmarks(display, hand, mp_hands.HAND_CONNECTIONS)
                    local = extract_local_landmarks(hand)

                if active_label is not None:
                    if local is not None:
                        active_frames.append(local.reshape(-1).astype(np.float32))
                    status = f"RECORDING {active_label.upper()} ROCK: {len(active_frames)}/{FRAMES_PER_TRIAL}"
                    if len(active_frames) >= FRAMES_PER_TRIAL:
                        trials.append(np.stack(active_frames[:FRAMES_PER_TRIAL]))
                        labels.append(active_label)
                        previous_status = f"Saved {active_label} trial #{sum(x == active_label for x in labels)}"
                        print(f"[SAVED] {previous_status}")
                        active_label = None
                        active_frames = []
                        status = previous_status
                else:
                    status = previous_status

                cv2.putText(display, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.58, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(display, "Hold a steady ROCK during each capture.", (12, 56),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(display, "R: right hand | L: left hand | Q: save and quit", (12, 84),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("Compare normalized hand features", display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                if active_label is None and key in (ord("r"), ord("l")):
                    active_label = "right" if key == ord("r") else "left"
                    active_frames = []
                    print(f"[START] {active_label} hand: hold a steady Rock gesture.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        save_results(trials, labels)


if __name__ == "__main__":
    main()
