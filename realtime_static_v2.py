"""
realtime_static_v2.py
----------------------
Dự đoán từng frame bằng model M7 (11 đặc trưng từng ngón + nhãn báo trước 3 frame).

Cần cùng thư mục: features.py, static_model_v2.pkl
Chạy: python realtime_static_v2.py      Thoát: bấm Q
"""

import time

import cv2
import joblib
import mediapipe as mp
import numpy as np

from features import extract_features, engineered_features_batch

MODEL_PATH = "static_model_v2.pkl"

SHOW_LANDMARKS = True      # đặt False để bỏ vẽ khung xương (nhẹ hơn một chút)
SHOW_FPS = True

COUNTER = {"keo": "bua", "bua": "bao", "bao": "keo"}
DISPLAY_NAME = {"keo": "KEO", "bua": "BUA", "bao": "BAO"}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def main():
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    classes = bundle["label_encoder"].classes_

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Khong mo duoc camera. Kiem tra camera da bat chua.")
        return

    last_t, fps = time.perf_counter(), 0.0

    with mp_hands.Hands(max_num_hands=1,
                        model_complexity=0,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            label, conf = None, 0.0
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                if SHOW_LANDMARKS:
                    mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                feats = extract_features(hand.landmark)          # 63 số, chuẩn hóa
                if feats is not None:
                    x = engineered_features_batch(np.array(feats))  # (1, 11)
                    probs = model.predict_proba(x)[0]
                    i = int(np.argmax(probs))
                    label, conf = classes[i], float(probs[i])

            if label is not None:
                msg = (f"Ban: {DISPLAY_NAME[label]} {conf:.2f}  ->  "
                       f"MAY: {DISPLAY_NAME[COUNTER[label]]}")
                color = (0, 255, 0)
            else:
                msg, color = "Khong thay tay", (0, 0, 255)
            cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if SHOW_FPS:
                now = time.perf_counter()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_t, 1e-6))
                last_t = now
                cv2.putText(frame, f"{fps:.0f} FPS", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Keo Bua Bao v2", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
