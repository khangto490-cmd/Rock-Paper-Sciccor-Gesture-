"""
realtime_static.py
--------------------
Dự đoán TRÊN TỪNG FRAME độc lập, kết quả nhảy liên tục theo đúng hình
dạng bàn tay ngay tại thời điểm đó -- không cần buffer, không cần chờ
đủ số frame như bản LSTM trước.

Chạy: python realtime_static.py
Thoát: bấm Q
"""

import cv2
import joblib
import mediapipe as mp
import numpy as np

from features import extract_features

MODEL_PATH = "static_model.pkl"

COUNTER = {"keo": "bua", "bua": "bao", "bao": "keo"}
DISPLAY_NAME = {"keo": "KEO", "bua": "BUA", "bao": "BAO"}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def main():
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    encoder = bundle["label_encoder"]
    has_proba = hasattr(model, "predict_proba")

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(max_num_hands=1,
                        model_complexity=0,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            pred_label = None
            conf_text = ""

            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                feats = extract_features(hand_lms.landmark)
                if feats is not None:
                    x = np.array(feats).reshape(1, -1)
                    if has_proba:
                        probs = model.predict_proba(x)[0]
                        idx = int(np.argmax(probs))
                        conf_text = f"{probs[idx]:.2f}"
                    else:
                        idx = int(model.predict(x)[0])
                    pred_label = encoder.classes_[idx]

            if pred_label is not None:
                counter_move = COUNTER[pred_label]
                msg = (f"Ban: {DISPLAY_NAME[pred_label]} {conf_text}  ->  "
                       f"MAY: {DISPLAY_NAME[counter_move]}")
                color = (0, 255, 0)
            else:
                msg = "Khong thay tay"
                color = (0, 0, 255)

            cv2.putText(frame, msg, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Keo Bua Bao - Du doan tung frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()