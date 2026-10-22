import time

import cv2
import joblib
import mediapipe as mp
import numpy as np

from features import extract_features

MODEL_PATH = "static_model.pkl"
REPORT_EVERY = 100

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def main():
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    print("Loai model dang dung:", type(model).__name__)
    if hasattr(model, "n_estimators"):
        print("So cay:", model.n_estimators)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("KHONG MO DUOC CAMERA. Kiem tra camera da bat chua, "
              "va co ung dung nao khac (Zoom, trinh duyet...) dang dung khong.")
        return
    print("Do phan giai camera:",
          cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stages = ["doc_camera", "mediapipe", "features", "model", "ve_hien_thi"]
    totals = {s: 0.0 for s in stages}
    n = 0
    t_start = time.perf_counter()

    with mp_hands.Hands(max_num_hands=1,
                        model_complexity=0,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                print("Khong doc duoc khung hinh tu camera, dung chuong trinh.")
                break
            t1 = time.perf_counter()

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            t2 = time.perf_counter()

            feats = None
            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                feats = extract_features(hand_lms.landmark)
            t3 = time.perf_counter()

            if feats is not None:
                x = np.array(feats).reshape(1, -1)
                if hasattr(model, "predict_proba"):
                    model.predict_proba(x)
                else:
                    model.predict(x)
            t4 = time.perf_counter()

            if results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.multi_hand_landmarks[0],
                    mp_hands.HAND_CONNECTIONS)
            cv2.imshow("profile", frame)
            key = cv2.waitKey(1) & 0xFF
            t5 = time.perf_counter()

            for s, dt in zip(stages, [t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4]):
                totals[s] += dt
            n += 1

            if n % REPORT_EVERY == 0:
                elapsed = time.perf_counter() - t_start
                print(f"\n--- {REPORT_EVERY} frame gan nhat (ms/frame) ---")
                for s in stages:
                    print(f"  {s:<12}: {totals[s] / REPORT_EVERY * 1000:6.1f}")
                total_ms = sum(totals.values()) / REPORT_EVERY * 1000
                print(f"  {'TONG':<12}: {total_ms:6.1f}   (~{REPORT_EVERY / elapsed:.1f} FPS)")
                totals = {s: 0.0 for s in stages}
                t_start = time.perf_counter()

            if key == ord('q'):
                break

    # In nốt phần trung bình của các frame chưa đủ 100 (để bấm Q sớm vẫn có số)
    remain = n % REPORT_EVERY
    if remain > 0:
        print(f"\n--- {remain} frame cuoi (ms/frame) ---")
        for s in stages:
            print(f"  {s:<12}: {totals[s] / remain * 1000:6.1f}")
        print(f"  {'TONG':<12}: {sum(totals.values()) / remain * 1000:6.1f}")
    if n == 0:
        print("Khong doc duoc frame nao tu camera.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
