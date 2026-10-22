import cv2
import mediapipe as mp
import time

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)
mp_draw = mp.solutions.drawing_utils

# ================= WEBCAM =================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Không mở được webcam")
    exit()

prev_time = 0
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không đọc được frame")
        break

    frame = cv2.flip(frame, 1)

    # ===== FPS (smooth) =====
    curr_time = time.time()
    if prev_time != 0:
        fps = 0.9 * fps + 0.1 * (1 / (curr_time - prev_time))
    prev_time = curr_time

    # ===== Mediapipe =====
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    left_detected = False
    right_detected = False

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(
            result.multi_hand_landmarks,
            result.multi_handedness
        ):
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            hand_label = handedness.classification[0].label  # Left / Right

            # Lấy vị trí cổ tay để hiện chữ gần tay
            wrist = hand_landmarks.landmark[0]
            h, w, _ = frame.shape
            cx, cy = int(wrist.x * w), int(wrist.y * h)

            if hand_label == "Left":
                left_detected = True
                color = (255, 0, 0)
            else:
                right_detected = True
                color = (0, 0, 255)

            cv2.putText(
                frame,
                hand_label.upper(),
                (cx - 30, cy - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

    # ===== UI GÓC TRÊN TRÁI =====
    x = 10
    y = 40
    line_gap = 40

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    if left_detected:
        cv2.putText(
            frame,
            "LEFT",
            (x, y + line_gap),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

    if right_detected:
        cv2.putText(
            frame,
            "RIGHT",
            (x, y + 2 * line_gap),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow("Gesture Webcam", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
