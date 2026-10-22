from collections import deque
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

MODEL_PATH = "checkpoints/tiny_gru_rps_best.pth"
META_PATH = "checkpoints/tiny_gru_rps_meta.pkl"

PROCESS_WIDTH = 640
PROCESS_EVERY = 1
CAMERA_FPS = 30
TRAIN_FPS = 120

OAN_DURATION = 0.45
TU_DURATION = 0.45
MAX_DECISION_TIME = 0.80

HAND_READY_FRAMES = 4
REAL_MODEL_FRAMES = 5
REQUIRED_CAPTURE_FRAMES = 7
MIN_CONFIDENCE = 0.68

# Adaptive lock: chỉ chốt khi tay đã ổn định
STABLE_MOTION_THRESHOLD = 0.015
STABLE_FRAMES_REQUIRED = 3
WINDOW_WEIGHTS = [0.20, 0.30, 0.50]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


class TinyGRU(nn.Module):
    def __init__(self, input_size=63, hidden_size=32, num_classes=3, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.classifier(self.dropout(hidden[-1]))


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False,
)
meta = joblib.load(META_PATH)

INPUT_SIZE = int(meta["input_size"])
HIDDEN_SIZE = int(meta["hidden_size"])
WINDOW_SIZE = int(meta["window_size"])
LABEL_NAMES = list(meta["label_names"])

feature_mean = np.asarray(meta["feature_mean"], dtype=np.float32).reshape(-1)
feature_std = np.asarray(meta["feature_std"], dtype=np.float32).reshape(-1)

model = TinyGRU(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_classes=len(LABEL_NAMES),
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print("Labels:", LABEL_NAMES)
print("Train window:", WINDOW_SIZE)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


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


def safe_normalize(v, eps=1e-6):
    norm = np.linalg.norm(v)
    if norm < eps:
        return None
    return v / norm


def extract_local_landmarks(hand_landmarks):
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


def flatten_features(local_landmarks):
    return local_landmarks.reshape(-1).astype(np.float32)


def calculate_motion(current_local, previous_local):
    if current_local is None or previous_local is None:
        return None

    diff = current_local[:, :2] - previous_local[:, :2]
    distances = np.linalg.norm(diff, axis=1)
    return float(np.mean(distances))


def normalize_window(window):
    x = np.asarray(window, dtype=np.float32)
    return (x - feature_mean) / feature_std


def temporal_resample(sequence, target_length):
    sequence = np.asarray(sequence, dtype=np.float32)
    source_length = len(sequence)

    if source_length == target_length:
        return sequence

    old_t = np.linspace(0.0, 1.0, source_length)
    new_t = np.linspace(0.0, 1.0, target_length)

    output = np.empty(
        (target_length, sequence.shape[1]),
        dtype=np.float32,
    )

    for j in range(sequence.shape[1]):
        output[:, j] = np.interp(
            new_t,
            old_t,
            sequence[:, j],
        )

    return output


def infer_probs(sequence):
    sequence = temporal_resample(sequence, WINDOW_SIZE)
    sequence = normalize_window(sequence)

    x = (
        torch.from_numpy(sequence)
        .float()
        .unsqueeze(0)
        .to(device)
    )

    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    return probs.cpu().numpy()


def weighted_recent_vote(capture_buffer):
    frames = list(capture_buffer)

    if len(frames) < REQUIRED_CAPTURE_FRAMES:
        return None, None, None

    # Ưu tiên các frame gần cuối hơn để tránh bị phụ thuộc
    # vào trạng thái trung gian Paper -> Scissors, v.v.
    windows = [
        frames[0:5],
        frames[1:6],
        frames[2:7],
    ]

    weighted_probs = None

    for weight, window in zip(WINDOW_WEIGHTS, windows):
        probs = infer_probs(np.stack(window))

        if weighted_probs is None:
            weighted_probs = weight * probs
        else:
            weighted_probs += weight * probs

    idx = int(np.argmax(weighted_probs))
    label = LABEL_NAMES[idx]
    confidence = float(weighted_probs[idx])

    return label, confidence, weighted_probs


def cpu_winning_move(player_label):
    return {
        "rock": "paper",
        "paper": "scissors",
        "scissors": "rock",
    }[player_label.lower()]


def draw_probability_bar(frame, label, probability, x, y):
    width = 300
    height = 24

    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        (80, 80, 80),
        2,
    )

    fill = int(width * probability)

    cv2.rectangle(
        frame,
        (x, y),
        (x + fill, y + height),
        (255, 255, 255),
        -1,
    )

    cv2.putText(
        frame,
        f"{label}: {probability * 100:.1f}%",
        (x, y - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


STATE_IDLE = "idle"
STATE_WAIT_HAND = "wait_hand"
STATE_OAN = "oan"
STATE_TU = "tu"
STATE_TI = "ti"
STATE_RESULT = "result"
STATE_RETRY = "retry"

state = STATE_IDLE
state_started = time.perf_counter()

capture_buffer = deque(maxlen=REQUIRED_CAPTURE_FRAMES)
hand_ready_count = 0
previous_local = None
stable_count = 0

locked_label = None
locked_conf = 0.0
locked_probs = None
cpu_label = None


def set_state(new_state):
    global state, state_started
    state = new_state
    state_started = time.perf_counter()


def new_round():
    global hand_ready_count
    global previous_local, stable_count
    global locked_label, locked_conf, locked_probs, cpu_label

    capture_buffer.clear()
    hand_ready_count = 0
    previous_local = None
    stable_count = 0

    locked_label = None
    locked_conf = 0.0
    locked_probs = None
    cpu_label = None

    set_state(STATE_WAIT_HAND)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

if not cap.isOpened():
    raise RuntimeError("Không mở được camera.")

frame_index = 0

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.60,
    min_tracking_confidence=0.60,
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        display = frame.copy()

        now = time.perf_counter()
        elapsed = now - state_started

        if state == STATE_OAN and elapsed >= OAN_DURATION:
            set_state(STATE_TU)

        elif state == STATE_TU and elapsed >= TU_DURATION:
            # Chỉ bắt đầu quan sát gesture cuối từ thời điểm TÌ.
            capture_buffer.clear()
            previous_local = None
            stable_count = 0
            set_state(STATE_TI)

        if frame_index % PROCESS_EVERY == 0:
            small = resize_frame(frame)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False

            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]

                mp_drawing.draw_landmarks(
                    display,
                    hand,
                    mp_hands.HAND_CONNECTIONS,
                )

                if state == STATE_WAIT_HAND:
                    hand_ready_count += 1

                    if hand_ready_count >= HAND_READY_FRAMES:
                        hand_ready_count = 0
                        set_state(STATE_OAN)

                elif state == STATE_TI:
                    local_landmarks = extract_local_landmarks(hand)

                    if local_landmarks is not None:
                        motion = calculate_motion(
                            local_landmarks,
                            previous_local,
                        )

                        previous_local = local_landmarks.copy()
                        capture_buffer.append(
                            flatten_features(local_landmarks)
                        )

                        if motion is not None:
                            if motion <= STABLE_MOTION_THRESHOLD:
                                stable_count += 1
                            else:
                                # Tay vẫn đang đổi gesture -> chưa lock.
                                stable_count = 0

            else:
                if state == STATE_WAIT_HAND:
                    hand_ready_count = 0
                elif state == STATE_TI:
                    stable_count = 0
                    previous_local = None

        if state == STATE_TI:
            # Chỉ chốt khi gesture cuối đã ổn định vài frame liên tiếp.
            if (
                stable_count >= STABLE_FRAMES_REQUIRED
                and len(capture_buffer) >= REQUIRED_CAPTURE_FRAMES
            ):
                label, confidence, probs = weighted_recent_vote(
                    capture_buffer
                )

                if (
                    label is not None
                    and confidence >= MIN_CONFIDENCE
                ):
                    locked_label = label
                    locked_conf = confidence
                    locked_probs = probs
                    cpu_label = cpu_winning_move(locked_label)
                    set_state(STATE_RESULT)
                else:
                    set_state(STATE_RETRY)

            elif elapsed >= MAX_DECISION_TIME:
                # Timeout: vẫn thử dùng buffer gần nhất để tránh chờ vô hạn.
                if len(capture_buffer) >= REQUIRED_CAPTURE_FRAMES:
                    label, confidence, probs = weighted_recent_vote(
                        capture_buffer
                    )

                    if (
                        label is not None
                        and confidence >= MIN_CONFIDENCE
                    ):
                        locked_label = label
                        locked_conf = confidence
                        locked_probs = probs
                        cpu_label = cpu_winning_move(locked_label)
                        set_state(STATE_RESULT)
                    else:
                        set_state(STATE_RETRY)
                else:
                    set_state(STATE_RETRY)

        if state == STATE_IDLE:
            cv2.putText(
                display,
                "PRESS SPACE TO START",
                (60, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

        elif state == STATE_WAIT_HAND:
            cv2.putText(
                display,
                "GET READY",
                (185, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (255, 255, 255),
                4,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                "Show your hand to camera",
                (120, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                f"Hand ready: {hand_ready_count}/{HAND_READY_FRAMES}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        elif state == STATE_OAN:
            cv2.putText(
                display,
                "OAN...",
                (220, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.8,
                (255, 255, 255),
                4,
                cv2.LINE_AA,
            )

        elif state == STATE_TU:
            cv2.putText(
                display,
                "TU...",
                (235, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.8,
                (255, 255, 255),
                4,
                cv2.LINE_AA,
            )

        elif state == STATE_TI:
            cv2.putText(
                display,
                "TI!",
                (260, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (255, 255, 255),
                4,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                f"Stable: {stable_count}/{STABLE_FRAMES_REQUIRED}  "
                f"Buffer: {len(capture_buffer)}/{REQUIRED_CAPTURE_FRAMES}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        elif state == STATE_RESULT:
            cv2.putText(
                display,
                "ROUND RESULT",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                f"YOU: {locked_label.upper()} {locked_conf * 100:.1f}%",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                f"CPU: {cpu_label.upper()}",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                "CPU WINS",
                (20, 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

            if locked_probs is not None:
                y = 240

                for i, label in enumerate(LABEL_NAMES):
                    draw_probability_bar(
                        display,
                        label.upper(),
                        float(locked_probs[i]),
                        20,
                        y,
                    )
                    y += 60

        elif state == STATE_RETRY:
            cv2.putText(
                display,
                "TRY AGAIN",
                (190, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (255, 255, 255),
                4,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                "Keep your final gesture visible at TI!",
                (45, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            display,
            "SPACE = Start/New round | ESC = Exit",
            (20, display.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(
            "Tiny GRU RPS - Adaptive Lock V6",
            display,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 32:
            new_round()

        elif key == 27:
            break

cap.release()
cv2.destroyAllWindows()
