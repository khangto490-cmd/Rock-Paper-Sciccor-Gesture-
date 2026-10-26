from collections import deque
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from gesture_gate import gate_check

MODEL_PATH = "checkpoints/tiny_gru_rps_best.pth"
META_PATH = "checkpoints/tiny_gru_rps_meta.pkl"

PROCESS_WIDTH = 640
PROCESS_EVERY = 1
CAMERA_FPS = 30
TRAIN_FPS = 120

OAN_DURATION = 0.45
TU_DURATION = 0.45
MAX_DECISION_TIME = 0.80
MIN_DECISION_TIME = 0.08  # giảm thời gian chờ nhưng vẫn tránh chốt ngay lập tức

HAND_READY_FRAMES = 4
REAL_MODEL_FRAMES = 5
REQUIRED_CAPTURE_FRAMES = 7
MIN_CONFIDENCE = 0.80

# Balanced lock V7.2: palm + finger motion, responsive EMA, hysteresis
STABILITY_PALM_POINTS = np.array([0, 5, 9, 13, 17], dtype=np.int64)
STABILITY_FINGER_POINTS = np.array(
    [3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20],
    dtype=np.int64,
)
STABLE_MOTION_THRESHOLD = 0.015
MOVING_RESET_THRESHOLD = 0.030
MOTION_EMA_ALPHA = 0.70  # phản ứng nhanh hơn với thay đổi cử chỉ
STABLE_FRAMES_REQUIRED = 1  # giảm thêm độ trễ; vẫn cần đủ buffer và vote nhiều cửa sổ
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


# ===== PERFORMANCE PROFILING =====
PROFILE_EVERY = 120

profile_data = {
    name: deque(maxlen=PROFILE_EVERY)
    for name in (
        "camera_read",
        "preprocess",
        "mediapipe",
        "landmark_drawing",
        "inference_prepare",
        "model_inference",
        "loop",
    )
}


def record_profile(name, elapsed_seconds):
    profile_data[name].append(elapsed_seconds * 1000.0)


def profile_average(name):
    values = profile_data[name]
    if not values:
        return None
    return sum(values) / len(values)


def profile_text(name):
    value = profile_average(name)
    return f"{value:.2f} ms" if value is not None else "N/A"


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
    """Đo cả chuyển động khung bàn tay và thay đổi hình dạng ngón."""
    if current_local is None or previous_local is None:
        return None

    def mean_displacement(indices):
        diff = (
            current_local[indices, :2]
            - previous_local[indices, :2]
        )
        return float(
            np.mean(np.linalg.norm(diff, axis=1))
        )

    palm_motion = mean_displacement(STABILITY_PALM_POINTS)
    finger_motion = mean_displacement(STABILITY_FINGER_POINTS)

    # max giúp không bỏ qua trường hợp lòng bàn tay đứng yên
    # nhưng các ngón vẫn đang chuyển từ Paper sang Scissors.
    return max(palm_motion, finger_motion)


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

def infer_probs_batch(sequences):
    """Chạy nhiều window trong một forward pass thay vì gọi GRU nhiều lần."""
    prepare_started = time.perf_counter()
    prepared = []

    for sequence in sequences:
        sequence = temporal_resample(sequence, WINDOW_SIZE)
        sequence = normalize_window(sequence)
        prepared.append(sequence)

    record_profile(
        "inference_prepare",
        time.perf_counter() - prepare_started,
    )           

    batch = np.stack(prepared).astype(np.float32)

    
    x = torch.from_numpy(batch).to(device)

    # Đồng bộ GPU để đo thời gian thực thi model chính xác hơn.
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    inference_started = time.perf_counter()

    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    record_profile(
        "model_inference",
        time.perf_counter() - inference_started,
    )

    return probs.cpu().numpy()



def weighted_recent_vote(capture_buffer):
    frames = list(capture_buffer)

    if len(frames) < REQUIRED_CAPTURE_FRAMES:
        return None, None, None

    # Các window chồng lấp; window cuối có trọng số lớn nhất.
    windows = [
        np.stack(frames[0:5]),
        np.stack(frames[1:6]),
        np.stack(frames[2:7]),
    ]

    # Một lần gọi model cho cả batch, giảm overhead ở frame lock.
    all_probs = infer_probs_batch(windows)
    weighted_probs = np.sum(
        all_probs * np.asarray(WINDOW_WEIGHTS, dtype=np.float32)[:, None],
        axis=0,
    )

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
motion_ema = None

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
    global previous_local, stable_count, motion_ema
    global locked_label, locked_conf, locked_probs, cpu_label

    capture_buffer.clear()
    hand_ready_count = 0
    previous_local = None
    stable_count = 0
    motion_ema = None

    locked_label = None
    locked_conf = 0.0
    locked_probs = None
    cpu_label = None

    set_state(STATE_WAIT_HAND)

def try_lock(label, confidence, probs):
    global locked_label, locked_conf, locked_probs, cpu_label

    if label is None or confidence is None or probs is None:
        print("[DECISION] No valid prediction; reject.")
        set_state(STATE_RETRY)
        return

    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    if probs.size < 2:
        print(f"[DECISION] Invalid probability vector: {probs}")
        set_state(STATE_RETRY)
        return

    # Diagnostic only: log the margin before applying the current threshold.
    # Do not reject by margin yet; calibrate it using known and unknown samples.
    sorted_probs = np.sort(probs)
    margin = float(sorted_probs[-1] - sorted_probs[-2])

    print(
        f"[DECISION] {label} "
        f"confidence={confidence:.3f} margin={margin:.3f} "
        f"probs={np.round(probs, 3)}"
    )

    if confidence < MIN_CONFIDENCE:
        print(
            f"[DECISION] REJECT: confidence {confidence:.3f} "
            f"< MIN_CONFIDENCE {MIN_CONFIDENCE:.2f}"
        )
        set_state(STATE_RETRY)
        return

    ok, info = gate_check(capture_buffer, label)
    print(f"[GATE] {'OK' if ok else 'REJECT'} | {info}")

    if not ok:
        set_state(STATE_RETRY)
        return

    locked_label = label
    locked_conf = confidence
    locked_probs = probs
    cpu_label = cpu_winning_move(label)
    set_state(STATE_RESULT)


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
        loop_started = time.perf_counter()

        read_started = time.perf_counter()
        ret, frame = cap.read()
        record_profile(
            "camera_read",
            time.perf_counter() - read_started,
        )
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
            motion_ema = None
            set_state(STATE_TI)

        if frame_index % PROCESS_EVERY == 0:
            preprocess_started = time.perf_counter()

            small = resize_frame(frame)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False

            record_profile(
                "preprocess",
                time.perf_counter() - preprocess_started,
            )

            mediapipe_started = time.perf_counter()
            results = hands.process(rgb)
            record_profile(
                "mediapipe",
                time.perf_counter() - mediapipe_started,
            )

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]

                drawing_started = time.perf_counter()

                mp_drawing.draw_landmarks(
                    display,
                    hand,
                    mp_hands.HAND_CONNECTIONS,
                )

                record_profile(
                    "landmark_drawing",
                    time.perf_counter() - drawing_started,
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
                            if motion_ema is None:
                                motion_ema = motion
                            else:
                                motion_ema = (
                                    MOTION_EMA_ALPHA * motion
                                    + (1.0 - MOTION_EMA_ALPHA) * motion_ema
                                )

                            if motion_ema <= STABLE_MOTION_THRESHOLD:
                                stable_count += 1
                            elif motion_ema >= MOVING_RESET_THRESHOLD:
                                # Chuyển động rõ ràng: reset để không lock gesture trung gian.
                                stable_count = 0
                            else:
                                # Vùng lưng chừng/jitter nhẹ: giảm dần, không reset mạnh.
                                stable_count = max(0, stable_count - 1)

            else:
                if state == STATE_WAIT_HAND:
                    hand_ready_count = 0
                elif state == STATE_TI:
                    stable_count = 0
                    previous_local = None
                    motion_ema = None

        if state == STATE_TI:
            # Tính thời gian từ lúc vào trạng thái TI, không dùng elapsed của TU.
            decision_elapsed = time.perf_counter() - state_started
            buffer_full = len(capture_buffer) >= REQUIRED_CAPTURE_FRAMES
            # Chỉ chốt khi gesture cuối đã ổn định vài frame liên tiếp.
            if (
                decision_elapsed >= MIN_DECISION_TIME
                and stable_count >= STABLE_FRAMES_REQUIRED
                and buffer_full
            ):
                try_lock(*weighted_recent_vote(capture_buffer))

            elif decision_elapsed >= MAX_DECISION_TIME:
                if buffer_full:
                    try_lock(*weighted_recent_vote(capture_buffer))
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
            "Tiny GRU RPS - V7.2 Profiling",
            display,
        )

        key = cv2.waitKey(1) & 0xFF

        record_profile(
            "loop",
            time.perf_counter() - loop_started,
        )

        if frame_index > 0 and frame_index % PROFILE_EVERY == 0:
            loop_ms = profile_average("loop")
            fps = 1000.0 / loop_ms if loop_ms else 0.0

            print(
                f"\n[PROFILE] Frame {frame_index} | "
                f"FPS ~ {fps:.1f}"
            )
            print(
                f"Camera read : {profile_text('camera_read')}"
            )
            print(
                f"Preprocess  : {profile_text('preprocess')}"
            )
            print(
                f"MediaPipe   : {profile_text('mediapipe')}"
            )
            print(
                f"Draw points : {profile_text('landmark_drawing')}"
            )
            print(
                f"Prepare seq : {profile_text('inference_prepare')}"
            )
            print(
                f"Model       : {profile_text('model_inference')}"
            )
            print(
                f"Total loop  : {profile_text('loop')}"
            )


        if key == 32:
            new_round()

        elif key == 27:
            break

cap.release()
cv2.destroyAllWindows()
