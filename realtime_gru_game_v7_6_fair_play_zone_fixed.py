import time
import random

import cv2
import joblib
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from gesture_gate_stable import gate_check
from collections import deque

MODEL_PATH = "checkpoints/tiny_gru_rps_best.pth"
META_PATH = "checkpoints/tiny_gru_rps_meta.pkl"

PROCESS_WIDTH = 640
PROCESS_EVERY = 1
CAMERA_FPS = 30
TRAIN_FPS = 120

OAN_DURATION = 0.45
TU_DURATION = 0.45
# Fixed reveal window: capture starts exactly at TI! and closes after this duration.
CAPTURE_DURATION = 1.20
CAPTURE_BUFFER_FRAMES = 36  # xấp xỉ 1.2 giây ở 30 FPS
MIN_CAPTURE_FRAMES = 7      # số frame tối thiểu để tạo các cửa sổ dự đoán

# Fair Play Zone (tọa độ tương đối theo khung hình camera).
ZONE_X_MIN = 0.20
ZONE_Y_MIN = 0.10
ZONE_X_MAX = 0.80
ZONE_Y_MAX = 0.90

HAND_READY_FRAMES = 4
REAL_MODEL_FRAMES = 5
MIN_CONFIDENCE = 0.80
HAND_SIDE_CONFIRM_FRAMES = 2  # lọc nhãn handedness chập chờn trong 1 frame
MAX_MISSED_HAND_FRAMES = 1  # chịu được 1 frame mất tracking; mất lâu hơn thì xóa buffer

# Current tested inference configuration: flip local Z for the physical left hand.
# This is an inference-time transform only; the checkpoint/training data are unchanged.
Z_FLIP_LEFT = True
# The input passed to MediaPipe is the raw webcam frame (not cv2.flip'ed).
# MediaPipe Hands handedness labels assume mirrored selfie input; when the
# input is unmirrored, swap its Left/Right label to estimate the physical hand.
INPUT_IMAGE_IS_MIRRORED = False

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
print(f"Fair Play Zone: x={ZONE_X_MIN:.2f}-{ZONE_X_MAX:.2f}, y={ZONE_Y_MIN:.2f}-{ZONE_Y_MAX:.2f}")
print(f"Fixed capture: {CAPTURE_DURATION:.2f}s, buffer={CAPTURE_BUFFER_FRAMES} frames")


def warmup_model(repeats=3):
    """Warm up the exact GRU batch shape used by weighted_recent_vote().

    Warm-up runs before the webcam loop and is intentionally excluded from
    runtime profiling so the first game round is less likely to absorb CUDA
    context/cuDNN initialization cost.
    """
    dummy = torch.zeros(
        (3, WINDOW_SIZE, INPUT_SIZE),
        dtype=torch.float32,
        device=device,
    )
    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(max(1, int(repeats))):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    print(f"Model warm-up: {repeats} passes, {elapsed_ms:.2f} ms (excluded from profile)")


warmup_model()


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
    values = profile_data[name]
    if not values:
        return "N/A (n=0)"
    average = sum(values) / len(values)
    maximum = max(values)
    return f"avg {average:.2f} ms | max {maximum:.2f} ms | n={len(values)}"


def clear_profile_interval():
    """Start a fresh measurement window after each PROFILE_EVERY frames."""
    for values in profile_data.values():
        values.clear()


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def get_physical_hand_side(results):
    """Return (physical_side, raw_mediapipe_label).

    Handedness is only a best-effort diagnostic; check the printed label while
    testing. MediaPipe Hands assumes mirrored selfie input for Left/Right.
    """
    if not getattr(results, "multi_handedness", None):
        return "unknown", "unknown"

    try:
        raw_label = results.multi_handedness[0].classification[0].label
    except (AttributeError, IndexError, TypeError):
        return "unknown", "unknown"

    raw_side = raw_label.strip().lower()
    if raw_side not in {"left", "right"}:
        return "unknown", str(raw_label)

    if INPUT_IMAGE_IS_MIRRORED:
        physical_side = raw_side
    else:
        # Raw, unmirrored camera input: MediaPipe's assumed selfie label is swapped.
        physical_side = "left" if raw_side == "right" else "right"

    return physical_side, str(raw_label)


def apply_hand_specific_transform(local_landmarks, physical_hand_side):
    """Apply the tested left-hand local-Z transform and report whether applied."""
    transformed = np.asarray(local_landmarks, dtype=np.float32).copy()
    z_flipped = bool(Z_FLIP_LEFT and physical_hand_side == "left")
    if z_flipped:
        transformed[:, 2] *= -1.0
    return transformed, z_flipped


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


def hand_inside_fair_zone(hand_landmarks):
    """Require the full detected hand landmark bounding box to stay inside the ROI."""
    xs = np.asarray([lm.x for lm in hand_landmarks.landmark], dtype=np.float32)
    ys = np.asarray([lm.y for lm in hand_landmarks.landmark], dtype=np.float32)
    return bool(
        xs.min() >= ZONE_X_MIN
        and xs.max() <= ZONE_X_MAX
        and ys.min() >= ZONE_Y_MIN
        and ys.max() <= ZONE_Y_MAX
    )


def draw_fair_zone(frame, status):
    """Draw the fixed player zone. status is inside, outside, or missing."""
    h, w = frame.shape[:2]
    x1, y1 = int(ZONE_X_MIN * w), int(ZONE_Y_MIN * h)
    x2, y2 = int(ZONE_X_MAX * w), int(ZONE_Y_MAX * h)
    color = {
        "inside": (0, 220, 0),
        "outside": (0, 0, 255),
        "missing": (0, 200, 255),
    }.get(status, (0, 200, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = {
        "inside": "FAIR ZONE: OK",
        "outside": "WARNING: MOVE HAND INTO ZONE",
        "missing": "PLACE ONE HAND INSIDE THE ZONE",
    }.get(status, "FAIR PLAY ZONE")
    cv2.putText(frame, label, (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


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

    if len(frames) < MIN_CAPTURE_FRAMES:
        return None, None, None

    # Lấy 7 mốc trải đều trên toàn bộ cửa sổ capture, không chỉ 7 frame đầu.
    sample_indices = np.linspace(0, len(frames) - 1, 7).round().astype(int)
    sampled = [frames[i] for i in sample_indices]
    # Ba cửa sổ chồng lấp; cửa sổ cuối đại diện cho cử chỉ gần thời điểm khóa nhất.
    windows = [
        np.stack(sampled[0:5]),
        np.stack(sampled[1:6]),
        np.stack(sampled[2:7]),
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


def choose_cpu_move_before_round():
    """Commit the CPU move before the player reveals their gesture."""
    return random.choice(("rock", "paper", "scissors"))


def get_round_outcome(player_label, computer_label):
    if player_label == computer_label:
        return "DRAW"
    winning_pairs = {
        ("rock", "scissors"),
        ("paper", "rock"),
        ("scissors", "paper"),
    }
    return "YOU WIN" if (player_label, computer_label) in winning_pairs else "CPU WINS"


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

capture_buffer = deque(maxlen=CAPTURE_BUFFER_FRAMES)
hand_ready_count = 0
previous_local = None
stable_count = 0
motion_ema = None
# Lock the physical hand side for a single TI/capture window. This prevents
# left-hand Z-flipped frames and right-hand unflipped frames entering one GRU window.
capture_hand_side = "unknown"
capture_zflip_applied = False
pending_hand_side = "unknown"
pending_hand_side_count = 0
missed_hand_frames = 0

locked_label = None
locked_conf = 0.0
locked_probs = None
cpu_label = None
current_physical_hand_side = "unknown"
current_mediapipe_handedness_label = "unknown"


def set_state(new_state):
    global state, state_started
    state = new_state
    state_started = time.perf_counter()


def new_round():
    global hand_ready_count
    global previous_local, stable_count, motion_ema
    global capture_hand_side, capture_zflip_applied
    global pending_hand_side, pending_hand_side_count, missed_hand_frames
    global locked_label, locked_conf, locked_probs, cpu_label

    capture_buffer.clear()
    hand_ready_count = 0
    previous_local = None
    stable_count = 0
    motion_ema = None
    capture_hand_side = "unknown"
    capture_zflip_applied = False
    pending_hand_side = "unknown"
    pending_hand_side_count = 0
    missed_hand_frames = 0

    locked_label = None
    locked_conf = 0.0
    locked_probs = None
    # CPU commits its move before countdown/capture, never after seeing the player.
    cpu_label = choose_cpu_move_before_round()
    print(f"[CPU COMMIT] move={cpu_label}")

    set_state(STATE_WAIT_HAND)

def try_lock(label, confidence, probs, capture_frames=None):
    global locked_label, locked_conf, locked_probs, cpu_label

    if label is None or confidence is None or probs is None:
        print("[DECISION] No valid prediction; reject.")
        set_state(STATE_RETRY)
        return

    if capture_hand_side not in {"left", "right"}:
        print("[CAPTURE] Physical hand side could not be verified; rejecting this round.")
        set_state(STATE_RETRY)
        return

    print(
        f"[HAND] captured_physical={capture_hand_side} "
        f"| current_physical={current_physical_hand_side} "
        f"| MediaPipe_label={current_mediapipe_handedness_label} "
        f"| zflip_applied={capture_zflip_applied}"
    )

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

    gate_input = capture_frames if capture_frames is not None else tuple(capture_buffer)
    ok, info = gate_check(gate_input, label)
    print(f"[GATE] {'OK' if ok else 'REJECT'} | {info}")

    if not ok:
        set_state(STATE_RETRY)
        return

    locked_label = label
    locked_conf = confidence
    locked_probs = probs
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
            capture_hand_side = "unknown"
            capture_zflip_applied = False
            pending_hand_side = "unknown"
            pending_hand_side_count = 0
            missed_hand_frames = 0
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
                current_physical_hand_side, current_mediapipe_handedness_label = (
                    get_physical_hand_side(results)
                )
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

                hand_in_zone_now = hand_inside_fair_zone(hand)

                if state == STATE_WAIT_HAND:
                    if hand_in_zone_now:
                        hand_ready_count += 1
                    else:
                        hand_ready_count = 0

                    if hand_ready_count >= HAND_READY_FRAMES:
                        hand_ready_count = 0
                        set_state(STATE_OAN)

                elif state in (STATE_OAN, STATE_TU) and not hand_in_zone_now:
                    # Do not continue the countdown if the player leaves the zone.
                    hand_ready_count = 0
                    set_state(STATE_WAIT_HAND)

                elif state == STATE_TI and not hand_in_zone_now:
                    print("[FAIR ZONE] Hand left the zone during capture; round invalidated.")
                    capture_buffer.clear()
                    set_state(STATE_RETRY)

                elif state == STATE_TI and hand_in_zone_now:
                    local_landmarks = extract_local_landmarks(hand)

                    if local_landmarks is not None:
                        detected_side = current_physical_hand_side

                        # A single-frame handedness glitch should not discard a valid
                        # capture. Require two consecutive frames before locking a side
                        # or switching from one physical hand to the other.
                        missed_hand_frames = 0
                        accept_frame = False
                        if detected_side not in {"left", "right"}:
                            if capture_buffer or capture_hand_side != "unknown":
                                print("[CAPTURE] Hand side unknown; clearing current window.")
                            capture_buffer.clear()
                            previous_local = None
                            stable_count = 0
                            motion_ema = None
                            capture_hand_side = "unknown"
                            capture_zflip_applied = False
                            pending_hand_side = "unknown"
                            pending_hand_side_count = 0
                        elif detected_side == capture_hand_side:
                            # Confirmed current hand: cancel any one-frame opposite-side glitch.
                            pending_hand_side = "unknown"
                            pending_hand_side_count = 0
                            accept_frame = True
                        else:
                            # First frame of a new/unknown side starts a candidate.
                            if detected_side == pending_hand_side:
                                pending_hand_side_count += 1
                            else:
                                pending_hand_side = detected_side
                                pending_hand_side_count = 1

                            if pending_hand_side_count >= HAND_SIDE_CONFIRM_FRAMES:
                                if capture_hand_side != "unknown":
                                    print(
                                        f"[CAPTURE] Hand changed {capture_hand_side} -> {detected_side}; "
                                        "round invalidated to prevent mixed-hand capture."
                                    )
                                    capture_buffer.clear()
                                    pending_hand_side = "unknown"
                                    pending_hand_side_count = 0
                                    set_state(STATE_RETRY)
                                    accept_frame = False
                                else:
                                    capture_hand_side = detected_side
                                    pending_hand_side = "unknown"
                                    pending_hand_side_count = 0
                                    accept_frame = True

                        if accept_frame and state == STATE_TI:
                            local_landmarks, capture_zflip_applied = apply_hand_specific_transform(
                                local_landmarks, capture_hand_side
                            )
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
                current_physical_hand_side = "unknown"
                current_mediapipe_handedness_label = "unknown"
                if state == STATE_WAIT_HAND:
                    hand_ready_count = 0
                elif state in (STATE_OAN, STATE_TU):
                    hand_ready_count = 0
                    set_state(STATE_WAIT_HAND)
                elif state == STATE_TI:
                    # A one-frame MediaPipe dropout is common enough to tolerate.
                    # Keep the current same-hand window, but break motion continuity.
                    missed_hand_frames += 1
                    stable_count = 0
                    previous_local = None
                    motion_ema = None
                    pending_hand_side = "unknown"
                    pending_hand_side_count = 0

                    if missed_hand_frames > MAX_MISSED_HAND_FRAMES:
                        # Longer tracking loss: discard old data so a later hand
                        # cannot be mixed into the previous GRU window.
                        capture_buffer.clear()
                        capture_hand_side = "unknown"
                        capture_zflip_applied = False
                        print("[CAPTURE] Tracking lost for too long; round will be rejected.")
                        set_state(STATE_RETRY)

        if state == STATE_TI:
            # Fixed deadline: do not predict early and never extend this window.
            capture_elapsed = time.perf_counter() - state_started
            if capture_elapsed >= CAPTURE_DURATION:
                print(
                    f"[CAPTURE LOCK] elapsed={capture_elapsed:.3f}s "
                    f"frames={len(capture_buffer)}/{CAPTURE_BUFFER_FRAMES}"
                )
                if len(capture_buffer) == CAPTURE_BUFFER_FRAMES:
                    # Freeze an exact-length snapshot. No partial capture is sent to GRU.
                    frozen_capture = tuple(capture_buffer)
                    print(
                        f"[CAPTURE] Complete fixed window: "
                        f"{len(frozen_capture)}/{CAPTURE_BUFFER_FRAMES} frames."
                    )
                    prediction = weighted_recent_vote(frozen_capture)
                    try_lock(*prediction, capture_frames=frozen_capture)
                else:
                    print(
                        f"[CAPTURE] Incomplete fixed window: "
                        f"{len(capture_buffer)}/{CAPTURE_BUFFER_FRAMES} frames; "
                        "round rejected."
                    )
                    capture_buffer.clear()
                    set_state(STATE_RETRY)
              

        # Draw the same normalized ROI over the displayed camera frame.
        if results.multi_hand_landmarks:
            zone_status = "inside" if hand_inside_fair_zone(results.multi_hand_landmarks[0]) else "outside"
        else:
            zone_status = "missing"
        draw_fair_zone(display, zone_status)

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
            capture_elapsed = min(time.perf_counter() - state_started, CAPTURE_DURATION)
            capture_progress = capture_elapsed / CAPTURE_DURATION
            cv2.putText(
                display,
                "TI! - HOLD YOUR GESTURE",
                (75, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (255, 255, 255),
                4,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                f"Capture: {capture_elapsed:.2f}/{CAPTURE_DURATION:.2f}s  "
                f"Frames: {len(capture_buffer)}/{CAPTURE_BUFFER_FRAMES}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            bar_x, bar_y, bar_w, bar_h = 20, 95, 260, 14
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)
            cv2.rectangle(display, (bar_x, bar_y),
                          (bar_x + int(bar_w * capture_progress), bar_y + bar_h),
                          (0, 220, 0), -1)

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
                f"CPU: {cpu_label.upper() if cpu_label else 'NOT SET'}",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            outcome_text = get_round_outcome(locked_label, cpu_label)
            cv2.putText(
                display,
                outcome_text,
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
                "Keep your whole hand inside the green zone and retry.",
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

        
        mode_text = (
            f"Hand: {current_physical_hand_side} | "
            f"Capture: {capture_hand_side} | "
            f"Side check: {pending_hand_side_count}/{HAND_SIDE_CONFIRM_FRAMES} | "
            f"Missed: {missed_hand_frames} | "
            f"Z-flip applied: {capture_zflip_applied}"
        )
        cv2.putText(
            display,
            mode_text,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(
            "Tiny GRU RPS - Fair Play Zone V7.6",
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
                f"\n[PROFILE V7.6] Frame {frame_index} | "
                f"FPS ~ {fps:.1f} | interval={PROFILE_EVERY} frames"
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
            clear_profile_interval()


        if key == 32:
            new_round()

        elif key == 27:
            break

cap.release()
cv2.destroyAllWindows()
