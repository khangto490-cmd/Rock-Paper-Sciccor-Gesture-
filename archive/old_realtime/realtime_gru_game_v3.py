from collections import deque, defaultdict
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "checkpoints/tiny_gru_rps_best.pth"
META_PATH = "checkpoints/tiny_gru_rps_meta.pkl"

PROCESS_WIDTH = 640
PROCESS_EVERY = 1

CAMERA_FPS = 30
TRAIN_FPS = 120

EMA_ALPHA = 0.60

# Nhịp chơi
OAN_DURATION = 0.45
TU_DURATION = 0.45
TI_DURATION = 0.20
CAPTURE_DURATION = 0.22

# Chỉ nhận vote nếu confidence đủ cao
MIN_CONFIDENCE = 0.72

# Nếu tổng vote hợp lệ quá ít thì yêu cầu chơi lại
MIN_VALID_VOTES = 2

REAL_BUFFER_SIZE = 6
# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# MODEL
# ============================================================

class TinyGRU(nn.Module):
    def __init__(
        self,
        input_size=63,
        hidden_size=32,
        num_classes=3,
        dropout=0.2,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            hidden_size,
            num_classes,
        )

    def forward(self, x):
        _, hidden = self.gru(x)
        last_hidden = hidden[-1]
        last_hidden = self.dropout(last_hidden)
        return self.classifier(last_hidden)


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

feature_mean = np.asarray(
    meta["feature_mean"],
    dtype=np.float32,
).reshape(-1)

feature_std = np.asarray(
    meta["feature_std"],
    dtype=np.float32,
).reshape(-1)

model = TinyGRU(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_classes=len(LABEL_NAMES),
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()

print("Labels:", LABEL_NAMES)
print("Window size:", WINDOW_SIZE)


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# ============================================================
# HELPERS
# ============================================================

def resize_frame(frame):
    h, w = frame.shape[:2]

    if w <= PROCESS_WIDTH:
        return frame

    scale = PROCESS_WIDTH / w

    return cv2.resize(
        frame,
        (
            PROCESS_WIDTH,
            int(h * scale),
        ),
        interpolation=cv2.INTER_AREA,
    )


def safe_normalize(v, eps=1e-6):
    norm = np.linalg.norm(v)

    if norm < eps:
        return None

    return v / norm


def extract_features(hand_landmarks):
    landmarks = np.array(
        [
            [lm.x, lm.y, lm.z]
            for lm in hand_landmarks.landmark
        ],
        dtype=np.float32,
    )

    wrist = landmarks[0].copy()
    centered = landmarks - wrist

    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    pinky_mcp = landmarks[17]

    x_axis = safe_normalize(
        index_mcp - pinky_mcp
    )

    if x_axis is None:
        return None

    y_temp = middle_mcp - wrist

    y_temp = (
        y_temp
        - np.dot(
            y_temp,
            x_axis,
        ) * x_axis
    )

    y_axis = safe_normalize(y_temp)

    if y_axis is None:
        return None

    z_axis = safe_normalize(
        np.cross(
            x_axis,
            y_axis,
        )
    )

    if z_axis is None:
        return None

    y_axis = safe_normalize(
        np.cross(
            z_axis,
            x_axis,
        )
    )

    if y_axis is None:
        return None

    basis = np.stack(
        [
            x_axis,
            y_axis,
            z_axis,
        ],
        axis=1,
    )

    local_landmarks = centered @ basis

    palm_scale = np.linalg.norm(
        middle_mcp - wrist
    )

    if palm_scale > 1e-6:
        local_landmarks /= palm_scale

    return (
        local_landmarks
        .reshape(-1)
        .astype(np.float32)
    )


def normalize_window(window):
    x = window.astype(np.float32)

    x = (
        x - feature_mean
    ) / feature_std

    return x


def update_ema(previous, current):
    if previous is None:
        return current.copy()

    return (
        EMA_ALPHA * current
        + (1.0 - EMA_ALPHA) * previous
    )


def cpu_winning_move(player_label):
    win_map = {
        "rock": "paper",
        "paper": "scissors",
        "scissors": "rock",
    }

    return win_map.get(
        player_label.lower()
    )


def temporal_resample(
    sequence,
    target_length,
):
    sequence = np.asarray(
        sequence,
        dtype=np.float32,
    )

    source_length = sequence.shape[0]

    if source_length == target_length:
        return sequence

    old_time = np.linspace(
        0.0,
        1.0,
        source_length,
    )

    new_time = np.linspace(
        0.0,
        1.0,
        target_length,
    )

    output = np.zeros(
        (
            target_length,
            sequence.shape[1],
        ),
        dtype=np.float32,
    )

    for feature_idx in range(
        sequence.shape[1]
    ):
        output[:, feature_idx] = np.interp(
            new_time,
            old_time,
            sequence[:, feature_idx],
        )

    return output


def predict_from_buffer(
    sequence_buffer,
    ema_probs,
):
    if len(sequence_buffer) < REAL_BUFFER_SIZE:
        return ema_probs, None, None

    window = np.stack(
        sequence_buffer
    )

    window = temporal_resample(
        window,
        WINDOW_SIZE,
    )

    window = normalize_window(
        window
    )

    x = (
        torch
        .from_numpy(window)
        .float()
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        logits = model(x)

        probs = torch.softmax(
            logits,
            dim=1,
        )[0]

    probs = (
        probs
        .cpu()
        .numpy()
    )

    ema_probs = update_ema(
        ema_probs,
        probs,
    )

    idx = int(
        np.argmax(
            ema_probs
        )
    )

    label = LABEL_NAMES[idx]
    conf = float(
        ema_probs[idx]
    )

    return (
        ema_probs,
        label,
        conf,
    )


def choose_vote_winner(votes):
    total_votes = sum(
        len(v)
        for v in votes.values()
    )

    if total_votes < MIN_VALID_VOTES:
        return None, 0.0, total_votes

    best_label = None
    best_score = -1.0
    best_conf = 0.0

    for label, confs in votes.items():
        if not confs:
            continue

        count = len(confs)
        mean_conf = float(
            np.mean(confs)
        )

        score = (
            count * mean_conf
        )

        if score > best_score:
            best_score = score
            best_label = label
            best_conf = mean_conf

    return (
        best_label,
        best_conf,
        total_votes,
    )


def draw_probability_bar(
    frame,
    label,
    probability,
    x,
    y,
):
    width = 300
    height = 24

    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        (80, 80, 80),
        2,
    )

    fill = int(
        width * probability
    )

    cv2.rectangle(
        frame,
        (x, y),
        (x + fill, y + height),
        (255, 255, 255),
        -1,
    )

    text = (
        f"{label}: "
        f"{probability * 100:.1f}%"
    )

    cv2.putText(
        frame,
        text,
        (x, y - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


# ============================================================
# GAME STATE
# ============================================================

STATE_IDLE = "idle"
STATE_OAN = "oan"
STATE_TU = "tu"
STATE_TI = "ti"
STATE_CAPTURE = "capture"
STATE_RESULT = "result"
STATE_RETRY = "retry"

state = STATE_IDLE
state_start_time = time.time()

locked_label = None
locked_conf = 0.0
locked_probs = None
cpu_label = None

capture_votes = defaultdict(list)

ema_probs = None


def change_state(new_state):
    global state
    global state_start_time

    state = new_state
    state_start_time = time.time()


def start_new_round():
    global locked_label
    global locked_conf
    global locked_probs
    global cpu_label
    global capture_votes
    global ema_probs

    sequence_buffer.clear()

    locked_label = None
    locked_conf = 0.0
    locked_probs = None
    cpu_label = None

    capture_votes = defaultdict(list)
    ema_probs = None

    change_state(
        STATE_OAN
    )


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW,
)

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1,
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640,
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480,
)

cap.set(
    cv2.CAP_PROP_FPS,
    CAMERA_FPS,
)

if not cap.isOpened():
    raise RuntimeError(
        "Không mở được camera."
    )


WINDOW_DURATION = (
    WINDOW_SIZE / TRAIN_FPS
)

REAL_BUFFER_SIZE = max(
    2,
    round(
        WINDOW_DURATION
        * CAMERA_FPS
    ),
)

print(
    "Realtime buffer:",
    REAL_BUFFER_SIZE,
    "frames",
)

sequence_buffer = deque(
    maxlen=REAL_BUFFER_SIZE
)

frame_index = 0


# ============================================================
# REALTIME LOOP
# ============================================================

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

        now = time.time()
        elapsed = now - state_start_time

        # --------------------------------------------
        # AUTO STATE TRANSITIONS
        # --------------------------------------------

        if (
            state == STATE_OAN
            and elapsed >= OAN_DURATION
        ):
            change_state(STATE_TU)

        elif (
            state == STATE_TU
            and elapsed >= TU_DURATION
        ):
            change_state(STATE_TI)

        elif (
            state == STATE_TI
            and elapsed >= TI_DURATION
        ):
            # Quan trọng:
            # bỏ toàn bộ frame trung gian trước "Tì!"
            sequence_buffer.clear()
            ema_probs = None
            capture_votes = defaultdict(list)

            change_state(
                STATE_CAPTURE
            )

        elif (
            state == STATE_CAPTURE
            and elapsed >= CAPTURE_DURATION
        ):
            (
                winner,
                winner_conf,
                total_votes,
            ) = choose_vote_winner(
                capture_votes
            )

            if winner is None:
                change_state(
                    STATE_RETRY
                )

            else:
                locked_label = winner
                locked_conf = winner_conf

                if ema_probs is not None:
                    locked_probs = (
                        ema_probs.copy()
                    )

                cpu_label = (
                    cpu_winning_move(
                        locked_label
                    )
                )

                change_state(
                    STATE_RESULT
                )

        # --------------------------------------------
        # MEDIAPIPE + MODEL
        # --------------------------------------------

        if (
            frame_index
            % PROCESS_EVERY
            == 0
        ):
            small = resize_frame(
                frame
            )

            rgb = cv2.cvtColor(
                small,
                cv2.COLOR_BGR2RGB,
            )

            rgb.flags.writeable = False

            results = hands.process(
                rgb
            )

            if results.multi_hand_landmarks:
                hand = (
                    results
                    .multi_hand_landmarks[0]
                )

                mp_drawing.draw_landmarks(
                    display,
                    hand,
                    mp_hands.HAND_CONNECTIONS,
                )

                # Chỉ dùng model trong CAPTURE.
                if state == STATE_CAPTURE:
                    features = extract_features(
                        hand
                    )

                    if features is not None:
                        sequence_buffer.append(
                            features
                        )

                    (
                        ema_probs,
                        current_label,
                        current_conf,
                    ) = predict_from_buffer(
                        sequence_buffer,
                        ema_probs,
                    )

                    if (
                        current_label is not None
                        and
                        current_conf
                        >= MIN_CONFIDENCE
                    ):
                        capture_votes[
                            current_label
                        ].append(
                            current_conf
                        )

            else:
                if state == STATE_CAPTURE:
                    sequence_buffer.clear()

        # --------------------------------------------
        # UI
        # --------------------------------------------

        if state == STATE_IDLE:
            cv2.putText(
                display,
                "PRESS SPACE TO START",
                (60, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

        elif state == STATE_OAN:
            cv2.putText(
                display,
                "OAN...",
                (220, 210),
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
                (235, 210),
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
                (255, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (255, 255, 255),
                4,
                cv2.LINE_AA,
            )

        elif state == STATE_CAPTURE:
            remaining = max(
                0.0,
                CAPTURE_DURATION
                - (
                    now - state_start_time
                )
            )

            cv2.putText(
                display,
                "HOLD YOUR GESTURE!",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                (
                    f"Capture: "
                    f"{remaining:.2f}s"
                ),
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                (
                    f"Buffer: "
                    f"{len(sequence_buffer)}/"
                    f"{REAL_BUFFER_SIZE}"
                ),
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if ema_probs is not None:
                top_idx = int(
                    np.argmax(
                        ema_probs
                    )
                )

                top_label = (
                    LABEL_NAMES[top_idx]
                )

                top_conf = float(
                    ema_probs[top_idx]
                )

                cv2.putText(
                    display,
                    (
                        f"Live: "
                        f"{top_label.upper()} "
                        f"{top_conf * 100:.1f}%"
                    ),
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                y = 205

                for i, label in enumerate(
                    LABEL_NAMES
                ):
                    draw_probability_bar(
                        display,
                        label.upper(),
                        float(
                            ema_probs[i]
                        ),
                        20,
                        y,
                    )

                    y += 60

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
                (
                    f"YOU: "
                    f"{locked_label.upper()} "
                    f"{locked_conf * 100:.1f}%"
                ),
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                (
                    f"CPU: "
                    f"{cpu_label.upper()}"
                ),
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

                for i, label in enumerate(
                    LABEL_NAMES
                ):
                    draw_probability_bar(
                        display,
                        label.upper(),
                        float(
                            locked_probs[i]
                        ),
                        20,
                        y,
                    )

                    y += 60

        elif state == STATE_RETRY:
            cv2.putText(
                display,
                "NO RELIABLE GESTURE",
                (40, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                "PRESS SPACE TO TRY AGAIN",
                (45, 235),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
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
            "Tiny GRU RPS Game",
            display,
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == 32:
            start_new_round()

        if key == 27:
            break


cap.release()
cv2.destroyAllWindows()
