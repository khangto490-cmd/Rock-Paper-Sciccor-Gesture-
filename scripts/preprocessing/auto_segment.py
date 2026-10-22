from pathlib import Path
import csv

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_ROOT = Path("dataset_120fps")
OUTPUT_ROOT = Path("dataset_segmented")
LOG_ROOT = Path("segment_logs")

MODEL_PATH = Path("rps_model.pkl")

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".m4v",
}

VALID_LABELS = {
    "rock",
    "paper",
    "scissors",
}

# MediaPipe không cần xử lý Full HD
PROCESS_WIDTH = 640

# Video 120 FPS -> chạy AI mỗi 2 frame
# ~60 inference / second
PROCESS_EVERY = 2

# Probability smoothing
EMA_ALPHA = 0.40

# Điều kiện prediction hợp lệ
CONF_THRESHOLD = 0.85
MARGIN_THRESHOLD = 0.25

# Các giá trị dưới đây tính theo GIÂY
START_STABLE_SECONDS = 0.12
END_GAP_SECONDS = 0.12

MIN_SEGMENT_SECONDS = 0.40
MAX_SEGMENT_SECONDS = 5.00

# Lấy thêm một chút trước khi model chắc chắn
PRE_ROLL_SECONDS = 0.08


# ============================================================
# LOAD RANDOM FOREST
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy model: {MODEL_PATH}"
    )

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
label_encoder = bundle["label_encoder"]

print(
    "Model classes:",
    list(label_encoder.classes_)
)


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands


# ============================================================
# RESIZE
# ============================================================

def resize_frame(frame):

    h, w = frame.shape[:2]

    if w <= PROCESS_WIDTH:
        return frame

    scale = PROCESS_WIDTH / w

    new_h = int(h * scale)

    return cv2.resize(
        frame,
        (PROCESS_WIDTH, new_h),
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# FEATURE EXTRACTION
#
# Phải giống với lúc train Random Forest:
#
# 21 landmarks
# x, y, z
# relative to wrist
#
# => 63 features
# ============================================================

def extract_features(hand_landmarks):

    wrist = hand_landmarks.landmark[0]

    features = []

    for lm in hand_landmarks.landmark:

        features.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z,
        ])

    features = np.asarray(
        features,
        dtype=np.float32,
    )

    if features.shape[0] != 63:

        raise ValueError(
            f"Sai số feature: "
            f"{features.shape[0]} != 63"
        )

    return features


# ============================================================
# RANDOM FOREST PROBABILITIES
# ============================================================

def predict_probabilities(features):

    if hasattr(
        model,
        "feature_names_in_"
    ):

        feature_names = list(
            model.feature_names_in_
        )

        if len(feature_names) != len(features):

            raise ValueError(
                "Số feature realtime "
                "không giống model."
            )

        x = pd.DataFrame(
            [features],
            columns=feature_names,
        )

    else:

        x = features.reshape(
            1,
            -1,
        )

    probs = model.predict_proba(x)[0]

    # Chuyển probability về dictionary:
    #
    # {
    #   "paper": ...,
    #   "rock": ...,
    #   "scissors": ...
    # }

    prob_dict = {}

    for i, encoded_class in enumerate(
        model.classes_
    ):

        label = label_encoder.inverse_transform(
            [encoded_class]
        )[0]

        prob_dict[label] = float(
            probs[i]
        )

    return prob_dict


# ============================================================
# EMA
# ============================================================

def update_ema(
    previous,
    current,
):

    if previous is None:
        return current.copy()

    output = {}

    for label in VALID_LABELS:

        output[label] = (
            EMA_ALPHA
            * current.get(label, 0.0)
            +
            (1.0 - EMA_ALPHA)
            * previous.get(label, 0.0)
        )

    return output


# ============================================================
# GET TOP PREDICTION
# ============================================================

def get_top_prediction(probs):

    ordered = sorted(
        probs.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    top1_label, top1_prob = ordered[0]
    _, top2_prob = ordered[1]

    margin = (
        top1_prob
        - top2_prob
    )

    return (
        top1_label,
        float(top1_prob),
        float(margin),
    )


# ============================================================
# SAVE VIDEO SEGMENT
#
# Không giữ Full-HD frames trong RAM.
# Sau khi biết start/end mới đọc lại video.
# ============================================================

def save_segment(
    video_path,
    start_frame,
    end_frame,
    fps,
    person,
    label,
    segment_index,
):

    if end_frame <= start_frame:
        return False

    duration = (
        end_frame
        - start_frame
        + 1
    ) / fps

    if duration < MIN_SEGMENT_SECONDS:
        return False

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        return False

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    output_dir = (
        OUTPUT_ROOT
        / person
        / label
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / (
            f"{video_path.stem}_"
            f"{label}_"
            f"{segment_index:03d}.mp4"
        )
    )

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (width, height),
    )

    if not writer.isOpened():

        cap.release()
        return False

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame,
    )

    current = start_frame

    while current <= end_frame:

        ret, frame = cap.read()

        if not ret:
            break

        writer.write(frame)

        current += 1

    writer.release()
    cap.release()

    print(
        f"\nSaved: {output_path}"
    )

    print(
        f"       label    : {label}"
    )

    print(
        f"       frames   : "
        f"{start_frame} -> {end_frame}"
    )

    print(
        f"       duration : "
        f"{duration:.2f}s"
    )

    return True


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(video_path):

    person = video_path.parent.name

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print(
            f"[ERROR] Không mở được "
            f"{video_path}"
        )

        return

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration = (
        total_frames / fps
        if fps > 0
        else 0
    )

    print()
    print("=" * 70)

    print(
        f"Video    : {video_path}"
    )

    print(
        f"Person   : {person}"
    )

    print(
        f"FPS      : {fps:.2f}"
    )

    print(
        f"Frames   : {total_frames}"
    )

    print(
        f"Duration : {duration:.2f}s"
    )

    # --------------------------------------------------------
    # Convert seconds -> number of AI predictions
    # --------------------------------------------------------

    inference_fps = (
        fps / PROCESS_EVERY
    )

    start_required = max(
        2,
        round(
            START_STABLE_SECONDS
            * inference_fps
        )
    )

    end_gap_required = max(
        2,
        round(
            END_GAP_SECONDS
            * inference_fps
        )
    )

    pre_roll_frames = round(
        PRE_ROLL_SECONDS
        * fps
    )

    max_segment_frames = round(
        MAX_SEGMENT_SECONDS
        * fps
    )

    print(
        f"AI rate  : "
        f"~{inference_fps:.1f} inference/s"
    )

    print(
        f"Start stable: "
        f"{start_required} predictions"
    )

    print(
        f"End gap     : "
        f"{end_gap_required} predictions"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    processed_count = 0
    detected_count = 0
    valid_count = 0

    # --------------------------------------------------------
    # Prediction state
    # --------------------------------------------------------

    ema_probs = None

    pending_label = None
    pending_count = 0
    pending_start_frame = None

    active_label = None
    active_start_frame = None

    gap_count = 0

    switch_label = None
    switch_count = 0
    switch_start_frame = None

    # --------------------------------------------------------
    # Result segments
    # --------------------------------------------------------

    segments = []

    # --------------------------------------------------------
    # Debug CSV
    # --------------------------------------------------------

    LOG_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        LOG_ROOT
        / person
        / f"{video_path.stem}.csv"
    )

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_file = open(
        log_path,
        "w",
        newline="",
        encoding="utf-8",
    )

    log_writer = csv.writer(
        log_file
    )

    log_writer.writerow([
        "frame",
        "time",
        "hand_detected",
        "prediction",
        "confidence",
        "margin",
        "rock",
        "paper",
        "scissors",
        "valid",
    ])

    # ========================================================
    # MEDIAPIPE
    # ========================================================

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.60,
        min_tracking_confidence=0.60,
    ) as hands:

        frame_index = -1

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_index += 1

            # -----------------------------------------------
            # Frame skipping:
            # source vẫn 120 FPS
            # AI chỉ chạy khoảng 60 inference/s
            # -----------------------------------------------

            if (
                frame_index
                % PROCESS_EVERY
                != 0
            ):
                continue

            processed_count += 1

            small = resize_frame(
                frame
            )

            rgb = cv2.cvtColor(
                small,
                cv2.COLOR_BGR2RGB,
            )

            # giúp MediaPipe tránh copy không cần thiết
            rgb.flags.writeable = False

            results = hands.process(
                rgb
            )

            detected = bool(
                results.multi_hand_landmarks
            )

            label = None
            confidence = 0.0
            margin = 0.0

            raw_probs = {
                "rock": 0.0,
                "paper": 0.0,
                "scissors": 0.0,
            }

            valid = False

            # -----------------------------------------------
            # Hand detected
            # -----------------------------------------------

            if detected:

                detected_count += 1

                hand = (
                    results
                    .multi_hand_landmarks[0]
                )

                features = extract_features(
                    hand
                )

                raw_probs = (
                    predict_probabilities(
                        features
                    )
                )

                ema_probs = update_ema(
                    ema_probs,
                    raw_probs,
                )

                (
                    label,
                    confidence,
                    margin,
                ) = get_top_prediction(
                    ema_probs
                )

                valid = (
                    label in VALID_LABELS
                    and
                    confidence
                    >= CONF_THRESHOLD
                    and
                    margin
                    >= MARGIN_THRESHOLD
                )

                if valid:
                    valid_count += 1

            else:

                # Khi mất tay,
                # không giữ EMA mãi mãi
                ema_probs = None

            # -----------------------------------------------
            # SAVE DEBUG
            # -----------------------------------------------

            log_writer.writerow([
                frame_index,
                frame_index / fps,
                int(detected),
                label or "",
                confidence,
                margin,
                (
                    ema_probs.get(
                        "rock",
                        0.0
                    )
                    if ema_probs
                    else 0.0
                ),
                (
                    ema_probs.get(
                        "paper",
                        0.0
                    )
                    if ema_probs
                    else 0.0
                ),
                (
                    ema_probs.get(
                        "scissors",
                        0.0
                    )
                    if ema_probs
                    else 0.0
                ),
                int(valid),
            ])

            # ====================================================
            # STATE MACHINE
            # ====================================================

            candidate = (
                label
                if valid
                else None
            )

            # ----------------------------------------------------
            # Không có active segment
            # ----------------------------------------------------

            if active_label is None:

                if candidate is None:

                    pending_label = None
                    pending_count = 0
                    pending_start_frame = None

                else:

                    if (
                        candidate
                        == pending_label
                    ):

                        pending_count += 1

                    else:

                        pending_label = (
                            candidate
                        )

                        pending_count = 1

                        pending_start_frame = (
                            frame_index
                        )

                    # --------------------------------------------
                    # đủ stable -> mở segment
                    # --------------------------------------------

                    if (
                        pending_count
                        >= start_required
                    ):

                        active_label = (
                            pending_label
                        )

                        active_start_frame = max(
                            0,
                            pending_start_frame
                            - pre_roll_frames,
                        )

                        pending_label = None
                        pending_count = 0

                        gap_count = 0

                        switch_label = None
                        switch_count = 0

            # ----------------------------------------------------
            # Đã có active segment
            # ----------------------------------------------------

            else:

                # cùng gesture
                if candidate == active_label:

                    gap_count = 0

                    switch_label = None
                    switch_count = 0

                # không chắc gesture
                elif candidate is None:

                    gap_count += 1

                    switch_label = None
                    switch_count = 0

                    if (
                        gap_count
                        >= end_gap_required
                    ):

                        end_frame = max(
                            active_start_frame,
                            frame_index
                            -
                            gap_count
                            * PROCESS_EVERY,
                        )

                        segments.append((
                            active_start_frame,
                            end_frame,
                            active_label,
                        ))

                        active_label = None
                        active_start_frame = None

                        gap_count = 0

                # một gesture khác đang xuất hiện
                else:

                    gap_count = 0

                    if (
                        candidate
                        == switch_label
                    ):

                        switch_count += 1

                    else:

                        switch_label = (
                            candidate
                        )

                        switch_count = 1

                        switch_start_frame = (
                            frame_index
                        )

                    # --------------------------------------------
                    # class mới đủ ổn định
                    # --------------------------------------------

                    if (
                        switch_count
                        >= start_required
                    ):

                        end_frame = max(
                            active_start_frame,
                            switch_start_frame - 1,
                        )

                        segments.append((
                            active_start_frame,
                            end_frame,
                            active_label,
                        ))

                        # class mới bắt đầu
                        active_label = (
                            switch_label
                        )

                        active_start_frame = max(
                            0,
                            switch_start_frame
                            - pre_roll_frames,
                        )

                        switch_label = None
                        switch_count = 0

                # -----------------------------------------------
                # tránh segment quá dài
                # -----------------------------------------------

                if (
                    active_label is not None
                    and
                    frame_index
                    - active_start_frame
                    >= max_segment_frames
                ):

                    segments.append((
                        active_start_frame,
                        frame_index,
                        active_label,
                    ))

                    active_label = None
                    active_start_frame = None

                    gap_count = 0
                    switch_label = None
                    switch_count = 0

            # ====================================================
            # TERMINAL STATUS
            # ====================================================

            if processed_count % 120 == 0:

                progress = (
                    frame_index
                    / total_frames
                    * 100
                )

                detection_rate = (
                    detected_count
                    / processed_count
                    * 100
                )

                valid_rate = (
                    valid_count
                    / processed_count
                    * 100
                )

                print(
                    f"\r"
                    f"{progress:6.2f}% "
                    f"| hand "
                    f"{detection_rate:5.1f}% "
                    f"| valid "
                    f"{valid_rate:5.1f}% "
                    f"| pred "
                    f"{label or '-':8s} "
                    f"| conf "
                    f"{confidence:.2f}",
                    end="",
                )

    cap.release()
    log_file.close()

    # ========================================================
    # VIDEO END
    # ========================================================

    if active_label is not None:

        segments.append((
            active_start_frame,
            total_frames - 1,
            active_label,
        ))

    print()

    # ========================================================
    # FILTER SEGMENTS
    # ========================================================

    clean_segments = []

    for (
        start_frame,
        end_frame,
        label
    ) in segments:

        segment_duration = (
            end_frame
            - start_frame
            + 1
        ) / fps

        if (
            segment_duration
            >= MIN_SEGMENT_SECONDS
        ):

            clean_segments.append((
                start_frame,
                end_frame,
                label,
            ))

    # ========================================================
    # STATISTICS
    # ========================================================

    detection_rate = (
        detected_count
        / processed_count
        * 100
        if processed_count > 0
        else 0
    )

    valid_rate = (
        valid_count
        / processed_count
        * 100
        if processed_count > 0
        else 0
    )

    print()
    print("--- SUMMARY ---")

    print(
        f"AI processed    : "
        f"{processed_count}"
    )

    print(
        f"Hands detected  : "
        f"{detected_count}"
    )

    print(
        f"Detection rate  : "
        f"{detection_rate:.2f}%"
    )

    print(
        f"Valid predictions: "
        f"{valid_count}"
    )

    print(
        f"Valid rate      : "
        f"{valid_rate:.2f}%"
    )

    print(
        f"Segments found  : "
        f"{len(clean_segments)}"
    )

    print(
        f"Prediction log  : "
        f"{log_path}"
    )

    # ========================================================
    # SAVE SEGMENTS
    # ========================================================

    segment_counter = {
        "rock": 0,
        "paper": 0,
        "scissors": 0,
    }

    for (
        start_frame,
        end_frame,
        label
    ) in clean_segments:

        segment_counter[label] += 1

        save_segment(
            video_path=video_path,
            start_frame=start_frame,
            end_frame=end_frame,
            fps=fps,
            person=person,
            label=label,
            segment_index=(
                segment_counter[label]
            ),
        )

    print()
    print(
        "Segments:",
        segment_counter
    )

    # ========================================================
    # DIAGNOSIS
    # ========================================================

    if detected_count == 0:

        print()
        print(
            "[WARNING] MediaPipe "
            "không detect được bàn tay."
        )

    elif detection_rate < 50:

        print()
        print(
            "[WARNING] Detection rate thấp."
        )

    elif valid_count == 0:

        print()
        print(
            "[WARNING] MediaPipe nhận tay "
            "nhưng Random Forest không có "
            "prediction đủ confidence."
        )

    elif len(clean_segments) == 0:

        print()
        print(
            "[WARNING] Có prediction hợp lệ "
            "nhưng chưa đủ ổn định để tạo segment."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT_ROOT.exists():

        print(
            "Không tìm thấy "
            "dataset_120fps."
        )

        return

    videos = sorted([
        p
        for p in INPUT_ROOT.rglob("*")
        if (
            p.is_file()
            and
            p.suffix.lower()
            in VIDEO_EXTENSIONS
        )
    ])

    # ========================================================
    # TEST 1 VIDEO TRƯỚC
    # ========================================================

    videos = videos[:1]

    print(
        f"Tìm thấy {len(videos)} "
        f"video để test."
    )

    for index, video_path in enumerate(
        videos,
        1,
    ):

        print(
            f"\n[{index}/{len(videos)}]"
        )

        process_video(
            video_path
        )

    print()
    print("=" * 70)
    print("HOÀN TẤT AUTO SEGMENT")


if __name__ == "__main__":
    main()