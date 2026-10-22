from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


# ============================================================
# CONFIG
# ============================================================

INPUT_ROOT = Path("dataset_labeled")
OUTPUT_ROOT = Path("sequences_labeled")

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

PROCESS_WIDTH = 640

# Video 120 FPS -> chạy MediaPipe mỗi frame
# vì sequence training cần giữ timeline 120 FPS đầy đủ
PROCESS_EVERY = 1


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
# LANDMARK EXTRACTION
# ============================================================

def safe_normalize(v, eps=1e-6):
    norm = np.linalg.norm(v)

    if norm < eps:
        return None

    return v / norm


def landmarks_to_features(hand_landmarks):

    # (21, 3)
    landmarks = np.array(
        [
            [lm.x, lm.y, lm.z]
            for lm in hand_landmarks.landmark
        ],
        dtype=np.float32,
    )

    # ========================================================
    # 1. ORIGIN = WRIST
    # ========================================================

    wrist = landmarks[0].copy()

    centered = landmarks - wrist


    # ========================================================
    # 2. TẠO HỆ TRỤC LOCAL CỦA LÒNG BÀN TAY
    #
    # landmark 5  = index MCP
    # landmark 9  = middle MCP
    # landmark 17 = pinky MCP
    # ========================================================

    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    pinky_mcp = landmarks[17]


    # --------------------------------------------------------
    # X-axis:
    # pinky -> index
    # --------------------------------------------------------

    x_axis = safe_normalize(
        index_mcp - pinky_mcp
    )


    # --------------------------------------------------------
    # Y-axis:
    # wrist -> middle MCP
    # --------------------------------------------------------

    y_temp = (
        middle_mcp
        - wrist
    )


    if x_axis is None:
        # fallback về normalization cũ
        distances = np.linalg.norm(
            centered[:, :2],
            axis=1
        )

        scale = np.max(distances)

        if scale > 1e-6:
            centered /= scale

        return centered.reshape(-1).astype(
            np.float32
        )


    # Loại component của Y nằm trên X
    y_temp = (
        y_temp
        - np.dot(y_temp, x_axis)
        * x_axis
    )

    y_axis = safe_normalize(
        y_temp
    )


    if y_axis is None:

        distances = np.linalg.norm(
            centered[:, :2],
            axis=1
        )

        scale = np.max(distances)

        if scale > 1e-6:
            centered /= scale

        return centered.reshape(-1).astype(
            np.float32
        )


    # --------------------------------------------------------
    # Z-axis = pháp tuyến lòng bàn tay
    # --------------------------------------------------------

    z_axis = safe_normalize(
        np.cross(
            x_axis,
            y_axis
        )
    )


    if z_axis is None:

        distances = np.linalg.norm(
            centered[:, :2],
            axis=1
        )

        scale = np.max(distances)

        if scale > 1e-6:
            centered /= scale

        return centered.reshape(-1).astype(
            np.float32
        )


    # Làm Y vuông góc hoàn toàn lại
    y_axis = safe_normalize(
        np.cross(
            z_axis,
            x_axis
        )
    )


    # ========================================================
    # 3. CHUYỂN 21 LANDMARK VỀ HỆ TỌA ĐỘ BÀN TAY
    # ========================================================

    basis = np.stack(
        [
            x_axis,
            y_axis,
            z_axis
        ],
        axis=1
    )

    # (21,3) @ (3,3)
    local_landmarks = (
        centered @ basis
    )


    # ========================================================
    # 4. SCALE NORMALIZATION
    #
    # dùng wrist -> middle MCP
    # ========================================================

    palm_scale = np.linalg.norm(
        middle_mcp
        - wrist
    )

    if palm_scale > 1e-6:
        local_landmarks /= palm_scale


    # ========================================================
    # 5. FLATTEN
    # ========================================================

    features = local_landmarks.reshape(
        -1
    ).astype(
        np.float32
    )


    if features.shape[0] != 63:

        raise ValueError(
            f"Sai số feature: "
            f"{features.shape[0]}"
        )


    return features

# ============================================================
# INTERPOLATE MISSING
# ============================================================

def interpolate_missing(sequence):

    sequence = sequence.copy()

    T, F = sequence.shape

    x = np.arange(T)

    for feature_idx in range(F):

        values = sequence[
            :,
            feature_idx
        ]

        valid = ~np.isnan(values)

        valid_count = np.sum(valid)

        # Không detect được feature nào
        if valid_count == 0:
            sequence[
                :,
                feature_idx
            ] = 0.0

            continue

        # Chỉ có 1 giá trị hợp lệ
        if valid_count == 1:
            sequence[
                :,
                feature_idx
            ] = values[valid][0]

            continue

        sequence[
            :,
            feature_idx
        ] = np.interp(
            x,
            x[valid],
            values[valid],
        )

    return sequence


# ============================================================
# PROCESS VIDEO
# ============================================================

def process_video(
    video_path,
    output_path,
    person,
    label,
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print(
            f"[ERROR] Không mở được "
            f"{video_path}"
        )

        return False

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print()
    print("-" * 70)

    print(
        f"Video  : {video_path}"
    )

    print(
        f"Person : {person}"
    )

    print(
        f"Label  : {label}"
    )

    print(
        f"FPS    : {fps:.2f}"
    )

    print(
        f"Frames : {total_frames}"
    )

    sequence = []

    detected_count = 0
    missing_count = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.60,
        min_tracking_confidence=0.60,
    ) as hands:

        frame_index = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_index += 1

            if (
                (frame_index - 1)
                % PROCESS_EVERY
                != 0
            ):
                continue

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

                features = (
                    landmarks_to_features(
                        hand
                    )
                )

                sequence.append(
                    features
                )

                detected_count += 1

            else:

                # giữ đúng vị trí temporal
                sequence.append(
                    np.full(
                        63,
                        np.nan,
                        dtype=np.float32,
                    )
                )

                missing_count += 1

    cap.release()

    if len(sequence) == 0:

        print(
            "[ERROR] Sequence rỗng."
        )

        return False

    sequence = np.stack(
        sequence
    ).astype(
        np.float32
    )

    # ========================================================
    # DETECTION RATE
    # ========================================================

    detection_rate = (
        detected_count
        / len(sequence)
    )

    # ========================================================
    # FIX MISSING
    # ========================================================

    sequence = interpolate_missing(
        sequence
    )

    if np.isnan(sequence).any():

        print(
            "[ERROR] Sequence vẫn còn NaN."
        )

        return False

    # ========================================================
    # SAVE
    # ========================================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,

        sequence=sequence,

        label=np.array(
            label
        ),

        person=np.array(
            person
        ),

        fps=np.float32(
            fps
        ),

        total_frames=np.int32(
            len(sequence)
        ),

        detection_rate=np.float32(
            detection_rate
        ),

        source_video=np.array(
            str(video_path)
        ),
    )

    print(
        f"Shape          : "
        f"{sequence.shape}"
    )

    print(
        f"Detected       : "
        f"{detected_count}"
    )

    print(
        f"Missing        : "
        f"{missing_count}"
    )

    print(
        f"Detection rate : "
        f"{detection_rate * 100:.2f}%"
    )

    print(
        f"Saved          : "
        f"{output_path}"
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT_ROOT.exists():

        print(
            "Không tìm thấy "
            "dataset_labeled."
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

    print(
        f"Tìm thấy {len(videos)} "
        f"video đã label."
    )

    success = 0
    failed = 0

    for index, video_path in enumerate(
        videos,
        1,
    ):

        relative = video_path.relative_to(
            INPUT_ROOT
        )

        parts = relative.parts

        # dataset_labeled/
        # person_01/
        # rock/
        # clip.mp4

        if len(parts) < 3:

            print(
                f"[SKIP] Sai cấu trúc: "
                f"{relative}"
            )

            failed += 1
            continue

        person = parts[0]

        label = parts[1].lower()

        if label not in VALID_LABELS:

            print(
                f"[SKIP] Label sai: "
                f"{label}"
            )

            failed += 1
            continue

        output_path = (
            OUTPUT_ROOT
            / relative
        ).with_suffix(
            ".npz"
        )

        print()
        print(
            f"[{index}/{len(videos)}]"
        )

        ok = process_video(
            video_path=video_path,
            output_path=output_path,
            person=person,
            label=label,
        )

        if ok:
            success += 1
        else:
            failed += 1

    print()
    print("=" * 70)

    print(
        "HOÀN TẤT EXTRACT LANDMARKS"
    )

    print(
        f"Success : {success}"
    )

    print(
        f"Failed  : {failed}"
    )

    print(
        f"Output  : "
        f"{OUTPUT_ROOT.resolve()}"
    )


if __name__ == "__main__":
    main()