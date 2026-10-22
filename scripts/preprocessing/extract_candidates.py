from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np


# ============================================================
# CONFIG
# ============================================================

INPUT_ROOT = Path("dataset_120fps")
OUTPUT_ROOT = Path("candidate_segments")

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".m4v",
}

PROCESS_WIDTH = 640

# Video 120 FPS -> xử lý mỗi 2 frame
# ~60 lần MediaPipe / giây
PROCESS_EVERY = 2


# ============================================================
# MOTION SETTINGS
# ============================================================

# EMA làm mượt motion
MOTION_EMA_ALPHA = 0.35

# Motion thấp hơn mức này -> coi là tay ổn định
STABLE_THRESHOLD = 0.012

# Motion cao hơn mức này -> coi là đang đổi gesture
MOVING_THRESHOLD = 0.025


# Tay phải ổn định ít nhất từng này giây
STABLE_SECONDS = 0.25

# Khi tay chuyển động liên tục đủ lâu -> kết thúc gesture
MOVING_SECONDS = 0.10

# mất tay bao lâu thì kết thúc segment
MISSING_SECONDS = 0.15


# ============================================================
# SEGMENT LENGTH
# ============================================================

MIN_SEGMENT_SECONDS = 0.40
MAX_SEGMENT_SECONDS = 3.00

# lấy thêm một ít trước/sau gesture
PRE_ROLL_SECONDS = 0.10
POST_ROLL_SECONDS = 0.08


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
# LANDMARK NORMALIZATION
# ============================================================

def get_normalized_landmarks(hand):

    landmarks = np.array(
        [
            [lm.x, lm.y, lm.z]
            for lm in hand.landmark
        ],
        dtype=np.float32,
    )

    # --------------------------------------------
    # 1. wrist-relative
    # --------------------------------------------

    wrist = landmarks[0].copy()

    landmarks -= wrist


    # --------------------------------------------
    # 2. normalize scale
    #
    # tránh tay gần / xa camera làm motion sai
    # --------------------------------------------

    xy_distance = np.linalg.norm(
        landmarks[:, :2],
        axis=1,
    )

    scale = np.max(
        xy_distance
    )

    if scale > 1e-6:
        landmarks /= scale

    return landmarks


# ============================================================
# MOTION
# ============================================================

def calculate_motion(
    current,
    previous,
):

    if (
        current is None
        or previous is None
    ):
        return None

    # Chỉ dùng XY để tính motion.
    # Z của MediaPipe nhiễu hơn.
    diff = (
        current[:, :2]
        - previous[:, :2]
    )

    distances = np.linalg.norm(
        diff,
        axis=1,
    )

    motion = float(
        np.mean(distances)
    )

    return motion


# ============================================================
# SAVE SEGMENT
# ============================================================

def save_segment(
    video_path,
    start_frame,
    end_frame,
    fps,
    person,
    index,
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
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / (
            f"{video_path.stem}_"
            f"{index:03d}.mp4"
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

    frame_id = start_frame

    while frame_id <= end_frame:

        ret, frame = cap.read()

        if not ret:
            break

        writer.write(frame)

        frame_id += 1

    writer.release()
    cap.release()

    print()
    print(
        f"Saved: {output_path}"
    )

    print(
        f"       Frames   : "
        f"{start_frame} -> {end_frame}"
    )

    print(
        f"       Duration : "
        f"{duration:.2f}s"
    )

    return True


# ============================================================
# PROCESS VIDEO
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

    inference_fps = (
        fps / PROCESS_EVERY
    )


    # ========================================================
    # SECOND -> COUNTS
    # ========================================================

    stable_required = max(
        2,
        round(
            STABLE_SECONDS
            * inference_fps
        )
    )

    moving_required = max(
        2,
        round(
            MOVING_SECONDS
            * inference_fps
        )
    )

    missing_required = max(
        2,
        round(
            MISSING_SECONDS
            * inference_fps
        )
    )

    pre_roll_frames = round(
        PRE_ROLL_SECONDS
        * fps
    )

    post_roll_frames = round(
        POST_ROLL_SECONDS
        * fps
    )

    max_segment_frames = round(
        MAX_SEGMENT_SECONDS
        * fps
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

    print(
        f"AI rate  : "
        f"~{inference_fps:.1f}/s"
    )

    print(
        f"Stable required : "
        f"{stable_required}"
    )

    print(
        f"Moving required : "
        f"{moving_required}"
    )


    # ========================================================
    # STATE
    # ========================================================

    previous_landmarks = None

    motion_ema = None

    stable_count = 0
    moving_count = 0
    missing_count = 0

    active = False

    segment_start = None

    segments = []

    processed_count = 0
    detected_count = 0

    stable_frames = 0
    moving_frames = 0


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


            # ==================================================
            # FRAME SAMPLING
            # ==================================================

            if (
                frame_index
                % PROCESS_EVERY
                != 0
            ):
                continue

            processed_count += 1


            # ==================================================
            # MEDIAPIPE
            # ==================================================

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

            hand_detected = bool(
                results.multi_hand_landmarks
            )


            # ==================================================
            # NO HAND
            # ==================================================

            if not hand_detected:

                missing_count += 1

                previous_landmarks = None
                motion_ema = None

                stable_count = 0
                moving_count = 0


                if (
                    active
                    and
                    missing_count
                    >= missing_required
                ):

                    end_frame = max(
                        segment_start,
                        frame_index
                        - missing_required * PROCESS_EVERY
                        + post_roll_frames,

                    )

                    segments.append((
                        segment_start,
                        end_frame,
                    ))

                    active = False
                    segment_start = None

                continue


            # ==================================================
            # HAND DETECTED
            # ==================================================

            detected_count += 1
            missing_count = 0

            hand = (
                results
                .multi_hand_landmarks[0]
            )

            landmarks = (
                get_normalized_landmarks(
                    hand
                )
            )


            # ==================================================
            # MOTION
            # ==================================================

            raw_motion = calculate_motion(
                landmarks,
                previous_landmarks,
            )

            previous_landmarks = (
                landmarks
            )


            if raw_motion is None:
                continue


            # ==================================================
            # EMA MOTION
            # ==================================================

            if motion_ema is None:

                motion_ema = raw_motion

            else:

                motion_ema = (
                    MOTION_EMA_ALPHA
                    * raw_motion
                    +
                    (
                        1.0
                        - MOTION_EMA_ALPHA
                    )
                    * motion_ema
                )


            # ==================================================
            # CLASSIFY MOTION STATE
            # ==================================================

            if (
                motion_ema
                <= STABLE_THRESHOLD
            ):

                state = "stable"

                stable_count += 1
                moving_count = 0

                stable_frames += 1


            elif (
                motion_ema
                >= MOVING_THRESHOLD
            ):

                state = "moving"

                moving_count += 1
                stable_count = 0

                moving_frames += 1


            else:

                # vùng giữa 2 threshold
                # không thay đổi state mạnh
                state = "transition"

                stable_count = max(
                    0,
                    stable_count - 1
                )

                moving_count = max(
                    0,
                    moving_count - 1
                )


            # ==================================================
            # START SEGMENT
            # ==================================================

            if not active:

                if (
                    stable_count
                    >= stable_required
                ):

                    active = True

                    segment_start = max(
                        0,
                        frame_index
                        -
                        (
                            stable_required
                            * PROCESS_EVERY
                        )
                        -
                        pre_roll_frames
                    )

                    moving_count = 0


            # ==================================================
            # ACTIVE SEGMENT
            # ==================================================

            else:

                # Gesture bắt đầu chuyển động
                if (
                    moving_count
                    >= moving_required
                ):

                    end_frame = min(
                        total_frames - 1,
                        frame_index
                        -
                        (
                            moving_required
                            * PROCESS_EVERY
                        )
                        +
                        post_roll_frames
                    )

                    segments.append((
                        segment_start,
                        end_frame,
                    ))

                    active = False
                    segment_start = None

                    stable_count = 0
                    moving_count = 0


                # tránh một clip dài quá
                elif (
                    frame_index
                    - segment_start
                    >= max_segment_frames
                ):

                    segments.append((
                        segment_start,
                        frame_index,
                    ))

                    active = False
                    segment_start = None

                    stable_count = 0
                    moving_count = 0


            # ==================================================
            # STATUS
            # ==================================================

            if (
                processed_count
                % 120
                == 0
            ):

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

                print(
                    f"\r"
                    f"{progress:6.2f}% "
                    f"| hand "
                    f"{detection_rate:5.1f}% "
                    f"| motion "
                    f"{motion_ema:.4f} "
                    f"| {state:10s} "
                    f"| active "
                    f"{active}",
                    end=""
                )


    cap.release()


    # ========================================================
    # VIDEO END
    # ========================================================

    if (
        active
        and
        segment_start is not None
    ):

        segments.append((
            segment_start,
            total_frames - 1,
        ))


    print()


    # ========================================================
    # FILTER SEGMENTS
    # ========================================================

    clean_segments = []

    for start, end in segments:

        clip_duration = (
            end
            - start
            + 1
        ) / fps

        if (
            MIN_SEGMENT_SECONDS
            <= clip_duration
            <= MAX_SEGMENT_SECONDS
        ):

            clean_segments.append(
                (start, end)
            )


    # ========================================================
    # SUMMARY
    # ========================================================

    detection_rate = (
        detected_count
        / processed_count
        * 100
        if processed_count > 0
        else 0
    )

    print()
    print("--- SUMMARY ---")

    print(
        f"AI processed   : "
        f"{processed_count}"
    )

    print(
        f"Hand detected  : "
        f"{detected_count}"
    )

    print(
        f"Detection rate : "
        f"{detection_rate:.2f}%"
    )

    print(
        f"Stable frames  : "
        f"{stable_frames}"
    )

    print(
        f"Moving frames  : "
        f"{moving_frames}"
    )

    print(
        f"Segments found : "
        f"{len(clean_segments)}"
    )


    # ========================================================
    # SAVE SEGMENTS
    # ========================================================

    saved = 0

    for index, (
        start,
        end,
    ) in enumerate(
        clean_segments,
        1,
    ):

        if save_segment(
            video_path,
            start,
            end,
            fps,
            person,
            index,
        ):

            saved += 1


    print()
    print(
        f"Saved segments : {saved}"
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
    print(
        "HOÀN TẤT MOTION SEGMENTATION"
    )


if __name__ == "__main__":
    main()