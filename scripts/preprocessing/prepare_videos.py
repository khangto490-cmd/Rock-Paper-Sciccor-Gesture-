from pathlib import Path
import cv2
import shutil
import time


INPUT_ROOT = Path("dataset_raw")
OUTPUT_ROOT = Path("dataset_120fps")

TARGET_FPS = 120.0

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".m4v",
}


def get_video_info(path):
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    duration = frames / fps if fps > 0 else 0

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frames": frames,
        "duration": duration,
    }


def classify_fps(fps):
    if 115 <= fps <= 125:
        return "120"

    if 55 <= fps <= 65:
        return "60"

    return "unsupported"


def interpolate_60_to_120(input_path, output_path):

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        print(f"[ERROR] Không mở được {input_path}")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        TARGET_FPS,
        (width, height)
    )

    if not writer.isOpened():
        print("[ERROR] Không tạo được video output.")
        cap.release()
        return False

    ret, previous = cap.read()

    if not ret:
        print("[ERROR] Không đọc được frame đầu.")
        cap.release()
        writer.release()
        return False

    current_index = 1

    start = time.time()

    while True:

        ret, current = cap.read()

        if not ret:
            break

        # frame thật
        writer.write(previous)

        # frame nội suy giữa 2 frame
        middle = cv2.addWeighted(
            previous,
            0.5,
            current,
            0.5,
            0
        )

        writer.write(middle)

        previous = current

        current_index += 1

        if current_index % 120 == 0:

            progress = (
                current_index / total_frames * 100
                if total_frames > 0
                else 0
            )

            print(
                f"\rProcessing: "
                f"{progress:.1f}% "
                f"({current_index}/{total_frames})",
                end=""
            )

    # frame cuối
    writer.write(previous)
    writer.write(previous)

    cap.release()
    writer.release()

    print()

    print(
        f"Hoàn thành sau "
        f"{time.time() - start:.2f}s"
    )

    return True


def main():

    if not INPUT_ROOT.exists():
        print("Không tìm thấy dataset_raw.")
        return

    videos = sorted([
        path
        for path in INPUT_ROOT.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in VIDEO_EXTENSIONS
        )
    ])

    print(f"Tìm thấy {len(videos)} video.")

    for index, input_path in enumerate(videos, 1):

        print()
        print("=" * 60)

        print(
            f"[{index}/{len(videos)}] "
            f"{input_path}"
        )

        info = get_video_info(input_path)

        if info is None:
            print("[ERROR] Không đọc được video.")
            continue

        fps = info["fps"]

        print(
            f"Resolution : "
            f"{info['width']}x{info['height']}"
        )

        print(
            f"FPS        : "
            f"{fps:.2f}"
        )

        print(
            f"Frames     : "
            f"{info['frames']}"
        )

        print(
            f"Duration   : "
            f"{info['duration']:.2f}s"
        )

        fps_type = classify_fps(fps)

        relative = input_path.relative_to(
            INPUT_ROOT
        )

        # ============================
        # Video đã 120 FPS
        # ============================

        if fps_type == "120":

            output_path = (
                OUTPUT_ROOT
                / relative
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            print(
                "→ Video đã ~120 FPS."
            )

            print(
                "→ Giữ nguyên video."
            )

            shutil.copy2(
                input_path,
                output_path
            )

        # ============================
        # Video 60 FPS
        # ============================

        elif fps_type == "60":

            output_path = (
                OUTPUT_ROOT
                / relative
            ).with_suffix(".mp4")

            print(
                "→ Video ~60 FPS."
            )

            print(
                "→ Nội suy lên 120 FPS."
            )

            success = interpolate_60_to_120(
                input_path,
                output_path
            )

            if not success:
                continue

        else:

            print(
                f"[SKIP] FPS {fps:.2f} "
                f"không thuộc nhóm 60/120."
            )

            continue

        # ============================
        # VERIFY
        # ============================

        output_info = get_video_info(
            output_path
        )

        if output_info is None:

            print(
                "[ERROR] Không kiểm tra "
                "được output."
            )

            continue

        print("--- OUTPUT ---")

        print(
            f"FPS      : "
            f"{output_info['fps']:.2f}"
        )

        print(
            f"Frames   : "
            f"{output_info['frames']}"
        )

        print(
            f"Duration : "
            f"{output_info['duration']:.2f}s"
        )

        print(
            f"Saved    : "
            f"{output_path}"
        )

    print()
    print("=" * 60)
    print("HOÀN TẤT")


if __name__ == "__main__":
    main()