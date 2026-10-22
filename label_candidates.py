from pathlib import Path
import cv2
import shutil


# ============================================================
# CONFIG
# ============================================================

INPUT_ROOT = Path("candidate_segments")
OUTPUT_ROOT = Path("dataset_labeled")

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".m4v",
}


# ============================================================
# COPY LABELED FILE
# ============================================================

def save_labeled_clip(
    video_path,
    person,
    label,
):

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
        / video_path.name
    )

    shutil.copy2(
        video_path,
        output_path,
    )

    print(
        f"Saved -> {output_path}"
    )


# ============================================================
# PLAY VIDEO
# ============================================================

def play_clip(video_path):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        return None

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30

    delay = max(
        1,
        int(1000 / fps)
    )

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return None

    # phát loop cho tới khi user bấm phím
    while True:

        for frame in frames:

            display = frame.copy()

            h, w = display.shape[:2]

            # resize để vừa màn hình
            max_width = 900

            if w > max_width:

                scale = (
                    max_width / w
                )

                display = cv2.resize(
                    display,
                    (
                        max_width,
                        int(h * scale)
                    )
                )

            # hướng dẫn
            cv2.putText(
                display,
                "R=Rock  P=Paper  S=Scissors  X=Skip  ESC=Exit",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                video_path.name,
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Label Candidate",
                display,
            )

            key = (
                cv2.waitKey(delay)
                & 0xFF
            )

            if key != 255:
                return key


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT_ROOT.exists():

        print(
            "Không tìm thấy "
            "candidate_segments."
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
        f"candidate clips."
    )

    if len(videos) == 0:
        return


    counts = {
        "rock": 0,
        "paper": 0,
        "scissors": 0,
        "skip": 0,
    }


    for index, video_path in enumerate(
        videos,
        1,
    ):

        # candidate_segments/person_01/file.mp4
        person = video_path.parent.name

        print()
        print(
            f"[{index}/{len(videos)}]"
        )

        print(
            video_path
        )

        key = play_clip(
            video_path
        )

        if key is None:
            print(
                "Không đọc được clip."
            )
            continue


        # ====================================================
        # ROCK
        # ====================================================

        if key in (
            ord("r"),
            ord("R"),
        ):

            label = "rock"

            save_labeled_clip(
                video_path,
                person,
                label,
            )

            counts[label] += 1


        # ====================================================
        # PAPER
        # ====================================================

        elif key in (
            ord("p"),
            ord("P"),
        ):

            label = "paper"

            save_labeled_clip(
                video_path,
                person,
                label,
            )

            counts[label] += 1


        # ====================================================
        # SCISSORS
        # ====================================================

        elif key in (
            ord("s"),
            ord("S"),
        ):

            label = "scissors"

            save_labeled_clip(
                video_path,
                person,
                label,
            )

            counts[label] += 1


        # ====================================================
        # SKIP
        # ====================================================

        elif key in (
            ord("x"),
            ord("X"),
        ):

            print(
                "Skipped."
            )

            counts["skip"] += 1


        # ====================================================
        # ESC
        # ====================================================

        elif key == 27:

            print(
                "Dừng labeling."
            )

            break


        else:

            print(
                "Phím không hợp lệ, "
                "bỏ clip."
            )


    cv2.destroyAllWindows()


    print()
    print("=" * 60)

    print(
        "HOÀN TẤT LABEL"
    )

    print(
        f"Rock     : "
        f"{counts['rock']}"
    )

    print(
        f"Paper    : "
        f"{counts['paper']}"
    )

    print(
        f"Scissors : "
        f"{counts['scissors']}"
    )

    print(
        f"Skipped  : "
        f"{counts['skip']}"
    )

    print(
        f"Output   : "
        f"{OUTPUT_ROOT.resolve()}"
    )


if __name__ == "__main__":
    main()