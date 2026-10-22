from pathlib import Path
import numpy as np


# ============================================================
# CONFIG
# ============================================================

INPUT_ROOT = Path("sequences_labeled")
OUTPUT_FILE = Path("gru_dataset.npz")

VALID_LABELS = {
    "rock",
    "paper",
    "scissors",
}

LABEL_TO_ID = {
    "rock": 0,
    "paper": 1,
    "scissors": 2,
}

# 36 frames @ 120 FPS ≈ 0.30 s
WINDOW_SIZE = 36

# Bước trượt
# 12 frames = 0.10 s
STRIDE = 12


# ============================================================
# LOAD ONE SEQUENCE
# ============================================================

def load_sequence(npz_path):
    data = np.load(
        npz_path,
        allow_pickle=True
    )

    sequence = data["sequence"].astype(
        np.float32
    )

    label = str(
        data["label"].item()
    )

    person = str(
        data["person"].item()
    )

    fps = float(
        data["fps"]
    )

    if "source_video" in data.files:
        source_video = data["source_video"].item()

    elif "source_file" in data.files:
        source_video = data["source_file"].item()

    else:
        source_video = npz_path.name    

    return (
        sequence,
        label,
        person,
        fps,
        source_video,
    )


# ============================================================
# CREATE WINDOWS
# ============================================================

def create_windows(
    sequence,
    window_size,
    stride,
):
    T = sequence.shape[0]

    windows = []

    if T < window_size:
        return windows

    for start in range(
        0,
        T - window_size + 1,
        stride
    ):
        end = start + window_size

        window = sequence[
            start:end
        ]

        windows.append(
            window
        )

    return windows


# ============================================================
# MAIN
# ============================================================

def main():
    if not INPUT_ROOT.exists():
        print(
            "Không tìm thấy "
            "sequences_labeled."
        )
        return

    npz_files = sorted(
        INPUT_ROOT.rglob("*.npz")
    )

    print(
        f"Tìm thấy {len(npz_files)} "
        f"sequence files."
    )

    X = []
    y = []

    persons = []
    clip_ids = []
    source_videos = []

    class_counts = {
        "rock": 0,
        "paper": 0,
        "scissors": 0,
    }

    skipped_short = 0

    for index, npz_path in enumerate(
        npz_files,
        1
    ):
        (
            sequence,
            label,
            person,
            fps,
            source_video,
        ) = load_sequence(
            npz_path
        )

        if label not in VALID_LABELS:
            print(
                f"[SKIP] Label sai: "
                f"{label}"
            )
            continue

        if (
            sequence.ndim != 2
            or sequence.shape[1] != 63
        ):
            print(
                f"[SKIP] Shape sai: "
                f"{npz_path} "
                f"{sequence.shape}"
            )
            continue

        windows = create_windows(
            sequence,
            WINDOW_SIZE,
            STRIDE,
        )

        if len(windows) == 0:
            print(
                f"[SKIP] Sequence quá ngắn: "
                f"{npz_path} "
                f"{sequence.shape}"
            )

            skipped_short += 1
            continue

        clip_id = (
            f"{person}/"
            f"{label}/"
            f"{npz_path.stem}"
        )

        for window in windows:
            X.append(
                window
            )

            y.append(
                LABEL_TO_ID[label]
            )

            persons.append(
                person
            )

            clip_ids.append(
                clip_id
            )

            source_videos.append(
                source_video
            )

            class_counts[label] += 1

        print(
            f"[{index}/{len(npz_files)}] "
            f"{label:8s} "
            f"| sequence {sequence.shape} "
            f"| windows {len(windows)}"
        )

    if len(X) == 0:
        print(
            "Không tạo được window nào."
        )
        return

    X = np.stack(
        X
    ).astype(
        np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int64
    )

    persons = np.asarray(
        persons
    )

    clip_ids = np.asarray(
        clip_ids
    )

    source_videos = np.asarray(
        source_videos
    )

    # ========================================================
    # SAVE
    # ========================================================

    np.savez_compressed(
        OUTPUT_FILE,

        X=X,
        y=y,

        persons=persons,
        clip_ids=clip_ids,
        source_videos=source_videos,

        window_size=np.int32(
            WINDOW_SIZE
        ),

        stride=np.int32(
            STRIDE
        ),

        label_names=np.array([
            "rock",
            "paper",
            "scissors",
        ]),
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)

    print(
        "HOÀN TẤT BUILD GRU DATASET"
    )

    print(
        f"X shape : {X.shape}"
    )

    print(
        f"y shape : {y.shape}"
    )

    print()

    print(
        f"Rock windows     : "
        f"{class_counts['rock']}"
    )

    print(
        f"Paper windows    : "
        f"{class_counts['paper']}"
    )

    print(
        f"Scissors windows : "
        f"{class_counts['scissors']}"
    )

    print(
        f"Skipped short    : "
        f"{skipped_short}"
    )

    print(
        f"Saved            : "
        f"{OUTPUT_FILE.resolve()}"
    )


if __name__ == "__main__":
    main()