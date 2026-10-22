from pathlib import Path
import numpy as np

# ============================================================
# CONFIG
# ============================================================

INPUT_ROOT = Path(
    r"C:\Users\letri\Downloads\Landmarks_120fps-20261009T033818Z-1-001\Landmarks_120fps"
)

OUTPUT_ROOT = Path(
    r"C:\Users\letri\OneDrive\Desktop\Hand-project\gesture_project\sequences_labeled\person_02"
)

CLASS_MAP = {
    "Rock": "rock",
    "Paper": "paper",
    "Scissors": "scissors",
}

EPS = 1e-6


# ============================================================
# NORMALIZATION
# ============================================================

def safe_normalize(v, eps=EPS):
    norm = np.linalg.norm(v)

    if norm < eps:
        return None

    return v / norm


def normalize_landmarks(frame_landmarks):
    """
    Input:
        frame_landmarks: (21, 3)

    Output:
        features: (63,)
    """

    landmarks = np.asarray(
        frame_landmarks,
        dtype=np.float32,
    )

    if landmarks.shape != (21, 3):
        return None

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

    y_axis = safe_normalize(
        y_temp
    )

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

    local_landmarks = (
        centered @ basis
    )

    palm_scale = np.linalg.norm(
        middle_mcp - wrist
    )

    if palm_scale > EPS:
        local_landmarks /= palm_scale

    features = (
        local_landmarks
        .reshape(-1)
        .astype(np.float32)
    )

    if features.shape[0] != 63:
        return None

    return features


# ============================================================
# INTERPOLATION
# ============================================================

def interpolate_missing(sequence):
    """
    sequence: (T, 63), có thể chứa NaN
    """

    sequence = np.asarray(
        sequence,
        dtype=np.float32,
    )

    T, F = sequence.shape

    x = np.arange(T)

    for feature_idx in range(F):
        values = sequence[:, feature_idx]

        valid = np.isfinite(values)

        if valid.sum() == 0:
            sequence[:, feature_idx] = 0.0
            continue

        if valid.sum() == 1:
            sequence[:, feature_idx] = (
                values[valid][0]
            )
            continue

        sequence[:, feature_idx] = np.interp(
            x,
            x[valid],
            values[valid],
        )

    return sequence


# ============================================================
# PROCESS FILE
# ============================================================

def process_file(
    npz_path,
    output_dir,
    class_label,
):
    data = np.load(
        npz_path,
        allow_pickle=True,
    )

    required_keys = {
        "landmarks",
        "mask",
        "fps",
        "label",
    }

    missing_keys = (
        required_keys
        - set(data.files)
    )

    if missing_keys:
        raise ValueError(
            f"Thiếu keys: {missing_keys}"
        )

    landmarks = np.asarray(
        data["landmarks"],
        dtype=np.float32,
    )

    mask = np.asarray(
        data["mask"]
    ).astype(bool)

    fps = float(
        np.asarray(data["fps"]).item()
    )

    if (
        landmarks.ndim != 3
        or landmarks.shape[1:] != (21, 3)
    ):
        raise ValueError(
            f"landmarks shape sai: "
            f"{landmarks.shape}"
        )

    if len(mask) != len(landmarks):
        raise ValueError(
            "mask length không khớp landmarks"
        )

    total_frames = len(landmarks)

    sequence = np.full(
        (total_frames, 63),
        np.nan,
        dtype=np.float32,
    )

    valid_count = 0

    for i in range(total_frames):
        if not mask[i]:
            continue

        features = normalize_landmarks(
            landmarks[i]
        )

        if features is None:
            continue

        sequence[i] = features
        valid_count += 1

    if valid_count == 0:
        raise ValueError(
            "Không có frame landmark hợp lệ"
        )

    sequence = interpolate_missing(
        sequence
    )

    detection_rate = (
        valid_count
        / total_frames
        * 100.0
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{npz_path.stem}.npz"
    )

    np.savez_compressed(
        output_path,
        sequence=sequence,
        label=class_label,
        person="person_02",
        fps=np.float32(fps),
        total_frames=np.int32(
            total_frames
        ),
        detection_rate=np.float32(
            detection_rate
        ),
        source_file=npz_path.name,
    )

    return (
        output_path,
        sequence.shape,
        detection_rate,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    if not INPUT_ROOT.exists():
        print(
            f"Không tìm thấy input:\n"
            f"{INPUT_ROOT}"
        )
        return

    success = 0
    failed = 0

    print("=" * 70)
    print("CONVERT PERSON_02 LANDMARKS")
    print("=" * 70)

    for source_folder, class_label in CLASS_MAP.items():
        input_dir = (
            INPUT_ROOT
            / source_folder
        )

        output_dir = (
            OUTPUT_ROOT
            / class_label
        )

        if not input_dir.exists():
            print(
                f"[WARN] Không có folder: "
                f"{input_dir}"
            )
            continue

        files = sorted(
            input_dir.glob("*.npz")
        )

        print()
        print(
            f"{source_folder}: "
            f"{len(files)} files"
        )

        for npz_path in files:
            try:
                (
                    output_path,
                    shape,
                    detection_rate,
                ) = process_file(
                    npz_path,
                    output_dir,
                    class_label,
                )

                success += 1

                print(
                    f"[OK] "
                    f"{npz_path.name} "
                    f"-> {shape} "
                    f"| detect "
                    f"{detection_rate:.1f}%"
                )

            except Exception as e:
                failed += 1

                print(
                    f"[FAIL] "
                    f"{npz_path.name}: "
                    f"{e}"
                )

    print()
    print("=" * 70)
    print("HOÀN TẤT CONVERT PERSON_02")
    print(f"Success : {success}")
    print(f"Failed  : {failed}")
    print(f"Output  : {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
