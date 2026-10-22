from pathlib import Path
import csv
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

DATASET_PATH = Path("gru_dataset.npz")
CHECKPOINT_PATH = Path("checkpoints/tiny_gru_rps_best.pth")
REPORT_DIR = Path("reports")
SEED = 42
BATCH_SIZE = 256


class TinyGRU(nn.Module):
    def __init__(self, input_size=63, hidden_size=32, num_classes=3, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.classifier(self.dropout(hidden[-1]))


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy {DATASET_PATH}; đặt file tại thư mục gốc.")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy {CHECKPOINT_PATH}; kiểm tra checkpoint.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    data = np.load(DATASET_PATH, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    clip_ids = data["clip_ids"].astype(str)
    dataset_labels = [str(v) for v in data["label_names"]]

    # Reproduce the same deterministic split used by train_gru.py.
    split1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, temp_idx = next(split1.split(X, y, groups=clip_ids))
    X_temp, y_temp = X[temp_idx], y[temp_idx]
    temp_groups = clip_ids[temp_idx]

    split2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    _, test_local_idx = next(split2.split(X_temp, y_temp, groups=temp_groups))
    test_idx = temp_idx[test_local_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    y_train = y[train_idx]

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    labels = [str(v) for v in checkpoint["label_names"]]
    if dataset_labels != labels:
        raise ValueError(
            f"Thứ tự nhãn không khớp. Dataset={dataset_labels}; checkpoint={labels}. "
            "Hãy xác minh dataset/checkpoint đúng cặp."
        )
    if X.ndim != 3:
        raise ValueError(f"X cần dạng (samples, frames, features), nhận được {X.shape}")

    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    # Use training normalization saved inside this checkpoint.
    X_test = (X_test - mean) / std

    model = TinyGRU(
        input_size=int(checkpoint["input_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        num_classes=int(checkpoint["num_classes"]),
        dropout=0.20,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    weights = torch.ones(len(labels), dtype=torch.float32)
    for class_id, weight in zip(classes, class_weights):
        weights[int(class_id)] = float(weight)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    all_true, all_pred = [], []
    total_loss = 0.0
    with torch.inference_mode():
        for start in range(0, len(X_test), BATCH_SIZE):
            xb = torch.from_numpy(X_test[start:start + BATCH_SIZE]).float().to(device)
            yb = torch.from_numpy(y_test[start:start + BATCH_SIZE]).long().to(device)
            logits = model(xb)
            total_loss += criterion(logits, yb).item() * len(yb)
            all_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            all_true.extend(yb.cpu().numpy().tolist())

    y_true = np.asarray(all_true, dtype=np.int64)
    y_pred = np.asarray(all_pred, dtype=np.int64)
    if len(y_true) == 0:
        raise ValueError("Tập test rỗng; không thể đánh giá.")

    test_loss = total_loss / len(y_true)
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, labels=np.arange(len(labels)),
        target_names=labels, zero_division=0, digits=4
    )
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "evaluation_results.txt"
    matrix_path = REPORT_DIR / "confusion_matrix.csv"

    report_text = "\n".join([
        "GRU Model Evaluation",
        f"Timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"Dataset: {DATASET_PATH}",
        f"Checkpoint: {CHECKPOINT_PATH}",
        f"Device: {device}",
        f"Total dataset samples: {len(X)}",
        f"Train split samples: {len(train_idx)}",
        f"Test split samples: {len(test_idx)}",
        f"Test loss (class-weighted): {test_loss:.6f}",
        f"Test accuracy: {accuracy:.6f} ({accuracy * 100:.2f}%)",
        "",
        "Classification report:",
        report,
        "Confusion matrix (rows=true labels, columns=predicted labels):",
        str(matrix),
        "",
        "Class order: " + ", ".join(labels),
        "",
        "Validity note: results are valid for this checkpoint only if gru_dataset.npz "
        "is the same dataset version used to train it."
    ])
    report_path.write_text(report_text, encoding="utf-8")

    with matrix_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["true/predicted"] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + row.tolist())

    print("\n" + "=" * 70)
    print("TEST RESULT")
    print(f"Test samples: {len(y_true)}")
    print(f"Test loss (class-weighted): {test_loss:.4f}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:\n" + report)
    print("Confusion Matrix (rows=true, columns=predicted):")
    print(matrix)
    print("\nSaved report:", report_path.resolve())
    print("Saved confusion matrix:", matrix_path.resolve())


if __name__ == "__main__":
    main()
