from pathlib import Path
import random

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = Path("gru_dataset.npz")

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL_PATH = (
    CHECKPOINT_DIR
    / "tiny_gru_rps_best.pth"
)

META_PATH = (
    CHECKPOINT_DIR
    / "tiny_gru_rps_meta.pkl"
)


SEED = 42

BATCH_SIZE = 32
EPOCHS = 100

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

HIDDEN_SIZE = 32

DROPOUT = 0.20

PATIENCE = 15


# ============================================================
# SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"Device: {device}"
)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD DATASET
# ============================================================

data = np.load(
    DATASET_PATH,
    allow_pickle=True
)

X = data["X"].astype(
    np.float32
)

y = data["y"].astype(
    np.int64
)

clip_ids = data["clip_ids"].astype(
    str
)

persons = data["persons"].astype(
    str
)

label_names = data["label_names"].astype(
    str
)

print()
print(
    f"X shape: {X.shape}"
)

print(
    f"y shape: {y.shape}"
)

print(
    "Labels:",
    list(label_names)
)

print(
    f"Unique clips: "
    f"{len(np.unique(clip_ids))}"
)

print(
    f"Persons: "
    f"{np.unique(persons)}"
)


# ============================================================
# SPLIT BY CLIP
#
# 70% train
# 15% val
# 15% test
# ============================================================

splitter_1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=SEED,
)

train_idx, temp_idx = next(
    splitter_1.split(
        X,
        y,
        groups=clip_ids,
    )
)


X_train = X[train_idx]
y_train = y[train_idx]

X_temp = X[temp_idx]
y_temp = y[temp_idx]

temp_groups = clip_ids[temp_idx]


splitter_2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=SEED,
)

val_local_idx, test_local_idx = next(
    splitter_2.split(
        X_temp,
        y_temp,
        groups=temp_groups,
    )
)


X_val = X_temp[
    val_local_idx
]

y_val = y_temp[
    val_local_idx
]


X_test = X_temp[
    test_local_idx
]

y_test = y_temp[
    test_local_idx
]


print()
print("=== SPLIT ===")

print(
    f"Train: {len(X_train)}"
)

print(
    f"Val  : {len(X_val)}"
)

print(
    f"Test : {len(X_test)}"
)


# ============================================================
# CHECK CLASS DISTRIBUTION
# ============================================================

def print_distribution(
    name,
    labels,
):

    print()
    print(
        f"{name} distribution:"
    )

    for idx, class_name in enumerate(
        label_names
    ):
        count = int(
            np.sum(
                labels == idx
            )
        )

        print(
            f"  {class_name:8s}: "
            f"{count}"
        )


print_distribution(
    "Train",
    y_train,
)

print_distribution(
    "Val",
    y_val,
)

print_distribution(
    "Test",
    y_test,
)


# ============================================================
# NORMALIZATION
#
# Mean/std chỉ lấy từ TRAIN
# ============================================================

feature_mean = X_train.mean(
    axis=(0, 1),
    keepdims=True,
)

feature_std = X_train.std(
    axis=(0, 1),
    keepdims=True,
)

feature_std = np.where(
    feature_std < 1e-6,
    1.0,
    feature_std,
)


X_train = (
    X_train
    - feature_mean
) / feature_std


X_val = (
    X_val
    - feature_mean
) / feature_std


X_test = (
    X_test
    - feature_mean
) / feature_std


# ============================================================
# CLASS WEIGHTS
# ============================================================

classes = np.unique(
    y_train
)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train,
)

weights_tensor = torch.ones(
    len(label_names),
    dtype=torch.float32,
)

for class_id, weight in zip(
    classes,
    class_weights,
):
    weights_tensor[
        int(class_id)
    ] = float(weight)

weights_tensor = weights_tensor.to(
    device
)

print()
print(
    "Class weights:",
    weights_tensor
)


# ============================================================
# DATALOADER
# ============================================================

def make_loader(
    X_data,
    y_data,
    shuffle,
):

    X_tensor = torch.from_numpy(
        X_data
    ).float()

    y_tensor = torch.from_numpy(
        y_data
    ).long()

    dataset = TensorDataset(
        X_tensor,
        y_tensor,
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


train_loader = make_loader(
    X_train,
    y_train,
    shuffle=True,
)

val_loader = make_loader(
    X_val,
    y_val,
    shuffle=False,
)

test_loader = make_loader(
    X_test,
    y_test,
    shuffle=False,
)


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

        self.dropout = nn.Dropout(
            dropout
        )

        self.classifier = nn.Linear(
            hidden_size,
            num_classes,
        )

    def forward(self, x):

        # x:
        # (batch, time, feature)
        #
        # example:
        # (32, 36, 63)

        output, hidden = self.gru(
            x
        )

        # hidden:
        # (1, batch, hidden_size)

        last_hidden = hidden[-1]

        last_hidden = self.dropout(
            last_hidden
        )

        logits = self.classifier(
            last_hidden
        )

        return logits


model = TinyGRU(
    input_size=X.shape[2],
    hidden_size=HIDDEN_SIZE,
    num_classes=len(
        label_names
    ),
    dropout=DROPOUT,
).to(device)


print()
print(model)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=weights_tensor
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(loader):

    model.eval()

    total_loss = 0.0

    all_preds = []
    all_targets = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(
                device,
                non_blocking=True,
            )

            y_batch = y_batch.to(
                device,
                non_blocking=True,
            )

            logits = model(
                X_batch
            )

            loss = criterion(
                logits,
                y_batch
            )

            total_loss += (
                loss.item()
                * X_batch.size(0)
            )

            preds = torch.argmax(
                logits,
                dim=1,
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_targets.extend(
                y_batch.cpu().numpy()
            )

    average_loss = (
        total_loss
        / len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_targets,
        all_preds,
    )

    return (
        average_loss,
        accuracy,
        np.asarray(all_targets),
        np.asarray(all_preds),
    )


# ============================================================
# TRAIN
# ============================================================

best_val_loss = float(
    "inf"
)

patience_counter = 0


for epoch in range(
    1,
    EPOCHS + 1,
):

    model.train()

    total_train_loss = 0.0

    correct = 0
    total = 0


    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(
            device,
            non_blocking=True,
        )

        y_batch = y_batch.to(
            device,
            non_blocking=True,
        )


        optimizer.zero_grad()


        logits = model(
            X_batch
        )


        loss = criterion(
            logits,
            y_batch
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )


        optimizer.step()


        total_train_loss += (
            loss.item()
            * X_batch.size(0)
        )


        preds = torch.argmax(
            logits,
            dim=1,
        )


        correct += (
            preds == y_batch
        ).sum().item()


        total += y_batch.size(0)


    train_loss = (
        total_train_loss
        / len(train_loader.dataset)
    )

    train_acc = (
        correct / total
    )


    (
        val_loss,
        val_acc,
        _,
        _,
    ) = evaluate(
        val_loader
    )


    print(
        f"Epoch "
        f"{epoch:03d}/{EPOCHS} "
        f"| train_loss "
        f"{train_loss:.4f} "
        f"| train_acc "
        f"{train_acc:.4f} "
        f"| val_loss "
        f"{val_loss:.4f} "
        f"| val_acc "
        f"{val_acc:.4f}"
    )


    # ========================================================
    # BEST MODEL
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        patience_counter = 0

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "input_size":
                    X.shape[2],

                "hidden_size":
                    HIDDEN_SIZE,

                "num_classes":
                    len(label_names),

                "window_size":
                    X.shape[1],

                "label_names":
                    list(label_names),

                "feature_mean":
                    feature_mean.astype(
                        np.float32
                    ),

                "feature_std":
                    feature_std.astype(
                        np.float32
                    ),
            },
            BEST_MODEL_PATH,
        )

    else:

        patience_counter += 1


    if (
        patience_counter
        >= PATIENCE
    ):

        print()
        print(
            "Early stopping."
        )

        break


# ============================================================
# LOAD BEST MODEL
# ============================================================

checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=device,
    weights_only=False,
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


# ============================================================
# TEST
# ============================================================

(
    test_loss,
    test_acc,
    y_true,
    y_pred,
) = evaluate(
    test_loader
)


print()
print("=" * 70)

print(
    "TEST RESULT"
)

print(
    f"Loss     : "
    f"{test_loss:.4f}"
)

print(
    f"Accuracy : "
    f"{test_acc * 100:.2f}%"
)


print()
print(
    "Classification Report:"
)

print(
    classification_report(
        y_true,
        y_pred,
        labels=np.arange(
            len(label_names)
        ),
        target_names=label_names,
        zero_division=0,
    )
)


print(
    "Confusion Matrix:"
)

print(
    confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(
            len(label_names)
        ),
    )
)


# ============================================================
# SAVE META
# ============================================================

joblib.dump(
    {
        "label_names":
            list(label_names),

        "window_size":
            int(X.shape[1]),

        "input_size":
            int(X.shape[2]),

        "hidden_size":
            HIDDEN_SIZE,

        "feature_mean":
            feature_mean,

        "feature_std":
            feature_std,
    },
    META_PATH,
)


print()
print(
    f"Best model saved:"
)

print(
    BEST_MODEL_PATH.resolve()
)

print()
print(
    f"Meta saved:"
)

print(
    META_PATH.resolve()
)