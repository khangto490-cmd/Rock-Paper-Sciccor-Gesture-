"""
train_static.py
-----------------
Đọc frame_dataset.csv (đã gán nhãn lại theo từng frame), train + so sánh
vài model phân loại TĨNH (không phải LSTM -- mỗi frame độc lập).

Chạy: python train_static.py
Output: static_model.pkl
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

CSV_PATH = "frame_dataset.csv"
MODEL_PATH = "static_model.pkl"


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Tong so frame: {len(df)}")
    print(df["label"].value_counts())

    X = df.drop(columns=["label"]).values
    y_raw = df["label"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    # Lưu ý: chia train/test theo TỪNG FRAME ở đây là hợp lý, khác với
    # LSTM trước đó -- vì giờ mỗi frame được coi là 1 mẫu độc lập, không
    # còn khái niệm "chuỗi" nữa nên không có nguy cơ rò rỉ giữa các bản
    # cắt của cùng 1 lần vung tay như trước.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM-RBF": SVC(kernel="rbf", C=10, gamma="scale", probability=True),
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
    }

    results = {}
    print("\n" + "=" * 45)
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        results[name] = (acc, model)
        print(f"{name:<20} accuracy = {acc:.4f}")
    print("=" * 45)

    best_name = max(results, key=lambda k: results[k][0])
    best_acc, best_model = results[best_name]
    print(f"\nModel tot nhat: {best_name} ({best_acc:.4f})")

    y_pred = best_model.predict(X_test)
    print("\nBao cao chi tiet:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    print("Confusion matrix:")
    print(encoder.classes_)
    print(confusion_matrix(y_test, y_pred))

    joblib.dump({"model": best_model, "label_encoder": encoder}, MODEL_PATH)
    print(f"\nDa luu model vao {MODEL_PATH}")


if __name__ == "__main__":
    main()