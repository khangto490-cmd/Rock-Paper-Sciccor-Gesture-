"""
train_static_v2.py
-------------------
Train RandomForest trên frame_dataset_v2.csv (11 đặc trưng từng ngón, nhãn báo trước).

Khác train_static.py:
  1. Đánh giá bằng kiểm tra chéo CHIA THEO CHUỖI (GroupKFold theo cột seq_id),
     nên accuracy không bị thổi phồng bởi các frame gần giống nhau của cùng 1 lần vung.
  2. Sau khi đánh giá xong, train lại trên TOÀN BỘ dữ liệu để có model cuối.
  3. Lưu model với n_jobs=1: lúc realtime mỗi lần chỉ dự đoán 1 mẫu, dùng nhiều
     luồng chỉ tốn thêm thời gian tạo luồng.

Chạy: python train_static_v2.py
Output: static_model_v2.pkl   (KHÔNG ghi đè static_model.pkl cũ, để còn so sánh)
"""

import csv

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

CSV_PATH = "frame_dataset_v2.csv"
MODEL_PATH = "static_model_v2.pkl"
N_TREES = 100      # giữ 100 để khớp với con số đã đo ở compare_methods.py
N_FOLDS = 5


def load_csv():
    X, y, groups = [], [], []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)                                   # bỏ dòng tiêu đề
        for row in r:
            X.append([float(v) for v in row[:-2]])
            y.append(row[-2])
            groups.append(int(row[-1]))
    return np.array(X, dtype=np.float32), np.array(y), np.array(groups)


def main():
    X, y_raw, groups = load_csv()
    print(f"{len(X)} frame, {len(set(groups))} chuoi, {X.shape[1]} dac trung")

    enc = LabelEncoder()
    y = enc.fit_transform(y_raw)

    model = RandomForestClassifier(n_estimators=N_TREES, random_state=42, n_jobs=-1)

    # Mỗi chuỗi nằm trọn ở train hoặc test trong mỗi lần chia.
    pred = cross_val_predict(model, X, y, groups=groups,
                             cv=GroupKFold(n_splits=N_FOLDS))

    print("\nBao cao (du doan ngoai mau, chia theo chuoi):")
    print(classification_report(y, pred, target_names=enc.classes_, digits=3))
    print("Confusion matrix (hang = that, cot = doan):")
    print(enc.classes_)
    print(confusion_matrix(y, pred))
    print("\nLuu y: day la do chinh xac theo FRAME. Muon biet 'som' bao nhieu frame")
    print("va bua bi kich hoat nham bao nhieu, dung compare_methods.py (M7).")

    # Model cuối: train trên toàn bộ dữ liệu.
    model.fit(X, y)
    model.set_params(n_jobs=1)
    joblib.dump({"model": model, "label_encoder": enc}, MODEL_PATH)
    print(f"\nDa luu {MODEL_PATH}")


if __name__ == "__main__":
    main()
