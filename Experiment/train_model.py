"""
train_model.py
--------------
Huấn luyện model dùng cho play_realtime.py từ dữ liệu thu bằng collect_approach.py.
Model: RandomForest đoán cử chỉ (kéo/búa/bao) từ đặc trưng của MỘT khung hình, giống "model" trong
analyze_approach.py (huấn luyện trên mọi khung có tay, nhãn = cử chỉ người chơi định làm).

Chạy:
    python train_model.py                                  (đọc approach_data_v2, ghi rps_model.joblib)
    python train_model.py --outdir approach_data_v2 --person hi --out rps_model.joblib

Lưu ý: file model chỉ dùng được với cùng phiên bản scikit-learn đã huấn luyện (cùng môi trường rps_env).
"""

import argparse
from collections import Counter

import joblib
from sklearn.ensemble import RandomForestClassifier

from rps_core import GESTURES, load_training_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="approach_data_v2")
    ap.add_argument("--person", default=None)
    ap.add_argument("--out", default="rps_model.joblib")
    ap.add_argument("--trees", type=int, default=100)
    ap.add_argument("--min-leaf", type=int, default=10)
    args = ap.parse_args()

    X, y, n_rounds = load_training_frames(args.outdir, args.person)
    if n_rounds < 15:
        print(f"Chi co {n_rounds} van hop le trong '{args.outdir}', qua it de huan luyen (can it nhat 15).")
        return
    cnt = Counter(y.tolist())
    print(f"Huan luyen tren {n_rounds} van, {len(y)} khung: " +
          ", ".join(f"{g} {cnt[i]}" for i, g in enumerate(GESTURES)))
    rf = RandomForestClassifier(n_estimators=args.trees, min_samples_leaf=args.min_leaf,
                                random_state=0, n_jobs=-1).fit(X, y)
    rf.n_jobs = 1                       # luc chay that du doan tung mau, khong can song song
    joblib.dump(dict(model=rf, version=1, rounds=n_rounds, frames=len(y), source=args.outdir), args.out)
    print(f"Da luu {args.out}")


if __name__ == "__main__":
    main()
