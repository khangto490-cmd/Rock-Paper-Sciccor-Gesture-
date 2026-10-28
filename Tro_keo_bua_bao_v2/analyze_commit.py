"""
analyze_commit.py
------------------
So sánh 3 cách "chốt" quyết định trên CÙNG xác suất từng khung của cùng một model, để xem
cách nào cho đánh đổi tốt nhất giữa tốc độ (thời gian từ lúc thấy tay) và độ chính xác:

  F (thoi gian co dinh): chốt sau T ms kể từ lúc thấy tay.
  P (vi tri o vuong)   : chốt khi tâm tay vào ô vuông (+ chờ thêm k khung trong ô).
  C (do tin cay)       : chốt ngay khi lớp dẫn đầu có xác suất >= theta ở DEBOUNCE khung liên tiếp
                         (nếu chưa tới ngưỡng thì chờ; hết dữ liệu thì chốt bằng khung cuối).

Câu hỏi chính: chốt theo độ tin cậy có chốt SỚM các ván rõ ràng và chờ lâu hơn các ván mơ hồ,
nhờ đó tốt hơn việc chốt theo thời gian cố định hoặc theo ô vuông, ở cùng thời gian trung bình không?

Chạy: python analyze_commit.py
      python analyze_commit.py --drop-c        (loại các ván búa loại C trong bua_flagged.csv)
Kết quả: in ra màn hình, ket_qua_commit.csv, commit_curves.png

Cần analyze_approach.py cùng thư mục. Kiểm tra chéo THEO VÁN; dữ liệu của một người.
"""

import argparse
import csv
import os
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from analyze_approach import GESTURES, G_IDX, load_rounds, prepare

BASE_SQUARE = 0.45                 # ô dùng làm mốc "đã vào ô"
SQUARES = [0.30, 0.45, 0.60]
EXTRA_K = [0, 3, 6]
F_TIMES = [100, 150, 200, 250, 300, 350, 400, 450, 500, 600]
THETAS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
DEBOUNCE = 2
TARGET_TIMES = [200, 250, 300, 350, 400]


# ------------------------------------------------------------------ chuẩn bị dữ liệu
def build(rounds):
    items = []
    for r in rounds:
        base = prepare(r, BASE_SQUARE)
        if base is None:
            continue
        entries = {}
        for f in SQUARES:
            p = prepare(r, f)           # cùng các khung giữ lại cho mọi cỡ ô -> chỉ số khớp nhau
            entries[f] = None if p is None else p["entry"]
        base["entries"] = entries
        base["t_ms"] = (base["tt"] - base["tt"][0]) * 1000.0
        items.append(base)
    return items


# ------------------------------------------------------------------ 3 quy tắc chốt
def rule_f(probs, t_ms, T):
    idx = int(np.searchsorted(t_ms, T, side="right") - 1)
    idx = max(0, min(idx, len(t_ms) - 1))
    return int(np.argmax(probs[idx])), float(t_ms[idx])


def rule_p(probs, t_ms, entry, k):
    if entry is None:
        return None
    idx = min(entry + k, len(t_ms) - 1)
    return int(np.argmax(probs[idx])), float(t_ms[idx])


def rule_c(probs, t_ms, theta, debounce=DEBOUNCE):
    cls, conf = probs.argmax(axis=1), probs.max(axis=1)
    ok = conf >= theta
    for i in range(debounce - 1, len(cls)):
        c = cls[i]
        if all(ok[j] and cls[j] == c for j in range(i - debounce + 1, i + 1)):
            return int(c), float(t_ms[i])
    return int(cls[-1]), float(t_ms[-1])          # chưa đủ tự tin tới hết dữ liệu -> chốt khung cuối


class Acc:
    def __init__(self):
        self.true, self.pred, self.time, self.early = [], [], [], []

    def add(self, true, pred, time, ref_time):
        self.true.append(true)
        self.pred.append(pred)
        self.time.append(time)
        self.early.append(time < ref_time)

    def summary(self):
        t, p, tm = np.array(self.true), np.array(self.pred), np.array(self.time)
        out = dict(n=len(t), acc=100 * float(np.mean(t == p)),
                   mean_ms=float(tm.mean()), med_ms=float(np.median(tm)), p90_ms=float(np.percentile(tm, 90)),
                   pct_before_entry=100 * float(np.mean(self.early)))
        for g in GESTURES:
            m = t == G_IDX[g]
            out["acc_" + g] = 100 * float(np.mean(p[m] == t[m])) if m.any() else float("nan")
        return out


# ------------------------------------------------------------------ chạy thí nghiệm
def experiment(items, reps):
    y = np.array([G_IDX[p["gesture"]] for p in items])
    n_splits = min(5, int(min(np.bincount(y, minlength=3)[np.bincount(y, minlength=3) > 0])))
    accs = {}

    def get(key):
        return accs.setdefault(key, Acc())

    for rep in range(reps):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rep)
        for tr, te in skf.split(np.zeros(len(y)), y):
            X = np.concatenate([items[i]["eng"] for i in tr])
            lab = np.concatenate([np.full(len(items[i]["eng"]), y[i]) for i in tr])
            rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=10,
                                        random_state=rep, n_jobs=-1).fit(X, lab)
            for i in te:
                p, true = items[i], y[i]
                probs = rf.predict_proba(p["eng"])
                t_ms = p["t_ms"]
                ref = t_ms[p["entries"][BASE_SQUARE]]          # thời điểm vào ô mốc (0.45)

                for T in F_TIMES:
                    pred, tm = rule_f(probs, t_ms, T)
                    get(("F", f"T={T} ms")).add(true, pred, tm, ref)
                for f in SQUARES:
                    for k in EXTRA_K:
                        res = rule_p(probs, t_ms, p["entries"][f], k)
                        if res is not None:
                            get(("P", f"o={f:.2f}, +{k} khung")).add(true, res[0], res[1], ref)
                for th in THETAS:
                    pred, tm = rule_c(probs, t_ms, th)
                    get(("C", f"theta={th:.2f}")).add(true, pred, tm, ref)
    return accs, n_splits


def best_under(rows, family, target):
    cand = [r for r in rows if r["family"] == family and r["mean_ms"] <= target]
    if not cand:
        return None
    return max(cand, key=lambda r: r["acc"])


def make_plot(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(khong co matplotlib nen bo qua ve hinh)")
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    style = {"F": ("tab:gray", "o", "F: thoi gian co dinh"),
             "P": ("tab:blue", "s", "P: vi tri o vuong (+ cho them khung)"),
             "C": ("tab:red", "^", "C: do tin cay")}
    for fam, (color, marker, label) in style.items():
        pts = sorted([(r["mean_ms"], r["acc"]) for r in rows if r["family"] == fam])
        if fam == "P":
            ax.scatter(*zip(*pts), color=color, marker=marker, label=label)
        else:
            ax.plot(*zip(*pts), color=color, marker=marker, label=label)
    ax.set_xlabel("thoi gian chot trung binh ke tu luc thay tay (ms)")
    ax.set_ylabel("ti le van doan dung (%)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Da luu hinh: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="approach_data")
    ap.add_argument("--person", default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--drop-c", action="store_true")
    args = ap.parse_args()

    rounds = load_rounds(args.outdir, args.person)
    if args.drop_c:
        with open("bua_flagged.csv", newline="", encoding="utf-8") as f:
            drop = {r["file"] for r in csv.DictReader(f) if r["category"].startswith("C")}
        before = len(rounds)
        rounds = [r for r in rounds if os.path.basename(r["path"]) not in drop]
        print(f"[--drop-c] Da loai {before - len(rounds)} van bua loai C.")
    items = build(rounds)
    if len(items) < 15:
        print(f"Chi co {len(items)} van dung duoc, qua it.")
        return
    print(f"{len(items)} van; kiem tra cheo theo van, lap {args.reps} lan.")
    accs, n_splits = experiment(items, args.reps)

    rows = []
    for (fam, param), acc in accs.items():
        s = acc.summary()
        rows.append(dict(family=fam, param=param, **s))

    names = {"F": "F - THOI GIAN CO DINH", "P": "P - VI TRI O VUONG", "C": "C - DO TIN CAY (>= theta, 2 khung lien tiep)"}
    for fam in ("F", "P", "C"):
        print("\n" + "=" * 92)
        print(names[fam])
        print("=" * 92)
        print(f"{'tham so':<22}{'TB ms':>8}{'trung vi':>10}{'p90':>8}{'chinh xac':>11}{'keo':>8}{'bua':>8}{'bao':>8}{'som hon o':>11}")
        for r in [r for r in rows if r["family"] == fam]:
            print(f"{r['param']:<22}{r['mean_ms']:>8.0f}{r['med_ms']:>10.0f}{r['p90_ms']:>8.0f}"
                  f"{r['acc']:>10.1f}%{r['acc_keo']:>7.0f}%{r['acc_bua']:>7.0f}%{r['acc_bao']:>7.0f}%{r['pct_before_entry']:>10.0f}%")
    print("\n('som hon o' = % van chot truoc luc tay vao o vuong 0.45)")

    print("\n" + "=" * 92)
    print("SO SANH CUNG THOI GIAN: voi moi moc thoi gian TB toi da, cach nao dung nhieu nhat?")
    print("=" * 92)
    print(f"{'TB toi da':<12}" + "".join(f"{fam + ' (cau hinh tot nhat)':<38}" for fam in ("F", "P", "C")))
    for target in TARGET_TIMES:
        cells = []
        for fam in ("F", "P", "C"):
            b = best_under(rows, fam, target)
            cells.append(f"{b['acc']:.1f}% [{b['param']}]" if b else "-")
        print(f"{target:<12}" + "".join(f"{c:<38}" for c in cells))
    print(f"\nLuu y: ~{len(items)} van, do lech chuan ngau nhien moi con so khoang 4-5 diem %; chenh lech nho hon ~5 diem % chua dang tin.")

    with open("ket_qua_commit.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Da luu ket_qua_commit.csv")
    make_plot(rows, "commit_curves.png")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        main()
