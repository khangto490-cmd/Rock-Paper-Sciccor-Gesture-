"""
analyze_approach.py
--------------------
Phân tích dữ liệu thu bằng collect_approach.py (thư mục approach_data/).

Trả lời các câu hỏi:
  1. Dữ liệu thu có ổn không (số ván, ván lỗi, FPS thật, nhãn tay của MediaPipe)?
  2. Từ lúc thấy tay đến lúc tay vào ô vuông có bao nhiêu khung hình (cửa sổ bằng chứng)?
  3. Ngón tay có bắt đầu duỗi TRƯỚC khi tay vào ô không, và sớm bao nhiêu?
  4. Quyết định ở lúc vào ô chính xác bao nhiêu nếu:
       - chỉ nhìn khung hình hiện tại (lam = 0),
       - cộng dồn đều các khung trước đó (lam = 1),
       - cộng dồn có điểm cũ phai dần (0 < lam < 1),
     và nếu chờ thêm k khung nữa trong ô.
  5. Chốt theo ĐỘ TỰ TIN (không cộng dồn): mỗi khung tự quyết, chốt ngay khi xác suất cao nhất
     vượt ngưỡng (có thể TRƯỚC khi tay vào ô); nếu đến hạn mà chưa đủ tự tin thì chốt theo khung
     cuối. Báo độ chính xác và độ trễ so với lúc tay vào ô (ms, âm = chốt trước khi vào ô).
     Mục tiêu độ trễ ~30 ms = một khung hình, tức hạn chốt là khung đầu tiên trong ô.

Chạy: python analyze_approach.py
      python analyze_approach.py --person hi --square 0.2 0.25 0.3
      python analyze_approach.py --min-frames 4      # số khung tối thiểu có tay trước khi được chốt sớm

Kết quả "chính xác" ở mục 4 được tính bằng kiểm tra chéo THEO VÁN (mỗi ván chỉ nằm ở
train hoặc test). Dữ liệu của một người nên con số chỉ nói về người đó.
"""

import argparse
import csv
import glob
import json
import os
from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

GESTURES = ["keo", "bua", "bao"]          # chỉ số 0, 1, 2
G_IDX = {g: i for i, g in enumerate(GESTURES)}
PALM_IDS = [0, 5, 9, 13, 17]
TIPS5, TIPS4, MCPS4 = [4, 8, 12, 16, 20], [8, 12, 16, 20], [5, 9, 13, 17]

LAMBDAS = [0.0, 0.5, 0.7, 0.9, 1.0]
EXTRA_FRAMES = [0, 3, 6]
POLICY_WAITS = [0, 3]                         # hạn chốt: khung vào ô + W khung (W=0 ~ 30 ms)
THRESHOLDS = [0.60, 0.70, 0.80, 0.90, 0.95, 1.01]   # 1.01 = không bao giờ chốt sớm (chốt ở hạn)


# ------------------------------------------------------------------ nạp dữ liệu
def load_rounds(outdir, person=None):
    rounds = []
    for path in sorted(glob.glob(os.path.join(outdir, "person_*", "session_*", "r*.npz"))):
        d = np.load(path)
        meta = json.loads(str(d["meta_json"]))
        if not meta.get("valid"):
            continue
        if person and meta["person"] != person:
            continue
        rounds.append(dict(meta=meta, t=d["t"], lm=d["landmarks"], det=d["detected"],
                           mp_label=d["mp_label"], path=path))
    return rounds


def invalid_reasons(outdir):
    p = os.path.join(outdir, "rounds_index.csv")
    if not os.path.exists(p):
        return Counter()
    with open(p, newline="", encoding="utf-8") as f:
        return Counter(r["reason"] for r in csv.DictReader(f) if r["valid"] != "True")


# ------------------------------------------------------------------ đặc trưng
def normalize(lm):
    """(n,21,3) toạ độ gốc -> chuẩn hóa theo cổ tay và chiều dài cổ tay->gốc ngón giữa.
    Giống hệt features.extract_features của dự án."""
    scale = np.linalg.norm(lm[:, 9] - lm[:, 0], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (lm - lm[:, 0:1]) / scale[:, None, None]


def eng_features(P):
    ext = np.linalg.norm(P[:, TIPS5], axis=2)
    curl = np.linalg.norm(P[:, TIPS4] - P[:, MCPS4], axis=2)
    spread = np.stack([np.linalg.norm(P[:, 8] - P[:, 12], axis=1),
                       np.linalg.norm(P[:, 12] - P[:, 16], axis=1)], axis=1)
    return np.concatenate([ext, curl, spread], axis=1)


def ext_keo(P):
    return (np.linalg.norm(P[:, 8], axis=1) + np.linalg.norm(P[:, 12], axis=1)) / 2


def ext_bao(P):
    return np.mean([np.linalg.norm(P[:, i], axis=1) for i in TIPS4], axis=0)


def square_bounds(w, h, frac):
    side = frac * min(w, h)
    return (0.5 - side / (2 * w), 0.5 + side / (2 * w),
            0.5 - side / (2 * h), 0.5 + side / (2 * h))


def prepare(r, frac):
    """Giữ các khung có tay và đặc trưng hợp lệ; tìm khung đầu tiên tâm tay vào ô."""
    det = np.where(r["det"])[0]
    if len(det) == 0:
        return None
    Pn = normalize(r["lm"][det])
    ok = np.isfinite(Pn).all(axis=(1, 2))
    det, Pn = det[ok], Pn[ok]
    if len(det) == 0:
        return None
    raw = r["lm"][det]
    palm = raw[:, PALM_IDS, :2].mean(axis=1)
    x0, x1, y0, y1 = square_bounds(r["meta"]["frame_w"], r["meta"]["frame_h"], frac)
    inside = (palm[:, 0] >= x0) & (palm[:, 0] <= x1) & (palm[:, 1] >= y0) & (palm[:, 1] <= y1)
    if not inside.any():
        return None
    return dict(entry=int(np.argmax(inside)), tt=r["t"][det], eng=eng_features(Pn),
                ek=ext_keo(Pn), eb=ext_bao(Pn), gesture=r["meta"]["gesture"],
                hand=r["meta"]["hand_cue_player"], idx_full=det)


# ------------------------------------------------------------------ ngưỡng từ dữ liệu mới
def thresholds(prepared):
    bua_ek = np.concatenate([p["ek"] for p in prepared if p["gesture"] == "bua"])
    bua_eb = np.concatenate([p["eb"] for p in prepared if p["gesture"] == "bua"])
    start = dict(keo=float(np.percentile(bua_ek, 95)), bao=float(np.percentile(bua_eb, 95)))
    last_k = np.array([p["ek"][-5:].mean() for p in prepared if p["gesture"] == "keo"])
    last_b = np.array([p["eb"][-5:].mean() for p in prepared if p["gesture"] == "bao"])
    formed = dict(keo=float(np.percentile(last_k, 10)), bao=float(np.percentile(last_b, 10)))
    return start, formed


def first_true(mask):
    i = np.where(mask)[0]
    return int(i[0]) if len(i) else None


def pctl(x, ps=(10, 50, 90)):
    return " / ".join(f"{np.percentile(x, p):6.1f}" for p in ps) if len(x) else "n/a"


# ------------------------------------------------------------------ 1. chất lượng dữ liệu
def data_quality(rounds, outdir):
    print("=" * 72)
    print("1. CHAT LUONG DU LIEU")
    print("=" * 72)
    cnt = Counter((r["meta"]["gesture"], r["meta"]["hand_cue_player"]) for r in rounds)
    print(f"So van hop le: {len(rounds)}")
    for (g, h), n in sorted(cnt.items()):
        print(f"  cu chi {g:<4} tay {h:<4}: {n}")
    bad = invalid_reasons(outdir)
    if bad:
        print("Van loi da bi loai (tu rounds_index.csv):")
        for reason, n in bad.most_common():
            print(f"  {reason}: {n}")
    dts = np.concatenate([np.diff(r["t"]) for r in rounds if len(r["t"]) > 2])
    print(f"FPS thuc te (trung vi): {1.0 / np.median(dts):.1f}")
    det_rate = np.mean([r["det"].mean() for r in rounds])
    print(f"Ti le khung thay tay trong van (trung binh): {100 * det_rate:.0f}%")
    table = Counter()
    for r in rounds:
        labs = [l for l, d in zip(r["mp_label"], r["det"]) if d and l]
        if labs:
            table[(r["meta"]["hand_cue_player"], Counter(labs).most_common(1)[0][0])] += 1
    print("Nhan Left/Right cua MediaPipe so voi tay theo LENH:")
    for (hand, lab), n in sorted(table.items()):
        print(f"  lenh tay {hand}: MediaPipe bao '{lab}' trong {n} van")


# ------------------------------------------------------------------ 2 + 3. cửa sổ bằng chứng, thời điểm bắt đầu duỗi
def window_and_onset(prepared, start, formed, rows_out):
    n_ev, dur, drop = [], [], []
    for p in prepared:
        e = p["entry"]
        n_ev.append(e + 1)
        dur.append((p["tt"][e] - p["tt"][0]) * 1000)
        span = p["idx_full"][e] - p["idx_full"][0] + 1
        drop.append(100 * (1 - (e + 1) / span))
    print("\n2. CUA SO BANG CHUNG (tu luc thay tay den luc tay vao o)   [p10 / trung vi / p90]")
    print(f"   so khung co tay:        {pctl(n_ev)}")
    print(f"   thoi gian (ms):         {pctl(dur)}")
    print(f"   ti le khung bi mat tay: {pctl(drop)} %")

    print("\n3. NGON TAY BAT DAU DUOI TRUOC KHI VAO O?  (duong = ngon bat dau duoi SOM hon luc vao o)")
    print(f"   nguong 'roi dang nam' (p95 cua bua): keo {start['keo']:.2f}, bao {start['bao']:.2f}")
    print(f"   nguong 'dinh hinh ro' (p10 cuoi van): keo {formed['keo']:.2f}, bao {formed['bao']:.2f}")
    for g, key in (("keo", "ek"), ("bao", "eb")):
        ps = [p for p in prepared if p["gesture"] == g]
        onset, form, ext_entry = [], [], []
        before = 0
        for p in ps:
            e = p["entry"]
            o = first_true(p[key] >= start[g])
            f = first_true(p[key] >= formed[g])
            ext_entry.append(p[key][e])
            if o is not None:
                lead = (p["tt"][e] - p["tt"][o]) * 1000
                onset.append(lead)
                before += lead > 0
            if f is not None:
                form.append((p["tt"][e] - p["tt"][f]) * 1000)
        print(f"   {g.upper()} ({len(ps)} van):")
        print(f"     do duoi luc vao o:            {pctl(ext_entry, (10, 50, 90))}")
        print(f"     bat dau duoi truoc vao o:     {100 * before / max(len(ps), 1):.0f}% so van")
        print(f"     som hon luc vao o (ms):       {pctl(onset)}")
        print(f"     da dinh hinh ro luc vao o:    {100 * np.mean([x > 0 for x in form]) if form else 0:.0f}% so van")
    bua = [p for p in prepared if p["gesture"] == "bua"]
    fk = np.mean([bool((p["ek"][: p["entry"] + 1] >= start["keo"]).any()) for p in bua])
    fb = np.mean([bool((p["eb"][: p["entry"] + 1] >= start["bao"]).any()) for p in bua])
    print(f"   BUA ({len(bua)} van): co khung vuot nguong 'roi dang nam' truoc khi vao o: "
          f"theo do keo {100 * fk:.0f}%, theo do bao {100 * fb:.0f}%")

    for p in prepared:
        e = p["entry"]
        rows_out.append(dict(gesture=p["gesture"], hand=p["hand"], n_evidence=e + 1,
                             window_ms=round((p["tt"][e] - p["tt"][0]) * 1000, 1),
                             ext_keo_at_entry=round(float(p["ek"][e]), 3),
                             ext_bao_at_entry=round(float(p["eb"][e]), 3)))


# ------------------------------------------------------------------ 4. quyết định: từng khung / cộng dồn
def decide(probs, entry, k, lam):
    """probs: xác suất từng khung (n,3). Dùng các khung 0..entry+k, cộng log-xác suất có phai dần."""
    L = np.log(np.clip(probs[: entry + 1 + k], 1e-3, 1.0))
    s = np.zeros(3)
    for row in L:
        s = lam * s + row
    return int(np.argmax(s))


def fire(probs, entry, wait, thr, min_frames):
    """Chốt theo độ tự tin. Trả về (chỉ số khung chốt, cử chỉ).
    Từ khung min_frames-1 trở đi, chốt ngay khi xác suất cao nhất >= thr;
    nếu đến hạn (entry + wait) mà chưa đủ thì chốt theo khung hạn."""
    last = min(entry + wait, len(probs) - 1)
    start = min(min_frames - 1, last)
    for f in range(start, last + 1):
        if probs[f].max() >= thr:
            return f, int(probs[f].argmax())
    return last, int(probs[last].argmax())


def print_policy(pol, reps):
    print("\n5. CHOT THEO DO TU TIN (moi khung tu quyet, khong cong don)")
    print("   Chot ngay khi xac suat cao nhat >= nguong; den han ma chua du thi chot theo khung han.")
    print("   tre = thoi gian tu luc tay vao o den luc chot (ms); am = chot TRUOC khi vao o.")
    for W in POLICY_WAITS:
        print(f"\n   Han chot: khung vao o + {W} khung" + ("  (~30 ms)" if W == 0 else f"  (+{W} khung)"))
        print("   nguong   chinh xac    keo    bua    bao   chot truoc o   tre p50    p90")
        for thr in THRESHOLDS:
            r = pol[(W, thr)]
            ok, g = np.array(r["ok"]), np.array(r["gest"])
            per = [100 * ok[g == gi].mean() if (g == gi).any() else float("nan") for gi in range(3)]
            lat = np.array(r["lat"])
            name = "khong som" if thr > 1 else f"{thr:.2f}"
            print(f"   {name:<9}{100 * ok.mean():>7.1f}%   {per[0]:>5.1f}  {per[1]:>5.1f}  {per[2]:>5.1f}"
                  f"   {100 * np.mean(r['early']):>9.0f}%   {np.percentile(lat, 50):>7.0f}  {np.percentile(lat, 90):>5.0f}")
    print("   Dong 'khong som' la cach hien tai (chot o han, khong chot truoc). Neu cac dong con lai khong")
    print("   hon no qua ~5 diem % o cung han chot thi chot som khong co gia tri. Sai so ngau nhien ~4-5 diem %.")


def decision_experiment(prepared, reps, median_dt_ms, min_frames=3):
    y = np.array([G_IDX[p["gesture"]] for p in prepared])
    n_splits = min(5, int(min(Counter(y).values())))
    if n_splits < 2:
        print("   Khong du van moi loai de kiem tra cheo.")
        return
    hits = {(lam, k): [] for lam in LAMBDAS for k in EXTRA_FRAMES}
    by_g = {key: Counter() for key in hits}
    tot_g = Counter(y.tolist())
    by_hand = {}
    pol = {(W, thr): dict(ok=[], gest=[], lat=[], early=[])
           for W in POLICY_WAITS for thr in THRESHOLDS}

    for rep in range(reps):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rep)
        for tr, te in skf.split(np.zeros(len(y)), y):
            X = np.concatenate([prepared[i]["eng"] for i in tr])
            lab = np.concatenate([np.full(len(prepared[i]["eng"]), y[i]) for i in tr])
            rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=10,
                                        random_state=rep, n_jobs=-1).fit(X, lab)
            for i in te:
                p = prepared[i]
                cut = min(len(p["eng"]), p["entry"] + 1 + max(EXTRA_FRAMES))
                probs = rf.predict_proba(p["eng"][:cut])
                for lam in LAMBDAS:
                    for k in EXTRA_FRAMES:
                        ok = decide(probs, p["entry"], k, lam) == y[i]
                        hits[(lam, k)].append(ok)
                        if ok:
                            by_g[(lam, k)][y[i]] += 1
                        if (lam, k) == (0.7, 0):
                            by_hand.setdefault(p["hand"], []).append(ok)
                for W in POLICY_WAITS:
                    for thr in THRESHOLDS:
                        f, pred = fire(probs, p["entry"], W, thr, min_frames)
                        rec = pol[(W, thr)]
                        rec["ok"].append(pred == y[i])
                        rec["gest"].append(int(y[i]))
                        rec["lat"].append((p["tt"][f] - p["tt"][p["entry"]]) * 1000)
                        rec["early"].append(f < p["entry"])

    print(f"\n   Do chinh xac quyet dinh (ti le van doan dung, {reps} lan kiem tra cheo x {n_splits} phan)")
    print(f"   hang = he so phai lam (0 = chi nhin khung hien tai, 1 = cong deu); cot = cho them k khung trong o")
    print("   lam \\ k   " + "".join(f"{k:>10}" for k in EXTRA_FRAMES))
    for lam in LAMBDAS:
        print(f"   {lam:>6.1f}    " + "".join(f"{100 * np.mean(hits[(lam, k)]):>9.1f}%" for k in EXTRA_FRAMES))
    print("   (cho them k khung = tre them khoang " +
          ", ".join(f"k={k}: {k * median_dt_ms:.0f} ms" for k in EXTRA_FRAMES if k) + ")")

    print("\n   Chia theo cu chi (k = 0):   " + "  ".join(f"lam={lam}" for lam in (0.0, 0.7, 1.0)))
    for g in GESTURES:
        gi = G_IDX[g]
        cells = [100 * by_g[(lam, 0)][gi] / (tot_g[gi] * reps) for lam in (0.0, 0.7, 1.0)]
        print(f"     {g:<4}: " + "          ".join(f"{c:5.1f}%" for c in cells))
    if by_hand:
        print("   Chia theo tay (lam = 0.7, k = 0): " +
              ", ".join(f"tay {h}: {100 * np.mean(v):.0f}%" for h, v in sorted(by_hand.items())))
    print(f"   (so van moi lop: " + ", ".join(f"{g}={tot_g[G_IDX[g]]}" for g in GESTURES) +
          ")\n   Luu y: voi ~120 van, do lech chuan ngau nhien cua moi con so khoang 4-5 diem %."
          "\n   Chenh lech giua hai cach nho hon ~5 diem % chua dang tin.")
    print_policy(pol, reps)


# ------------------------------------------------------------------ chạy
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="approach_data")
    ap.add_argument("--person", default=None)
    ap.add_argument("--square", type=float, nargs="+", default=[0.20, 0.25, 0.30],
                    help="cac co o vuong thu nghiem (ti le canh ngan khung hinh)")
    ap.add_argument("--reps", type=int, default=3, help="so lan lap kiem tra cheo")
    ap.add_argument("--min-frames", type=int, default=3,
                    help="so khung co tay toi thieu truoc khi duoc chot som (khung dau thuong bi cat o mep)")
    args = ap.parse_args()

    rounds = load_rounds(args.outdir, args.person)
    if len(rounds) < 15:
        print(f"Chi co {len(rounds)} van hop le, qua it de phan tich.")
        return
    data_quality(rounds, args.outdir)

    dts = np.concatenate([np.diff(r["t"]) for r in rounds if len(r["t"]) > 2])
    median_dt_ms = float(np.median(dts)) * 1000
    summary = []

    for frac in args.square:
        prepared = [p for p in (prepare(r, frac) for r in rounds) if p is not None]
        skipped = len(rounds) - len(prepared)
        print("\n" + "=" * 72)
        print(f"O VUONG co canh = {frac:.2f} x canh ngan khung hinh   "
              f"({len(prepared)} van dung duoc, bo qua {skipped} van khong vao toi o nay)")
        print("=" * 72)
        if skipped:
            print(f"Luu y: {skipped} van tay khong cham toi o nay bi bo qua, nen do chinh xac chi tinh tren cac van con lai.")
        if len(prepared) < 15 or len({p["gesture"] for p in prepared}) < 3:
            print("Khong du du lieu cho co o nay.")
            continue
        start, formed = thresholds(prepared)
        rows = []
        window_and_onset(prepared, start, formed, rows)
        for row in rows:
            row["square_frac"] = frac
        summary.extend(rows)
        print("\n4. QUYET DINH LUC TAY VAO O")
        decision_experiment(prepared, args.reps, median_dt_ms, args.min_frames)

    if summary:
        with open("ket_qua_approach.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        print("\nDa luu ket_qua_approach.csv (so lieu tung van)")


if __name__ == "__main__":
    main()