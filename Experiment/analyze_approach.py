"""
analyze_approach.py
--------------------
Phân tích dữ liệu thu bằng collect_approach.py (mặc định đọc thư mục approach_data_v2/).

Trả lời các câu hỏi:
  1. Dữ liệu thu có ổn không? (số ván, ván lỗi, từng phiên thu, FPS thật, thời gian MediaPipe mỗi khung,
     nhãn tay của MediaPipe có khớp lệnh không, từng phiên có dùng cùng một tay cho cả hai lệnh không)
 1b. Cử chỉ ở cuối ván có khớp lệnh hiển thị không (phát hiện ván người chơi làm sai cử chỉ).
  2. Từ lúc thấy tay đến lúc tay vào ô vuông có bao nhiêu khung hình (cửa sổ bằng chứng)?
  3. Ngón tay có bắt đầu duỗi TRƯỚC khi tay vào ô không, và sớm bao nhiêu?
  4. Quyết định ở lúc vào ô chính xác bao nhiêu nếu chỉ nhìn khung hiện tại, cộng dồn đều, cộng dồn
     phai dần, và nếu chờ thêm k khung nữa trong ô.
  5. Chốt theo ĐỘ TỰ TIN (không cộng dồn): chốt ngay khi xác suất cao nhất vượt ngưỡng (có thể trước khi
     tay vào ô); đến hạn mà chưa đủ thì chốt theo khung hạn. Báo độ trễ so với lúc tay vào ô.
  6. Thử cải thiện model ở hạn chốt = khung đầu tiên trong ô (~30 ms):
       - huấn luyện chỉ trên các khung quanh lúc vào ô,
       - thêm đặc trưng "ngón đang duỗi nhanh hay chậm" lấy từ các khung TRƯỚC (không thêm độ trễ),
       - cả hai;
     đồng thời báo tỉ lệ máy THẮNG/HÒA/THUA khi máy chọn (a) khắc chế cử chỉ đoán nhiều nhất hoặc
     (b) nước có kỳ vọng thắng cao nhất theo xác suất, và ma trận nhầm lẫn.

  7. Đo theo mốc "tay bắt đầu đổi dáng" (không phải lúc tay vào ô): mốc = khung đầu tiên ngón vượt mức
     nắm tay. Với mỗi số khung sau mốc (và vài khung trước mốc), đo độ chính xác khi chỉ nhìn khung hiện
     tại và khi cộng dồn từ đầu ván (phai dần / cộng đều). Búa không có thay đổi thật nên chỉ có "mốc giả".

Cách chạy:
    python analyze_approach.py                          (đọc approach_data_v2)
    python analyze_approach.py --outdir approach_data_v2 --square 0.2 0.25 0.3
    python analyze_approach.py --cv round               (ép kiểm tra chéo theo ván)
    python analyze_approach.py --drop-suspect           (loại ván nghi sai cử chỉ ở mục 1b)
    python analyze_approach.py --no-variants            (bỏ mục 6 cho chạy nhanh)

Kiểm tra chéo (--cv auto): nếu có từ 2 người trở lên thì huấn luyện trên người này, kiểm tra trên người
khác; nếu chỉ một người nhưng nhiều buổi thì theo buổi; chỉ khi một người một buổi mới phải chia theo
ván (dễ lạc quan, vì các ván cùng buổi giống nhau hơn so với ván của buổi/người khác).
"""

import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict

GESTURES = ["keo", "bua", "bao"]          # chỉ số 0, 1, 2
G_IDX = {g: i for i, g in enumerate(GESTURES)}
PALM_IDS = [0, 5, 9, 13, 17]
TIPS5, TIPS4, MCPS4 = [4, 8, 12, 16, 20], [8, 12, 16, 20], [5, 9, 13, 17]

LAMBDAS = [0.0, 0.5, 0.7, 0.9, 1.0]
EXTRA_FRAMES = [0, 3, 6]
POLICY_WAITS = [0, 3]                         # hạn chốt: khung vào ô + W khung (W=0 ~ 30 ms)
THRESHOLDS = [0.60, 0.70, 0.80, 0.90, 0.95, 1.01]   # 1.01 = không bao giờ chốt sớm (chốt ở hạn)
BUDGET_MS = 30.0                              # ngân sách thời gian mỗi khung
NEAR_ENTRY = 3                                # huấn luyện quanh lúc vào ô: +-3 khung
DYN_LAG = 3                                   # đặc trưng động: so với 3 khung trước

OFFSETS = [-3, 0, 1, 2, 3, 6, 9]              # mục 7: số khung so với mốc đổi dáng (âm = trước mốc)
ONSET_LAMBDAS = [0.0, 0.7, 1.0]               # mục 7: 0 = chỉ khung hiện tại, 1 = cộng đều

# Luật chơi: BEATS[m] = cử chỉ mà nước m thắng (keo thắng bao, bua thắng keo, bao thắng bua)
BEATS = {0: 2, 1: 0, 2: 1}
PAY = np.array([[0 if m == g else (1 if BEATS[m] == g else -1) for g in range(3)] for m in range(3)], float)
COUNTER = {g: m for m in range(3) for g in range(3) if BEATS[m] == g}   # nước khắc chế cử chỉ g
CV_TEXT = {"round": "theo van", "person": "theo NGUOI", "session": "theo BUOI"}


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
        proc = d["proc_ms"] if "proc_ms" in d.files else None      # bản thu cũ không có
        rounds.append(dict(meta=meta, t=d["t"], lm=d["landmarks"], det=d["detected"],
                           mp_label=d["mp_label"], proc=proc, path=path))
    return rounds


def invalid_reasons(outdir, person=None):
    p = os.path.join(outdir, "rounds_index.csv")
    if not os.path.exists(p):
        return Counter()
    with open(p, newline="", encoding="utf-8") as f:
        return Counter(r["reason"] for r in csv.DictReader(f)
                       if r["valid"] != "True" and (not person or r["person"] == person))


def session_key(r):
    return f'{r["meta"].get("person", "?")}/{r["meta"].get("session", "?")}'


def majority_label(r):
    labs = [l for l, d in zip(r["mp_label"], r["det"]) if d and l]
    return Counter(labs).most_common(1)[0][0] if labs else ""


def noise_pts(n, p=0.9):
    """Nửa độ rộng khoảng tin cậy ~95% (điểm %) của một tỉ lệ ~90% đo trên n ván."""
    return 200.0 * np.sqrt(p * (1 - p) / max(n, 1))


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


def feats(p, dyn):
    """Đặc trưng từng khung. dyn=True: thêm 'thay đổi so với DYN_LAG khung trước' (chỉ dùng khung đã qua)."""
    e = p["eng"]
    if not dyn:
        return e
    prev = np.maximum(np.arange(len(e)) - DYN_LAG, 0)
    return np.concatenate([e, e - e[prev]], axis=1)


def proba3(rf, X):
    """predict_proba luôn trả về đủ 3 cột (kéo, búa, bao), kể cả khi tập huấn luyện thiếu một lớp."""
    out = np.zeros((len(X), 3))
    out[:, rf.classes_.astype(int)] = rf.predict_proba(X)
    return out


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
                hand=r["meta"]["hand_cue_player"], idx_full=det, path=r["path"],
                person=str(r["meta"].get("person", "?")), skey=session_key(r))


# ------------------------------------------------------------------ chia kiểm tra chéo
def resolve_cv(rounds, mode):
    if mode != "auto":
        return mode
    if len({r["meta"].get("person") for r in rounds}) >= 2:
        return "person"
    if len({session_key(r) for r in rounds}) >= 2:
        return "session"
    return "round"


def make_splits(prepared, y, cv_mode, rep):
    n = len(y)
    if cv_mode in ("person", "session"):
        key = "person" if cv_mode == "person" else "skey"
        _, g = np.unique([p[key] for p in prepared], return_inverse=True)
        k = min(5, len(set(g)))
        if k >= 2:
            yield from GroupKFold(n_splits=k).split(np.zeros(n), y, g)
        return
    k = min(5, int(min(Counter(y).values())))
    if k >= 2:
        yield from StratifiedKFold(n_splits=k, shuffle=True, random_state=rep).split(np.zeros(n), y)


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
def data_quality(rounds, outdir, person=None):
    print("=" * 72)
    print("1. CHAT LUONG DU LIEU")
    print("=" * 72)
    cnt = Counter((r["meta"]["gesture"], r["meta"]["hand_cue_player"]) for r in rounds)
    print(f"So van hop le: {len(rounds)}")
    for (g, h), n in sorted(cnt.items()):
        print(f"  cu chi {g:<4} tay {h:<4}: {n}")
    bad = invalid_reasons(outdir, person)
    if bad:
        print("Van loi da bi loai (tu rounds_index.csv):")
        for reason, n in bad.most_common():
            print(f"  {reason}: {n}")

    # ---- từng phiên thu
    sess = defaultdict(list)
    for r in rounds:
        sess[session_key(r)].append(r)
    print("Cac phien thu:")
    for key, rs in sorted(sess.items()):
        m = rs[0]["meta"]
        dts = np.concatenate([np.diff(r["t"]) for r in rs if len(r["t"]) > 2])
        info = (f", o dich {m['target_frac']}, o kiem tra {m.get('gate_frac')}, "
                f"tay thuan {m.get('dominant_hand') or '?'}") if "target_frac" in m else ", dinh dang cu (khong co o dich)"
        print(f"  {key}: {len(rs)} van, FPS {1.0 / np.median(dts):.1f}{info}")
    targets = {r["meta"].get("target_frac") for r in rounds}
    if len(targets) > 1:
        print("  CANH BAO: cac phien co o dich khac nhau (hoac lan dinh dang cu): "
              f"{sorted(str(t) for t in targets)}. Nen phan tich rieng tung nhom.")
    fps_set = {round(float(1.0 / np.median(np.concatenate([np.diff(r['t']) for r in rs if len(r['t']) > 2]))), -1)
               for rs in sess.values()}
    if len(fps_set) > 1:
        print(f"  CANH BAO: FPS thuc te khac nhau giua cac phien ({sorted(fps_set)}); tinh theo khung nen khong cung don vi.")

    dts = np.concatenate([np.diff(r["t"]) for r in rounds if len(r["t"]) > 2])
    print(f"FPS thuc te (trung vi): {1.0 / np.median(dts):.1f}")
    det_rate = np.mean([r["det"].mean() for r in rounds])
    print(f"Ti le khung thay tay trong van (trung binh): {100 * det_rate:.0f}%")

    # ---- thời gian xử lý MediaPipe
    procs = [r["proc"] for r in rounds if r["proc"] is not None]
    if procs:
        pr = np.concatenate(procs)
        pr = pr[np.isfinite(pr)]
        print(f"Thoi gian MediaPipe moi khung (ms): trung vi {np.median(pr):.1f}, p95 {np.percentile(pr, 95):.1f}, "
              f"p99 {np.percentile(pr, 99):.1f}, lon nhat {pr.max():.1f}")
        print(f"  {100 * np.mean(pr > BUDGET_MS):.1f}% so khung vuot {BUDGET_MS:.0f} ms "
              "(chi tinh MediaPipe; chua gom trich dac trung, RandomForest, do tre camera)")
    else:
        print("(Du lieu khong co proc_ms: thu bang ban collect cu)")

    # ---- nhãn Left/Right của MediaPipe so với tay theo lệnh
    table = Counter()
    per_sess = defaultdict(lambda: defaultdict(Counter))
    for r in rounds:
        lab = majority_label(r)
        if lab:
            table[(r["meta"]["hand_cue_player"], lab)] += 1
            per_sess[session_key(r)][r["meta"]["hand_cue_player"]][lab] += 1
    print("Nhan Left/Right cua MediaPipe so voi tay theo LENH:")
    for (hand, lab), n in sorted(table.items()):
        print(f"  lenh tay {hand}: MediaPipe bao '{lab}' trong {n} van")
    for key, tab in sorted(per_sess.items()):
        if len(tab) == 2:
            hands = sorted(tab)
            fl = [tab[h]["Left"] / max(sum(tab[h].values()), 1) for h in hands]
            if abs(fl[0] - fl[1]) < 0.4:
                print(f"  CANH BAO phien {key}: nhan MediaPipe cua hai tay gan nhu giong nhau "
                      f"(ti le 'Left': tay {hands[0]} {100 * fl[0]:.0f}%, tay {hands[1]} {100 * fl[1]:.0f}%). "
                      "Co the nguoi choi dung cung mot tay cho ca hai lenh -> nen loai phien nay.")


# ------------------------------------------------------------------ 1b. cử chỉ cuối ván có khớp lệnh không
def final_features(r):
    det = np.where(r["det"])[0]
    if len(det) < 5:
        return None
    Pn = normalize(r["lm"][det])
    Pn = Pn[np.isfinite(Pn).all(axis=(1, 2))]
    if len(Pn) < 5:
        return None
    return eng_features(Pn[-5:]).mean(axis=0)


def label_check(rounds):
    print("\n" + "=" * 72)
    print("1b. CU CHI CUOI VAN CO KHOP LENH KHONG (phat hien van lam sai cu chi)")
    print("=" * 72)
    X, idx = [], []
    for i, r in enumerate(rounds):
        f = final_features(r)
        if f is not None:
            X.append(f)
            idx.append(i)
    y = np.array([G_IDX[rounds[i]["meta"]["gesture"]] for i in idx])
    if len(idx) < 15 or min(Counter(y).values()) < 2:
        print("Qua it van de kiem tra.")
        return set()
    k = min(5, int(min(Counter(y).values())))
    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=0, n_jobs=-1)
    P = cross_val_predict(rf, np.array(X), y, cv=StratifiedKFold(k, shuffle=True, random_state=0),
                          method="predict_proba")
    pred, conf = P.argmax(1), P.max(1)
    sus = (pred != y) & (conf >= 0.8)
    print(f"Do chinh xac nhan dien cu chi o cuoi van (kiem tra cheo): {100 * np.mean(pred == y):.1f}%  "
          "(nen gan 100%; thap nghia la nhan sai hoac cu chi khong chuan)")
    print(f"Van nghi lam sai cu chi (may doan khac lenh voi do tin cay >= 80%): {int(sus.sum())}/{len(idx)}")
    flagged = set()
    for j in np.where(sus)[0]:
        r = rounds[idx[j]]
        flagged.add(r["path"])
        print(f"  {os.path.basename(r['path'])}: lenh {GESTURES[y[j]]}, may thay {GESTURES[pred[j]]} ({100 * conf[j]:.0f}%)")
    if flagged:
        print("  Xem lai cac van nay; them --drop-suspect de loai chung khoi phan tich.")
    return flagged


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
        rows_out.append(dict(person=p["person"], session=p["skey"], gesture=p["gesture"], hand=p["hand"],
                             n_evidence=e + 1, window_ms=round((p["tt"][e] - p["tt"][0]) * 1000, 1),
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


def print_policy(pol, n):
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
    print(f"   hon no qua ~{noise_pts(n):.0f} diem % o cung han chot thi chot som khong co gia tri.")


def decision_experiment(prepared, reps, median_dt_ms, min_frames, cv_mode):
    y = np.array([G_IDX[p["gesture"]] for p in prepared])
    n_folds = len(list(make_splits(prepared, y, cv_mode, 0)))
    if n_folds < 2:
        print("   Khong du van/nhom de kiem tra cheo.")
        return
    reps_eff = reps if cv_mode == "round" else 1
    hits = {(lam, k): [] for lam in LAMBDAS for k in EXTRA_FRAMES}
    by_g = {key: Counter() for key in hits}
    tot_g = Counter(y.tolist())
    by_hand = {}
    pol = {(W, thr): dict(ok=[], gest=[], lat=[], early=[]) for W in POLICY_WAITS for thr in THRESHOLDS}

    for rep in range(reps_eff):
        for tr, te in make_splits(prepared, y, cv_mode, rep):
            X = np.concatenate([prepared[i]["eng"] for i in tr])
            lab = np.concatenate([np.full(len(prepared[i]["eng"]), y[i]) for i in tr])
            rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=10,
                                        random_state=rep, n_jobs=-1).fit(X, lab)
            for i in te:
                p = prepared[i]
                cut = min(len(p["eng"]), p["entry"] + 1 + max(EXTRA_FRAMES))
                probs = proba3(rf, p["eng"][:cut])
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

    print(f"\n   Do chinh xac quyet dinh (ti le van doan dung, kiem tra cheo {CV_TEXT[cv_mode]}: "
          f"{reps_eff} lan x {n_folds} phan)")
    print("   hang = he so phai lam (0 = chi nhin khung hien tai, 1 = cong deu); cot = cho them k khung trong o")
    print("   lam \\ k   " + "".join(f"{k:>10}" for k in EXTRA_FRAMES))
    for lam in LAMBDAS:
        print(f"   {lam:>6.1f}    " + "".join(f"{100 * np.mean(hits[(lam, k)]):>9.1f}%" for k in EXTRA_FRAMES))
    print("   (cho them k khung = tre them khoang " +
          ", ".join(f"k={k}: {k * median_dt_ms:.0f} ms" for k in EXTRA_FRAMES if k) + ")")

    print("\n   Chia theo cu chi (k = 0):   " + "  ".join(f"lam={lam}" for lam in (0.0, 0.7, 1.0)))
    for g in GESTURES:
        gi = G_IDX[g]
        cells = [100 * by_g[(lam, 0)][gi] / (tot_g[gi] * reps_eff) for lam in (0.0, 0.7, 1.0)]
        print(f"     {g:<4}: " + "          ".join(f"{c:5.1f}%" for c in cells))
    if by_hand:
        print("   Chia theo tay (lam = 0.7, k = 0): " +
              ", ".join(f"tay {h}: {100 * np.mean(v):.0f}%" for h, v in sorted(by_hand.items())))
    print("   (so van moi lop: " + ", ".join(f"{g}={tot_g[G_IDX[g]]}" for g in GESTURES) + ")")
    print(f"   Luu y: voi {len(y)} van, sai so ngau nhien (do tin cay ~95%) cua moi con so khoang "
          f"+-{noise_pts(len(y)):.1f} diem %; chenh lech nho hon muc nay chua dang tin.")
    print_policy(pol, len(y))


# ------------------------------------------------------------------ 6. thử cải thiện model ở hạn 30 ms
VARIANTS = [
    ("hien tai (moi khung, dac trung tinh)", False, False),
    ("huan luyen quanh luc vao o", False, True),
    ("them dac trung dong", True, False),
    ("ca hai", True, True),
]


def summarize_variant(res):
    P, t = np.array(res["probs"]), np.array(res["true"])
    pred = P.argmax(1)
    acc = 100 * np.mean(pred == t)
    per = [100 * np.mean(pred[t == g] == g) if (t == g).any() else float("nan") for g in range(3)]

    def wdl(m):
        out = PAY[m, t]
        return tuple(100 * np.mean(out == v) for v in (1, 0, -1))

    m_arg = np.array([COUNTER[int(a)] for a in pred])
    m_ev = (P @ PAY.T).argmax(1)
    return acc, per, wdl(m_arg), wdl(m_ev), pred, t


def variant_experiment(prepared, reps, cv_mode):
    y = np.array([G_IDX[p["gesture"]] for p in prepared])
    n_folds = len(list(make_splits(prepared, y, cv_mode, 0)))
    if n_folds < 2:
        return
    reps_eff = reps if cv_mode == "round" else 1
    res = {name: dict(probs=[], true=[]) for name, _, _ in VARIANTS}
    for rep in range(reps_eff):
        for tr, te in make_splits(prepared, y, cv_mode, rep):
            for name, dyn, near in VARIANTS:
                Xs, ys = [], []
                for i in tr:
                    p = prepared[i]
                    F = feats(p, dyn)
                    if near:
                        lo, hi = max(0, p["entry"] - NEAR_ENTRY), min(len(F), p["entry"] + NEAR_ENTRY + 1)
                        F = F[lo:hi]
                    else:
                        F = F
                    Xs.append(F)
                    ys.append(np.full(len(F), y[i]))
                rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=rep,
                                            n_jobs=-1).fit(np.concatenate(Xs), np.concatenate(ys))
                for i in te:
                    p = prepared[i]
                    F = feats(p, dyn)
                    res[name]["probs"].append(proba3(rf, F[p["entry"]][None, :])[0])
                    res[name]["true"].append(int(y[i]))

    print("\n6. THU CAI THIEN MODEL (quyet dinh o khung dau tien trong o, han ~30 ms, khong cong don)")
    print("   may thang/hoa/thua (%): (a) doan roi khac che    (b) chon theo ky vong thang tu xac suat")
    print("   (may ngau nhien: 33/33/33; doan dung hoan toan: 100/0/0)")
    print(f"   {'bien the':<38}{'chinh xac':>10}{'keo':>7}{'bua':>7}{'bao':>7}   {'(a) T/H/T':>16}   {'(b) T/H/T':>16}")
    base_conf = None
    for name, _, _ in VARIANTS:
        acc, per, wa, wb, pred, t = summarize_variant(res[name])
        print(f"   {name:<38}{acc:>9.1f}%{per[0]:>7.1f}{per[1]:>7.1f}{per[2]:>7.1f}   "
              f"{wa[0]:>5.1f}/{wa[1]:>4.1f}/{wa[2]:>4.1f}   {wb[0]:>5.1f}/{wb[1]:>4.1f}/{wb[2]:>4.1f}")
        if base_conf is None:
            base_conf = (pred, t)
    pred, t = base_conf
    print("\n   Ma tran nham lan cua bien the hien tai (hang = cu chi that, cot = may doan, % theo hang):")
    print("              " + "".join(f"{g:>8}" for g in GESTURES))
    for g in range(3):
        m = t == g
        row = [100 * np.mean(pred[m] == c) if m.any() else float("nan") for c in range(3)]
        print(f"     {GESTURES[g]:<8} " + "".join(f"{v:>7.1f}%" for v in row))
    print(f"   Luu y: sai so ngau nhien khoang +-{noise_pts(len(y)):.1f} diem % (do tin cay ~95%); "
          "chi tin bien the nao hon 'hien tai' ro ret hon muc nay.")


# ------------------------------------------------------------------ 7. đo theo mốc bắt đầu đổi dáng
def prepare_seq(r):
    """Chuỗi khung có tay của cả ván (không phụ thuộc cỡ ô)."""
    det = np.where(r["det"])[0]
    if len(det) == 0:
        return None
    Pn = normalize(r["lm"][det])
    ok = np.isfinite(Pn).all(axis=(1, 2))
    det, Pn = det[ok], Pn[ok]
    if len(det) < 3:
        return None
    return dict(tt=r["t"][det], eng=eng_features(Pn), ek=ext_keo(Pn), eb=ext_bao(Pn),
                gesture=r["meta"]["gesture"], hand=r["meta"]["hand_cue_player"], path=r["path"],
                person=str(r["meta"].get("person", "?")), skey=session_key(r))


def onset_experiment(seqs, reps, cv_mode):
    print("\n" + "=" * 72)
    print("7. DO TU LUC TAY BAT DAU DOI DANG (30 ms tinh tu moc nay, khong tinh tu luc vao o)")
    print("=" * 72)
    if len(seqs) < 15 or len({p["gesture"] for p in seqs}) < 3:
        print("Khong du du lieu.")
        return
    start, _ = thresholds(seqs)
    for p in seqs:
        p["onset"] = first_true((p["ek"] >= start["keo"]) | (p["eb"] >= start["bao"]))
    y = np.array([G_IDX[p["gesture"]] for p in seqs])
    tot, has = Counter(y.tolist()), Counter(int(g) for g, p in zip(y, seqs) if p["onset"] is not None)
    print(f"   Moc = khung dau tien co ngon vuot muc 'roi dang nam' (do duoi ngon tro+giua >= {start['keo']:.2f}"
          f" hoac 4 ngon >= {start['bao']:.2f}, tuc p95 cua bua). Bo phat hien nay khong biet cu chi la gi.")
    print("   Van co moc: " + ", ".join(f"{g} {has[G_IDX[g]]}/{tot[G_IDX[g]]}" for g in GESTURES) +
          "   (bua co moc = MOC GIA do nhieu; tinh tren ca van nen la can tren)")
    print(f"   Van bua KHONG co moc nao: {tot[1] - has[1]}/{tot[1]} -> o van nay may chi chot bua duoc theo han cho, "
          "khong do do tre tu luc doi dang.")
    if not any(p["onset"] is not None for p in seqs):
        return
    n_folds = len(list(make_splits(seqs, y, cv_mode, 0)))
    if n_folds < 2:
        print("   Khong du van/nhom de kiem tra cheo.")
        return
    reps_eff = reps if cv_mode == "round" else 1
    res = {(lam, j): {0: [], 1: [], 2: []} for lam in ONSET_LAMBDAS for j in OFFSETS}
    tms = {j: [] for j in OFFSETS}
    for rep in range(reps_eff):
        for tr, te in make_splits(seqs, y, cv_mode, rep):
            X = np.concatenate([seqs[i]["eng"] for i in tr])
            lab = np.concatenate([np.full(len(seqs[i]["eng"]), y[i]) for i in tr])
            rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=rep,
                                        n_jobs=-1).fit(X, lab)
            for i in te:
                p = seqs[i]
                if p["onset"] is None:
                    continue
                probs = proba3(rf, p["eng"])
                for j in OFFSETS:
                    f = p["onset"] + j
                    if f < 0 or f >= len(probs):
                        continue
                    tms[j].append((p["tt"][f] - p["tt"][p["onset"]]) * 1000)
                    for lam in ONSET_LAMBDAS:
                        res[(lam, j)][int(y[i])].append(decide(probs, f, 0, lam) == y[i])

    print(f"\n   Do chinh xac tai j khung sau moc (j<0 la TRUOC moc); kiem tra cheo {CV_TEXT[cv_mode]}: "
          f"{reps_eff} lan x {n_folds} phan")
    print("   'gop' = ti le dung chung cua ca ba nhom (keo, bao, bua-moc-gia) = tinh huong that khi may thay moc.")
    names = {0.0: "chi nhin khung hien tai (lam = 0)", 0.7: "cong don phai dan (lam = 0.7)", 1.0: "cong don deu (lam = 1.0)"}
    reach = {}
    for lam in ONSET_LAMBDAS:
        print(f"\n   --- {names[lam]} ---")
        print("   khung    ms     n(keo/bao/bua)      keo     bao   bua(gia)     gop")
        reach[lam] = None
        for j in OFFSETS:
            r = res[(lam, j)]
            allv = r[0] + r[1] + r[2]
            if not allv:
                continue
            cell = [f"{100 * np.mean(r[g]):>6.1f}%" if r[g] else "     n/a" for g in (0, 2, 1)]
            gop = 100 * np.mean(allv)
            cnt = f"{len(r[0]) // reps_eff}/{len(r[2]) // reps_eff}/{len(r[1]) // reps_eff}"
            print(f"   {j:>+4d} {np.median(tms[j]):>7.0f}   {cnt:>14}   {cell[0]}  {cell[1]}  {cell[2]}  {gop:>6.1f}%")
            if j >= 0 and reach[lam] is None and gop >= 90:
                reach[lam] = (j, float(np.median(tms[j])))
    print("\n   So khung can sau moc de 'gop' dat >= 90%:")
    for lam in ONSET_LAMBDAS:
        txt = (f"{reach[lam][0]} khung (~{reach[lam][1]:.0f} ms)" if reach[lam]
               else f"chua dat trong {OFFSETS[-1]} khung")
        print(f"     {names[lam]:<38}: {txt}")
    n_ev = sum(1 for p in seqs if p["onset"] is not None)
    print(f"   Luu y: chi {n_ev} van co moc, sai so ngau nhien cua moi con so co the lon (+-{noise_pts(n_ev):.0f} diem %).")
    print("   Neu cong don khong tot hon 'chi nhin khung hien tai' o cac khung ngay sau moc thi cong don khong")
    print("   giup duoc muc 30 ms tinh tu luc doi dang; neu j<0 (truoc moc) da cao thi bang chung truoc moc co thong tin.")


# ------------------------------------------------------------------ chạy
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="approach_data_v2")
    ap.add_argument("--person", default=None)
    ap.add_argument("--square", type=float, nargs="+", default=None,
                    help="cac co o vuong thu nghiem (ti le canh ngan khung hinh); "
                         "mac dinh 0.20 0.25 0.30 va co o dich ghi trong du lieu")
    ap.add_argument("--reps", type=int, default=3, help="so lan lap kiem tra cheo (chi dung khi chia theo van)")
    ap.add_argument("--min-frames", type=int, default=3,
                    help="so khung co tay toi thieu truoc khi duoc chot som (khung dau thuong bi cat o mep)")
    ap.add_argument("--cv", default="auto", choices=["auto", "round", "person", "session"],
                    help="cach chia kiem tra cheo: auto = theo nguoi neu >=2 nguoi, theo buoi neu >=2 buoi, con lai theo van")
    ap.add_argument("--drop-suspect", action="store_true", help="loai cac van nghi sai cu chi (muc 1b)")
    ap.add_argument("--no-variants", action="store_true", help="bo muc 6 (thu cai thien model) cho chay nhanh")
    args = ap.parse_args()

    rounds = load_rounds(args.outdir, args.person)
    if len(rounds) < 15:
        print(f"Chi co {len(rounds)} van hop le trong '{args.outdir}', qua it de phan tich "
              "(can it nhat 15). Kiem tra --outdir co dung va nam trong thu muc dang chay khong.")
        return
    data_quality(rounds, args.outdir, args.person)
    suspect = label_check(rounds)
    if args.drop_suspect and suspect:
        rounds = [r for r in rounds if r["path"] not in suspect]
        print(f"Da loai {len(suspect)} van nghi sai cu chi; con {len(rounds)} van.")

    cv_mode = resolve_cv(rounds, args.cv)
    print("\nCach kiem tra cheo: " + CV_TEXT[cv_mode] +
          (" (CANH BAO: chi co mot nguoi va mot buoi -> ket qua de lac quan; nen thu them nguoi/buoi khac)"
           if cv_mode == "round" else ""))

    targets = sorted({round(float(r["meta"]["target_frac"]), 2) for r in rounds if "target_frac" in r["meta"]})
    squares = sorted(set(args.square)) if args.square else sorted(set([0.20, 0.25, 0.30] + targets))

    dts = np.concatenate([np.diff(r["t"]) for r in rounds if len(r["t"]) > 2])
    median_dt_ms = float(np.median(dts)) * 1000
    summary = []

    for frac in squares:
        prepared = [p for p in (prepare(r, frac) for r in rounds) if p is not None]
        skipped = len(rounds) - len(prepared)
        print("\n" + "=" * 72)
        tag = "   [o dich luc thu]" if round(frac, 2) in targets else ""
        print(f"O VUONG co canh = {frac:.2f} x canh ngan khung hinh{tag}   "
              f"({len(prepared)} van dung duoc = {100 * len(prepared) / len(rounds):.0f}%, "
              f"bo qua {skipped} van khong vao toi o nay)")
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
        decision_experiment(prepared, args.reps, median_dt_ms, args.min_frames, cv_mode)
        if not args.no_variants:
            variant_experiment(prepared, args.reps, cv_mode)

    seqs = [q for q in (prepare_seq(r) for r in rounds) if q is not None]
    onset_experiment(seqs, args.reps, cv_mode)

    if summary:
        with open("ket_qua_approach.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        print("\nDa luu ket_qua_approach.csv (so lieu tung van)")


if __name__ == "__main__":
    main()
