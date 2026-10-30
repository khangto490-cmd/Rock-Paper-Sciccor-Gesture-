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

  8. Đánh giá theo HÌNH TAY THẤY ĐƯỢC ở khung vào ô (luật: cử chỉ của người chơi = hình tay ở đúng khung
     máy quyết định). Hình tay thấy được xác định bằng luật hình học trên 21 mốc, không dựa vào nhãn ý
     định: ngón duỗi khi đầu ngón xa cổ tay hơn khớp giữa. Báo: hình tay lúc vào ô khớp ý định bao nhiêu,
     và độ chính xác / tỉ lệ thắng-hòa-thua của máy so với hình thấy được (chỉ nhìn khung vs cộng dồn).
     Trước đó có bước kiểm tra luật hình học trên 5 khung cuối ván (lúc tay đã thành hình).

  9. Hình tay thành hình nhanh đến đâu sau khi tay vào ô: tỉ lệ ván đã thành hình sau T ms, thời gian
     thành hình (trung vị / p90), và với mỗi khung quyết định đặt ở j khung sau lúc vào ô: tỉ lệ hình thấy
     được khớp ý định, độ chính xác và tỉ lệ thắng-hòa-thua của máy. Dùng để chọn khung quyết định.

 10. Luật "chờ đến khi hình tay đọc rõ" (tối đa --wait-frames khung, mặc định 6 ~ 190 ms): từ khung vào ô,
     chốt ở khung đầu tiên mà luật hình học đọc cùng một hình rõ ở S khung liên tiếp (S = 1, 2, 3); đến hạn
     mà chưa rõ thì chốt theo model. So hai cách đọc (luật hình học / model) và các mốc chờ cố định.
     Báo độ trễ, tỉ lệ bị ép ở hạn, và thắng-hòa-thua so với ý định và so với hình thấy được.

 11. Chờ cố định --wait-frames khung (mặc định 6) rồi, NẾU độ tin cậy của model còn thấp (dưới ngưỡng), chờ
     thêm tối đa 1-3 khung và chốt ngay khi đủ tin cậy. So với chờ cố định +6, +7, +8, +9 khung: độ trễ,
     tỉ lệ ván phải chờ thêm, độ chính xác và thắng-hòa-thua (so với ý định và so với hình thấy được).

Cách chạy:
    python analyze_approach.py                          (đọc approach_data_v2)
    python analyze_approach.py --outdir approach_data_v2 --square 0.2 0.25 0.3
    python analyze_approach.py --cv round               (ép kiểm tra chéo theo ván)
    python analyze_approach.py --drop-suspect           (loại ván nghi sai cử chỉ ở mục 1b)
    python analyze_approach.py --no-variants            (bỏ mục 6 cho chạy nhanh)
    python analyze_approach.py --no-visible             (bỏ mục 8)
    python analyze_approach.py --ext-hi 1.2 --ext-lo 0.95   (chỉnh ngưỡng luật hình học của mục 8)

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

EXT_HI, EXT_LO = 1.15, 1.00                   # mục 8: tỉ lệ (đầu ngón / khớp giữa) tới cổ tay: >= HI duỗi, <= LO co
FINGERS = [(6, 8), (10, 12), (14, 16), (18, 20)]   # (khớp giữa, đầu ngón): trỏ, giữa, áp út, út
LAMS8 = [0.0, 0.5, 0.7, 1.0]
SHAPE_NAMES = ["keo", "bua", "bao", "khong ro"]
WAIT_FRAMES = 6                               # mục 10: hạn chờ mặc định (khung sau lúc vào ô)
PERSIST = [1, 2, 3]                           # mục 10: số khung liên tiếp phải cùng đọc ra một hình rõ
EXT_THRS = [0.7, 0.8, 0.9]                    # mục 11: ngưỡng tin cậy để khỏi phải chờ thêm
EXT_FRAMES = [1, 2, 3]                        # mục 11: số khung chờ thêm tối đa
FORM_RUN = 3                                  # mục 9: 'thành hình' = luật đọc đúng ý định >= 3 khung liên tiếp
FORM_OFFSETS = [0, 3, 6, 9, 12]               # mục 9: khung quyết định đặt j khung sau lúc vào ô
FORM_T = [0, 100, 200, 300, 400]              # mục 9: mốc thời gian (ms) kể từ lúc vào ô

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
                person=str(r["meta"].get("person", "?")), skey=session_key(r), Pn=Pn)


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


# ------------------------------------------------------------------ 8. hình tay thấy được (luật hình học)
def visible_shape(P):
    """P (n,21,3) đã chuẩn hóa -> (n,) 0 kéo, 1 búa, 2 bao, -1 không rõ.
    Ngón duỗi khi |đầu ngón - cổ tay| / |khớp giữa - cổ tay| >= EXT_HI, co khi <= EXT_LO, còn lại lưng chừng.
    Kéo = trỏ+giữa duỗi, áp út+út co; bao = 4 ngón duỗi; búa = 4 ngón co; còn lại = không rõ. Bỏ qua ngón cái."""
    st = np.zeros((len(P), 4), int)
    for k, (pip, tip) in enumerate(FINGERS):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.linalg.norm(P[:, tip], axis=1) / np.linalg.norm(P[:, pip], axis=1)
        st[:, k] = np.where(r >= EXT_HI, 1, np.where(r <= EXT_LO, -1, 0))
    shp = np.full(len(P), -1)
    shp[(st == -1).all(axis=1)] = 1
    shp[(st == 1).all(axis=1)] = 2
    shp[(st[:, 0] == 1) & (st[:, 1] == 1) & (st[:, 2] == -1) & (st[:, 3] == -1)] = 0
    return shp


def print_shape_matrix(intent, shape, header):
    print(header)
    print("                " + "".join(f"{n:>10}" for n in SHAPE_NAMES) + "      n")
    for g in range(3):
        m = intent == g
        if not m.any():
            continue
        cells = [100 * np.mean(shape[m] == c) for c in (0, 1, 2, -1)]
        print(f"     {GESTURES[g]:<9}  " + "".join(f"{v:>9.1f}%" for v in cells) + f"   {int(m.sum()):>4}")


def rule_check(rounds):
    print("\n" + "=" * 72)
    print("8a. KIEM TRA LUAT HINH HOC (ap dung o 5 khung cuoi van, luc tay da thanh hinh)")
    print("=" * 72)
    intent, shape = [], []
    for r in rounds:
        det = np.where(r["det"])[0]
        if len(det) < 5:
            continue
        Pn = normalize(r["lm"][det])
        Pn = Pn[np.isfinite(Pn).all(axis=(1, 2))]
        if len(Pn) < 5:
            continue
        intent.append(G_IDX[r["meta"]["gesture"]])
        shape.append(Counter(visible_shape(Pn[-5:]).tolist()).most_common(1)[0][0])
    if len(intent) < 15:
        print("Qua it van de kiem tra.")
        return None
    intent, shape = np.array(intent), np.array(shape)
    print(f"   Nguong: ngon duoi khi ti le >= {EXT_HI}, co khi <= {EXT_LO} (chinh bang --ext-hi / --ext-lo)")
    print_shape_matrix(intent, shape, "   Hinh luat doc duoc (hang = y dinh, cot = hinh luat doc, % theo hang):")
    match = 100 * np.mean(shape == intent)
    print(f"   Luat khop y dinh o cuoi van: {match:.1f}%  (tay luc nay da thanh hinh nen nen cao)")
    if match < 85:
        print("   CANH BAO: luat khop thap -> hinh 'thay duoc' o muc 8 KHONG dang tin; thu doi --ext-hi/--ext-lo.")
    return float(match)


def visible_experiment(prepared, reps, cv_mode):
    y = np.array([G_IDX[p["gesture"]] for p in prepared])
    vis = np.array([visible_shape(p["Pn"][p["entry"]: p["entry"] + 1])[0] for p in prepared])
    print("\n8. DANH GIA THEO HINH TAY THAY DUOC O KHUNG VAO O (cu chi nguoi choi = hinh tay o khung may quyet dinh)")
    print_shape_matrix(y, vis, "   Hinh tay thay duoc luc vao o (hang = y dinh, cot = hinh luat doc, % theo hang):")
    n_formed = int(np.sum(vis == y))
    print(f"   Da thanh hinh dung y dinh luc vao o: {n_formed}/{len(y)} van ({100 * n_formed / len(y):.0f}%); "
          f"chua thanh hinh / khong ro / khac: {len(y) - n_formed}")
    n_folds = len(list(make_splits(prepared, y, cv_mode, 0)))
    if n_folds < 2:
        return
    reps_eff = reps if cv_mode == "round" else 1
    preds = {lam: [] for lam in LAMS8}
    for rep in range(reps_eff):
        for tr, te in make_splits(prepared, y, cv_mode, rep):
            X = np.concatenate([prepared[i]["eng"] for i in tr])
            lab = np.concatenate([np.full(len(prepared[i]["eng"]), y[i]) for i in tr])
            rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=rep,
                                        n_jobs=-1).fit(X, lab)
            for i in te:
                p = prepared[i]
                probs = proba3(rf, p["eng"][: p["entry"] + 1])
                for lam in LAMS8:
                    preds[lam].append((i, decide(probs, p["entry"], 0, lam)))
    print(f"\n   Do chinh xac cua may o khung vao o (kiem tra cheo {CV_TEXT[cv_mode]}); T/H/T = may thang/hoa/thua (%)")
    print("   'vs y dinh' = so voi cu chi nguoi choi dinh lam; 'vs hinh thay duoc' chi tinh van luat doc ro.")
    print(f"   {'lam':>5} {'vs y dinh':>10} {'(da thanh hinh':>15} {'chua thanh hinh)':>17}   "
          f"{'vs hinh thay duoc':>18}   {'T/H/T vs hinh thay duoc':>24}")
    for lam in LAMS8:
        idx = np.array([i for i, _ in preds[lam]])
        pr = np.array([q for _, q in preds[lam]])
        yy, vv = y[idx], vis[idx]
        formed = vv == yy
        clear = vv >= 0
        a_int = 100 * np.mean(pr == yy)
        a_f = 100 * np.mean(pr[formed] == yy[formed]) if formed.any() else float("nan")
        a_u = 100 * np.mean(pr[~formed] == yy[~formed]) if (~formed).any() else float("nan")
        a_v = 100 * np.mean(pr[clear] == vv[clear]) if clear.any() else float("nan")
        out = PAY[np.array([COUNTER[int(q)] for q in pr[clear]]), vv[clear]] if clear.any() else np.array([])
        wdl = [100 * np.mean(out == v) if len(out) else float("nan") for v in (1, 0, -1)]
        print(f"   {lam:>5.1f} {a_int:>9.1f}% {a_f:>14.1f}% {a_u:>16.1f}%   {a_v:>17.1f}%   "
              f"{wdl[0]:>8.1f}/{wdl[1]:>4.1f}/{wdl[2]:>4.1f}  (n={int(clear.sum()) // reps_eff})")
    print(f"   Van chua thanh hinh luc vao o: {len(y) - n_formed}/{len(y)}; van luat doc ro: {int((vis >= 0).sum())}/{len(y)}.")
    print("   Neu 'vs hinh thay duoc' cao hon 'vs y dinh' nhieu thi phan lon 'loi' so voi y dinh la do tay chua")
    print("   thanh hinh (theo luat chot o khung may quyet dinh thi khong phai loi cua may). Luat hinh hoc chi la xap xi.")
    print(f"   Luu y: sai so ngau nhien khoang +-{noise_pts(len(y)):.1f} diem % (do tin cay ~95%).")


def formation_experiment(prepared, reps, cv_mode, rule_match=None):
    y = np.array([G_IDX[p["gesture"]] for p in prepared])
    shp_all = [visible_shape(p["Pn"]) for p in prepared]
    print("\n9. HINH TAY THANH HINH NHANH DEN DAU SAU KHI VAO O")
    print(f"   Thanh hinh = luat hinh hoc doc dung y dinh o >= {FORM_RUN} khung lien tiep (tinh tu khung vao o tro di).")
    if rule_match is not None:
        print(f"   Luat khop y dinh o cuoi van {rule_match:.1f}% -> khoang {100 - rule_match:.0f}% van se khong bao gio 'thanh hinh' "
              "chi vi sai so cua luat.")

    ftime = []                                  # ms ke tu luc vao o den luc thanh hinh (None = chua)
    for i, p in enumerate(prepared):
        sq = shp_all[i][p["entry"]:]
        ok = sq == y[i]
        t = None
        for k in range(0, len(sq) - FORM_RUN + 1):
            if ok[k:k + FORM_RUN].all():
                t = float((p["tt"][p["entry"] + k] - p["tt"][p["entry"]]) * 1000)
                break
        ftime.append(t)

    print("\n   Ti le van DA thanh hinh sau T ms ke tu luc tay vao o (T = 0: ngay khung vao o):")
    print("      T(ms)" + "".join(f"{g:>9}" for g in GESTURES) + f"{'chung':>9}")
    groups = {g: [ftime[i] for i in range(len(prepared)) if y[i] == G_IDX[g]] for g in GESTURES}
    groups["chung"] = list(ftime)
    for T in FORM_T:
        cells = [100 * np.mean([t is not None and t <= T for t in groups[g]]) for g in GESTURES + ["chung"]]
        print(f"      {T:>5}" + "".join(f"{c:>8.1f}%" for c in cells))
    print("   Chua thanh hinh trong thoi gian ghi: " +
          ", ".join(f"{g} {100 * np.mean([t is None for t in groups[g]]):.0f}%" for g in GESTURES))
    parts = []
    for g in GESTURES:
        v = [t for t in groups[g] if t is not None]
        parts.append(f"{g} {np.percentile(v, 50):.0f} / {np.percentile(v, 90):.0f}" if v else f"{g} n/a")
    print("   Thoi gian thanh hinh (ms, trung vi / p90, chi cac van da thanh hinh): " + ", ".join(parts))

    n_folds = len(list(make_splits(prepared, y, cv_mode, 0)))
    if n_folds < 2:
        return
    reps_eff = reps if cv_mode == "round" else 1
    maxj = max(FORM_OFFSETS)
    preds = {j: [] for j in FORM_OFFSETS}
    for rep in range(reps_eff):
        for tr, te in make_splits(prepared, y, cv_mode, rep):
            X = np.concatenate([prepared[i]["eng"] for i in tr])
            lab = np.concatenate([np.full(len(prepared[i]["eng"]), y[i]) for i in tr])
            rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=rep,
                                        n_jobs=-1).fit(X, lab)
            for i in te:
                p = prepared[i]
                probs = proba3(rf, p["eng"][: p["entry"] + 1 + maxj])
                for j in FORM_OFFSETS:
                    if p["entry"] + j < len(probs):
                        preds[j].append((i, int(probs[p["entry"] + j].argmax())))

    print(f"\n   Quyet dinh dat o j khung SAU luc vao o (chi nhin khung hien tai; kiem tra cheo {CV_TEXT[cv_mode]}):")
    print("   'vs y dinh' = so voi cu chi dinh lam; 'vs hinh thay' = so voi hinh luat doc o khung quyet dinh (chi van doc ro).")
    print(f"   {'khung':>5} {'ms':>5} {'n':>4}   {'thanh hinh':>10} {'khong ro':>9} {'hinh khac':>10}   "
          f"{'may vs y dinh':>14} {'vs hinh thay':>13}   {'T/H/T vs hinh thay':>20}")
    for j in FORM_OFFSETS:
        avail = [i for i, p in enumerate(prepared) if p["entry"] + j < len(p["eng"])]
        if len(avail) < 10:
            continue
        vis_f = np.array([shp_all[i][prepared[i]["entry"] + j] for i in avail])
        yy = y[avail]
        ms = np.median([(prepared[i]["tt"][prepared[i]["entry"] + j] - prepared[i]["tt"][prepared[i]["entry"]]) * 1000
                        for i in avail])
        formed, unclear = np.mean(vis_f == yy), np.mean(vis_f == -1)
        other = np.mean((vis_f != yy) & (vis_f >= 0))
        idx = np.array([i for i, _ in preds[j]])
        pr = np.array([q for _, q in preds[j]])
        vv = np.array([shp_all[i][prepared[i]["entry"] + j] for i in idx])
        acc_i = 100 * np.mean(pr == y[idx])
        clear = vv >= 0
        acc_v = 100 * np.mean(pr[clear] == vv[clear]) if clear.any() else float("nan")
        out = PAY[np.array([COUNTER[int(q)] for q in pr[clear]]), vv[clear]] if clear.any() else np.array([])
        wdl = [100 * np.mean(out == v) if len(out) else float("nan") for v in (1, 0, -1)]
        print(f"   {j:>+5d} {ms:>5.0f} {len(avail):>4}   {100 * formed:>9.1f}% {100 * unclear:>8.1f}% {100 * other:>9.1f}%   "
              f"{acc_i:>13.1f}% {acc_v:>12.1f}%   {wdl[0]:>6.1f}/{wdl[1]:>4.1f}/{wdl[2]:>4.1f}")
    print("   Doc: chon khung quyet dinh sao cho ti le 'thanh hinh' du cao (phan con lai la van chua co cu chi ro")
    print("   rang de tinh) ma van chap nhan duoc do tre. Luat hinh hoc chi la xap xi, nen mọi con so co sai so.".replace("mọi", "moi"))
    print(f"   Luu y: sai so ngau nhien khoang +-{noise_pts(len(y)):.1f} diem % (do tin cay ~95%); n giam o j lon vi van ket thuc som.")


def first_ready(shp, entry, last, S):
    """Khung đầu tiên f trong [entry, last] mà S khung liên tiếp kết thúc ở f cùng đọc ra MỘT hình rõ (>= 0)."""
    for f in range(entry, last + 1):
        if f - S + 1 < 0:
            continue
        w = shp[f - S + 1: f + 1]
        if w[0] >= 0 and (w == w[0]).all():
            return f
    return None


def wait_policy_experiment(prepared, reps, cv_mode, wait_frames, median_dt_ms):
    y = np.array([G_IDX[p["gesture"]] for p in prepared])
    shp_all = [visible_shape(p["Pn"]) for p in prepared]
    n_folds = len(list(make_splits(prepared, y, cv_mode, 0)))
    if n_folds < 2:
        return
    reps_eff = reps if cv_mode == "round" else 1
    W = wait_frames
    policies = [("luat", S) for S in PERSIST] + [("model", S) for S in PERSIST] + [("co dinh", j) for j in (0, 3, 6)]
    res = {pol: [] for pol in policies}
    for rep in range(reps_eff):
        for tr, te in make_splits(prepared, y, cv_mode, rep):
            X = np.concatenate([prepared[i]["eng"] for i in tr])
            lab = np.concatenate([np.full(len(prepared[i]["eng"]), y[i]) for i in tr])
            rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=rep,
                                        n_jobs=-1).fit(X, lab)
            for i in te:
                p, shp = prepared[i], shp_all[i]
                entry = p["entry"]
                last = min(entry + W, len(p["eng"]) - 1)
                mp = proba3(rf, p["eng"][: last + 1]).argmax(1)       # doan cua model o tung khung
                for kind, S in policies:
                    if kind == "co dinh":
                        f, forced = min(entry + S, last), False
                        pred = int(mp[f])
                    else:
                        f0 = first_ready(shp, entry, last, S)
                        forced = f0 is None
                        f = last if forced else f0
                        pred = int(shp[f]) if (kind == "luat" and shp[f] >= 0) else int(mp[f])
                    res[(kind, S)].append(dict(y=int(y[i]), vis=int(shp[f]), pred=pred, forced=forced,
                                               lat=float((p["tt"][f] - p["tt"][entry]) * 1000), now=(f == entry)))

    print("\n10. CHO DEN KHI HINH TAY DOC RO ROI MOI QUYET DINH (toi da "
          f"{W} khung ~ {W * median_dt_ms:.0f} ms sau luc vao o)")
    print("   Quy tac: tu khung vao o, chot o khung DAU TIEN ma luat hinh hoc doc cung mot hinh ro o S khung lien tiep;")
    print("   den han ma chua ro thi chot theo model o khung han. 'luat' = may chot theo hinh luat doc; 'model' = may chot theo")
    print("   model (khung chot van do luat chon). 'co dinh' = khong cho doi hinh ro, chot o +j khung (de so sanh).")
    names = {("luat", S): f"luat, S={S}" for S in PERSIST}
    names.update({("model", S): f"model, S={S}" for S in PERSIST})
    names.update({("co dinh", j): f"co dinh +{j} khung" for j in (0, 3, 6)})

    print(f"\n   A. Do tre va so voi cu chi DINH LAM (n={len(y)}); T/H/T = may thang/hoa/thua (%)")
    print(f"   {'cach':<18} {'chot ngay':>10} {'tre p50':>8} {'p90':>6} {'bi ep han':>10}   "
          f"{'dung y dinh':>12}   {'T/H/T vs y dinh':>17}   {'hinh ro nhung khac y dinh':>26}")
    for pol in policies:
        r = res[pol]
        yy = np.array([q["y"] for q in r]); pr = np.array([q["pred"] for q in r])
        lat = np.array([q["lat"] for q in r]); vis = np.array([q["vis"] for q in r])
        out = PAY[np.array([COUNTER[int(a)] for a in pr]), yy]
        wdl = [100 * np.mean(out == v) for v in (1, 0, -1)]
        forced = 100 * np.mean([q["forced"] for q in r]) if pol[0] != "co dinh" else float("nan")
        now = 100 * np.mean([q["now"] for q in r])
        other = 100 * np.mean((vis >= 0) & (vis != yy))
        ftxt = f"{forced:>9.0f}%" if forced == forced else f"{'-':>10}"
        print(f"   {names[pol]:<18} {now:>9.0f}% {np.percentile(lat, 50):>8.0f} {np.percentile(lat, 90):>6.0f} {ftxt}   "
              f"{100 * np.mean(pr == yy):>11.1f}%   {wdl[0]:>6.1f}/{wdl[1]:>4.1f}/{wdl[2]:>4.1f}   {other:>25.1f}%")
    print("\n   B. So voi HINH THAY DUOC o khung chot (chi van luat doc ro o khung do)")
    print(f"   {'cach':<18} {'dung hinh thay':>15}   {'T/H/T vs hinh thay':>20}   {'n ro':>5}")
    for pol in policies:
        r = res[pol]
        clear = [q for q in r if q["vis"] >= 0]
        if not clear:
            continue
        pr = np.array([q["pred"] for q in clear]); vv = np.array([q["vis"] for q in clear])
        out = PAY[np.array([COUNTER[int(a)] for a in pr]), vv]
        wdl = [100 * np.mean(out == v) for v in (1, 0, -1)]
        print(f"   {names[pol]:<18} {100 * np.mean(pr == vv):>14.1f}%   {wdl[0]:>8.1f}/{wdl[1]:>4.1f}/{wdl[2]:>4.1f}   "
              f"{len(clear) // reps_eff:>5}")
    print("   Doc: 'bi ep han' cao nghia la nhieu van den het thoi gian cho van chua doc ro. 'hinh ro nhung khac y dinh'")
    print("   la hinh trung gian (vd ngon dang mo) bi coi la cu chi ngay khi vua doc ro; tang S giam hien tuong nay nhung")
    print("   tang do tre. Luat hinh hoc chi la xap xi nen moi con so co sai so.")
    print(f"   Luu y: sai so ngau nhien khoang +-{noise_pts(len(y)):.1f} diem % (do tin cay ~95%).")


def extend_policy_experiment(prepared, reps, cv_mode, wait_frames, median_dt_ms):
    y = np.array([G_IDX[p["gesture"]] for p in prepared])
    shp_all = [visible_shape(p["Pn"]) for p in prepared]
    n_folds = len(list(make_splits(prepared, y, cv_mode, 0)))
    if n_folds < 2:
        return
    reps_eff = reps if cv_mode == "round" else 1
    W0, Emax = wait_frames, max(EXT_FRAMES)
    policies = [("co dinh", E, None) for E in (0,) + tuple(EXT_FRAMES)] + \
               [("them", E, thr) for thr in EXT_THRS for E in EXT_FRAMES]
    res = {pol: [] for pol in policies}
    for rep in range(reps_eff):
        for tr, te in make_splits(prepared, y, cv_mode, rep):
            X = np.concatenate([prepared[i]["eng"] for i in tr])
            lab = np.concatenate([np.full(len(prepared[i]["eng"]), y[i]) for i in tr])
            rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=rep,
                                        n_jobs=-1).fit(X, lab)
            for i in te:
                p, shp = prepared[i], shp_all[i]
                entry = p["entry"]
                cut = min(len(p["eng"]) - 1, entry + W0 + Emax) + 1
                probs = proba3(rf, p["eng"][:cut])
                top, am = probs.max(1), probs.argmax(1)
                at = lambda f: min(f, cut - 1)
                for pol in policies:
                    kind, E, thr = pol
                    if kind == "co dinh":
                        f, extended = at(entry + W0 + E), E > 0
                    else:
                        f = at(entry + W0)
                        extended = top[f] < thr
                        if extended:
                            for k in range(1, E + 1):
                                f = at(entry + W0 + k)
                                if top[f] >= thr:
                                    break
                    res[pol].append(dict(y=int(y[i]), pred=int(am[f]), vis=int(shp[f]), extended=bool(extended),
                                         low_end=bool(kind == "them" and top[f] < thr),
                                         lat=float((p["tt"][f] - p["tt"][entry]) * 1000)))

    print(f"\n11. CHO {W0} KHUNG, NEU DO TIN CAY THAP THI CHO THEM (toi da {Emax} khung ~ {Emax * median_dt_ms:.0f} ms)")
    print(f"   Quy tac: chot o khung thu {W0} sau luc vao o; neu xac suat cao nhat < nguong thi cho them tung khung (toi da E khung),")
    print("   chot ngay khi xac suat >= nguong, het cho ma van thap thi chot theo khung cuoi. 'co dinh +j' = luon cho j khung.")
    print("   'them%' = ti le van phai cho them; 'thap%' = ti le van het cho ma van chua du tin cay; T/H/T = may thang/hoa/thua (%).")
    print(f"   {'cach':<22} {'tre p50':>8} {'p90':>6} {'max':>6} {'them%':>6} {'thap%':>6}   {'dung y dinh':>12} {'T/H/T vs y dinh':>18}"
          f"   {'dung hinh thay':>15} {'T/H/T vs hinh thay':>20}")
    for pol in policies:
        kind, E, thr = pol
        name = f"co dinh +{W0 + E}" if kind == "co dinh" else f"{W0} + toi da {E}, nguong {thr:.1f}"
        r = res[pol]
        yy = np.array([q["y"] for q in r]); pr = np.array([q["pred"] for q in r]); vv = np.array([q["vis"] for q in r])
        lat = np.array([q["lat"] for q in r])
        out = PAY[np.array([COUNTER[int(a)] for a in pr]), yy]
        wdl = [100 * np.mean(out == v) for v in (1, 0, -1)]
        clear = vv >= 0
        outv = PAY[np.array([COUNTER[int(a)] for a in pr[clear]]), vv[clear]] if clear.any() else np.array([])
        wv = [100 * np.mean(outv == v) if len(outv) else float("nan") for v in (1, 0, -1)]
        acc_v = 100 * np.mean(pr[clear] == vv[clear]) if clear.any() else float("nan")
        them = 100 * np.mean([q["extended"] for q in r])
        thap = 100 * np.mean([q["low_end"] for q in r]) if kind == "them" else float("nan")
        thap_t = f"{thap:>5.0f}%" if thap == thap else f"{'-':>6}"
        print(f"   {name:<22} {np.percentile(lat, 50):>8.0f} {np.percentile(lat, 90):>6.0f} {lat.max():>6.0f} {them:>5.0f}% {thap_t}   "
              f"{100 * np.mean(pr == yy):>11.1f}% {wdl[0]:>6.1f}/{wdl[1]:>4.1f}/{wdl[2]:>4.1f}   "
              f"{acc_v:>14.1f}% {wv[0]:>8.1f}/{wv[1]:>4.1f}/{wv[2]:>4.1f}")
    print("   Doc: chi tin mot cach hon 'co dinh +6' khi so van thua giam ro ret (hon sai so) VA thoi gian cho (p90, max) van")
    print("   nam trong muc ban chap nhan. 'co dinh +7/+8/+9' la gia phai tra neu cho them cho TAT CA van, de so voi cho co chon loc.")
    print(f"   Luu y: sai so ngau nhien khoang +-{noise_pts(len(y)):.1f} diem % (do tin cay ~95%); so van thua rat nho nen cang lon sai so.")


# ------------------------------------------------------------------ chạy
def main():
    global EXT_HI, EXT_LO
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
    ap.add_argument("--no-visible", action="store_true", help="bo muc 8 (danh gia theo hinh tay thay duoc)")
    ap.add_argument("--wait-frames", type=int, default=WAIT_FRAMES,
                    help="muc 10: so khung toi da cho them sau luc vao o de hinh tay doc ro")
    ap.add_argument("--ext-hi", type=float, default=EXT_HI, help="nguong ngon duoi cua luat hinh hoc (muc 8)")
    ap.add_argument("--ext-lo", type=float, default=EXT_LO, help="nguong ngon co cua luat hinh hoc (muc 8)")
    args = ap.parse_args()
    EXT_HI, EXT_LO = args.ext_hi, args.ext_lo

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

    rule_match = None
    if not args.no_visible:
        rule_match = rule_check(rounds)

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
        if not args.no_visible:
            visible_experiment(prepared, args.reps, cv_mode)
            formation_experiment(prepared, args.reps, cv_mode, rule_match)
            wait_policy_experiment(prepared, args.reps, cv_mode, args.wait_frames, median_dt_ms)
            extend_policy_experiment(prepared, args.reps, cv_mode, args.wait_frames, median_dt_ms)

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
