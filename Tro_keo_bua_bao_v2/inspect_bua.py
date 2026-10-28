"""
inspect_bua.py
---------------
Xem kỹ các ván BÚA mà lúc tay vào ô trông như đang mở (độ duỗi cao giống kéo/bao).

Phân loại mỗi ván búa bị đánh dấu:
  A. "mo roi nam"   : tay mở nhiều khung liền rồi nắm lại trước cuối ván
                      -> hành vi của người chơi (đưa tay vào khi còn mở, nắm sau)
  B. "dot bien ngan": chỉ cao 1-3 khung rồi hạ xuống
                      -> giống nhiễu của MediaPipe (có thể lọc bằng trung vị vài khung)
  C. "mo suot van"  : vẫn cao ở cuối ván
                      -> nắm không chặt hoặc ra nhầm cử chỉ; nên xem lại ván đó

Ngoài ra đo: tay có mở ngay từ khung đầu tiên thấy tay không, tay có bị cắt ở mép khung
không, và (với loại A) tay còn mở bao lâu SAU khi vào ô -- đây là thời gian máy cần chờ.

Chạy: python inspect_bua.py            (mặc định ô vuông 0.45, như lúc thu)
      python inspect_bua.py --square 0.30
Kết quả: in ra màn hình, bua_flagged.csv, inspect_bua.png

Cần analyze_approach.py nằm cùng thư mục.
"""

import argparse
import csv
import os
import warnings
from collections import Counter

import numpy as np

from analyze_approach import PALM_IDS, ext_bao, ext_keo, load_rounds, normalize, square_bounds


def prep(r, frac):
    det = np.where(r["det"])[0]
    if len(det) == 0:
        return None
    lm = r["lm"][det]
    Pn = normalize(lm)
    ok = np.isfinite(Pn).all(axis=(1, 2))
    det, lm, Pn = det[ok], lm[ok], Pn[ok]
    if len(det) == 0:
        return None
    palm = lm[:, PALM_IDS, :2].mean(axis=1)
    x0, x1, y0, y1 = square_bounds(r["meta"]["frame_w"], r["meta"]["frame_h"], frac)
    inside = (palm[:, 0] >= x0) & (palm[:, 0] <= x1) & (palm[:, 1] >= y0) & (palm[:, 1] <= y1)
    if not inside.any():
        return None
    touch = ((lm[:, :, 0].min(axis=1) < 0.01) | (lm[:, :, 0].max(axis=1) > 0.99) |
             (lm[:, :, 1].min(axis=1) < 0.01) | (lm[:, :, 1].max(axis=1) > 0.99))
    labs = [l for l in r["mp_label"][det] if l]
    mp = Counter(labs).most_common(1)[0][0] if labs else ""
    return dict(name=os.path.basename(r["path"]), gesture=r["meta"]["gesture"],
                hand=r["meta"]["hand_cue_player"], t=r["t"][det], ek=ext_keo(Pn), eb=ext_bao(Pn),
                palm=palm, entry=int(np.argmax(inside)), touch=touch, mp=mp)


def run_around(high, i0):
    """Đoạn các khung 'cao' liên tiếp chứa khung i0 (hoặc rỗng nếu i0 không cao)."""
    if not high[i0]:
        return None
    a = b = i0
    while a > 0 and high[a - 1]:
        a -= 1
    while b < len(high) - 1 and high[b + 1]:
        b += 1
    return a, b


def pctl(x, ps=(10, 50, 90)):
    return " / ".join(f"{np.percentile(x, p):6.0f}" for p in ps) if len(x) else "n/a"


def make_plot(prepared, flagged_names, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(khong co matplotlib nen bo qua ve hinh)")
        return
    grid = np.arange(-500, 601, 33.0)

    def aligned(p, key):
        x = (p["t"] - p["t"][p["entry"]]) * 1000
        return x, p[key]

    def mean_curve(group, key):
        ys = [np.interp(grid, *aligned(p, key), left=np.nan, right=np.nan) for p in group]
        if not ys:
            return None
        with warnings.catch_warnings():            # các điểm lưới chưa có ván nào phủ tới -> NaN, bỏ qua cảnh báo
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(np.array(ys), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for ax, key, title in ((axes[0], "ek", "Do duoi ngon tro + giua"), (axes[1], "eb", "Do duoi ca 4 ngon")):
        for p in prepared:
            if p["gesture"] != "bua":
                continue
            flagged = p["name"] in flagged_names
            x, y = aligned(p, key)
            ax.plot(x, y, color="red" if flagged else "0.75", lw=1.2 if flagged else 0.6,
                    alpha=0.8 if flagged else 0.6, zorder=3 if flagged else 1)
        for g, color, label in (("keo", "tab:blue", "keo (trung binh)"), ("bao", "tab:green", "bao (trung binh)")):
            m = mean_curve([p for p in prepared if p["gesture"] == g], key)
            if m is not None:
                ax.plot(grid, m, color=color, lw=2.5, label=label, zorder=4)
        ax.axvline(0, color="black", ls="--", lw=1)
        ax.set_title(title)
        ax.set_xlabel("thoi gian so voi luc tay vao o (ms)")
        ax.set_ylabel("do duoi")
    axes[0].plot([], [], color="red", label="bua bi danh dau (cao luc vao o)")
    axes[0].plot([], [], color="0.75", label="bua binh thuong")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Da luu hinh: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="approach_data")
    ap.add_argument("--person", default=None)
    ap.add_argument("--square", type=float, default=0.45)
    ap.add_argument("--high-keo", type=float, default=1.5, help="nguong 'cao' cua do duoi ngon tro+giua")
    ap.add_argument("--high-bao", type=float, default=1.3, help="nguong 'cao' cua do duoi ca 4 ngon")
    args = ap.parse_args()

    rounds = load_rounds(args.outdir, args.person)
    prepared = [p for p in (prep(r, args.square) for r in rounds) if p is not None]
    bua = [p for p in prepared if p["gesture"] == "bua"]
    if len(bua) < 5:
        print("Qua it van bua de phan tich.")
        return

    for p in prepared:
        p["high"] = (p["ek"] >= args.high_keo) | (p["eb"] >= args.high_bao)
        e = p["entry"]
        p["flag"] = bool(p["high"][e])
        p["end_ek"], p["end_eb"] = float(p["ek"][-5:].mean()), float(p["eb"][-5:].mean())

    normal = [p for p in bua if not p["flag"]]
    ref_ek = float(np.median([p["end_ek"] for p in normal])) if normal else 1.1
    ref_eb = float(np.median([p["end_eb"] for p in normal])) if normal else 1.0
    flagged = [p for p in bua if p["flag"]]

    print("=" * 74)
    print(f"VAN BUA: {len(bua)} van, o vuong {args.square}")
    print(f"Nguong 'cao': do duoi keo >= {args.high_keo} hoac do duoi bao >= {args.high_bao} luc tay vao o")
    print(f"Van bua bi danh dau (luc vao o trong nhu dang mo): {len(flagged)}/{len(bua)} "
          f"= {100 * len(flagged) / len(bua):.0f}%")
    print(f"Muc 'nam chat' cua cac van binh thuong o cuoi van: keo {ref_ek:.2f}, bao {ref_eb:.2f}")
    print("=" * 74)

    rows, cats = [], Counter()
    close_ms = []
    for p in flagged:
        a, b = run_around(p["high"], p["entry"])
        n_run = b - a + 1
        end_low = (p["end_ek"] <= ref_ek + 0.25) and (p["end_eb"] <= ref_eb + 0.25)
        if not end_low:
            cat = "C mo suot van"
        elif n_run <= 3:
            cat = "B dot bien ngan"
        else:
            cat = "A mo roi nam"
        closing = None
        if b + 1 < len(p["high"]):
            closing = (p["t"][b + 1] - p["t"][p["entry"]]) * 1000
        if cat.startswith("A") and closing is not None:
            close_ms.append(closing)
        cats[cat] += 1
        rows.append(dict(file=p["name"], hand=p["hand"], mediapipe_label=p["mp"], category=cat,
                         ext_keo_at_entry=round(float(p["ek"][p["entry"]]), 2),
                         ext_bao_at_entry=round(float(p["eb"][p["entry"]]), 2),
                         high_run_frames=n_run, open_from_first_frame=bool(a == 0),
                         hand_touches_border=bool(p["touch"][a:b + 1].any()),
                         ms_open_after_entry=round(closing, 0) if closing is not None else "",
                         end_ext_keo=round(p["end_ek"], 2), end_ext_bao=round(p["end_eb"], 2)))

    print("\nPHAN LOAI cac van bua bi danh dau:")
    for cat in ("A mo roi nam", "B dot bien ngan", "C mo suot van"):
        print(f"  {cat:<18}: {cats[cat]}")
    if rows:
        print(f"\n  Mo ngay tu khung dau tien thay tay: {sum(r['open_from_first_frame'] for r in rows)}/{len(rows)}")
        print(f"  Tay bi cat o mep khung trong luc cao: {sum(r['hand_touches_border'] for r in rows)}/{len(rows)}")
    if close_ms:
        print(f"\n  Loai A: tay con mo SAU khi vao o (ms)  [p10 / trung vi / p90]: {pctl(close_ms)}")
        print("  -> day la thoi gian may can cho de thay tay nam lai.")

    print("\nCHI TIET (toi da 30 van):")
    print(f"  {'file':<28}{'tay':<6}{'MP':<7}{'loai':<17}{'keo@vao':>8}{'bao@vao':>8}{'khung':>6}{'ms mo sau vao':>14}")
    for r in rows[:30]:
        print(f"  {r['file']:<28}{r['hand']:<6}{r['mediapipe_label']:<7}{r['category']:<17}{r['ext_keo_at_entry']:>8}"
              f"{r['ext_bao_at_entry']:>8}{r['high_run_frames']:>6}{str(r['ms_open_after_entry']):>14}")
    print("  (MP = nhan Left/Right cua MediaPipe; tay PHAI binh thuong se la 'Left', tay TRAI la 'Right' vi anh khong lat)")

    if rows:
        with open("bua_flagged.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("\nDa luu bua_flagged.csv")
    make_plot(prepared, {p["name"] for p in flagged}, "inspect_bua.png")


if __name__ == "__main__":
    main()
