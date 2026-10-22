
import os

import numpy as np

DATA_DIR = "sequences"
INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 8, 12, 16, 20


def dist(frame, idx):
    x, y, z = frame[idx * 3: idx * 3 + 3]
    return float(np.sqrt(x * x + y * y + z * z))


def ext_index_middle(frame):
    return (dist(frame, INDEX_TIP) + dist(frame, MIDDLE_TIP)) / 2


def ext_all(frame):
    return float(np.mean([dist(frame, i)
                          for i in (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)]))


def load(label):
    d = os.path.join(DATA_DIR, label)
    return [np.load(os.path.join(d, f)) for f in sorted(os.listdir(d))
            if f.endswith(".npy")]


def report(name, seqs, bua_seqs, ext_fn, thresholds):
    ext_curves = [np.array([ext_fn(f) for f in s]) for s in seqs]
    bua_vals = np.array([ext_fn(f) for s in bua_seqs for f in s])

    print(f"\n=== {name} ===")
    print(f"{'nguong':>7} | {'som (% chuoi, trung vi)':>24} | "
          f"{'khong vuot':>10} | {'xung dot voi bua':>16}")
    for t in thresholds:
        fracs, never = [], 0
        for curve in ext_curves:
            above = np.where(curve >= t)[0]
            if len(above) == 0:
                never += 1
            else:
                fracs.append(above[0] / len(curve))
        som = f"{np.median(fracs) * 100:5.1f}%" if fracs else "   n/a"
        conflict = (bua_vals >= t).mean() * 100
        print(f"{t:7.2f} | {som:>24} | {never:>10} | {conflict:15.1f}%")


def main():
    keo, bua, bao = load("keo"), load("bua"), load("bao")

    report("KEO (do ngon tro + giua), hien tai T_KEO = 1.47",
           keo, bua, ext_index_middle, np.arange(1.10, 1.55, 0.05))
    report("BAO (do ca 4 ngon), hien tai T_BAO = 1.34",
           bao, bua, ext_all, np.arange(1.00, 1.45, 0.05))

    print("\nCach chon: tim hang co 'xung dot voi bua' van chap nhan duoc")
    print("(vi du duoi 2-3%), roi lay hang co 'som' nho nhat trong so do.")


if __name__ == "__main__":
    main()
