"""
play_realtime.py
----------------
Chạy thật: camera -> MediaPipe -> model -> máy ra nước khắc chế.

Luật đã chốt:
  - Cử chỉ của người chơi được tính ở ĐÚNG khung hình máy quyết định.
  - Từ khung tay vào ô vuông, máy chờ thêm --wait-frames khung (mặc định 6 ~ 190 ms) rồi quyết định
    (kết quả phân tích: chờ cố định 6 khung tốt hơn "chờ đến khi hình tay đọc rõ").
  - Máy chọn nước khắc chế cử chỉ mà model đoán ở khung đó, rồi hiện lên màn hình.

Nước đi của máy hiện trên màn hình bằng HÌNH (kéo/búa/bao) ở góc phải trên. Mặc định dùng hình vẽ sẵn;
muốn dùng ảnh của bạn thì đặt keo.png, bua.png, bao.png (nền trong suốt càng tốt) vào một thư mục và chạy
với --images <thư mục>.

Mỗi ván: rút tay ra khỏi khung hình -> đưa tay từ ngoài vào ô vuông (ô viền xanh) -> máy hiện nước đi.
Phím: Q thoát | D bật/tắt thông tin gỡ lỗi | 1/2/3 = cử chỉ THẬT của người chơi ở ván vừa rồi
(1 kéo, 2 búa, 3 bao) để tính thắng/hòa/thua | X bỏ ván đang chạy.

Chạy:
    python train_model.py                       (một lần, tạo rps_model.joblib)
    python play_realtime.py
    python play_realtime.py --wait-frames 4 --target 0.25 --fps 60 --mjpg

Mỗi ván được ghi vào rps_play_log_<ngày giờ>.csv, mỗi phiên một file, không ghi đè phiên trước (thời gian chờ, xác suất model, nước máy ra, hình luật hình học đọc
được, nhãn thật nếu bạn bấm phím, thời gian xử lý). Cuối phiên in thống kê thời gian xử lý thật.
"""

import argparse
import csv
import time
from collections import Counter

import numpy as np

from rps_core import (COUNTER, GESTURES, PAY, eng_features, in_square, normalize, palm_center,
                      proba3, square_bounds, visible_shape)

CLEAR_S = 0.5             # phải không còn tay trong khung chừng này giây thì mới nhận ván mới
LOST_S = 1.0              # mất tay lâu hơn mức này khi tay còn ở ngoài ô thì bỏ ván
WAIT_LOST_S = 0.15        # mất tay lâu hơn mức này trong lúc chờ quyết định thì bỏ ván
STALE_S = 0.07            # ép quyết định chỉ dùng khung có tay cũ hơn mức này (giây)
MAX_WAIT_S = 0.25         # quá thời gian này sau lúc vào ô mà chưa đủ khung thì ép quyết định (khung có tay mới nhất)
TIMEOUT_S = 6.0           # tay ở ngoài ô quá lâu thì bỏ ván
OUT_NAMES = {1: "thang", 0: "hoa", -1: "thua"}
CSV_COLS = ["round", "wait_ms", "frames_waited", "forced", "pred", "conf", "p_keo", "p_bua", "p_bao",
            "machine_move", "rule_shape", "operator_label", "outcome_vs_rule", "outcome_vs_label",
            "mp_ms", "model_ms", "compute_ms", "shown_ms"]


# ====================================================================
# Máy trạng thái của ván chơi (không cần camera, đã kiểm tra riêng)
# ====================================================================
class GameMachine:
    def __init__(self, w, h, target=0.25, wait_frames=6, hold_s=1.5, on_row=None, max_wait_s=MAX_WAIT_S):
        self.w, self.h, self.target = w, h, target
        self.max_wait_s = max_wait_s
        self.wait_frames, self.hold_s, self.on_row = wait_frames, hold_s, on_row
        self.state, self.clear_since = "clear", None
        self.round_id = 0
        self.last = None            # kết quả ván gần nhất (dict)
        self.pending = None         # ván chưa ghi log (chờ nhãn thật từ người chơi)
        self.msg = ""
        self.score_rule, self.score_label = Counter(), Counter()
        self.cancelled = Counter()

    # ---- tiện ích
    def inside(self, palm):
        return in_square(palm[0], palm[1], self.w, self.h, self.target)

    def _flush(self):
        if self.pending is not None and self.on_row is not None:
            self.on_row(self.pending)
        self.pending = None

    def _cancel(self, reason):
        self.cancelled[reason] += 1
        self.msg = f"BO VAN: {reason}"
        self.state, self.clear_since = "clear", None
        return f"cancel:{reason}"

    def label(self, g):
        """Người chơi báo cử chỉ THẬT (0 kéo, 1 búa, 2 bao) của ván vừa rồi."""
        res = self.pending
        if res is None:
            return
        if res["operator"] >= 0:                                   # sửa nhãn cũ
            self.score_label[OUT_NAMES[int(PAY[res["move"], res["operator"]])]] -= 1
        res["operator"] = int(g)
        self.score_label[OUT_NAMES[int(PAY[res["move"], g])]] += 1

    def abort(self):
        if self.state in ("approach", "wait"):
            self._cancel("bam X")

    def close(self):
        self._flush()

    # ---- cập nhật mỗi khung hình
    def update(self, t, lm, decide_fn):
        """lm: (21,3) toạ độ gốc của khung này hoặc None. Trả về tên sự kiện hoặc None."""
        detected = lm is not None
        palm = palm_center(lm) if detected else None

        if self.state == "clear":
            if detected:
                self.clear_since = None
            elif self.clear_since is None:
                self.clear_since = t
            elif t - self.clear_since >= CLEAR_S:
                self.state = "ready"
                self.msg = ""
            return None

        if self.state == "ready":
            if detected:
                if self.inside(palm):
                    self.cancelled["tay da o trong o"] += 1
                    self.msg = "RUT TAY RA KHOI O"
                    self.state, self.clear_since = "clear", None
                    return "cancel:tay da o trong o"
                self._flush()
                self.round_id += 1
                self.state, self.t_appear, self.last_seen = "approach", t, t
                self.msg = ""
                return "round_start"
            return None

        if self.state == "approach":
            if detected:
                self.last_seen = t
                if self.inside(palm):
                    self.state, self.entry_t, self.count, self.last_lm = "wait", t, 0, lm
                    return self._wait_step(t, lm, decide_fn, first=True)
            elif t - self.last_seen > LOST_S:
                return self._cancel("mat tay")
            if t - self.t_appear > TIMEOUT_S:
                return self._cancel("het gio")
            return None

        if self.state == "wait":
            return self._wait_step(t, lm, decide_fn, first=False)

        if self.state == "show":
            if t >= self.show_until:
                self.state, self.clear_since = "clear", None
                return "show_end"
            return None
        return None

    def _wait_step(self, t, lm, decide_fn, first):
        if lm is not None:
            self.last_seen, self.last_lm = t, lm
            if not first:
                self.count += 1
            if self.count >= self.wait_frames:
                return self._decide(t, lm, decide_fn, forced=False)
        if t - self.entry_t > self.max_wait_s and t - self.last_seen <= STALE_S:   # het han cho, chua du khung
            return self._decide(t, self.last_lm, decide_fn, forced=True)
        if lm is None and t - self.last_seen > WAIT_LOST_S:
            return self._cancel("mat tay luc cho")
        return None

    def _decide(self, t, lm, decide_fn, forced):
        pred, probs, shape, model_ms = decide_fn(lm)
        move = COUNTER[int(pred)]
        self.last = dict(round=self.round_id, wait_ms=(t - self.entry_t) * 1000.0, frames=self.count,
                         forced=forced, pred=int(pred), conf=float(np.max(probs)), probs=np.asarray(probs),
                         move=int(move), shape=int(shape), operator=-1, model_ms=float(model_ms),
                         mp_ms=float("nan"), compute_ms=float("nan"), shown_ms=float("nan"))
        if shape >= 0:
            self.score_rule[OUT_NAMES[int(PAY[move, shape])]] += 1
        self.pending = self.last
        self.state, self.show_until = "show", t + self.hold_s
        self.msg = ""
        return "decided"


def make_decide(model):
    """Hàm quyết định: từ toạ độ gốc của khung -> (cử chỉ đoán, xác suất, hình luật đọc, ms của model)."""
    def decide(lm):
        t0 = time.perf_counter()
        P = normalize(lm[None].astype(float))
        probs = proba3(model, eng_features(P))[0]
        ms = (time.perf_counter() - t0) * 1000.0
        return int(np.argmax(probs)), probs, int(visible_shape(P)[0]), ms
    return decide


def row_to_csv(res):
    ob = lambda m, g: OUT_NAMES[int(PAY[m, g])] if g >= 0 else ""
    p = res["probs"]
    return [res["round"], f"{res['wait_ms']:.1f}", res["frames"], int(res["forced"]), GESTURES[res["pred"]],
            f"{res['conf']:.3f}", f"{p[0]:.3f}", f"{p[1]:.3f}", f"{p[2]:.3f}", GESTURES[res["move"]],
            GESTURES[res["shape"]] if res["shape"] >= 0 else "khong ro",
            GESTURES[res["operator"]] if res["operator"] >= 0 else "", ob(res["move"], res["shape"]),
            ob(res["move"], res["operator"]), f"{res['mp_ms']:.1f}", f"{res['model_ms']:.2f}",
            f"{res['compute_ms']:.1f}", f"{res['shown_ms']:.1f}"]


# ====================================================================
# Hình nước đi của máy (vẽ sẵn bằng OpenCV hoặc đọc từ thư mục ảnh)
# ====================================================================
def _capsule(cv2, img, p1, p2, r, color):
    cv2.line(img, p1, p2, color, thickness=max(1, 2 * r))
    cv2.circle(img, p1, r, color, -1)
    cv2.circle(img, p2, r, color, -1)


def draw_icon(cv2, g, size=256):
    """Hình bàn tay đơn giản (BGRA, nền trong suốt): g = 0 kéo, 1 búa, 2 bao."""
    S = size
    P = lambda x, y: (int(x * S), int(y * S))
    R = lambda r: max(1, int(r * S))
    skin, edge = (70, 200, 255, 255), (30, 30, 30, 255)
    if g == 1:                                                    # búa: nắm tay
        shapes = [("cap", (.40, .64), (.60, .64), .22)] + [("cap", (x, .42), (x, .42), .095) for x in (.32, .44, .56, .68)]
        front = [("cap", (.30, .72), (.52, .60), .075)]
    elif g == 2:                                                  # bao: xòe tay
        shapes = [("cap", (.40, .72), (.60, .72), .19)]
        shapes += [("cap", (x, .52), (x, top), .055) for x, top in ((.34, .22), (.45, .12), (.56, .10), (.67, .20))]
        front = [("cap", (.27, .66), (.12, .45), .06)]
    else:                                                         # kéo: trỏ + giữa chữ V
        shapes = [("cap", (.40, .78), (.60, .78), .15), ("cap", (.43, .64), (.30, .14), .06),
                  ("cap", (.57, .64), (.70, .14), .06), ("cap", (.66, .68), (.66, .68), .07),
                  ("cap", (.76, .74), (.76, .74), .06)]
        front = [("cap", (.30, .76), (.50, .68), .07)]
    img = np.zeros((S, S, 4), np.uint8)
    grow = R(0.02)
    for _, a, b, r in shapes:                                     # vòng 1: viền
        _capsule(cv2, img, P(*a), P(*b), R(r) + grow, edge)
    for _, a, b, r in shapes:                                     # vòng 2: tô màu da
        _capsule(cv2, img, P(*a), P(*b), R(r), skin)
    for _, a, b, r in front:                                      # ngón cái nằm phía trước
        _capsule(cv2, img, P(*a), P(*b), R(r) + grow, edge)
        _capsule(cv2, img, P(*a), P(*b), R(r), skin)
    return img


def load_icons(cv2, folder, size):
    """{0,1,2: BGRA vuông cạnh size}. Có ảnh trong folder thì dùng, không thì dùng hình vẽ sẵn."""
    import os
    icons = {}
    for g, name in enumerate(GESTURES):
        img = None
        if folder:
            path = os.path.join(folder, name + ".png")
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is not None and img.ndim == 3 and img.shape[2] == 3:
                    img = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
                elif img is not None and img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        if img is None:
            img = draw_icon(cv2, g, 256)
        ih, iw = img.shape[:2]
        k = size / max(ih, iw)
        icons[g] = cv2.resize(img, (max(1, int(iw * k)), max(1, int(ih * k))), interpolation=cv2.INTER_AREA)
    return icons


def paste_icon(disp, icon, x, y):
    ih, iw = icon.shape[:2]
    Hh, Ww = disp.shape[:2]
    x2, y2 = min(x + iw, Ww), min(y + ih, Hh)
    if x < 0 or y < 0 or x2 <= x or y2 <= y:
        return
    sub = icon[: y2 - y, : x2 - x]
    a = sub[:, :, 3:4].astype(np.float32) / 255.0
    roi = disp[y:y2, x:x2].astype(np.float32)
    disp[y:y2, x:x2] = (sub[:, :, :3].astype(np.float32) * a + roi * (1 - a)).astype(np.uint8)


# ====================================================================
# Chương trình chính (cần camera + MediaPipe)
# ====================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="rps_model.joblib")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--target", type=float, default=0.25, help="co o vuong (nen bang luc thu du lieu)")
    p.add_argument("--wait-frames", type=int, default=6, help="so khung cho them sau luc tay vao o")
    p.add_argument("--max-wait-ms", type=int, default=int(MAX_WAIT_S * 1000),
                   help="neu sau chung nay ms ke tu luc vao o van chua du khung thi ep quyet dinh")
    p.add_argument("--hold", type=float, default=1.5, help="thoi gian hien nuoc di cua may (giay)")
    p.add_argument("--fps", type=int, default=0)
    p.add_argument("--mjpg", action="store_true")
    p.add_argument("--complexity", type=int, default=0, choices=[0, 1],
                   help="model_complexity MediaPipe; dung dung gia tri luc thu du lieu")
    p.add_argument("--images", default="", help="thu muc chua keo.png, bua.png, bao.png (khong co thi dung hinh ve san)")
    p.add_argument("--icon-size", type=float, default=0.40, help="canh hinh nuoc di = ty le chieu cao khung hinh")
    p.add_argument("--log", default="", help="file log; de trong = tu dat ten theo gio (rps_play_log_NgayGio.csv) de khong ghi de phien truoc")
    return p.parse_args()


def main():
    args = parse_args()
    import cv2
    import joblib
    import mediapipe as mp

    bundle = joblib.load(args.model)
    model = bundle["model"]
    model.n_jobs = 1
    decide = make_decide(model)
    mp_hands, mp_draw = mp.solutions.hands, mp.solutions.drawing_utils

    cap = cv2.VideoCapture(args.camera)
    if args.mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if args.fps > 0:
        cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        print("Khong mo duoc camera.")
        return

    if not args.log:
        args.log = time.strftime("rps_play_log_%Y%m%d_%H%M%S.csv")
    print(f"Log phien nay: {args.log}")
    logf = open(args.log, "w", newline="", encoding="utf-8")
    writer = csv.writer(logf)
    writer.writerow(CSV_COLS)

    def on_row(res):
        writer.writerow(row_to_csv(res))
        logf.flush()

    machine, w, h, icons = None, None, None, None
    debug = True
    mp_all, dts, compute_all, model_all, wait_all, shown_all = [], [], [], [], [], []
    prev_t = None
    print(f"Dang chay: o {args.target}, cho {args.wait_frames} khung. Bam Q de thoat.")

    with mp_hands.Hands(max_num_hands=1, model_complexity=args.complexity,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            ok, raw = cap.read()
            if not ok:
                break
            t = time.perf_counter()
            if machine is None:
                h, w = raw.shape[:2]
                machine = GameMachine(w, h, args.target, args.wait_frames, args.hold, on_row, args.max_wait_ms / 1000.0)
                icons = load_icons(cv2, args.images, int(args.icon_size * h))
            res = hands.process(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
            mp_ms = (time.perf_counter() - t) * 1000.0
            mp_all.append(mp_ms)
            if prev_t is not None:
                dts.append(t - prev_t)
            prev_t = t

            lm = None
            if res.multi_hand_landmarks:
                pts = res.multi_hand_landmarks[0].landmark
                lm = np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float32)
                if not np.isfinite(normalize(lm[None].astype(float))).all():
                    lm = None

            ev = machine.update(t, lm, decide)
            if ev == "decided":
                machine.last["mp_ms"] = mp_ms
                machine.last["compute_ms"] = (time.perf_counter() - t) * 1000.0
                compute_all.append(machine.last["compute_ms"])
                model_all.append(machine.last["model_ms"])
                wait_all.append(machine.last["wait_ms"])

            # ---------------- hiển thị (ảnh lật gương cho người chơi) ----------------
            disp = raw.copy()
            if lm is not None:
                mp_draw.draw_landmarks(disp, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            disp = cv2.flip(disp, 1)
            x0, x1, y0, y1 = square_bounds(w, h, args.target)
            cv2.rectangle(disp, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (255, 200, 0), 2)
            F = cv2.FONT_HERSHEY_SIMPLEX
            if machine.state == "show" and machine.last is not None:
                mv = machine.last["move"]
                ic = icons[mv]
                ix, iy = w - ic.shape[1] - 10, 10
                paste_icon(disp, ic, ix, iy)
                cv2.putText(disp, GESTURES[mv].upper(), (ix + 10, iy + ic.shape[0] + 32), F, 1.1, (0, 255, 0), 3)
                cv2.putText(disp, "MAY RA:", (20, 50), F, 1.0, (0, 255, 0), 2)
            elif machine.state in ("clear", "ready"):
                cv2.putText(disp, machine.msg or ("RUT TAY RA" if machine.state == "clear" else "SAN SANG"),
                            (20, 50), F, 1.0, (200, 200, 200), 2)
            else:
                cv2.putText(disp, "...", (20, 50), F, 1.0, (200, 200, 200), 2)
            if debug:
                sr, sl = machine.score_rule, machine.score_label
                cv2.putText(disp, f"theo luat hinh hoc  T{sr['thang']} H{sr['hoa']} Th{sr['thua']}", (10, h - 50), F, 0.5,
                            (255, 255, 255), 1)
                cv2.putText(disp, f"theo phim 1/2/3     T{sl['thang']} H{sl['hoa']} Th{sl['thua']}", (10, h - 30), F, 0.5,
                            (255, 255, 255), 1)
                if machine.last is not None:
                    L = machine.last
                    cv2.putText(disp, f"doc: {GESTURES[L['pred']]} ({100 * L['conf']:.0f}%)  cho {L['frames']} khung "
                                      f"{L['wait_ms']:.0f}ms  xu ly {L['compute_ms']:.0f}ms",
                                (10, h - 10), F, 0.5, (255, 255, 0), 1)
            cv2.imshow("RPS realtime", disp)
            key = cv2.waitKey(1) & 0xFF                      # imshow + waitKey cung vong de hinh hien ngay
            if ev == "decided":
                machine.last["shown_ms"] = (time.perf_counter() - t) * 1000.0
                shown_all.append(machine.last["shown_ms"])
            if key == ord("q"):
                break
            if key == ord("d"):
                debug = not debug
            if key == ord("x"):
                machine.abort()
            if key in (ord("1"), ord("2"), ord("3")):
                machine.label(key - ord("1"))

    machine and machine.close()
    logf.close()
    cap.release()
    cv2.destroyAllWindows()

    print("\nTong ket phien:")
    if machine is not None:
        print(f"  So van may da quyet dinh: {len(wait_all)}; van bi bo: " +
              (", ".join(f"{k} {v}" for k, v in machine.cancelled.items()) or "0"))
        sr, sl = machine.score_rule, machine.score_label
        print(f"  May thang/hoa/thua theo luat hinh hoc: {sr['thang']}/{sr['hoa']}/{sr['thua']}; "
              f"theo phim ban bam: {sl['thang']}/{sl['hoa']}/{sl['thua']}")
    if len(dts) > 10:
        print(f"  FPS thuc te (trung vi): {1.0 / np.median(dts):.1f}")
    if len(mp_all) > 20:
        pa = np.array(mp_all[5:])
        print(f"  MediaPipe moi khung (ms): trung vi {np.median(pa):.1f}, p95 {np.percentile(pa, 95):.1f}, "
              f"lon nhat {pa.max():.1f}")
    if compute_all:
        c, m, wv = np.array(compute_all), np.array(model_all), np.array(wait_all)
        print(f"  Tu luc nhan khung quyet dinh den luc co nuoc di (ms): trung vi {np.median(c):.1f}, "
              f"lon nhat {c.max():.1f}  (trong do model {np.median(m):.1f})")
        print(f"  Thoi gian cho tu luc tay vao o den luc quyet dinh (ms): trung vi {np.median(wv):.0f}, "
              f"p90 {np.percentile(wv, 90):.0f}")
        if shown_all:
            sh = np.array(shown_all)
            print(f"  Tu luc nhan khung quyet dinh den luc hinh nuoc di da ve len cua so (ms): trung vi {np.median(sh):.1f}, "
                  f"lon nhat {sh.max():.1f}")
        print("  Luu y: chua gom do tre camera va do tre man hinh (tan so quet, thuong them vai den hon muoi ms).")


if __name__ == "__main__":
    main()
