"""
collect_approach.py
--------------------
Thu dữ liệu cho giao thức "đưa tay từ ngoài vào ô vuông giữa màn hình".

Mặc định mỗi người chơi có 6 điều kiện: 3 cử chỉ (kéo/búa/bao) x 1 phía vào (TRÁI)
x 2 tay (trái/phải). Có thể đổi bằng --side (trai | phai | both) và --hand (trai | phai | both).
Chương trình tự lập lịch ngẫu nhiên có cân bằng, tự phát hiện bắt đầu/kết thúc
từng ván, nên người chơi KHÔNG phải bấm phím trong lúc thu.

Cách chạy (ví dụ):
    python collect_approach.py --person hi --rounds 20              (6 điều kiện x 20 = 120 ván)
    python collect_approach.py --person hi --side both --rounds 10  (cả hai phía, 12 điều kiện)
    python collect_approach.py --person an --rounds 10 --video      (chỉ bật --video khi người chơi đồng ý bị quay)

Phím: SPACE = bắt đầu phiên, X = hủy ván đang thu (tính là ván lỗi), Q = thoát.

Quy ước (QUAN TRỌNG):
  - Ảnh lưu là ảnh GỐC, KHÔNG lật gương. Màn hình xem trước thì lật gương (cho tự nhiên).
  - "Phía vào" và "tay" trong lệnh là của NGƯỜI CHƠI nhìn vào màn hình-gương:
      "vao tu TRAI" = tay đi vào từ bên trái màn hình xem trước (= bên PHẢI của ảnh gốc).
      "tay TRAI" = tay trái của chính người chơi.
  - Nhãn "Left/Right" do MediaPipe trả về chỉ được lưu để THAM KHẢO. Theo tài liệu
    MediaPipe nhãn này giả định ảnh đã lật gương, nên với ảnh gốc có thể bị đảo.
    Nhãn đúng là nhãn theo LỆNH hiển thị cho người chơi.
"""

import argparse
import csv
import glob
import json
import os
import random
import time
from collections import Counter, deque
from datetime import datetime

import numpy as np

GESTURES = ["keo", "bua", "bao"]
SIDES = ["trai", "phai"]
HANDS = ["trai", "phai"]
GESTURE_TEXT = {"keo": "KEO", "bua": "BUA", "bao": "BAO"}
SIDE_TEXT = {"trai": "TRAI", "phai": "PHAI"}

MIRROR_PREVIEW = True     # màn hình xem trước lật gương
SQUARE_FRAC = 0.45        # cạnh ô vuông = 45% cạnh ngắn của khung hình (chỉ để dẫn đường)
HOLD_S = 0.7              # tay phải ở trong ô liên tục chừng này giây thì ván hoàn tất
TIMEOUT_S = 6.0           # quá thời gian này mà chưa xong thì ván lỗi
LOST_S = 1.0              # mất tay lâu hơn mức này (sau khi đã thấy tay) thì ván lỗi
CLEAR_S = 0.5             # phải không còn tay trong khung chừng này giây mới sang ván mới
CUE_S = 1.5               # thời gian hiện lệnh trước khi bắt đầu ghi
MSG_S = 1.0               # thời gian hiện kết quả ván trước đó

PALM_IDS = [0, 5, 9, 13, 17]   # cổ tay + 4 khớp gốc ngón: dùng tính tâm lòng bàn tay


# ====================================================================
# Phần logic thuần (không cần camera) -- đã kiểm tra riêng
# ====================================================================
def make_schedule(rounds_per_condition, seed=None, sides=SIDES, hands=HANDS):
    """Lịch cân bằng theo khối: mỗi khối chứa đủ mọi điều kiện (xáo trộn).
    Nếu phiên bị dừng giữa chừng, các điều kiện vẫn còn gần cân bằng."""
    rng = random.Random(seed)
    conditions = [(g, s, h) for g in GESTURES for s in sides for h in hands]
    schedule = []
    for _ in range(rounds_per_condition):
        block = conditions[:]
        rng.shuffle(block)
        schedule.extend(block)
    return schedule


def square_bounds(w, h, frac=SQUARE_FRAC):
    """Ô vuông ở giữa, trả về (x0, x1, y0, y1) theo toạ độ chuẩn hóa 0..1."""
    side = frac * min(w, h)
    return (0.5 - side / (2 * w), 0.5 + side / (2 * w),
            0.5 - side / (2 * h), 0.5 + side / (2 * h))


def in_square(cx, cy, w, h, frac=SQUARE_FRAC):
    x0, x1, y0, y1 = square_bounds(w, h, frac)
    return x0 <= cx <= x1 and y0 <= cy <= y1


def palm_center(lm):
    """lm: mảng (21, 3) toạ độ chuẩn hóa của ảnh gốc."""
    return float(np.mean(lm[PALM_IDS, 0])), float(np.mean(lm[PALM_IDS, 1]))


def expected_raw_side(cue_side, mirror=MIRROR_PREVIEW):
    """Đổi 'phía vào theo màn hình xem trước' sang 'phía của ảnh gốc'."""
    player_left = (cue_side == "trai")
    if mirror:
        return "right" if player_left else "left"
    return "left" if player_left else "right"


class RoundController:
    """Theo dõi phần ghi của 1 ván. update() trả về None khi ván còn đang diễn ra,
    hoặc (hop_le: bool, ly_do: str, phia_vao_anh_goc: str|None) khi ván kết thúc."""

    def __init__(self, cue_side, w, h, hold_s=HOLD_S, timeout_s=TIMEOUT_S, lost_s=LOST_S):
        self.expect = expected_raw_side(cue_side)
        self.w, self.h = w, h
        self.hold_s, self.timeout_s, self.lost_s = hold_s, timeout_s, lost_s
        self.first_t = None
        self.entry_side = None
        self.hold_start = None
        self.last_seen = None

    def update(self, t, detected, palm_xy):
        if t > self.timeout_s:
            return (False, "het gio", self.entry_side)

        if detected:
            cx, cy = palm_xy
            self.last_seen = t
            inside = in_square(cx, cy, self.w, self.h)

            if self.first_t is None:                      # lần đầu thấy tay trong ván này
                self.first_t = t
                if inside:
                    return (False, "tay da o trong o luc bat dau", None)
                self.entry_side = "left" if cx < 0.5 else "right"
                if self.entry_side != self.expect:
                    return (False, "vao sai phia", self.entry_side)

            if inside:
                if self.hold_start is None:
                    self.hold_start = t
                elif t - self.hold_start >= self.hold_s:
                    return (True, "ok", self.entry_side)
            else:
                self.hold_start = None
        else:
            self.hold_start = None
            if self.first_t is not None and t - self.last_seen > self.lost_s:
                return (False, "mat tay", self.entry_side)
        return None


class Recorder:
    """Giữ dữ liệu thô từng khung của 1 ván (khung không thấy tay -> NaN)."""

    def __init__(self):
        self.t, self.lm, self.det, self.label, self.score = [], [], [], [], []

    def add(self, t, lm, label, score):
        self.t.append(t)
        self.det.append(lm is not None)
        self.lm.append(lm if lm is not None else np.full((21, 3), np.nan, dtype=np.float32))
        self.label.append(label or "")
        self.score.append(float(score) if score is not None else np.nan)

    def majority_label(self):
        labs = [l for l, d in zip(self.label, self.det) if d and l]
        return Counter(labs).most_common(1)[0][0] if labs else ""


def save_round(path, rec, meta):
    np.savez_compressed(
        path,
        t=np.array(rec.t, dtype=np.float64),
        landmarks=np.array(rec.lm, dtype=np.float32),     # (T, 21, 3), x,y chuẩn hóa theo ảnh gốc
        detected=np.array(rec.det, dtype=bool),
        mp_label=np.array(rec.label, dtype="U8"),
        mp_score=np.array(rec.score, dtype=np.float32),
        meta_json=np.array(json.dumps(meta, ensure_ascii=False)),
    )


INDEX_FIELDS = ["person", "session", "attempt", "gesture", "side_cue", "hand_cue",
                "valid", "reason", "entry_side_raw", "n_frames", "n_detected",
                "duration_s", "mp_majority_label", "file"]


def append_index(index_path, row):
    new = not os.path.exists(index_path)
    with open(index_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


# ====================================================================
# Phần chạy với camera
# ====================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--person", required=True, help="mã người chơi, ví dụ: hi, an, binh")
    p.add_argument("--session", default=None, help="mã buổi thu (mặc định: ngày-giờ)")
    p.add_argument("--rounds", type=int, default=10, help="số ván MỖI điều kiện")
    p.add_argument("--side", default="trai", choices=["trai", "phai", "both"],
                   help="phía tay đi vào, theo màn hình xem trước (mặc định: trai)")
    p.add_argument("--hand", default="both", choices=["trai", "phai", "both"],
                   help="tay dùng để chơi (mặc định: both = cả hai tay)")
    p.add_argument("--outdir", default="approach_data")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--complexity", type=int, default=0, choices=[0, 1],
                   help="model_complexity của MediaPipe; dùng đúng giá trị sẽ dùng lúc chạy thật")
    p.add_argument("--video", action="store_true", help="lưu video thô mỗi ván (cần người chơi đồng ý)")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    import cv2
    import mediapipe as mp

    session = args.session or datetime.now().strftime("%Y%m%d_%H%M%S")
    sess_dir = os.path.join(args.outdir, f"person_{args.person}", f"session_{session}")
    os.makedirs(sess_dir, exist_ok=True)
    index_path = os.path.join(args.outdir, "rounds_index.csv")

    sides = SIDES if args.side == "both" else [args.side]
    hands_used = HANDS if args.hand == "both" else [args.hand]
    conditions = [(g, s_, h_) for g in GESTURES for s_ in sides for h_ in hands_used]
    schedule = deque(make_schedule(args.rounds, args.seed, sides, hands_used))
    total = len(schedule)
    valid_counts, hand_table = Counter(), Counter()
    attempt = len(glob.glob(os.path.join(sess_dir, "r*.npz")))

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Khong mo duoc camera.")
        return

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    phase = "idle"
    cond = ctrl = rec = writer = video_path = None
    clear_since, cue_start, t_go = None, 0.0, 0.0
    msg, msg_until = "", 0.0
    w = h = None
    done_valid = 0

    with mp_hands.Hands(max_num_hands=1, model_complexity=args.complexity,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            ok, raw = cap.read()
            if not ok:
                break
            t = time.perf_counter()
            if w is None:
                h, w = raw.shape[:2]
            res = hands.process(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))

            lm = label = score = None
            if res.multi_hand_landmarks:
                pts = res.multi_hand_landmarks[0].landmark
                lm = np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float32)
                cls = res.multi_handedness[0].classification[0]
                label, score = cls.label, cls.score
            detected = lm is not None
            palm = palm_center(lm) if detected else None

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            # ---------------- máy trạng thái ----------------
            if phase == "idle":
                if key == ord(" "):
                    phase = "clear"
                    clear_since = None

            elif phase == "clear":
                if detected:
                    clear_since = None
                elif clear_since is None:
                    clear_since = t
                elif t - clear_since >= CLEAR_S and t >= msg_until:
                    if not schedule:
                        break
                    cond = schedule.popleft()
                    cue_start, phase = t, "cue"

            elif phase == "cue":
                if detected:                                  # tay còn trong khung -> chờ lại
                    schedule.appendleft(cond)
                    phase, clear_since = "clear", None
                elif t - cue_start >= CUE_S:
                    ctrl = RoundController(cond[1], w, h)
                    rec = Recorder()
                    t_go = t
                    if args.video:
                        video_path = os.path.join(sess_dir, "tmp_round.avi")
                        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"MJPG"),
                                                 30.0, (w, h))
                    phase = "go"

            elif phase == "go":
                rec.add(t - t_go, lm, label, score)
                if writer is not None:
                    writer.write(raw)
                status = ctrl.update(t - t_go, detected, palm)
                if key == ord("x"):
                    status = (False, "huy thu cong", ctrl.entry_side)
                if status is not None:
                    valid, reason, entry_raw = status
                    attempt += 1
                    g, side, hand = cond
                    name = f"r{attempt:04d}_{g}_{side}_{hand}" + ("" if valid else "_invalid")
                    meta = dict(person=args.person, session=session, attempt=attempt,
                                gesture=g, side_cue_player=side, hand_cue_player=hand,
                                valid=valid, reason=reason, entry_side_raw=entry_raw,
                                frame_w=w, frame_h=h, mirror_preview=MIRROR_PREVIEW,
                                saved_frames_are_mirrored=False,
                                mediapipe_complexity=args.complexity,
                                square_frac=SQUARE_FRAC, hold_s=HOLD_S)
                    npz_path = os.path.join(sess_dir, name + ".npz")
                    save_round(npz_path, rec, meta)
                    if writer is not None:
                        writer.release()
                        writer = None
                        if valid:
                            os.replace(video_path, os.path.join(sess_dir, name + ".avi"))
                        else:
                            os.remove(video_path)
                    append_index(index_path, dict(
                        person=args.person, session=session, attempt=attempt, gesture=g,
                        side_cue=side, hand_cue=hand, valid=valid, reason=reason,
                        entry_side_raw=entry_raw or "", n_frames=len(rec.t),
                        n_detected=int(sum(rec.det)), duration_s=round(rec.t[-1], 3),
                        mp_majority_label=rec.majority_label(),
                        file=os.path.relpath(npz_path, args.outdir)))
                    if valid:
                        valid_counts[cond] += 1
                        hand_table[(hand, rec.majority_label())] += 1
                        done_valid += 1
                        msg = f"OK ({done_valid}/{total})"
                    else:
                        schedule.append(cond)                 # lặp lại điều kiện này ở cuối lịch
                        msg = f"LOI: {reason} - se lam lai"
                    msg_until = t + MSG_S
                    phase, clear_since = "clear", None

            # ---------------- hiển thị ----------------
            disp = raw.copy()
            if detected:
                mp_draw.draw_landmarks(disp, res.multi_hand_landmarks[0],
                                       mp_hands.HAND_CONNECTIONS)
            if MIRROR_PREVIEW:
                disp = cv2.flip(disp, 1)
            x0, x1, y0, y1 = square_bounds(w, h)
            cv2.rectangle(disp, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)),
                          (255, 200, 0), 2)

            def put(text, row, color=(0, 255, 0)):
                cv2.putText(disp, text, (10, 28 + 28 * row), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2)

            if phase == "idle":
                put("Bam SPACE de bat dau phien thu", 0)
                put(f"Nguoi choi: {args.person} | {total} van", 1, (200, 200, 200))
            elif phase == "clear":
                put("Rut tay ra khoi khung hinh...", 0, (0, 200, 255))
            elif phase in ("cue", "go"):
                g, side, hand = cond
                put(f"{GESTURE_TEXT[g]} - tay {SIDE_TEXT[hand]} - vao tu {SIDE_TEXT[side]}", 0)
                put("CHUAN BI..." if phase == "cue" else "DUA TAY VAO O VUONG!", 1,
                    (0, 200, 255) if phase == "cue" else (0, 0, 255))
            if msg and t < msg_until:
                put(msg, 3, (255, 255, 0))
            put(f"da xong {done_valid}/{total}", 4, (200, 200, 200))
            cv2.imshow("Thu du lieu - dua tay vao o vuong", disp)

    cap.release()
    if writer is not None:
        writer.release()
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
    cv2.destroyAllWindows()

    print(f"\nHoan tat {done_valid}/{total} van hop le. Du lieu o: {sess_dir}")
    print("So van hop le theo dieu kien (cu chi, phia, tay):")
    for c in conditions:
        print(f"  {c}: {valid_counts[c]}")
    print("\nNhan Left/Right cua MediaPipe so voi tay theo LENH (anh goc khong lat):")
    for (hand, lab), n in sorted(hand_table.items()):
        print(f"  lenh tay {hand}: MediaPipe bao '{lab}' trong {n} van")


if __name__ == "__main__":
    main()
