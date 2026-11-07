"""Demo v2: game oẳn tù tì A -> B chơi với AI qua webcam (cửa sổ OpenCV, dựa trên 22_GameSkeleton_AB).
Chạy:  python demo_v2.py            (webcam 0)
       python demo_v2.py --cam 1    (đổi camera)
Phím: SPACE chơi / chơi tiếp / chơi lại | r reset điểm | q thoát.

Luật (bản chốt 06/11/2026): SPACE rồi nắm BÚA đưa vào ô A, giữ 0,3 giây. Đi tay theo đường nối tới ô B; tới vạch phải còn là búa,
qua vạch thì ra kéo, búa hoặc bao tùy ý. Tay chạm B thì AI chốt đáp án (không random, không sửa) và ra quân khắc chế.
0,2 giây sau, cử chỉ thật của bạn (đọc bằng luật ngón tay, độc lập với model) giống lúc AI chốt thì ra kết quả, khác thì hủy ván."""
import argparse, sys, threading, time, unicodedata, wave
from pathlib import Path
import cv2
import numpy as np

from game_logic import GameAB, CLASSES, hand_shape, hand_point

HERE = Path(__file__).resolve().parent
MIRROR = True                                                  # lật gương hình webcam
WIN_NAME = "Oan tu ti AI (SPACE: choi | r: reset | q: thoat)"
COLORS = {0: (60, 60, 255), 1: (255, 150, 40), 2: (60, 210, 60)}      # BGR: paper đỏ, rock xanh dương, scissors xanh lá (chỉ dùng khi --debug)
GRAY, YELLOW, BLACK, WHITE = (190, 190, 190), (150, 255, 255), (0, 0, 0), (255, 255, 255)
RED, CYAN, ORANGE, GREEN = (40, 40, 255), (255, 230, 0), (60, 160, 255), (80, 230, 80)
BONES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
         (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
BANNER = {"ai": ("AI THANG", (60, 60, 255)), "player": ("BAN THANG", (60, 210, 60)),
          "draw": ("HOA", YELLOW), "unknown": ("KHONG XAC DINH", GRAY)}


# ---------- giao diện (OpenCV không vẽ được chữ có dấu nên bỏ dấu) ----------
def ascii_vn(s):
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def put_label(img, text, x, y, bg, scale=0.5):
    th = 1 if scale < 0.7 else 2                               # chữ nhỏ thì nét mảnh cho dễ đọc
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    cv2.rectangle(img, (x, y - h - base - 4), (x + w + 8, y + 2), bg, -1)
    cv2.putText(img, text, (x + 4, y - base), cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, th, cv2.LINE_AA)


def draw_hand(img, lm, color, mirror=True):
    h, w = img.shape[:2]
    px, py = (lm[:, 0] * w).astype(int), (lm[:, 1] * h).astype(int)
    if mirror:
        px = w - 1 - px
    for a, b in BONES:
        cv2.line(img, (px[a], py[a]), (px[b], py[b]), color, 2, cv2.LINE_AA)
    cv2.rectangle(img, (px.min() - 12, py.min() - 12), (px.max() + 12, py.max() + 12), color, 2)


def draw_zone(img, g):
    """Vẽ dải cho phép, đường nối, ô A, ô B và vạch. Chờ ở A: ô A vàng, đang giữ búa thì xanh lá + vòng tiến độ;
    đi: đỏ khi lệch đường; vạch cam (chưa chạm, phải giữ búa) -> xanh lá (đã chạm, được đổi tay)."""
    h, w = img.shape[:2]
    a, b = (int(g.A_n[0] * w), int(g.A_n[1] * h)), (int(g.B_n[0] * w), int(g.B_n[1] * h))
    r, need = int(g.box_r * h), g.state == "NEED_A"
    col = RED if g.off else CYAN
    col_a = (GREEN if g.a_hold > 0 else YELLOW) if need else col
    col_b = GRAY if need else col
    col_line = GRAY if need else col
    overlay = img.copy()
    cv2.line(overlay, a, b, col_line, max(2, int(2 * g.band * h)), cv2.LINE_AA)       # dải cho phép lệch (mờ)
    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
    cv2.line(img, a, b, col_line, 2, cv2.LINE_AA)
    for pt, name, c in ((a, "A", col_a), (b, "B", col_b)):
        cv2.circle(img, pt, r, c, 3, cv2.LINE_AA)
        put_label(img, name, pt[0] - 8, pt[1] - r - 6, c, 0.5)
    put_label(img, "BUA", a[0] - 18, a[1] + r + 22, col_a, 0.5)                       # nhắc: ô A phải là búa
    if need and g.a_hold > 0:                                                         # vòng tiến độ giữ búa
        cv2.ellipse(img, a, (r + 8, r + 8), -90, 0, int(360 * g.a_hold), GREEN, 4, cv2.LINE_AA)
    if g.state in ("NEED_A", "GO"):                                                   # vạch vuông góc với đường A->B tại gate_f
        d = np.array([b[0] - a[0], b[1] - a[1]], np.float32); d /= np.linalg.norm(d)
        perp = np.array([-d[1], d[0]])
        c = np.array(a, np.float32) + (np.array(b, np.float32) - np.array(a, np.float32)) * g.gate_f
        half = 1.3 * g.band * h
        gc = GREEN if (g.state == "GO" and g.passed) else (YELLOW if need else ORANGE)
        cv2.line(img, tuple((c - perp * half).astype(int)), tuple((c + perp * half).astype(int)), gc, 4, cv2.LINE_AA)


def draw_game_ab(img, g):
    h, w = img.shape[:2]
    if g.state == "GO":
        if g.off: txt, c = "LECH DUONG!", RED
        elif g.warn: txt, c = "CANH BAO: GIU BUA toi vach!", RED
        elif not g.passed: txt, c = "Giu BUA toi vach", YELLOW
        else: txt, c = "Di theo duong toi B", YELLOW
        put_label(img, txt, 10, h - 32, c, 0.55)
    elif g.state == "RESULT":
        text, col = BANNER[g.outcome]
        put_label(img, text, 10, 100, col, 1.1)
        if g.outcome == "unknown":
            put_label(img, "Khong doc duoc hinh tay sau khi AI chot", 10, h - 56, GRAY, 0.5)
        else:
            put_label(img, f"BAN ra: {CLASSES[g.player]}", 10, h - 80, COLORS[g.player], 0.6)
            put_label(img, f"AI doan ban ra {CLASSES[g.guess]} -> AI ra: {CLASSES[g.ai]}", 10, h - 56, COLORS[g.ai], 0.6)
        put_label(img, "Khung luc cham B | SPACE: choi tiep", 10, h - 32, YELLOW, 0.5)
    elif g.state == "CANCEL":
        put_label(img, "VAN HUY", 10, 100, ORANGE, 1.1)
        put_label(img, f"Tay luc cham B khac tay sau {g.confirm_sec:.1f}s", 10, h - 56, ORANGE, 0.55)
        put_label(img, "Khung luc cham B | SPACE: choi lai", 10, h - 32, YELLOW, 0.5)
    elif g.state != "CONFIRM":
        put_label(img, ascii_vn(g.msg), 10, h - 32, YELLOW, 0.55)
    s = g.score
    put_label(img, f"AI {s['ai']} - {s['player']} Nguoi | hoa {s['draw']} | huy {g.stats['cancel']}", 10, h - 10, YELLOW, 0.45)


# ---------- âm thanh (chỉ Windows; nơi khác hoặc khi lỗi thì chạy im lặng) ----------
try:
    import winsound
except ImportError:
    winsound = None
SOUNDS = {}


def init_sounds():
    """tick khi bắt đầu đi, tiếng cao khi tay chạm B. File wav tự tạo trong demo_v2/sounds."""
    if winsound is None:
        return
    try:
        snd = HERE / "sounds"; snd.mkdir(exist_ok=True)
        for name, f, ms in (("tick", 800, 120), ("ra", 1400, 350)):
            path = snd / f"{name}.wav"
            if not path.exists():
                sr = 44100; t = np.arange(int(sr * ms / 1000)) / sr
                x = np.sin(2 * np.pi * f * t) * np.minimum(1, np.minimum(t, t[-1] - t) * 200)     # mờ đầu và cuối, tránh tiếng "tạch"
                with wave.open(str(path), "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
                    w.writeframes((x * 0.6 * 32767).astype(np.int16).tobytes())
            SOUNDS[name] = str(path)
    except OSError:
        SOUNDS.clear()


def play(name):
    if name in SOUNDS:
        winsound.PlaySound(SOUNDS[name], winsound.SND_FILENAME | winsound.SND_ASYNC)    # không chặn vòng lặp webcam


# ---------- webcam ----------
def open_camera(i):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(i)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30); cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class Grabber(threading.Thread):
    """Đọc camera liên tục ở luồng riêng và CHỈ giữ khung mới nhất, để xử lý chậm hơn camera cũng không bị trễ dồn."""

    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap, self.cv, self.halt = cap, threading.Condition(), threading.Event()
        self.frame, self.t, self.seq, self.err = None, 0.0, 0, None

    def run(self):
        while not self.halt.is_set():
            ok, f = self.cap.read()
            t = time.perf_counter()                             # mốc thời gian lúc chụp
            with self.cv:
                if ok:
                    self.frame, self.t, self.seq = f, t, self.seq + 1
                else:
                    self.err = "Webcam ngừng trả hình."
                self.cv.notify_all()
            if not ok:
                return

    def get(self, last_seq, timeout=1.0):
        """Chờ khung mới hơn last_seq. Trả (khung, mốc chụp, seq); hết giờ chờ thì trả (None, None, last_seq)."""
        with self.cv:
            self.cv.wait_for(lambda: self.seq != last_seq or self.err, timeout)
            if self.err:
                raise RuntimeError(self.err)
            if self.seq == last_seq:
                return None, None, last_seq
            return self.frame, self.t, self.seq


def main():
    ap = argparse.ArgumentParser(description="Oẳn tù tì AI: luật A -> B, chơi bằng webcam")
    ap.add_argument("--cam", type=int, default=0, help="số camera (mặc định 0; không mở được thì thử 1)")
    ap.add_argument("--debug", action="store_true", help="hiện nhãn model trên hình (chỉ để kiểm tra, làm LỘ đáp án của AI)")
    ap.add_argument("--no-sound", action="store_true", help="tắt âm thanh")
    args = ap.parse_args()

    from detector import Detector                               # import ở đây để --help chạy nhanh
    print("Đang nạp model...")
    det = Detector()
    print(f"Model chạy trên: {det.dev}")
    cap = open_camera(args.cam)
    if not cap.isOpened():
        sys.exit(f"Không mở được webcam {args.cam} (đang có chương trình khác dùng camera? Thử: python demo_v2.py --cam 1)")
    if not args.no_sound:
        init_sounds()

    g = GameAB()                                                # đổi tham số luật ở đây, ví dụ GameAB(gate_f=0.3, hold_a=0.4)
    grab = Grabber(cap); grab.start()
    seq, evidence, fps, last, shown = 0, None, 0.0, time.perf_counter(), False
    print("Sẵn sàng. Nhấn SPACE trong cửa sổ game để chơi.")
    try:
        while True:
            frame, now, seq = grab.get(seq)                     # now = mốc lúc camera chụp khung này
            if frame is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            lm, pr = det.process(frame, now)
            sh = int(hand_shape(lm)) if lm is not None else None
            pt = hand_point(lm, MIRROR) if lm is not None else None
            before = g.state
            g.step(now, pt, pr, sh)                             # luật chạy trước khi vẽ
            if g.entered_go: play("tick")                       # tiếng tick khi bắt đầu đi
            if g.locked_now: play("ra")                         # tiếng cao khi tay chạm B (AI chốt)
            if g.locked_now:                                    # nhật ký kiểm chứng: đáp án AI ghi ngay lúc chốt, trước khi bạn đổi tay
                r = g.history[-1]
                print(f"[ván {r['round']}] AI chốt {r['ai']} (đoán bạn ra {r['guess']}) sau {r['tu_A_den_chot_s']:.2f}s kể từ lúc bắt đầu đi")
            if before == "CONFIRM" and g.state in ("RESULT", "CANCEL"):
                r = g.history[-1]
                print(f"[ván {r['round']}] tay sau xác nhận: {r['player']} -> {g.state}"
                      f"{'' if g.state == 'CANCEL' else ' ' + str(r['outcome'])} | điểm {g.score} | huỷ {g.stats['cancel']}")

            disp = cv2.flip(frame, 1) if MIRROR else frame
            draw_zone(disp, g)
            if lm is not None:
                col, c = CYAN, None                             # một màu cố định, không lộ model đang đoán gì
                if args.debug:
                    c, conf = int(pr.argmax()), float(pr.max())
                    col = COLORS[c] if conf >= g.thr else GRAY
                draw_hand(disp, lm, col, MIRROR)
                hh, ww = disp.shape[:2]
                cv2.circle(disp, (int(pt[0] * ww), int(pt[1] * hh)), 6, WHITE, -1)     # điểm đại diện của tay
                if args.debug:
                    put_label(disp, f"model: {CLASSES[c]} {conf:.0%} | hinh: {CLASSES[sh] if sh >= 0 else '?'}", 10, 42, col, 0.45)
            if g.locked_now:
                evidence = disp.copy()                          # khung lúc chạm B (có đường, ô, khung xương)
            if g.state in ("RESULT", "CANCEL") and evidence is not None:
                disp = evidence.copy()                          # kết quả hoặc hủy ván đứng yên trên khung lúc chạm B cho đến khi bấm SPACE
            fps, last = 0.9 * fps + 0.1 / max(now - last, 1e-6), now
            put_label(disp, f"FPS {fps:.0f}", 10, 20, YELLOW, 0.45)
            draw_game_ab(disp, g)
            cv2.imshow(WIN_NAME, disp)
            shown = True

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q") or (shown and cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1):
                break
            if k == 32:
                g.start(now)
            if k == ord("r"):
                g.reset_score()
    except RuntimeError as e:                                   # camera ngừng trả hình
        print("LỖI:", e)
    finally:
        grab.halt.set(); grab.join(timeout=1)
        cap.release(); cv2.destroyAllWindows(); det.close()
    print("Đã đóng. Số ván:", g.stats, "| điểm:", g.score)


if __name__ == "__main__":
    main()
