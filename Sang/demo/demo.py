"""Demo kéo-búa-bao: dự đoán sớm cử chỉ từ webcam hoặc file video (TCN task8_step2_distance).

Chạy:  python demo.py                       (webcam)
       python demo.py --video clip.mp4      (file video)
Phím:  q thoát | [ giảm ngưỡng tin cậy | ] tăng ngưỡng (mỗi lần 0.05)
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from features import LandmarkStream
from model import load_model, pick_device, predict_last

HERE = Path(__file__).resolve().parent
CLASSES = ["paper", "rock", "scissors"]                              # nhãn 0, 1, 2 theo thứ tự chữ cái
COLORS = {0: (60, 60, 255), 1: (255, 150, 40), 2: (60, 210, 60)}     # BGR: paper đỏ, rock xanh dương, scissors xanh lá
GRAY, YELLOW, BLACK = (190, 190, 190), (150, 255, 255), (0, 0, 0)
BONES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
         (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]


# ---------- giao diện (giống 08_WebcamWide: chữ đen trên nền màu) ----------
def put_label(img, text, x, y, bg, scale=0.8):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(img, (x, y - h - base - 6), (x + w + 10, y + 2), bg, -1)
    cv2.putText(img, text, (x + 5, y - base), cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 2, cv2.LINE_AA)


def draw_hand(img, lm, color, label, mirror):
    """Vẽ khung xương, khung bao quanh tay và nhãn."""
    h, w = img.shape[:2]
    px, py = (lm[:, 0] * w).astype(int), (lm[:, 1] * h).astype(int)
    if mirror:
        px = w - 1 - px
    for a, b in BONES:
        cv2.line(img, (px[a], py[a]), (px[b], py[b]), color, 3, cv2.LINE_AA)
    for i in range(21):
        cv2.circle(img, (px[i], py[i]), 6, color, -1, cv2.LINE_AA)
        cv2.circle(img, (px[i], py[i]), 6, BLACK, 1, cv2.LINE_AA)
    x0, y0, x1, y1 = px.min() - 15, py.min() - 15, px.max() + 15, py.max() + 15
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 3, cv2.LINE_AA)
    ly = y1 + 34 if (x0 < 220 and y0 < 125) else y0          # tránh đè lên khối FPS/ngưỡng ở góc trên trái
    put_label(img, label, max(x0, 0), min(max(ly, 32), h - 5), color)


def draw_hud(img, fps, thr):
    put_label(img, f"FPS {fps:.0f}", 10, 35, YELLOW)
    put_label(img, f"Nguong >= {thr:.2f}", 10, 75, YELLOW)
    for i, name in enumerate(CLASSES):
        put_label(img, name, img.shape[1] - 125, img.shape[0] - 10 - 40 * (2 - i), COLORS[i])


# ---------- MediaPipe và nguồn video ----------
def make_detector(task_path):
    """MediaPipe HandLandmarker chạy CPU, chế độ VIDEO, 1 tay. Trả về hàm detect(frame_bgr, ts_ms) -> (21,3) hoặc None."""
    import mediapipe as mp
    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(task_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=1))

    def detect(frame_bgr, ts_ms):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = landmarker.detect_for_video(mp.Image(mp.ImageFormat.SRGB, rgb), ts_ms)
        if not res.hand_landmarks:
            return None
        return np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[0]], np.float32)
    return detect


def open_source(args):
    """Trả về (cap, fps_nguồn, là_video)."""
    if args.video:
        cap = cv2.VideoCapture(str(args.video))
        if not cap.isOpened():
            sys.exit(f"Không mở được video: {args.video}")
        return cap, cap.get(cv2.CAP_PROP_FPS) or 30.0, True
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))      # MJPG giúp webcam đạt 30 fps ở 640x480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        sys.exit(f"Không mở được webcam số {args.camera}. Thử --camera 1, và kiểm tra quyền truy cập camera.")
    return cap, 30.0, False


def parse_args():
    ap = argparse.ArgumentParser(description="Demo kéo-búa-bao: dự đoán sớm cử chỉ tay bằng TCN landmark.")
    ap.add_argument("--video", type=Path, help="chạy trên file video thay vì webcam")
    ap.add_argument("--camera", type=int, default=0, help="số thứ tự webcam (mặc định 0)")
    ap.add_argument("--threshold", type=float, default=0.70, help="ngưỡng tin cậy ban đầu (mặc định 0.70)")
    ap.add_argument("--no-mirror", action="store_true", help="không lật gương webcam")
    ap.add_argument("--cpu", action="store_true", help="ép chạy bằng CPU")
    ap.add_argument("--checkpoint", type=Path, default=HERE / "TCN.pt")
    ap.add_argument("--task", type=Path, default=HERE / "hand_landmarker.task")
    return ap.parse_args()


def run(args, detect=None):
    for p in (args.checkpoint, args.task):
        if not p.exists():
            sys.exit(f"Không thấy file: {p}\nXem hướng dẫn trong RUN.md.")
    dev = pick_device(args.cpu)
    net, ck = load_model(args.checkpoint, dev)
    detect = detect or make_detector(args.task)
    cap, src_fps, is_video = open_source(args)
    mirror = (not is_video) and (not args.no_mirror)       # chỉ lật gương với webcam
    print(f"Đã nạp checkpoint (epoch {ck.get('epoch', '?')}) | thiết bị: {dev} | nguồn: {args.video or 'webcam'}")

    stream, thr = LandmarkStream(), args.threshold
    t_start = time.perf_counter()
    fps, last, last_ts, idx = 0.0, t_start, -1, 0
    try:
        while True:
            now = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / src_fps if is_video else now - t_start          # thời điểm của khung (giây)
            last_ts = max(int(t * 1000), last_ts + 1)                 # timestamp của MediaPipe phải luôn tăng
            lm = detect(frame, last_ts)                               # MediaPipe nhận ảnh gốc (không lật)
            stream.push(lm, t)
            out = predict_last(net, stream.window(), dev) if lm is not None else None

            disp = cv2.flip(frame, 1) if mirror else frame            # chỉ lật để hiển thị cho dễ chơi
            fps, last = 0.9 * fps + 0.1 / max(now - last, 1e-6), now
            if out is None:
                put_label(disp, "no hand", 10, 115, GRAY)
            else:
                c, conf = out
                sure = conf >= thr
                draw_hand(disp, lm, COLORS[c] if sure else GRAY,
                          f"{CLASSES[c]} {conf:.0%}" if sure else f"{CLASSES[c]}? {conf:.0%}", mirror)
            draw_hud(disp, fps, thr)
            if disp.shape[1] > 960:                                   # video lớn: thu nhỏ khi hiển thị
                disp = cv2.resize(disp, (960, int(disp.shape[0] * 960 / disp.shape[1])))

            cv2.imshow("RPS demo - task8_step2_distance (q: thoat)", disp)
            wait = max(1, int(1000 / src_fps - (time.perf_counter() - now) * 1000)) if is_video else 1
            k = cv2.waitKey(wait) & 0xFF
            if k == ord("q"):
                break
            if k == ord("["):
                thr = max(0.05, thr - 0.05)
            if k == ord("]"):
                thr = min(0.95, thr + 0.05)
            idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print("Đã đóng.")


if __name__ == "__main__":
    run(parse_args())
