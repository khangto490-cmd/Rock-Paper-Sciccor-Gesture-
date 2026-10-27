"""Demo dự đoán sớm kéo / búa / bao từ webcam hoặc file video (model step2 s1, landmark MediaPipe).

Chạy:  python demo.py                    (webcam)
       python demo.py --video clip.mp4   (file video)
Phím:  q thoát | [ giảm ngưỡng | ] tăng ngưỡng (mỗi lần 0.05)
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

from features import FrameBuffer, make_input
from model import CLASSES, load_model

HERE = Path(__file__).resolve().parent
CKPT, TASK = HERE / "TCN.pt", HERE / "hand_landmarker.task"

# ---- giao diện (giống 08_WebcamWide): màu theo lớp, chữ ĐEN trên nền màu ----
COLORS = {0: (60, 60, 255), 1: (255, 150, 40), 2: (60, 210, 60)}   # BGR: paper = đỏ, rock = xanh dương, scissors = xanh lá
GRAY, YELLOW, BLACK = (190, 190, 190), (150, 255, 255), (0, 0, 0)
BONES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
         (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]


def put_label(img, text, x, y, bg, scale=0.8):
    """Chữ đen trên hộp màu; (x, y) là góc dưới-trái của hộp. Không dấu vì OpenCV không vẽ được chữ có dấu."""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(img, (x, y - h - base - 6), (x + w + 10, y + 2), bg, -1)
    cv2.putText(img, text, (x + 5, y - base), cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 2, cv2.LINE_AA)


def draw_hand(img, lm, color, label, mirror):
    """Khung xương + khung bao quanh tay + nhãn cùng màu."""
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
    ly = y1 + 34 if (x0 < 220 and y0 < 125) else y0           # tránh đè lên khối FPS/ngưỡng ở góc trên trái
    put_label(img, label, max(x0, 0), min(max(ly, 32), h - 5), color)


def draw_hud(img, fps, thr):
    """FPS, ngưỡng và chú thích màu của 3 lớp."""
    put_label(img, f"FPS {fps:.0f}", 10, 35, YELLOW)
    put_label(img, f"Nguong >= {thr:.2f}", 10, 75, YELLOW)
    for i, name in enumerate(CLASSES):
        put_label(img, name, img.shape[1] - 125, img.shape[0] - 10 - 40 * (2 - i), COLORS[i])


def main():
    ap = argparse.ArgumentParser(description="Demo dự đoán sớm kéo/búa/bao (step2 s1)")
    ap.add_argument("--video", help="đường dẫn file video (bỏ trống = dùng webcam)")
    ap.add_argument("--camera", type=int, default=0, help="số thứ tự webcam (mặc định 0)")
    ap.add_argument("--threshold", type=float, default=0.60, help="ngưỡng tin cậy ban đầu (mặc định 0.60)")
    ap.add_argument("--no-mirror", action="store_true", help="không lật gương webcam")
    ap.add_argument("--cpu", action="store_true", help="ép chạy bằng CPU")
    args = ap.parse_args()

    for p in (CKPT, TASK):
        if not p.exists():
            sys.exit(f"Không thấy file: {p}")
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    net, epoch = load_model(CKPT, device)
    print(f"Đã nạp {CKPT.name} | epoch {epoch} | chạy trên: {device}")

    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(TASK)),     # MediaPipe luôn chạy CPU
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=1))

    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            sys.exit(f"Không mở được video: {args.video}")
        vfps = cap.get(cv2.CAP_PROP_FPS) or 60
        mirror = False                                        # video không lật gương
    else:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))     # MJPG giúp webcam đạt 30 fps ở 640x480
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480); cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        mirror = not args.no_mirror
        if not cap.isOpened():
            sys.exit("Không mở được webcam (thử --camera 1, hoặc đóng ứng dụng khác đang dùng camera).")

    buf, thr = FrameBuffer(), args.threshold
    t0, last_ts, idx = time.perf_counter(), -1, 0
    fps, last = 0.0, time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.perf_counter()
            t = idx / vfps if args.video else now            # video: thời gian theo số khung; webcam: đồng hồ thật
            ts = max(int((t if args.video else now - t0) * 1000), last_ts + 1)   # timestamp MediaPipe phải luôn tăng
            last_ts, idx = ts, idx + 1

            res = landmarker.detect_for_video(mp.Image(mp.ImageFormat.SRGB, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)
            lm = np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[0]], np.float32) if res.hand_landmarks else None
            buf.push(lm, t)

            out = None
            if lm is not None:
                with torch.no_grad():
                    conf, c = net(make_input(buf.window(), device)).softmax(-1)[0, -1].max(0)
                out = int(c), float(conf)

            disp = cv2.flip(frame, 1) if mirror else frame
            if disp.shape[1] > 960:                           # video lớn: thu nhỏ để vừa màn hình
                disp = cv2.resize(disp, (960, int(disp.shape[0] * 960 / disp.shape[1])))
            fps, last = 0.9 * fps + 0.1 / max(now - last, 1e-6), now
            if out is None:
                put_label(disp, "no hand", 10, 115, GRAY)
            else:
                c, conf = out
                sure = conf >= thr
                draw_hand(disp, lm, COLORS[c] if sure else GRAY,
                          f"{CLASSES[c]} {conf:.0%}" if sure else f"{CLASSES[c]}? {conf:.0%}", mirror)
            draw_hud(disp, fps, thr)

            cv2.imshow("RPS demo - step2 s1 (q: thoat)", disp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("["):
                thr = max(0.05, thr - 0.05)
            if k == ord("]"):
                thr = min(0.95, thr + 0.05)
    finally:
        cap.release(); cv2.destroyAllWindows(); landmarker.close()
    print("Đã đóng.")


if __name__ == "__main__":
    main()
