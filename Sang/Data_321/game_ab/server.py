"""Server game oẳn tù tì luật A -> B. Chạy:  python server.py [số camera]   rồi mở http://127.0.0.1:8000

  GET  /          giao diện web (thư mục web/)
  GET  /video     luồng hình webcam (MJPEG, đã lật gương, chỉ có khung xương tay; ô A, B, vạch do web vẽ đè lên)
                  Ở RESULT và CANCEL hình đứng yên trên khung lúc tay chạm B.
  GET  /state     trạng thái game dạng JSON (xem GameAB.snapshot, thêm status, error, has_hand, hand, fps)
  POST /cmd/play  như bấm SPACE (chơi / chơi tiếp / chơi lại)
  POST /cmd/reset như bấm r (reset điểm)
"""
import asyncio, queue, sys, threading, time
from contextlib import asynccontextmanager
from pathlib import Path
import cv2
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from detector import Detector
from game_logic import GameAB, CLASSES, hand_shape, hand_point

CAM_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0          # không mở được thì thử: python server.py 1
MIRROR = True                                                  # lật gương hình webcam (giao diện KHÔNG lật thêm)
HAND_MODE = "palm"                                             # điểm đại diện của tay: "palm" (giữa lòng bàn tay) hoặc "box" (tâm ô bao quanh tay)
DEBUG = False                                                  # True: hiện nhãn model trên hình (chỉ để dev, làm LỘ đáp án AI)
TIMING = False                                                 # True: in thời gian từng khâu mỗi 60 khung
WEB = Path(__file__).parent / "web"
assert WEB.exists(), f"Không thấy thư mục giao diện: {WEB}"

BONES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
         (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
CYAN = (255, 240, 0)                                           # BGR của neon cyan; một màu cố định, không đổi theo model

lock = threading.Lock()
cmds = queue.SimpleQueue()                                     # lệnh từ giao diện, chỉ luồng webcam được đụng vào Game
stop = threading.Event()
shared = {"jpeg": None, "state": {"status": "loading", "error": None}}


def draw_hand(img, lm):
    h, w = img.shape[:2]
    px, py = (lm[:, 0] * w).astype(int), (lm[:, 1] * h).astype(int)
    if MIRROR:
        px = w - 1 - px
    for a, b in BONES:
        cv2.line(img, (px[a], py[a]), (px[b], py[b]), CYAN, 2, cv2.LINE_AA)
    for x, y in zip(px, py):
        cv2.circle(img, (int(x), int(y)), 3, CYAN, -1, cv2.LINE_AA)


class Grabber(threading.Thread):
    """Đọc camera liên tục ở luồng riêng và CHỈ giữ khung mới nhất.
    Lý do: lúc có tay, xử lý (~40 ms) chậm hơn camera (~33 ms) nên khung xếp hàng trong bộ đệm camera và hình bị trễ dồn.
    Ở đây khung cũ bị bỏ, lần nào xử lý cũng lấy khung mới nhất."""

    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap, self.cv, self.halt = cap, threading.Condition(), threading.Event()
        self.frame, self.t, self.seq, self.err = None, 0.0, 0, None

    def run(self):
        while not (stop.is_set() or self.halt.is_set()):
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


def worker():
    """Luồng xử lý: lấy khung mới nhất -> MediaPipe + model -> luật GameAB -> cập nhật hình và trạng thái cho giao diện."""
    cap = det = grab = None
    try:
        det = Detector()
        cap = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30); cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(f"Không mở được webcam {CAM_ID} (đang có chương trình khác dùng camera? Thử: python server.py 1)")
        grab = Grabber(cap); grab.start()
        g = GameAB()                                            # đổi tham số luật ở đây, ví dụ GameAB(gate_f=0.3, hold_a=0.4)
        fps, last, seq, n_dbg, evidence = 0.0, time.perf_counter(), 0, 0, None
        print("Sẵn sàng. Mở http://127.0.0.1:8000 để chơi (Ctrl+C để tắt).")
        while not stop.is_set():
            ta = time.perf_counter()
            frame, now, seq = grab.get(seq)                     # now = mốc lúc camera chụp khung này
            if frame is None:
                continue
            tb = time.perf_counter()
            lm, pr = det.process(frame, now)
            tc = time.perf_counter()
            sh = int(hand_shape(lm)) if lm is not None else None
            pt = hand_point(lm, MIRROR, HAND_MODE) if lm is not None else None
            before = g.state
            g.step(now, pt, pr, sh)
            if g.locked_now:                                    # nhật ký kiểm chứng: đáp án AI + mốc chốt, ghi ngay lúc chốt, trước khi người chơi đổi tay
                r = g.history[-1]
                print(f"[ván {r['round']}] AI chốt {r['ai']} (đoán bạn ra {r['guess']}) sau {r['tu_A_den_chot_s']:.2f}s kể từ lúc bắt đầu đi")
            if before == "CONFIRM" and g.state in ("RESULT", "CANCEL"):
                r = g.history[-1]
                print(f"[ván {r['round']}] tay sau xác nhận: {r['player']} -> {g.state}"
                      f"{'' if g.state == 'CANCEL' else ' ' + str(r['outcome'])} | điểm {g.score} | đổi tay {g.stats['switch']} | không xác định {g.stats['unknown']}")
            while not cmds.empty():
                c = cmds.get()
                if c == "play":
                    g.start(now)
                elif c == "reset":
                    g.reset_score()
            if g.state not in ("RESULT", "CANCEL"):
                evidence = None

            disp = cv2.flip(frame, 1) if MIRROR else frame
            if lm is not None:
                draw_hand(disp, lm)
                if DEBUG:
                    cv2.putText(disp, f"model: {CLASSES[int(pr.argmax())]} {pr.max():.0%}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)
            fps, last = 0.9 * fps + 0.1 / max(now - last, 1e-6), now
            _, jpg = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 80])
            td = time.perf_counter()
            data = jpg.tobytes()
            if g.locked_now:
                evidence = data                                 # khung lúc tay chạm B
            if g.state in ("RESULT", "CANCEL") and evidence is not None:
                data = evidence                                 # kết quả hoặc huỷ ván đứng yên trên khung lúc chạm B cho đến khi bấm SPACE
            snap = g.snapshot()
            frozen = g.state in ("RESULT", "CANCEL")
            hand = None if (pt is None or frozen) else {"x": round(float(pt[0]), 4), "y": round(float(pt[1]), 4)}
            snap.update(status="ready", error=None, has_hand=lm is not None, hand=hand, fps=round(fps, 1))
            with lock:
                shared["jpeg"], shared["state"] = data, snap
            if TIMING:
                n_dbg += 1
                if n_dbg % 60 == 0:
                    mp_ms, model_ms = det.ms
                    print(f"fps {fps:.1f} | chờ khung {1000*(tb-ta):.0f} | MediaPipe {mp_ms:.0f} | model {model_ms:.0f} "
                          f"| nén JPEG {1000*(td-tc):.0f} | TRỄ chụp->gửi {1000*(td-now):.0f} ms | có tay: {lm is not None}")
    except Exception as e:                                      # lỗi nào cũng báo cho giao diện thay vì chết im lặng
        print("LỖI:", e)
        with lock:
            shared["state"] = {"status": "error", "error": str(e)}
    finally:
        if grab is not None:
            grab.halt.set(); grab.join(timeout=1)
        if cap is not None:
            cap.release()
        if det is not None:
            det.close()


@asynccontextmanager
async def lifespan(app):
    th = threading.Thread(target=worker, daemon=True)
    th.start()
    yield
    stop.set()
    th.join(timeout=3)


app = FastAPI(lifespan=lifespan)


@app.get("/state")
def get_state():
    with lock:
        return JSONResponse(dict(shared["state"]), headers={"Cache-Control": "no-store"})


@app.post("/cmd/{name}")
def post_cmd(name: str):
    if name not in ("play", "reset"):
        raise HTTPException(404, "Không có lệnh này")
    cmds.put(name)
    return {"ok": True}


async def mjpeg():
    last = None
    while True:
        with lock:
            jpg = shared["jpeg"]
        if jpg is not None and jpg is not last:
            last = jpg
            yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
        await asyncio.sleep(0.004)


@app.get("/video")
async def video():
    return StreamingResponse(mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})


app.mount("/", StaticFiles(directory=WEB, html=True), name="web")      # phải để cuối cùng


if __name__ == "__main__":
    print(f"Đang nạp model và mở camera {CAM_ID}...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
