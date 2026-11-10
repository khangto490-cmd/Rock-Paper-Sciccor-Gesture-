"""Model step2 s1 + MediaPipe (CPU): nhận một khung webcam, trả landmark bàn tay và xác suất (paper, rock, scissors).
Giữ nguyên cách tính của cell 2 và 3 trong 21_GameSkeleton_v2 (546 đặc trưng, nội suy lên 120 fps).
Bộ đệm 80 khung thay vì 128: TCN này (5 khối, kernel 3, dilation 1-2-4-8-16) chỉ nhìn lại 63 khung,
nên đầu ra của khung cuối GIỐNG HỆT khi dùng 80 hay 128 khung, chỉ tính nhanh hơn."""
import time, collections
from pathlib import Path
import numpy as np, torch, cv2
import torch.nn as nn, torch.nn.functional as F
import mediapipe as mp

BASE = Path.home() / "Downloads" / "Data_321"
TASK = BASE / "hand_landmarker.task"
CKPT = BASE / "checkpoints_task9" / "step2" / "seed1" / "TCN.pt"      # step2 s1
FPS_MODEL, WIN = 120, 80                                              # WIN phải >= 64 (tầm nhìn của TCN là 63 khung)


class Block(nn.Module):
    def __init__(self, c, d):
        super().__init__(); self.pad = 2 * d
        self.conv = nn.Conv1d(c, c, 3, dilation=d); self.drop = nn.Dropout(0.1)

    def forward(self, x):
        return x + self.drop(F.relu(self.conv(F.pad(x, (self.pad, 0)))))


class TCN(nn.Module):
    def __init__(self, in_features=126, c=64):
        super().__init__(); self.inp = nn.Linear(in_features, c)
        self.blocks = nn.Sequential(*[Block(c, d) for d in (1, 2, 4, 8, 16)])
        self.out = nn.Linear(c, 3)

    def forward(self, x):
        return self.out(self.blocks(self.inp(x).transpose(1, 2)).transpose(1, 2))


class Detector:
    def __init__(self):
        assert torch.cuda.is_available(), "Chưa bật CUDA; không chạy trên CPU."
        for p in (TASK, CKPT):
            assert p.exists(), f"Không thấy: {p}"
        self.dev = "cuda"
        self.pi, self.pj = torch.triu_indices(21, 21, offset=1, device=self.dev)      # 210 cặp khớp
        self.net = TCN(in_features=546).to(self.dev)
        ck = torch.load(CKPT, map_location=self.dev, weights_only=False)
        assert ck.get("cfg", {}).get("n_in") == 546, f"Checkpoint không phải họ step2: {ck.get('cfg')}"
        self.net.load_state_dict(ck["w"]); self.net.eval()
        with torch.no_grad():
            self.net(torch.zeros(1, WIN, 546, device=self.dev))                       # làm nóng GPU
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(TASK)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=1))
        self.t0, self.last_ts = time.perf_counter(), -1
        self.buf = collections.deque(maxlen=WIN)
        self.prev_lm = self.prev_t = None
        self.ms = (0.0, 0.0)                                                          # (MediaPipe, nội suy + model) của khung vừa xử lý, đơn vị ms

    def close(self):
        self.landmarker.close()

    # ---------- đặc trưng ----------
    def _features_from_pos(self, pos, hand_mask):
        """pos (b,T,21,3) -> (b,T,546): 126 gốc + 210 khoảng cách + 210 vận tốc khoảng cách."""
        pos = torch.where(hand_mask[:, :, None, None], pos, torch.zeros_like(pos))
        b, T = pos.shape[:2]
        flat = pos.reshape(b, T, 63)
        pv = (hand_mask[:, 1:] & hand_mask[:, :-1])[:, :, None]
        vel = torch.zeros_like(flat); vel[:, 1:] = (flat[:, 1:] - flat[:, :-1]) * pv
        dist = torch.linalg.vector_norm(pos[:, :, self.pi] - pos[:, :, self.pj], dim=-1) * hand_mask[:, :, None]
        dvel = torch.zeros_like(dist); dvel[:, 1:] = (dist[:, 1:] - dist[:, :-1]) * pv
        return torch.cat([flat, vel, dist, dvel], dim=-1)

    def _geo_from_base(self, X, hand_mask):
        return self._features_from_pos(X[..., :63].reshape(X.shape[0], X.shape[1], 21, 3), hand_mask)

    @staticmethod
    def _base_features(lm):
        has = ~np.isnan(lm[:, 0, 0])
        q = lm - lm[:, :1]
        q = np.nan_to_num(q / np.maximum(np.linalg.norm(q[:, 9], axis=1)[:, None, None], 1e-8)).astype(np.float32)
        q[~has] = 0
        pos = q.reshape(len(q), 63)
        vel = np.zeros_like(pos)
        if len(pos) > 1:
            vel[1:] = (pos[1:] - pos[:-1]) * (has[1:] & has[:-1])[:, None]
        return np.hstack([pos, vel]).astype(np.float32), has

    # ---------- webcam -> landmark -> xác suất ----------
    def _detect(self, frame_bgr, now):
        ts = max(int((now - self.t0) * 1000), self.last_ts + 1)                       # timestamp MediaPipe phải luôn tăng
        self.last_ts = ts
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.landmarker.detect_for_video(mp.Image(mp.ImageFormat.SRGB, rgb), ts)
        if not res.hand_landmarks:
            return None
        return np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[0]], np.float32)

    def _push(self, lm, t):
        prev, pt = self.prev_lm, self.prev_t
        n = 1 if pt is None else max(1, round((t - pt) * FPS_MODEL))                  # nội suy tuyến tính lên 120 fps
        if lm is None:
            self.buf.extend([np.full((21, 3), np.nan, np.float32)] * n)
        elif prev is None:
            self.buf.append(lm)
        else:
            self.buf.extend(prev + (lm - prev) * (k / n) for k in range(1, n + 1))
        self.prev_lm, self.prev_t = lm, t

    def _probs(self):
        """Xác suất (paper, rock, scissors) của khung cuối."""
        lm = np.stack(self.buf)
        base, has = self._base_features(lm)
        xb, mb = torch.from_numpy(base)[None].to(self.dev), torch.from_numpy(has)[None].to(self.dev)
        with torch.no_grad():
            return self.net(self._geo_from_base(xb, mb)).softmax(-1)[0, -1].cpu().numpy()

    def process(self, frame_bgr, now):
        """Trả (lm, probs): lm (21,3) hoặc None nếu không thấy tay; probs (3,) hoặc None."""
        a = time.perf_counter()
        lm = self._detect(frame_bgr, now)
        b = time.perf_counter()
        self._push(lm, now)
        probs = self._probs() if lm is not None else None
        c = time.perf_counter()
        self.ms = (1000 * (b - a), 1000 * (c - b))
        return lm, probs
