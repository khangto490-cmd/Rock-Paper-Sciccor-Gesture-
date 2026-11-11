"""Dựng đầu vào cho model: nội suy landmark webcam lên 120 fps và tạo 546 đặc trưng mỗi khung."""
import collections

import numpy as np
import torch

FPS_MODEL = 120   # model được train ở 120 fps
WIN = 128         # số khung gần nhất giữ lại (TCN chỉ nhìn 63 khung gần nhất nên đủ)
N_IN = 546        # 126 gốc + 210 khoảng cách khớp + 210 vận tốc khoảng cách


class FrameBuffer:
    """Giữ WIN khung landmark gần nhất ở 120 fps; NaN = mất tay."""

    def __init__(self):
        self.buf = collections.deque(maxlen=WIN)
        self.reset()

    def reset(self):
        self.buf.clear()
        self.prev, self.prev_t = None, None

    def push(self, lm, t):
        """lm: (21,3) hoặc None nếu mất tay; t: thời điểm (giây). Chèn các khung nội suy tuyến tính (chỉ dùng khung trước)."""
        n = 1 if self.prev_t is None else max(1, round((t - self.prev_t) * FPS_MODEL))
        if lm is None:
            self.buf.extend([np.full((21, 3), np.nan, np.float32)] * n)
        elif self.prev is None:
            self.buf.append(lm)
        else:
            self.buf.extend(self.prev + (lm - self.prev) * (k / n) for k in range(1, n + 1))
        self.prev, self.prev_t = lm, t

    def window(self):
        return np.stack(self.buf)


_PAIRS = {}


def _pairs(device):
    if device not in _PAIRS:                                  # 21 điểm có 21*20/2 = 210 cặp khác nhau
        _PAIRS[device] = torch.triu_indices(21, 21, offset=1, device=device)
    return _PAIRS[device]


def make_input(lm, device):
    """lm: (T,21,3), có thể chứa NaN -> tensor (1,T,546) trên device.

    126 gốc (tọa độ trừ cổ tay, chia cỡ bàn tay, vận tốc) + 210 khoảng cách giữa các khớp + 210 vận tốc khoảng cách.
    Khung mất tay điền 0 (giống lúc train).
    """
    has = ~np.isnan(lm[:, 0, 0])
    q = lm - lm[:, :1]                                        # cổ tay làm gốc
    q = np.nan_to_num(q / np.maximum(np.linalg.norm(q[:, 9], axis=1)[:, None, None], 1e-8)).astype(np.float32)
    q[~has] = 0
    pos, h = torch.from_numpy(q).to(device), torch.from_numpy(has).to(device)
    flat = pos.reshape(len(pos), 63)
    both = (h[1:] & h[:-1])[:, None]                          # chỉ tính vận tốc khi 2 khung liên tiếp đều có tay
    vel = torch.zeros_like(flat)
    vel[1:] = (flat[1:] - flat[:-1]) * both
    i, j = _pairs(device)
    dist = torch.linalg.vector_norm(pos[:, i] - pos[:, j], dim=-1) * h[:, None]
    dvel = torch.zeros_like(dist)
    dvel[1:] = (dist[1:] - dist[:-1]) * both
    return torch.cat([flat, vel, dist, dvel], dim=-1)[None]
