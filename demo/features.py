"""Đặc trưng cho TCN task8_step2_distance (546 số mỗi khung) và bộ đệm landmark từ camera.

Công thức giống 13_Task8_Step2.ipynb (webcam_base_features + features_from_pos):
  126 số gốc (63 tọa độ trừ cổ tay, chia cỡ bàn tay + 63 vận tốc)
  + 210 khoảng cách giữa 21 khớp + 210 vận tốc của khoảng cách = 546.
"""
import collections
import numpy as np

N_JOINTS = 21
PAIR_I, PAIR_J = np.triu_indices(N_JOINTS, k=1)   # 210 cặp khớp khác nhau
N_DIST = len(PAIR_I)
N_FEATURES = 126 + 2 * N_DIST                      # 546


def make_features(lm):
    """lm: (T,21,3), khung mất tay là NaN -> (T,546) float32."""
    has = ~np.isnan(lm[:, 0, 0])
    q = lm - lm[:, :1]                                         # cổ tay làm gốc
    scale = np.linalg.norm(q[:, 9], axis=1)[:, None, None]
    q = np.nan_to_num(q / np.maximum(scale, 1e-8)).astype(np.float32)   # chuẩn hóa theo cỡ bàn tay
    q[~has] = 0
    pos = q.reshape(len(q), 63)
    both = (has[1:] & has[:-1])[:, None]                       # vận tốc chỉ tính khi 2 khung liền kề đều có tay
    vel = np.zeros_like(pos)
    vel[1:] = (pos[1:] - pos[:-1]) * both
    dist = np.linalg.norm(q[:, PAIR_I] - q[:, PAIR_J], axis=-1) * has[:, None]
    dist_vel = np.zeros_like(dist)
    dist_vel[1:] = (dist[1:] - dist[:-1]) * both
    return np.hstack([pos, vel, dist, dist_vel]).astype(np.float32)


class LandmarkStream:
    """Nhận landmark từ camera (khoảng 30 fps), nội suy tuyến tính lên 120 fps (model train ở 120 fps)
    và giữ 128 khung gần nhất. Chỉ dùng khung trước và khung hiện tại, không nhìn trước."""

    def __init__(self, fps_model=120, win=128):
        self.fps_model = fps_model
        self.buf = collections.deque(maxlen=win)
        self.prev_lm, self.prev_t = None, None

    def reset(self):
        self.buf.clear()
        self.prev_lm, self.prev_t = None, None

    def push(self, lm, t):
        """lm: (21,3) hoặc None nếu mất tay; t: thời điểm của khung, tính bằng giây."""
        prev, pt = self.prev_lm, self.prev_t
        n = 1 if pt is None else max(1, round((t - pt) * self.fps_model))
        if lm is None:
            self.buf.extend([np.full((N_JOINTS, 3), np.nan, np.float32)] * n)
        elif prev is None:
            self.buf.append(lm)
        else:
            self.buf.extend(prev + (lm - prev) * (k / n) for k in range(1, n + 1))
        self.prev_lm, self.prev_t = lm, t

    def window(self):
        return np.stack(self.buf)
