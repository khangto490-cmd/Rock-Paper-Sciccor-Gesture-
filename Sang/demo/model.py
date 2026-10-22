"""TCN task8_step2_distance: kiến trúc, nạp checkpoint và dự đoán khung cuối."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from features import N_FEATURES, make_features


class Block(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.pad = 2 * d
        self.conv = nn.Conv1d(c, c, 3, dilation=d)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        return x + self.drop(F.relu(self.conv(F.pad(x, (self.pad, 0)))))


class TCN(nn.Module):
    def __init__(self, in_features=126, c=64):
        super().__init__()
        self.inp = nn.Linear(in_features, c)
        self.blocks = nn.Sequential(*[Block(c, d) for d in (1, 2, 4, 8, 16)])
        self.out = nn.Linear(c, 3)

    def forward(self, x):
        return self.out(self.blocks(self.inp(x).transpose(1, 2)).transpose(1, 2))


def pick_device(force_cpu=False):
    """Dùng GPU NVIDIA nếu có, không thì CPU (TCN nhỏ nên CPU vẫn chạy được)."""
    return "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"


def load_model(path, device):
    """Nạp checkpoint, trả về (net, checkpoint). Checkpoint do chính repo này cung cấp."""
    try:
        ck = torch.load(path, map_location=device, weights_only=True)
    except Exception:                       # checkpoint lưu thêm cấu hình/lịch sử nên có thể cần nạp đầy đủ
        ck = torch.load(path, map_location=device, weights_only=False)
    n_in = ck.get("cfg", {}).get("in_features", N_FEATURES)
    if n_in != N_FEATURES:
        raise ValueError(f"Checkpoint cần {n_in} đặc trưng, demo tạo {N_FEATURES}. Sai checkpoint?")
    net = TCN(in_features=n_in).to(device)
    net.load_state_dict(ck["w"])
    net.eval()
    with torch.inference_mode():
        net(torch.zeros(1, 128, N_FEATURES, device=device))   # khởi động trước để khung đầu không bị giật
    return net, ck


def predict_last(net, lm, device):
    """lm: (T,21,3) các khung gần nhất (NaN = mất tay). Trả về (lớp, độ tin cậy) của khung cuối, hoặc None nếu mất tay."""
    if np.isnan(lm[-1, 0, 0]):
        return None
    x = torch.from_numpy(make_features(lm))[None].to(device)
    with torch.inference_mode():
        probs = net(x).softmax(-1)[0, -1]
    conf, cls = probs.max(0)
    return int(cls), float(conf)
