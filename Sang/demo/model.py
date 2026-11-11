"""Kiến trúc TCN (nhân quả) và hàm nạp checkpoint step2 s1."""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from features import N_IN

CLASSES = ["paper", "rock", "scissors"]   # nhãn 0, 1, 2 theo thứ tự chữ cái


class Block(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.pad = 2 * d
        self.conv = nn.Conv1d(c, c, 3, dilation=d)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        return x + self.drop(F.relu(self.conv(F.pad(x, (self.pad, 0)))))


class TCN(nn.Module):
    def __init__(self, in_features=N_IN, c=64):
        super().__init__()
        self.inp = nn.Linear(in_features, c)
        self.blocks = nn.Sequential(*[Block(c, d) for d in (1, 2, 4, 8, 16)])
        self.out = nn.Linear(c, 3)

    def forward(self, x):                 # x: (batch, thời gian, đặc trưng) -> (batch, thời gian, 3)
        return self.out(self.blocks(self.inp(x).transpose(1, 2)).transpose(1, 2))


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    n_in = ck["w"]["inp.weight"].shape[1]
    if n_in != N_IN:
        sys.exit(f"Checkpoint cần {n_in} đặc trưng, nhưng demo dùng {N_IN}. Sai checkpoint, cần TCN.pt của step2.")
    net = TCN().to(device)
    net.load_state_dict(ck["w"])
    net.eval()
    with torch.no_grad():
        net(torch.zeros(1, 128, N_IN, device=device))      # khởi động để khung đầu không bị giật
    return net, ck.get("epoch", "?")
