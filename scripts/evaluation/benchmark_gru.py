
from pathlib import Path
import time

import torch
import torch.nn as nn


class TinyGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.classifier(self.dropout(hidden[-1]))


def load_model(path, device):
    checkpoint = torch.load(
        path, map_location=device, weights_only=False
    )
    model = TinyGRU(
        input_size=int(checkpoint["input_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        num_classes=int(checkpoint["num_classes"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(model, checkpoint, device, batch_size):
    x = torch.zeros(
        batch_size,
        int(checkpoint["window_size"]),
        int(checkpoint["input_size"]),
        device=device,
    )

    with torch.inference_mode():
        for _ in range(100):
            model(x)

        synchronize(device)
        start = time.perf_counter()

        for _ in range(1000):
            model(x)

        synchronize(device)
        elapsed = time.perf_counter() - start

    return elapsed * 1000 / 1000


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Device: CPU")

    h32_path = Path("checkpoints/tiny_gru_rps_best.pth")
    h16_path = Path("checkpoints/tiny_gru_rps_h16_best.pth")

    for path in (h32_path, h16_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {path}")

    h32, c32 = load_model(h32_path, device)
    h16, c16 = load_model(h16_path, device)

    print("\nLatency per model call (milliseconds)")
    print("Batch | H32 | H16 | H16 speed improvement")

    for batch in (1, 3):
        t32 = measure(h32, c32, device, batch)
        t16 = measure(h16, c16, device, batch)

        improvement = (t32 - t16) / t32 * 100 if t32 else 0
        print(
            f"{batch:5d} | {t32:.4f} | {t16:.4f} "
            f"| {improvement:+.2f}%"
        )


if __name__ == "__main__":
    main()
