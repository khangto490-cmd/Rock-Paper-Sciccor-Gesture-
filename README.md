# Rock-Paper-Scissors Gesture: phần của Sang

Dự đoán sớm cử chỉ kéo / búa / bao khi tay người chơi đang ra, từ landmark bàn tay (MediaPipe), để máy khắc chế lại.

**Kết quả chính:** TCN landmark `wide45_lownoise`, đạt tiêu chí 80% trên 33 video test, demo webcam chạy tốt.
Chi tiết từng notebook, thử nghiệm và lý do chọn/bỏ: xem [`NOTES.md`](NOTES.md).

## Cấu trúc

```
Sang/
├── README.md
├── NOTES.md              <- mỗi notebook làm gì, thử nghiệm gì, kết quả
├── requirements.txt
└── Data_321/
    ├── 01_...ipynb ... 15_...ipynb
    ├── manifest.csv
    ├── hand_landmarker.task
    └── checkpoints_v2/
```

## Lưu ý khi chạy

- Notebook được làm và chạy trên máy cá nhân (Windows, GPU NVIDIA), đọc dữ liệu video tại máy, **không chạy trực tiếp trên GitHub**. Output đã chạy được giữ trong notebook.
- Dataset không có trong repo vì nặng (hơn 340 MB). Tải tại: [điền link Drive], giải nén vào `Data_321/`.
- Demo webcam (`08_WebcamWide.ipynb`) chỉ cần checkpoint và `hand_landmarker.task` có sẵn trong repo, không cần dataset.

## Cài đặt

```
pip install -r requirements.txt
```
