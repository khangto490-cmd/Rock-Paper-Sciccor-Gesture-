# Nhận diện sớm cử chỉ Kéo – Búa – Bao (Rock-Paper-Scissors)

Hệ thống dự đoán cử chỉ **kéo / búa / bao** của người chơi ngay **trong lúc tay đang ra**, từ webcam hoặc file video, để máy có thể ra cử chỉ khắc chế.

> Thư mục này (`Sang/`, nhánh `Sang`) nằm trong repo `Rock-Paper-Sciccor-Gesture-`.
> Chi tiết quá trình làm, thử nghiệm và kết luận: xem [`NOTES.md`](NOTES.md). Hướng dẫn chạy demo: xem [`demo/RUN.md`](demo/RUN.md).

## 1. Tóm tắt

- **Cách làm:** MediaPipe lấy 21 điểm landmark của bàn tay ở mỗi khung hình → dựng đặc trưng → mô hình **TCN** (chỉ nhìn các khung trước) trả về xác suất kéo/búa/bao cho khung hiện tại.
- **Mô hình đang dùng:** `task8_step2_distance` (TCN, 546 đặc trưng/khung). Lý do chọn: xem `NOTES.md`, mục 5.
- **Kết quả trên tập test (33 video)** với xoay giả lập ở nhiều góc:

| Góc xoay | 0° | 15° | 30° | 45° | 60° |
|---|---|---|---|---|---|
| Độ chính xác mọi khung có tay (%) | 93.2 | 92.8 | 92.9 | 92.8 | 92.3 |
| Độ phủ ở ngưỡng tin cậy 0.7 (%) | 91.8 | 91.7 | 91.7 | 92.1 | 91.8 |

  Bảng đầy đủ và so sánh với các mô hình khác nằm trong `NOTES.md`.
- **Hạn chế chính:** test chỉ có 33 video, cùng nguồn quay với tập train, chưa kiểm chứng với người chơi hay camera mới (xem `NOTES.md`, mục 8).

## 2. Cấu trúc thư mục

```
Sang/
├── README.md             # file này
├── NOTES.md              # nhật ký: đã làm gì, thử nghiệm, kết luận, việc còn lại
├── requirements.txt      # thư viện cho các notebook trong Data_321 (môi trường train)
├── requirements-lock.txt # toàn bộ pip freeze của môi trường train
├── demo/                 # demo webcam / video, CHẠY ĐƯỢC (xem demo/RUN.md)
└── Data_321/             # notebook, manifest.csv, checkpoint trong quá trình nghiên cứu
    ├── 01_PrepareData.ipynb ... 15_Task8_Step3b.ipynb
    ├── manifest.csv
    └── checkpoints_v2/
```

## 3. Chạy demo

Demo không cần tải dataset. Làm theo [`demo/RUN.md`](demo/RUN.md). Tóm tắt:

```
cd Sang/demo
python -m venv .venv
# kích hoạt .venv (xem RUN.md cho từng hệ điều hành)
pip install -r requirements.txt
python demo.py                    # webcam
python demo.py --video clip.mp4   # file video
```

## 4. Lưu ý về thư mục `Data_321/`

Các notebook trong `Data_321/` là **notebook nghiên cứu, chạy riêng trên máy cá nhân** (Windows, GPU NVIDIA RTX 3050 Ti, môi trường `train-gpu`). **Người khác không chạy lại được các notebook này từ repo**, vì chúng đọc dữ liệu và file nằm ngoài repo, trong thư mục `~/Downloads/Data_321` của máy train:

- video gốc và file landmark (`.npz`) của dữ liệu tự thu (không nằm trong repo);
- file `hand_landmarker.task` của MediaPipe;
- GPU có CUDA (notebook dừng nếu không có CUDA, không train bằng CPU).

Vì vậy, để xem kết quả hãy đọc **output đã lưu trong notebook** (ví dụ `13_Task8_Step2.ipynb`) và các bảng tổng hợp trong `NOTES.md`. Phần có thể chạy lại độc lập là `demo/`.

Danh sách và vai trò từng notebook: `NOTES.md`, mục 4.

## 5. Dữ liệu

Toàn bộ dữ liệu là **dữ liệu tự thu** (gồm dữ liệu tự thu của cá nhân và dữ liệu tự thu của nhóm), không dùng dataset công khai. Dữ liệu không nằm trong repo.

| Nguồn | Nội dung | Tốc độ | Dung lượng |
|---|---|---|---|
| `Dataset_hand` | 27 video mỗi lớp (rock / paper / scissors) | 60 fps (nội suy lên 120 fps) | khoảng 133 MB |
| `dataset_labeled/person_01` | paper 48, rock 43, scissors 43 video | 120 fps | khoảng 208 MB |

Nhãn lấy từ tên thư mục. Tổng cộng 215 video, chia **theo video** (không chia theo khung hình): train 151 / val 31 / test 33, chia theo nguồn × lớp, seed 42.

## 6. Thông tin mô hình AI đang dùng

| Mục | Giá trị |
|---|---|
| Tên | `task8_step2_distance` (checkpoint `TCN.pt`) |
| Kiến trúc | TCN nhân quả: Linear(546→64) → 5 khối residual Conv1d (kernel 3, dilation 1, 2, 4, 8, 16, dropout 0.1) → Linear(64→3). Khoảng 97 nghìn tham số |
| Đầu vào mỗi khung | 546 số = 126 (tọa độ landmark trừ cổ tay, chia cỡ bàn tay, + vận tốc) + 210 khoảng cách giữa các cặp khớp + 210 vận tốc của khoảng cách |
| Đầu ra | Xác suất 3 lớp: paper = 0, rock = 1, scissors = 2 |
| Tốc độ chuỗi | 120 fps (webcam khoảng 30 fps được nội suy lên 120 fps) |
| Huấn luyện | CUDA, AdamW lr 1e-3, weight decay 1e-2, batch 16, tối đa 100 epoch, dừng sớm sau 18 epoch val loss không giảm (tốt nhất ở epoch 25, val loss 0.2104) |
| Augmentation | Xoay 3D (±40°, ±45°, ±40° theo 3 trục), lật gương (xác suất 0.5), nhiễu tọa độ (độ lệch chuẩn 0.005) |
| Checkpoint | `Data_321/checkpoints_v2/task8_step2_distance/TCN.pt`; bản dùng cho demo: `demo/TCN.pt` |

## 7. Môi trường

- Máy train: Windows, RTX 3050 Ti Laptop GPU (4 GB VRAM), RAM 15,7 GB. Thư viện chính: `torch 2.6.0+cu124`, `mediapipe 1.1.0`, `numpy 2.5.2`, `opencv-contrib-python 5.0.0.93`. Danh sách đầy đủ: [`requirements.txt`](requirements.txt) và [`requirements-lock.txt`](requirements-lock.txt).
- Demo: Python 3 (đã chạy thử với Python 3.13.5), CPU hoặc GPU. Thư viện của demo: `demo/requirements.txt`.
- MediaPipe cài bằng pip không bật được GPU, nên bước trích landmark và demo luôn chạy MediaPipe bằng CPU.
