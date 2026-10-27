# Nhận diện sớm cử chỉ Kéo – Búa – Bao (Rock-Paper-Scissors)

Hệ thống dự đoán cử chỉ **kéo / búa / bao** của người chơi ngay **trong lúc tay đang ra**, từ webcam hoặc file video, để máy có thể ra cử chỉ khắc chế.

> Thư mục này (`Sang/`, nhánh `Sang`) nằm trong repo `Rock-Paper-Sciccor-Gesture-`.
> Chi tiết quá trình làm, thử nghiệm và kết luận: xem [`NOTES.md`](NOTES.md). Hướng dẫn chạy demo: xem [`demo/RUN.md`](demo/RUN.md).

## 1. Tóm tắt

- **Cách làm:** MediaPipe lấy 21 điểm landmark của bàn tay ở mỗi khung hình → dựng đặc trưng → mô hình **TCN** (chỉ nhìn các khung trước) trả về xác suất kéo/búa/bao cho khung hiện tại.
- **Mô hình đang dùng:** `step2 s1` (TCN, 546 đặc trưng/khung; cấu hình `step2` train với seed 1 trên dữ liệu cũ + `Dataset_bonus`). Ngoài ra giữ thêm `step2 ens4` (ghép 4 seed của cùng cấu hình, lấy trung bình xác suất). Lý do chọn: xem `NOTES.md`, mục 5.6.
- **Kết quả trên tập test (48 video: 33 video cũ + 15 video bonus)** với xoay giả lập ngẫu nhiên ở nhiều góc:

| Góc xoay | 0° | 15° | 30° | 45° | 60° |
|---|---|---|---|---|---|
| Tỉ lệ video có khung vừa đúng vừa tự tin ≥ ngưỡng, trung bình 5 ngưỡng 0.5 → 0.9 (%) | 97.1 | 96.8 | 96.7 | 96.8 | 96.3 |
| Độ chính xác mọi khung có tay (%) | 85.9 | 85.8 | 85.9 | 86.6 | 86.0 |

  Ngoài ra: độ chính xác khi tin cậy ≥ 0.7 là 95.8%, độ phủ ở ngưỡng 0.7 là 81.8%, suy luận khoảng 1 ms mỗi dự đoán. Khi so 3 cấu hình TCN trên cùng test, cấu hình `step2` đạt trung bình 96.0 (chỉ số ở hàng đầu bảng, trung bình mọi góc và 4 seed), hơn `wide45_lownoise` (89.9) và `tcn_v2` (85.1). Bảng đầy đủ nằm trong `NOTES.md`, mục 5.6.

  Lưu ý: tập test đã đổi (trước đây 33 video) và mô hình được train lại với dữ liệu bổ sung, nên các số liệu này **không so trực tiếp** được với các bảng ở những bước trước trong `NOTES.md`.
- **Hạn chế chính:** test chỉ có 48 video (trong đó 15 video bonus), vẫn là dữ liệu tự thu cùng nguồn với tập train, chưa kiểm chứng với người chơi hay camera mới; trên 15 video bonus kết quả thấp hơn rõ so với 33 video cũ (xem `NOTES.md`, mục 5.6 và 8).

### Trạng thái hiện tại và hướng phát triển

Dự án đang ở giai đoạn **nghiên cứu và kiểm thử khả năng nhận diện**. **Chưa có giao diện chơi game**: demo hiện chỉ hiển thị khung xương tay, nhãn kèm độ tin cậy và FPS để kiểm tra mô hình. Các hướng đang cải tiến, tối ưu và nâng cấp:

- **Nhận diện nhanh hơn, đoán sớm hơn:** mô hình dự đoán trên từng khung hình và được đánh giá bằng thời điểm đoán đúng đầu tiên (ms). Đã thử làm mượt kết quả webcam để nhãn ít nhảy trong lúc tay đang ra, nhưng bỏ vì độ chính xác giảm, đoán đúng chậm hơn và webcam không thấy khác biệt.
- **Nhận đúng ở mọi góc tay:** đã cải thiện qua Data Augmentation (xoay 3D, lật gương, nhiễu), các bản góc xoay rộng (`wide45_lownoise`) và đặc trưng khoảng cách giữa các khớp (`task8_step2_distance`). Sau khi thêm dữ liệu (Task 9), cấu hình `step2` gần như không giảm khi xoay giả lập đến 60°. Khi tay nghiêng mạnh (khoảng 45° trở lên) trên webcam thật vẫn còn nhận sai, nên đây vẫn là hướng cải tiến chính.
- **Mở rộng dữ liệu:** đã thêm `Dataset_bonus` (90 video quay bằng điện thoại) và train lại (Task 9). Việc tiếp theo: thêm dữ liệu từ nhiều người chơi và camera khác để kiểm tra khả năng tổng quát.
- **Giao diện chơi (máy ra cử chỉ khắc chế):** chưa làm. Sẽ thực hiện sau khi mô hình nhận diện đủ nhanh và ổn định ở mọi góc.

## 2. Cấu trúc thư mục

```
Sang/
├── README.md             # file này
├── NOTES.md              # nhật ký: đã làm gì, thử nghiệm, kết luận, việc còn lại
├── requirements.txt      # thư viện cho các notebook trong Data_321 (môi trường train)
├── requirements-lock.txt # toàn bộ pip freeze của môi trường train
├── demo/                 # demo webcam / video, CHẠY ĐƯỢC (xem demo/RUN.md)
│   ├── demo.py           # giao diện và vòng lặp webcam / video
│   ├── model.py          # kiến trúc TCN, nạp checkpoint
│   ├── features.py       # nội suy lên 120 fps, dựng 546 đặc trưng
│   ├── requirements.txt  # thư viện của demo
│   ├── RUN.md            # hướng dẫn chạy
│   ├── TCN.pt            # checkpoint step2 s1
│   └── hand_landmarker.task  # model MediaPipe
└── Data_321/             # notebook, bảng danh sách video, checkpoint trong quá trình nghiên cứu
    ├── 01_PrepareData.ipynb ... 19_Task9_Train.ipynb
    ├── manifest.csv          # 215 video đầu (2 nguồn đầu)
    ├── manifest_bonus.csv    # 90 video Dataset_bonus
    ├── manifest_all.csv      # gộp 305 video, có cột chia train/val/test
    ├── checkpoints_v2/
    └── checkpoints_task9/    # step2/seed0..seed3 (TCN.pt), train_summary.csv, compare_task9.csv, final_model.json
```

## 3. Chạy demo

Demo không cần tải dataset. Làm theo [`demo/RUN.md`](demo/RUN.md). Tóm tắt:

```
cd Sang/demo
python --version                  # nên là Python 3.13
python -m venv .venv
# kích hoạt .venv (xem RUN.md cho từng hệ điều hành)
python -m pip install -r requirements.txt   # dùng "python -m pip", không dùng "pip" trần
python demo.py                    # webcam
python demo.py --video clip.mp4   # file video
```

Ngưỡng tin cậy hiển thị mặc định là 0.60, đổi bằng phím `[` `]` hoặc tùy chọn `--threshold`.

## 4. Lưu ý về thư mục `Data_321/`

Các notebook trong `Data_321/` là **notebook nghiên cứu, chạy riêng trên máy cá nhân** (Windows, GPU NVIDIA RTX 3050 Ti, môi trường `train-gpu`). **Người khác không chạy lại được các notebook này từ repo**, vì chúng đọc dữ liệu và file nằm ngoài repo, trong thư mục `~/Downloads/Data_321` của máy train:

- video gốc và file landmark (`.npz`) của dữ liệu tự thu (không nằm trong repo);
- file `hand_landmarker.task` của MediaPipe;
- GPU có CUDA (notebook dừng nếu không có CUDA, không train bằng CPU).

Vì vậy, để xem kết quả hãy đọc **output đã lưu trong notebook** (ví dụ `13_Task8_Step2.ipynb`, `19_Task9_Train.ipynb`) và các bảng tổng hợp trong `NOTES.md`. Phần có thể chạy lại độc lập là `demo/`.

Danh sách và vai trò từng notebook: `NOTES.md`, mục 4.

## 5. Dữ liệu

Toàn bộ dữ liệu là **dữ liệu tự thu** (gồm dữ liệu tự thu của cá nhân và dữ liệu tự thu của nhóm), không dùng dataset công khai. Dữ liệu không nằm trong repo.

| Nguồn | Nội dung | Tốc độ | Dung lượng |
|---|---|---|---|
| `Dataset_hand` | 27 video mỗi lớp (rock / paper / scissors) | 60 fps (nội suy lên 120 fps) | khoảng 133 MB |
| `dataset_labeled/person_01` | paper 48, rock 43, scissors 43 video | 120 fps | khoảng 208 MB |
| `Dataset_bonus` | 90 video (30 mỗi lớp), quay bằng điện thoại | báo 60 fps, một số video đã được làm chậm nên có thể không thật sự đạt 60 fps (nội suy lên 120 fps) | chưa ghi lại |

Nhãn lấy từ tên thư mục. Hai nguồn đầu (215 video) chia **theo video** (không chia theo khung hình): train 151 / val 31 / test 33, chia theo nguồn × lớp, seed 42. `Dataset_bonus` cũng chia theo video, tỉ lệ 70/15/15, seed 42: 63 / 12 / 15. Sau khi gộp: 305 video, train 214 / val 43 / test 48.

## 6. Thông tin mô hình AI đang dùng

| Mục | Giá trị |
|---|---|
| Tên | `step2 s1` (cấu hình `step2` train với seed 1; checkpoint `TCN.pt`) |
| Kiến trúc | TCN nhân quả: Linear(546→64) → 5 khối residual Conv1d (kernel 3, dilation 1, 2, 4, 8, 16, dropout 0.1) → Linear(64→3). Khoảng 97 nghìn tham số |
| Đầu vào mỗi khung | 546 số = 126 (tọa độ landmark trừ cổ tay, chia cỡ bàn tay, + vận tốc) + 210 khoảng cách giữa các cặp khớp + 210 vận tốc của khoảng cách |
| Đầu ra | Xác suất 3 lớp: paper = 0, rock = 1, scissors = 2 |
| Tốc độ chuỗi | 120 fps (webcam khoảng 30 fps được nội suy lên 120 fps) |
| Huấn luyện | CUDA, AdamW lr 1e-3, weight decay 1e-2, batch 16, tối đa 120 epoch, dừng sớm sau 20 epoch val loss không giảm (tốt nhất ở epoch 27, dừng ở epoch 47, val loss 0.3224, độ chính xác val 0.8651 trên 43 video val). Dữ liệu train: 214 video (cũ + `Dataset_bonus`) |
| Augmentation | Xoay 3D (±40°, ±45°, ±40° theo 3 trục), lật gương (xác suất 0.5), nhiễu tọa độ (độ lệch chuẩn 0.005) |
| Checkpoint | `Data_321/checkpoints_task9/step2/seed1/TCN.pt`; bản dùng cho demo: `demo/TCN.pt`. Bản trước Task 9: `Data_321/checkpoints_v2/task8_step2_distance/TCN.pt` |

`step2 ens4` (giữ để nghiên cứu tiếp) dùng 4 checkpoint `seed0` – `seed3` trong `Data_321/checkpoints_task9/step2/`, lấy trung bình xác suất; chậm hơn `step2 s1` khoảng 3 lần (2.9 ms so với 1.0 ms mỗi dự đoán).

## 7. Môi trường

- Máy train: Windows, RTX 3050 Ti Laptop GPU (4 GB VRAM), RAM 15,7 GB. Thư viện chính: `torch 2.6.0+cu124`, `mediapipe 1.1.0`, `numpy 2.5.2`, `opencv-contrib-python 5.0.0.93`. Danh sách đầy đủ: [`requirements.txt`](requirements.txt) và [`requirements-lock.txt`](requirements-lock.txt).
- Demo: Python 3.13 (đã chạy thử với 3.13.5), CPU hoặc GPU. Thư viện của demo: `demo/requirements.txt`, cài bằng `python -m pip install -r requirements.txt`.
- MediaPipe cài bằng pip không bật được GPU, nên bước trích landmark và demo luôn chạy MediaPipe bằng CPU.
