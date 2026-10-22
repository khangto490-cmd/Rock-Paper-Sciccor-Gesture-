# Rock-Paper-Sciccor-Gesture-

Dự án nhận diện cử chỉ bàn tay **Rock – Paper – Scissors (Kéo – Búa – Bao)** bằng webcam. Hệ thống sử dụng MediaPipe Hands để phát hiện các điểm mốc bàn tay và mô hình TinyGRU để phân loại chuỗi đặc trưng theo thời gian.

> **Trạng thái tài liệu:** Nội dung được soạn dựa trên mã nguồn `realtime_gru_game_v7_2_balanced.py` và `train_gru.py`. Các phiên bản thư viện, nguồn tải dữ liệu/checkpoint và kết quả đánh giá chưa được xác minh sẽ được đánh dấu rõ bên dưới.

## 1. Chức năng chính

- Thu nhận hình ảnh trực tiếp từ webcam.
- Phát hiện một bàn tay và trích xuất 21 landmarks bằng MediaPipe Hands.
- Chuẩn hóa landmarks thành vector đặc trưng 63 chiều cho mỗi frame.
- Sử dụng TinyGRU để phân loại chuỗi đặc trưng thành Rock, Paper hoặc Scissors.
- Kết hợp dự đoán từ nhiều cửa sổ thời gian, ngưỡng confidence và kiểm tra độ ổn định trước khi chốt kết quả.
- Hiển thị kết quả nhận diện và nước đi đối kháng trên cửa sổ demo.

## 2. Luồng xử lý

```text
Webcam
  ↓
MediaPipe Hands — phát hiện 21 landmarks
  ↓
Chuẩn hóa tọa độ bàn tay và tạo vector 63 đặc trưng
  ↓
Buffer chuỗi theo thời gian
  ↓
Nội suy thời gian + chuẩn hóa bằng mean/std từ lúc huấn luyện
  ↓
TinyGRU
  ↓
Kết hợp xác suất từ các cửa sổ chồng lấp
  ↓
Kiểm tra confidence và độ ổn định
  ↓
Hiển thị kết quả trò chơi
```

## 3. Công nghệ

| Thành phần | Công nghệ |
|---|---|
| Camera và xử lý ảnh | OpenCV |
| Phát hiện bàn tay | MediaPipe Hands |
| Trích xuất đặc trưng | 21 landmarks × 3 tọa độ = 63 giá trị/frame |
| Mô hình phân loại | TinyGRU được triển khai bằng PyTorch |
| Lưu metadata | Joblib |
| Huấn luyện và đánh giá | PyTorch, NumPy, scikit-learn |

## 4. Cấu trúc repository

```text
Rock-Paper-Sciccor-Gesture-/
├── README.md
├── requirements.txt
├── realtime_gru_game_v7_2_balanced.py
├── train_gru.py
├── build_gru_dataset.py
├── checkpoints/
│   ├── tiny_gru_rps_best.pth
│   └── tiny_gru_rps_meta.pkl
└── docs/
    ├── DEMO.md
    └── MODEL.md
```

Đây là cấu trúc mục tiêu để tham khảo. Giữ các script hiện có trong repository; không cần xóa các phiên bản thử nghiệm chỉ để khớp cây thư mục minh họa. Hai file trong `checkpoints/` là các artifact mà script demo hiện tại yêu cầu. Nếu chúng không được lưu trong GitHub, hãy tải chúng từ nguồn được nhóm cung cấp và đặt đúng đường dẫn.

## 5. Cài đặt môi trường

Yêu cầu hệ điều hành và phiên bản Python **cần được kiểm thử và xác nhận**. Script demo hiện dùng `cv2.CAP_DSHOW`, vì vậy hướng dẫn hiện tại nhắm đến Windows.

Tạo môi trường ảo tại thư mục gốc repository:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` hiện liệt kê các thư viện trực tiếp mà mã nguồn sử dụng, chưa khóa phiên bản. Sau khi kiểm thử thành công trên một môi trường sạch, hãy ghi lại các phiên bản đã kiểm thử để tăng khả năng tái lập.

PyTorch có thể chạy trên CPU hoặc GPU tùy bản cài đặt và phần cứng. Nếu cần GPU NVIDIA, cài bản PyTorch tương thích theo hướng dẫn chính thức của PyTorch, sau đó xác minh bằng:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 6. Chạy demo

Đảm bảo hai file sau tồn tại:

- `checkpoints/tiny_gru_rps_best.pth`
- `checkpoints/tiny_gru_rps_meta.pkl`

Chạy lệnh từ **thư mục gốc repository**:

```powershell
python realtime_gru_game_v7_2_balanced.py
```

Phím điều khiển:

- `SPACE`: bắt đầu ván mới.
- `ESC`: thoát chương trình.

Hướng dẫn chi tiết và xử lý lỗi: [docs/DEMO.md](docs/DEMO.md).

## 7. Thông tin mô hình

Mô hình phân loại là **TinyGRU**. Đầu vào của mỗi frame là vector 63 đặc trưng từ landmarks đã được chuyển sang hệ tọa độ cục bộ và chuẩn hóa. GRU xử lý chuỗi theo thời gian; lớp tuyến tính cuối tạo logits cho các lớp cử chỉ.

Các tham số `window_size`, `input_size`, `hidden_size`, danh sách nhãn và thống kê `feature_mean`/`feature_std` được đọc từ file metadata. Chi tiết: [docs/MODEL.md](docs/MODEL.md).

## 8. Dữ liệu và model artifacts

Dataset huấn luyện lớn không được yêu cầu để chạy inference, nhưng cần có khi huấn luyện lại mô hình. Dataset được lưu bên ngoài GitHub.

- **Nguồn tải dataset:** https://drive.google.com/drive/folders/1vuAldCAyX72r4UR-zXz7qcQGK3vYupPR?usp=drive_link
- **Nguồn tải checkpoint và metadata nếu không commit lên GitHub:** https://drive.google.com/drive/folders/1sh1_35dJXXE0l2EuNpm3JRI3rZ28qKht?usp=drive_link


## 9. Đánh giá mô hình

Mô hình GRU được đánh giá trên tập kiểm thử gồm 321 mẫu, với ba lớp cử chỉ bàn tay: Rock (nắm tay), Paper (xòe bàn tay) và Scissors (giơ hai ngón tay).

### 9.1. Kết quả tổng thể

| Chỉ số | Kết quả |
|---|---:|
| Test Loss (độ lỗi trên tập kiểm thử) | 0.2567 |
| Accuracy (độ chính xác) | 92.21% |
| Macro F1-score | 92.03% |
| Weighted F1-score | 92.09% |
 
### 9.2. Kết quả theo từng lớp

| Lớp cử chỉ | Precision | Recall | F1-score | Số mẫu |
|---|---:|---:|---:|---:|
| Rock | 91.67% | 97.06% | 94.29% | 102 |
| Paper | 98.78% | 81.00% | 89.01% | 100 |
| Scissors | 88.55% | 97.48% | 92.80% | 119 |

### 9.3. Phân tích kết quả

Mô hình đạt độ chính xác 92.21% trên tập kiểm thử được sử dụng trong lần đánh giá này.

- **Rock:** Đạt Recall 97.06%, cho thấy mô hình nhận diện đúng phần lớn các mẫu Rock trong tập kiểm thử.
- **Paper:** Đạt Precision 98.78% nhưng Recall là 81.00%. Điều này cho thấy các dự đoán Paper thường chính xác, tuy nhiên mô hình vẫn bỏ sót một số mẫu Paper và phân loại chúng thành các lớp khác.
- **Scissors:** Đạt Recall 97.48%, cho thấy mô hình nhận diện đúng phần lớn các mẫu Scissors trong tập kiểm thử.

Kết quả cho thấy mô hình có khả năng phân loại tương đối tốt trên tập dữ liệu được đánh giá. Tuy nhiên, cần tiếp tục kiểm thử với webcam trong các điều kiện thực tế khác nhau để đánh giá độ ổn định khi sử dụng sản phẩm.


## 10. Giới hạn đã biết

- Hệ thống hiện được thiết kế cho một bàn tay và ba lớp Rock, Paper, Scissors.
- Mô hình luôn tạo xác suất trên các lớp đã huấn luyện; ngưỡng confidence giúp yêu cầu thử lại nhưng không tự động tạo ra một lớp `unknown` đã được huấn luyện.
- Kết quả có thể bị ảnh hưởng bởi ánh sáng, góc bàn tay, che khuất, khoảng cách camera và cử chỉ ngoài phân phối dữ liệu huấn luyện.
- Ngưỡng confidence và các tham số ổn định trong demo là cấu hình triển khai; không thay thế cho đánh giá độ chính xác trên dữ liệu độc lập.

## 11. Tài liệu

- [Hướng dẫn chạy demo](docs/DEMO.md)
- [Tài liệu mô hình và huấn luyện](docs/MODEL.md)

## 12. Tài liệu tham khảo

- Zhang et al., *MediaPipe Hands: On-device Real-time Hand Tracking* (2020): https://arxiv.org/abs/2006.10214
- MediaPipe Hand Landmarker documentation: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker
- PyTorch `torch.nn.GRU`: https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html
- OpenCV `VideoCapture`: https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html
- scikit-learn `classification_report`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
- scikit-learn `confusion_matrix`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

## 13. Đóng góp và giấy phép

- Thành viên/đóng góp: To Tran Gia Khang - Doan Minh Sang - Ly Nam Binh
- License của mã nguồn: Không thêm license

Không nên ghi một license nếu nhóm chưa xác định quyền sử dụng mã nguồn và dữ liệu.
