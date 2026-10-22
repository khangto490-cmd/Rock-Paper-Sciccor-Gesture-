# Model Documentation

## 1. Tổng quan

Phiên bản demo dự kiến sử dụng **TinyGRU** để phân loại chuỗi đặc trưng bàn tay thành ba lớp: `rock`, `paper`, `scissors`.

Trong triển khai hiện tại, MediaPipe Hands thực hiện phát hiện bàn tay và cung cấp 21 landmarks. Mỗi landmark có tọa độ `x`, `y`, `z`. Các điểm được chuyển sang hệ tọa độ cục bộ, chuẩn hóa theo kích thước lòng bàn tay và làm phẳng thành 63 đặc trưng cho mỗi frame.

## 2. Kiến trúc mô hình

```text
Webcam frame
    ↓
MediaPipe Hands (21 landmarks)
    ↓
Local-coordinate transform
- trừ vị trí cổ tay
- xây dựng hệ trục bàn tay
- chuẩn hóa theo khoảng cách cổ tay–khớp giữa
    ↓
Flatten: 21 × 3 = 63 features/frame
    ↓
Temporal buffer and temporal resampling
    ↓
Feature normalization using training mean/std
    ↓
TinyGRU
    ↓
Dropout layer (training configuration)
    ↓
Linear classifier
    ↓
Logits for 3 gesture classes
```

### Cấu trúc TinyGRU

Mã nguồn `train_gru.py` và script inference định nghĩa mô hình với:

- GRU một tầng (`num_layers=1`).
- `batch_first=True`, nên tensor đầu vào có dạng `(batch, time, feature)`.
- Kích thước đầu vào mặc định: 63 đặc trưng/frame.
- Kích thước hidden mặc định trong mã nguồn: 32.
- Dropout được cấu hình ở mức 0.20 trong script huấn luyện.
- Lớp tuyến tính cuối ánh xạ hidden state cuối cùng sang số lớp phân loại.

Khi chạy inference, `input_size`, `hidden_size`, `window_size` và nhãn được đọc từ metadata. Vì vậy, metadata phải khớp với checkpoint; không nên thay các giá trị này độc lập.

## 3. Tiền xử lý dữ liệu

### 3.1. Trích xuất landmarks

Mỗi frame sử dụng 21 landmarks của bàn tay, với ba tọa độ cho mỗi điểm. Tổng số đặc trưng là:

`21 landmarks × 3 coordinates = 63 features/frame`.

### 3.2. Chuẩn hóa hình học

Mã nguồn:

1. Lấy cổ tay làm gốc tọa độ.
2. Xây dựng hệ trục cục bộ từ các khớp bàn tay.
3. Chuyển landmarks sang hệ trục đó.
4. Chuẩn hóa theo khoảng cách từ cổ tay đến khớp gốc ngón giữa.
5. Làm phẳng ma trận landmarks thành vector 63 chiều.

### 3.3. Chuẩn hóa đặc trưng

`train_gru.py` tính `feature_mean` và `feature_std` từ tập train, sau đó dùng cùng thống kê để chuẩn hóa train, validation và test. Thống kê này được lưu trong checkpoint và metadata để inference sử dụng lại.

### 3.4. Chuẩn hóa độ dài chuỗi

Ở inference, chuỗi được nội suy theo thời gian về `window_size` lấy từ metadata. Giá trị `window_size` chính thức của checkpoint cần đọc từ file metadata hoặc log khi chạy demo; không nên ghi một con số cố định khi chưa xác minh artifact đang dùng.

## 4. Dữ liệu và chia tập

Script huấn luyện đọc `gru_dataset.npz` với các trường:

- `X`: chuỗi đặc trưng.
- `y`: nhãn dạng số.
- `clip_ids`: định danh clip, dùng để chia nhóm.
- `persons`: thông tin người tham gia.
- `label_names`: tên các lớp.

Script chia dữ liệu theo `clip_ids` bằng `GroupShuffleSplit`, với mục tiêu xấp xỉ:

- Train: 70%.
- Validation: 15%.
- Test: 15%.

Tỷ lệ thực tế và số lượng mẫu từng tập cần lấy từ log huấn luyện. Do chia theo nhóm clip, số mẫu cuối cùng có thể không đúng chính xác các tỷ lệ mục tiêu.

- Nguồn dataset: **[CẦN BỔ SUNG]**
- Số clip và số chuỗi: **[CẦN LẤY TỪ LOG/DATASET]**
- Phân bố lớp: **[CẦN LẤY TỪ LOG]**
- Điều kiện cấp phép/chia sẻ dữ liệu: **[CẦN XÁC MINH]**

Dataset lớn được lưu ngoài GitHub. Cần bổ sung link tải, hướng dẫn truy cập, cấu trúc thư mục và checksum nếu nhóm có cung cấp.

## 5. Cấu hình huấn luyện trong `train_gru.py`

Các giá trị được khai báo trong mã nguồn hiện có:

| Tham số | Giá trị trong mã nguồn |
|---|---:|
| Seed | 42 |
| Batch size | 32 |
| Số epoch tối đa | 100 |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Hidden size | 32 |
| Dropout | 0.20 |
| Early-stopping patience | 15 |
| Optimizer | AdamW |
| Loss | Cross-entropy có trọng số lớp |
| Gradient clipping | max norm = 1.0 |
| Tiêu chí lưu model tốt nhất | Validation loss |

Đây là cấu hình khai báo trong script huấn luyện, không phải khẳng định rằng mọi epoch đều đã chạy đủ. Early stopping có thể kết thúc trước số epoch tối đa.

## 6. Suy luận thời gian thực và ổn định dự đoán

`realtime_gru_game_v7_2_balanced.py` thực hiện thêm một số bước sau model:

- Buffer thu thập 7 frame đặc trưng khi đến giai đoạn nhận diện.
- Tạo ba cửa sổ chồng lấp, mỗi cửa sổ lấy 5 frame.
- Nội suy mỗi cửa sổ về `window_size` lấy từ metadata.
- Chạy inference theo batch cho ba cửa sổ.
- Kết hợp xác suất theo trọng số `[0.20, 0.30, 0.50]`, ưu tiên cửa sổ mới nhất.
- Chỉ chốt dự đoán nếu confidence đạt ngưỡng `0.68` và điều kiện thời gian/độ ổn định được đáp ứng; nếu không, ứng dụng yêu cầu thử lại.

Các giá trị trên là cấu hình của bản demo hiện tại. Chúng không phải là kết quả đánh giá độ chính xác và cần được hiệu chỉnh bằng kiểm thử thực tế.

## 7. Artifact cần cho inference

| File | Vai trò |
|---|---|
| `checkpoints/tiny_gru_rps_best.pth` | Trọng số mô hình đã chọn theo validation loss và các trường checkpoint được lưu trong script huấn luyện |
| `checkpoints/tiny_gru_rps_meta.pkl` | Metadata gồm nhãn, `window_size`, `input_size`, `hidden_size`, `feature_mean`, `feature_std` |

Cần bảo đảm hai file được tạo từ cùng một lần huấn luyện và tương thích với script demo.

Nguồn tải artifact nếu không commit lên GitHub: **[CẦN BỔ SUNG LINK]**.

**Lưu ý an toàn:** `torch.load(..., weights_only=False)` và `joblib.load` có thể nạp dữ liệu tuần tự hóa. Chỉ sử dụng các artifact lấy từ nguồn đáng tin cậy.

## 8. Kết quả đánh giá

Không điền số liệu khi chưa xác minh log của lần huấn luyện được sử dụng cho checkpoint hiện tại.

| Chỉ số | Kết quả |
|---|---|
| Test loss | [CẦN BỔ SUNG TỪ LOG] |
| Test accuracy | [CẦN BỔ SUNG TỪ LOG] |
| Precision / Recall / F1-score từng lớp | [CẦN BỔ SUNG TỪ CLASSIFICATION REPORT] |
| Confusion matrix | [CẦN BỔ SUNG TỪ LOG] |
| FPS/latency trên thiết bị mục tiêu | [CẦN ĐO THỰC TẾ] |

Khi báo cáo kết quả, cần ghi rõ phiên bản checkpoint, dataset/split được đánh giá, phần cứng, môi trường phần mềm và phương pháp đo. Không dùng accuracy trên train thay cho kết quả test.

## 9. Giới hạn và hướng cải thiện

- Model được huấn luyện cho ba lớp mục tiêu. Cử chỉ ngoài các lớp này vẫn có thể bị gán vào lớp gần nhất.
- Ngưỡng confidence không thay thế một lớp `unknown` hoặc dữ liệu negative được huấn luyện và đánh giá riêng.
- Cần kiểm thử với người tham gia chưa xuất hiện trong train, các góc tay, ánh sáng và khoảng cách camera khác nhau.
- Cần đánh giá confusion matrix và false-positive cases trước khi kết luận rằng các chỉnh sửa về smoothing/threshold cải thiện mô hình.

## 10. Tài liệu tham khảo

- Zhang et al., *MediaPipe Hands: On-device Real-time Hand Tracking* (2020): https://arxiv.org/abs/2006.10214
- MediaPipe Hand Landmarker documentation: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker
- PyTorch `torch.nn.GRU`: https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html
- scikit-learn `GroupShuffleSplit`: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html
- scikit-learn `classification_report`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
- scikit-learn `confusion_matrix`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
