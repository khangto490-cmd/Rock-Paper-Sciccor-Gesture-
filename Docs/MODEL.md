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

### 8.1. Thiết lập đánh giá

Mô hình GRU được đánh giá trên tập kiểm thử gồm 321 mẫu thuộc ba lớp cử chỉ: Rock, Paper và Scissors.

Các chỉ số được sử dụng gồm Test Loss, Accuracy, Precision, Recall, F1-score và Confusion Matrix. Mục đích là đánh giá khả năng phân loại của mô hình và xác định những cặp cử chỉ thường bị nhận diện nhầm.

### 8.2. Kết quả tổng thể

| Chỉ số | Kết quả |
|---|---:|
| Số mẫu kiểm thử | 321 |
| Test Loss (class-weighted) | 0.2567 |
| Accuracy | 92.21% |
| Macro F1-score | 92.03% |
| Weighted F1-score | 92.09% |

Accuracy đạt 92.21%, tương ứng với 296/321 mẫu được phân loại đúng trên tập kiểm thử.

### 8.3. Kết quả theo từng lớp

| Lớp | Precision | Recall | F1-score | Số mẫu |
|---|---:|---:|---:|---:|
| Rock | 91.67% | 97.06% | 94.29% | 102 |
| Paper | 98.78% | 81.00% | 89.01% | 100 |
| Scissors | 88.55% | 97.48% | 92.80% | 119 |
| **Macro Average** | **93.00%** | **91.85%** | **92.03%** | **321** |
| **Weighted Average** | **92.21%** | **92.21%** | **92.09%** | **321** |

### 8.4. Confusion Matrix

Bảng dưới đây thể hiện số lượng mẫu theo nhãn thực tế và nhãn dự đoán. Hàng là nhãn thực tế, cột là nhãn được mô hình dự đoán.

| Nhãn thực tế \\ Dự đoán | Rock | Paper | Scissors |
|---|---:|---:|---:|
| Rock | 99 | 0 | 3 |
| Paper | 7 | 81 | 12 |
| Scissors | 2 | 1 | 116 |

Các phần tử trên đường chéo chính là số mẫu được phân loại đúng. Những phần tử ngoài đường chéo thể hiện các trường hợp phân loại sai.

### 8.5. Phân tích kết quả

- **Rock:** Mô hình phân loại đúng 99/102 mẫu, đạt Recall 97.06%. Có 3 mẫu Rock bị dự đoán thành Scissors.
- **Paper:** Mô hình phân loại đúng 81/100 mẫu, đạt Recall 81.00%. Có 7 mẫu Paper bị dự đoán thành Rock và 12 mẫu bị dự đoán thành Scissors.
- **Scissors:** Mô hình phân loại đúng 116/119 mẫu, đạt Recall 97.48%. Có 2 mẫu bị dự đoán thành Rock và 1 mẫu bị dự đoán thành Paper.

Paper là lớp cần ưu tiên cải thiện vì có Recall thấp nhất trong ba lớp. Cần kiểm tra các mẫu bị phân loại sai để xác định nguyên nhân, chẳng hạn đặc điểm landmark, sự khác biệt giữa người thực hiện cử chỉ hoặc chất lượng dữ liệu.

Các kết quả cần được đối chiếu với phiên bản dataset, checkpoint và quy trình chia tập thực tế trước khi sử dụng làm kết quả báo cáo chính thức.

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
