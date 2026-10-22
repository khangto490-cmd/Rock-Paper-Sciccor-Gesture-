# Model Documentation

## 1. Model Summary

| Component | Description |
|---|---|
| Hand landmark detection | MediaPipe Hands |
| Gesture classification | [Tên model phân loại thực tế] |
| Task | Phân loại Rock, Paper, Scissors |
| Inference mode | Real-time webcam |
| Input features | [Mô tả đặc trưng thực tế] |
| Output classes | Rock, Paper, Scissors |

## 2. Model Architecture

Mô tả luồng xử lý từ ảnh webcam, phát hiện bàn tay, trích xuất đặc trưng, tiền xử lý dữ liệu và phân loại cử chỉ.

Nếu sử dụng GRU, cần mô tả cách biểu diễn chuỗi đặc trưng theo thời gian và cấu hình GRU thực tế.

Nếu sử dụng Random Forest, cần mô tả vector đặc trưng đầu vào, số lượng cây, tiêu chí phân chia và các siêu tham số đã sử dụng.

## 3. Training Procedure

- Dataset source: [Nguồn dữ liệu].
- Number of classes: 3.
- Data preprocessing: [Phương pháp thực tế].
- Train/validation/test split: [Tỷ lệ hoặc cách chia thực tế].
- Training configuration: [Các tham số đã dùng].

## 4. Evaluation

Bổ sung các kết quả đo được trên tập kiểm thử:

- Accuracy.
- Precision, Recall và F1-score.
- Confusion matrix.
- Latency hoặc FPS nếu đã đo.

Không điền số liệu chưa được đo kiểm chứng.

## 5. Model Artifacts

Liệt kê các file model cần thiết để chạy demo, vai trò của từng file và vị trí lưu trữ.

## 6. Limitations

Mô tả các trường hợp nhận diện sai, điều kiện ánh sáng, góc bàn tay và những giới hạn đã quan sát trong quá trình thử nghiệm.

## 7. References

Liệt kê tài liệu chính thức, bài báo khoa học và nguồn tham khảo liên quan đến các thành phần được sử dụng.
