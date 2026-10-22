# Rock-Paper-Scissors Hand Gesture Recognition

## 1. Project Overview

Dự án xây dựng hệ thống nhận diện cử chỉ bàn tay để phân loại ba hành động: Rock, Paper và Scissors. Hệ thống sử dụng camera để thu nhận hình ảnh bàn tay, trích xuất đặc trưng và dự đoán cử chỉ theo thời gian thực.

## 2. Main Features

- Nhận diện cử chỉ bàn tay từ webcam.
- Phân loại Rock, Paper và Scissors.
- Xử lý chuỗi dữ liệu theo thời gian nếu sử dụng mô hình GRU.
- Hiển thị kết quả nhận diện trực tiếp trên màn hình.
- Ghi nhận kết quả và hỗ trợ đánh giá mô hình.

## 3. System Architecture

Camera Input → Hand Landmark Detection → Feature Preprocessing → Gesture Classification → Prediction Filtering → Display Result

## 4. Model Information

- Hand landmark detector: MediaPipe Hands.
- Gesture classification model: [Điền đúng model được sử dụng trong phiên bản demo].
- Input representation: [Điền đặc trưng đầu vào thực tế].
- Output classes: Rock, Paper, Scissors.

Chi tiết xem tại [Model Documentation](docs/MODEL.md).

## 5. Installation

Yêu cầu Python: [Điền phiên bản Python đã kiểm thử].

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 6. Run the Demo

Xem hướng dẫn chi tiết tại [DEMO.md](docs/DEMO.md).

## 7. Project Documentation

- [Demo Guide](docs/DEMO.md)
- [Model Documentation](docs/MODEL.md)
- [System Architecture](docs/ARCHITECTURE.md)

## 8. References

Liệt kê các bài báo, tài liệu chính thức và nguồn kỹ thuật được sử dụng trong quá trình phát triển.

## 9. Limitations

- Kết quả nhận diện phụ thuộc vào ánh sáng, góc camera và cách thực hiện cử chỉ.
- Độ chính xác cần được đánh giá trên dữ liệu kiểm thử phù hợp.
- Các trường hợp cử chỉ ngoài ba lớp mục tiêu cần được kiểm tra riêng.

## 10. License and Contributions

[Điền giấy phép sử dụng và thông tin đóng góp nếu có.]

