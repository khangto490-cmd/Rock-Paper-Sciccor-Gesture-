# Demo Guide

## 1. Prerequisites

- Windows 10/11 hoặc hệ điều hành đã được kiểm thử.
- Python [phiên bản đã kiểm thử].
- Webcam.
- Các thư viện được liệt kê trong `requirements.txt`.
- Các model/checkpoint cần thiết cho phiên bản demo.

## 2. Clone the Repository

```bash
git clone https://github.com/khangto490-cmd/Rock-Paper-Sciccor-Gesture-.git
cd Rock-Paper-Sciccor-Gesture-
git checkout Gia-Khang
```

## 3. Set Up the Environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Prepare Model Files and Data

Kiểm tra xem các file model cần thiết đã có trong repository hay chưa.

Nếu dữ liệu video hoặc checkpoint không được đưa lên GitHub, hãy cung cấp nguồn tải, hướng dẫn chuẩn bị thư mục và cách đặt tên file tương ứng.

## 5. Run the Application

Chạy script chính của dự án:

```powershell
python <ten_script_demo>.py
```

Thay `<ten_script_demo>.py` bằng tên script thực tế đã được kiểm thử.

## 6. How to Test

1. Cấp quyền truy cập webcam nếu được yêu cầu.
2. Đưa bàn tay vào vùng camera.
3. Thực hiện các cử chỉ Rock, Paper và Scissors.
4. Quan sát nhãn dự đoán và phản hồi của hệ thống.
5. Kiểm tra thêm các cử chỉ ngoài ba lớp mục tiêu để đánh giá khả năng từ chối dự đoán sai, nếu chức năng này đã được triển khai.

## 7. Troubleshooting

- **Không mở được webcam:** kiểm tra quyền truy cập camera và chỉ số camera.
- **Thiếu thư viện:** kích hoạt đúng môi trường rồi cài lại `requirements.txt`.
- **Không tìm thấy model:** kiểm tra đường dẫn checkpoint hoặc file `.pkl`.
- **Không chạy được GPU:** kiểm tra phiên bản PyTorch, driver và CUDA tương thích.

## 8. Demo Evidence

Bổ sung ảnh chụp màn hình hoặc video minh họa quá trình nhận diện, nếu có.
