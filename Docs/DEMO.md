# Demo Guide — V7.2 Balanced

Tài liệu này hướng dẫn cài đặt và chạy `realtime_gru_game_v7_2_balanced.py` trên Windows.

## 1. Điều kiện trước khi chạy

- Python: **[CẦN XÁC NHẬN PHIÊN BẢN ĐÃ KIỂM THỬ]**
- Webcam được Windows nhận diện.
- Các thư viện trong `requirements.txt`.
- Hai artifact của mô hình:
  - `checkpoints/tiny_gru_rps_best.pth`
  - `checkpoints/tiny_gru_rps_meta.pkl`

Demo không cần tải dataset huấn luyện nếu hai artifact trên đã có. Dataset chỉ cần cho quá trình xây dựng dataset hoặc huấn luyện lại.

## 2. Clone repository

Mở PowerShell hoặc terminal của VS Code:

```powershell
git clone --branch Gia-Khang --single-branch https://github.com/khangto490-cmd/Rock-Paper-Sciccor-Gesture-.git
cd Rock-Paper-Sciccor-Gesture-
```

Nếu repository hoặc tên nhánh đã thay đổi, hãy cập nhật lệnh theo địa chỉ thực tế trên GitHub.

## 3. Tạo môi trường Python

Tại thư mục gốc repository:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu PowerShell chặn kích hoạt môi trường, có thể dùng terminal Command Prompt:

```bat
.venv\Scripts\activate.bat
```

Không cài các gói vào một Python khác với Python đang dùng để chạy demo. Kiểm tra đường dẫn Python bằng:

```powershell
python --version
python -m pip --version
```

### PyTorch và GPU

Mã nguồn tự chọn `cuda` nếu PyTorch báo GPU CUDA khả dụng; nếu không, chương trình chạy trên CPU. Nếu cần GPU NVIDIA, hãy cài bản PyTorch phù hợp với hệ điều hành, Python và driver từ trang chính thức: https://pytorch.org/get-started/locally/

Kiểm tra trạng thái:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Phiên bản Python, PyTorch, MediaPipe và các thư viện cần được kiểm thử cùng nhau trước khi công bố môi trường được hỗ trợ.

## 4. Đặt checkpoint và metadata

Script hiện đọc các đường dẫn tương đối từ thư mục làm việc:

```text
checkpoints/
├── tiny_gru_rps_best.pth
└── tiny_gru_rps_meta.pkl
```

Nếu nhóm lưu các file này ngoài GitHub:

1. Tải chúng từ nguồn do nhóm cung cấp.
2. Tạo thư mục `checkpoints` tại thư mục gốc repository nếu chưa có.
3. Đặt đúng tên file như trên.
4. Không đổi tên hoặc dùng metadata từ một model khác.

Nguồn tải artifact: **[CẦN BỔ SUNG LINK VÀ QUYỀN TRUY CẬP]**.

Chỉ tải và nạp checkpoint/metadata từ nguồn đáng tin cậy. File `.pth` và `.pkl` có thể chứa dữ liệu tuần tự hóa; không nạp file không rõ nguồn gốc.

## 5. Chạy chương trình

Đảm bảo terminal đang ở thư mục gốc, nơi có `realtime_gru_game_v7_2_balanced.py`, sau đó chạy:

```powershell
python realtime_gru_game_v7_2_balanced.py
```

Ứng dụng mở webcam mặc định tại index `0`, với cấu hình khung hình yêu cầu 640 × 480 và 30 FPS. Đây là giá trị yêu cầu của chương trình; camera có thể không hỗ trợ đầy đủ các thông số này.

## 6. Cách sử dụng

1. Ngồi trước webcam, bảo đảm bàn tay nằm trong khung hình và đủ sáng.
2. Nhấn `SPACE` để bắt đầu ván mới.
3. Khi chương trình hiển thị các giai đoạn “OAN”, “TU” và “TI”, đưa ra cử chỉ cuối cùng và giữ tay đủ rõ, ổn định trong giai đoạn nhận diện.
4. Chương trình hiển thị lớp dự đoán, confidence và xác suất của các lớp.
5. Nếu xuất hiện `TRY AGAIN`, hãy đưa tay vào khung hình rõ hơn và giữ cử chỉ cuối ổn định hơn rồi nhấn `SPACE` để chơi lại.
6. Nhấn `SPACE` để bắt đầu ván tiếp theo; nhấn `ESC` để thoát.

Logic hiện tại chọn nước đi CPU khắc chế cử chỉ mà mô hình dự đoán. Vì vậy, màn hình kết quả hiện ghi `CPU WINS`; đây là hành vi được lập trình trong demo, không phải chỉ số đánh giá mô hình.

## 7. Kiểm tra sau khi cài đặt

- [ ] `python --version` hiển thị phiên bản đã được nhóm kiểm thử.
- [ ] Các import trong `requirements.txt` chạy không lỗi.
- [ ] Hai artifact có mặt đúng vị trí.
- [ ] Chương trình mở được webcam.
- [ ] MediaPipe vẽ được landmarks trên bàn tay.
- [ ] Đã thử Rock, Paper và Scissors trong nhiều điều kiện ánh sáng.
- [ ] Đã kiểm tra các cử chỉ ngoài ba lớp mục tiêu và ghi nhận lỗi nhận diện.
- [ ] Đã lưu thông tin môi trường và kết quả kiểm thử để tái lập.

## 8. Xử lý lỗi thường gặp

### `FileNotFoundError` với checkpoint hoặc metadata

Kiểm tra rằng terminal đang ở thư mục gốc và cả hai file sau tồn tại:

- `checkpoints/tiny_gru_rps_best.pth`
- `checkpoints/tiny_gru_rps_meta.pkl`

### `ModuleNotFoundError`

Kích hoạt đúng môi trường `.venv`, sau đó chạy:

```powershell
python -m pip install -r requirements.txt
```

### Không mở được camera

- Đóng các ứng dụng khác đang sử dụng webcam.
- Kiểm tra quyền truy cập camera trong Windows.
- Thử kiểm tra webcam bằng ứng dụng Camera của Windows.
- Mã hiện dùng camera index `0` và backend `cv2.CAP_DSHOW`; nếu máy dùng camera index khác, cần sửa mã nguồn tương ứng.

### MediaPipe không có `mp.solutions`

Đây có thể là vấn đề tương thích phiên bản MediaPipe với API mà script đang dùng. Kiểm tra phiên bản đã cài và xác định một phiên bản tương thích bằng thử nghiệm trong môi trường sạch; sau đó khóa phiên bản đã xác minh trong `requirements.txt`. Không tự đoán phiên bản.

### PyTorch không nhận GPU

Demo vẫn có thể dùng CPU nếu bản PyTorch hiện tại không hỗ trợ CUDA. Nếu cần GPU, kiểm tra bản PyTorch, driver NVIDIA và hướng dẫn cài đặt chính thức.

### Nhận diện sai hoặc báo `TRY AGAIN`

- Đảm bảo bàn tay không bị che khuất.
- Cải thiện ánh sáng và giữ bàn tay trong vùng camera.
- Tránh chuyển cử chỉ trong giai đoạn “TI”.
- Xác minh metadata đi kèm đúng checkpoint.
- Ghi nhận lỗi trên tập kiểm thử; không điều chỉnh ngưỡng chỉ để che khuất lỗi phân loại mà chưa đo tác động.

## 9. Ảnh/video minh họa

Bổ sung ảnh chụp màn hình hoặc video demo tại đây sau khi chạy thử thành công.

- Ảnh minh họa: `[CẦN BỔ SUNG]`
- Video demo: `[CẦN BỔ SUNG LINK NẾU CÓ]`
