# Hướng dẫn chạy demo

Demo dự đoán sớm cử chỉ **kéo / búa / bao** từ webcam (hoặc file video), dùng model TCN **`step2 s1`** (Task 9: họ `step2` seed 1, train trên data cũ + Dataset_bonus) trên landmark bàn tay của MediaPipe. Không cần tải dataset.

## 1. Yêu cầu

- Python 3.13 (bản đã chạy thử: 3.13.5). Kiểm tra bằng `python --version`.
- Webcam, hoặc một file video quay bàn tay.
- GPU NVIDIA không bắt buộc. Không có GPU thì demo tự chạy bằng CPU.
- Thư mục `demo/` phải có đủ: `demo.py`, `model.py`, `features.py`, `requirements.txt`, `TCN.pt` (checkpoint `step2 s1`) và `hand_landmarker.task` (model MediaPipe).

## 2. Cài đặt (làm một lần)

Mở terminal (Windows: **PowerShell**), đi vào thư mục `demo`, kiểm tra phiên bản Python rồi tạo môi trường ảo:

```
cd đường-dẫn-tới-repo/Sang/demo
python --version
python -m venv .venv
```

Nếu `python --version` không ra Python 3.13 (ví dụ máy có Anaconda), tạo venv bằng đúng bản 3.13: Windows dùng `py -3.13 -m venv .venv`, Mac/Linux dùng `python3.13 -m venv .venv`.

Kích hoạt môi trường ảo:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
  (nếu báo lỗi "running scripts is disabled", chạy `Set-ExecutionPolicy -Scope Process Bypass` rồi kích hoạt lại)
- Windows CMD: `.venv\Scripts\activate.bat`
- Mac / Linux: `source .venv/bin/activate`

Đầu dòng lệnh hiện `(.venv)` là đã kích hoạt. Kiểm tra python đang dùng đúng của venv:

```
python -c "import sys; print(sys.executable)"
```

Đường dẫn in ra phải nằm trong thư mục `demo/.venv/`. Nếu không, đừng cài tiếp, tạo lại venv như trên.

Cài thư viện. **Luôn gõ `python -m pip`, không gõ `pip` trần**, để chắc thư viện vào đúng môi trường ảo (máy có Anaconda hoặc nhiều bản Python thì `pip` trần dễ cài nhầm chỗ):

```
python -m pip install -r requirements.txt
python -c "import cv2, mediapipe, torch; print('OK', torch.__version__)"
```

Thấy in `OK ...` là cài xong. Muốn dùng GPU NVIDIA, xem ghi chú trong `requirements.txt` để cài torch bản CUDA.

## 3. Chạy

Vẫn trong thư mục `demo` và đã kích hoạt `(.venv)`:

```
python demo.py                      # dùng webcam
python demo.py --video clip.mp4     # dùng file video
```

Tùy chọn thêm:

| Tùy chọn | Ý nghĩa |
|---|---|
| `--camera 1` | dùng webcam số 1 (mặc định 0) |
| `--threshold 0.8` | đổi ngưỡng tin cậy ban đầu (mặc định 0.60) |
| `--no-mirror` | không lật gương webcam |
| `--cpu` | ép chạy bằng CPU |

Xong thì thoát môi trường ảo bằng lệnh `deactivate`.

## 4. Cách dùng và đọc màn hình

**Phím** (bấm khi đang chọn cửa sổ webcam): `q` thoát | `[` giảm ngưỡng | `]` tăng ngưỡng (mỗi lần 0.05).

**Màn hình:**
- Khung tay và khung xương đổi màu theo lớp: **paper = đỏ, rock = xanh dương, scissors = xanh lá**.
- Nhãn kèm độ tin cậy, ví dụ `rock 93%`.
- Khi độ tin cậy chưa đạt ngưỡng, khung màu xám và nhãn có dấu `?`.
- Góc trên trái: FPS và ngưỡng hiện tại. Góc dưới phải: chú thích màu.
- Không thấy tay: hiện `no hand`.

Model dự đoán trên từng khung hình, nên nhãn có thể đổi trong lúc tay đang ra. Nhìn kết quả lúc tay đã ra gần xong là ổn định nhất.

## 5. Cách hoạt động (tóm tắt)

1. MediaPipe lấy 21 điểm landmark của bàn tay từ mỗi khung hình.
2. Landmark webcam khoảng 30 fps được nội suy lên 120 fps (model được train ở 120 fps), giữ 128 khung gần nhất.
3. Mỗi khung dựng 546 đặc trưng: 126 số gốc (tọa độ trừ cổ tay, chia cỡ bàn tay, vận tốc) + 210 khoảng cách giữa các khớp + 210 vận tốc của khoảng cách. Khoảng cách không đổi khi xoay bàn tay nên model ít phụ thuộc góc nhìn.
4. TCN chỉ nhìn các khung trước, trả về xác suất kéo/búa/bao của khung hiện tại.

## 6. Lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'` (hoặc `mediapipe`, `torch`) | Thư viện bị cài nhầm môi trường. Kích hoạt `(.venv)`, rồi chạy lại `python -m pip install -r requirements.txt` (không dùng `pip` trần) |
| `Không thấy file: ...TCN.pt` hoặc `...hand_landmarker.task` | Hai file này phải nằm trong thư mục `demo/` |
| `Không mở được webcam` | Thử `--camera 1`; đóng ứng dụng khác đang dùng camera; Mac: cấp quyền camera cho Terminal |
| `Checkpoint cần ... đặc trưng` | Sai checkpoint, cần `TCN.pt` của `step2` (546 đặc trưng) |
| FPS thấp | Đóng bớt ứng dụng nặng; máy có GPU NVIDIA thì cài torch bản CUDA |
| Nhận sai khi tay quá nghiêng (khoảng 45° trở lên) | Giới hạn đã biết của model, xem `NOTES.md` |
| Lỗi liên quan GPU của MediaPipe | Không ảnh hưởng: MediaPipe luôn chạy CPU trong demo này |
