# Chạy demo_v2: game Oẳn Tù Tì chơi với AI (luật A → B)

Demo v2 là bản chơi game thật bằng webcam, dựa trên notebook `Data_321/22_GameSkeleton_AB.ipynb`. Khác `demo/` (chỉ test nhận diện: khung xương, nhãn, độ tin cậy), demo v2 **có luật chơi**: AI đoán sớm cử chỉ của bạn và ra quân khắc chế. Không cần tải dataset, không cần GPU.

## Luật chơi
1. Bấm **SPACE**, rồi **nắm tay thành BÚA** và đưa vào ô **A** (bên trái). Giữ búa khoảng 0,3 giây, vòng xanh chạy đầy thì bắt đầu đi.
2. Đưa tay theo đường nối từ **A** sang ô **B** (bên phải). Lệch ít thì cho qua; lệch xa, đường đỏ lên và quá 0,5 giây thì quay về A. Mất tay quá 0,3 giây cũng quay về A.
3. Từ A tới **vạch** (gần A) tay phải còn là búa. Đổi tay trước vạch thì có cảnh báo và phải làm lại từ A. Chạm vạch rồi thì ra **kéo, búa hoặc bao** tùy ý.
4. Tay chạm ô **B**: AI chốt đáp án ngay (không random, không sửa sau khi chốt) và ra quân khắc chế.
5. Đợi 0,2 giây, hệ thống đọc lại cử chỉ thật của bạn bằng luật ngón tay (độc lập với model) và so với **cử chỉ lúc chốt** (hình tay đa số trong khoảng 0,1 giây tới khung chốt), không so với dự đoán của AI. Giống nhau thì hiện kết quả, chấm thắng, thua hoặc hòa bình thường, kể cả khi AI đoán sai. Khác nhau là **đổi tay**: hủy ván, không tính điểm, không hiện đáp án AI. Mất tay suốt 0,2 giây thì "không xác định", không tính điểm.

Phím: `SPACE` chơi / chơi tiếp / chơi lại, `r` reset điểm, `q` thoát (hoặc đóng cửa sổ).

## Cách chạy
Demo v2 dùng chung `TCN.pt`, `hand_landmarker.task` và `requirements.txt` với thư mục `demo/`, nên **để `demo_v2/` nằm cạnh `demo/`** (đúng cấu trúc repo).

```
cd Sang/demo_v2
python --version                  # nên là Python 3.13
python -m venv .venv
# kích hoạt .venv (Windows: .venv\Scripts\activate | macOS, Linux: source .venv/bin/activate)
python -m pip install -r requirements.txt   # dùng "python -m pip", không dùng "pip" trần
python demo_v2.py
```

Tùy chọn:
- `--cam 1`: đổi camera (mặc định 0). Nếu báo không mở được webcam thì thử `--cam 1` và tắt các chương trình khác đang dùng camera.
- `--no-sound`: tắt âm thanh (âm thanh chỉ có trên Windows, máy khác tự chạy im lặng).
- `--debug`: hiện nhãn model trên hình (chỉ để kiểm tra, **làm lộ đáp án của AI**).

Ghi chú:
- Lần đầu mở sẽ chờ nạp model vài giây. Dòng "Model chạy trên: cpu" (hoặc cuda) cho biết đang dùng CPU hay GPU.
- Cửa sổ game hiển thị FPS. Máy yếu thì FPS thấp hơn nhưng vẫn chơi được.
- Terminal in nhật ký mỗi ván: lúc AI chốt (đáp án, thời gian từ lúc bắt đầu đi) và lúc ra kết quả hoặc hủy ván. Khi thoát, terminal in tổng kết: số ván hợp lệ, đổi tay, không xác định, tỷ lệ hủy, độ phủ và độ chính xác của AI lúc chốt (so dự đoán với cử chỉ lúc chốt đọc bằng luật ngón tay).
- Ngồi chỗ sáng, để bàn tay vào khung hình rõ. Ô A, B, vạch và đường nối được vẽ ngay trên hình webcam đã lật gương.

## Các file
| File | Việc |
|---|---|
| `demo_v2.py` | cửa sổ game và vòng lặp webcam |
| `game_logic.py` | luật chơi (lớp `GameAB`), luật đọc hình tay, không dính webcam hay model |
| `detector.py` | MediaPipe và model TCN `step2 s1` (CPU hoặc GPU) |
| `test_game_logic.py` | 17 nhóm phép thử luật bằng dữ liệu giả, chạy `python test_game_logic.py` (chỉ cần numpy) |
| `requirements.txt` | trỏ tới `../demo/requirements.txt` |

Muốn chỉnh luật (vị trí vạch, thời gian giữ búa...) thì sửa dòng `g = GameAB()` trong `demo_v2.py`, ví dụ `GameAB(gate_f=0.3, hold_a=0.4)`.
