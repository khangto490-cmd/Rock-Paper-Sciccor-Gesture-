# Kéo – Búa – Bao: nhận diện cử chỉ tay theo từng khung hình

Chương trình đọc bàn tay qua webcam, **dự đoán trên từng khung hình** xem tay đang ra kéo, búa hay bao, rồi hiện ngay cử chỉ máy sẽ ra để thắng (kéo → búa, búa → bao, bao → kéo). Nhãn thay đổi liên tục theo hình dạng bàn tay: tay còn nắm thì là búa, khi ngón tay bắt đầu duỗi thì chuyển sang kéo hoặc bao.

Dự án là một thử nghiệm về **nhận diện nhanh** cho bài toán trò chơi với robot, tham khảo hướng của robot Janken (Đại học Tokyo). Chạy hoàn toàn trên CPU với webcam thông thường.

## Cách hoạt động

```
Webcam → MediaPipe Hands (21 mốc bàn tay) → chuẩn hóa (vector 63 chiều)
       → RandomForest (dự đoán từng frame) → kéo / búa / bao → cử chỉ đối kháng
```

1. **MediaPipe Hands** lấy 21 mốc của bàn tay (`model_complexity=0` để nhanh hơn).
2. **Chuẩn hóa** (`features.py`): lấy cổ tay làm gốc tọa độ và chia cho khoảng cách từ cổ tay đến gốc ngón giữa, nên vector đặc trưng không phụ thuộc vị trí tay trong khung hình hay khoảng cách đến camera.
3. **Mô hình RandomForest** dự đoán nhãn của từng khung hình, độc lập với các khung khác.
4. **Tra bảng đối kháng** để hiện cử chỉ của máy.

### Gán nhãn từng frame (điểm quan trọng của dự án)

Dữ liệu được thu theo **chuỗi**: mỗi lần là một động tác từ lúc tay nắm đến lúc ra kéo/búa/bao. Nếu gán mọi frame của chuỗi kéo là "kéo" thì các frame đầu (tay còn nắm) giống hệt frame của chuỗi búa mà nhãn khác nhau, làm model bị nhiễu.

Cách giải quyết: **gán lại nhãn từng frame theo hình dạng tay tại frame đó**, dựa trên độ duỗi ngón tay (khoảng cách từ đầu ngón đến cổ tay, trong không gian đã chuẩn hóa):

| Chuỗi | Quy tắc gán nhãn từng frame |
|---|---|
| Kéo | độ duỗi ngón trỏ + giữa `>= T_KEO` → "kéo", ngược lại → "búa" |
| Bao | độ duỗi cả 4 ngón `>= T_BAO` → "bao", ngược lại → "búa" |
| Búa | luôn là "búa" |

Hai ngưỡng được chọn từ dữ liệu bằng `analyze_extension.py`: vùng "còn nắm" và vùng "đã duỗi rõ" tách biệt hoàn toàn, nên ban đầu đặt ngưỡng ở giữa (kéo ≈ 1.47, bao ≈ 1.34). Sau khi phân tích đánh đổi bằng `tune_threshold.py`, ngưỡng được hạ xuống **T_KEO = `<điền giá trị đang dùng>`, T_BAO = `<điền giá trị đang dùng>`** (khai báo ở đầu `build_frame_dataset.py`).

## Cấu trúc thư mục

| File / thư mục | Vai trò |
|---|---|
| `features.py` | Hàm trích đặc trưng dùng chung cho cả lúc thu dữ liệu và lúc chạy |
| `collect_sequences.py` | Thu dữ liệu: mỗi lần vung tay lưu thành một file `.npy` |
| `sequences/` | Dữ liệu đã thu (`keo/`, `bua/`, `bao/`) |
| `analyze_extension.py` | Phân tích độ duỗi ngón tay để tìm ngưỡng |
| `build_frame_dataset.py` | Gán lại nhãn từng frame, tạo `frame_dataset.csv` |
| `train_static.py` | Huấn luyện và so sánh các model, lưu model tốt nhất vào `static_model.pkl` |
| `realtime_static.py` | Chạy trò chơi theo thời gian thực |
| `static_model.pkl` | Model đã huấn luyện (nếu có trong repo) |
| `requirements.txt` | Danh sách thư viện |

Các công cụ đo đạc đi kèm (nếu có trong repo): `tune_threshold.py` (đánh đổi khi hạ ngưỡng), `measure_transition.py` (độ dài chuyển tiếp từ nắm đến duỗi), `realtime_profile.py` (đo thời gian từng công đoạn), `live_compare.py` (so sánh hai model trên cùng luồng webcam).

## Cài đặt và chạy

### Yêu cầu
- Python 3.9 (đã thử trên Windows; Linux và macOS chưa kiểm tra)
- Webcam
- Anaconda hoặc `venv` (nên dùng môi trường riêng)

### Các bước

**1. Tải mã nguồn**
```bash
git clone <URL-repo>
cd <tên-thư-mục-repo>
```

**2. Tạo môi trường riêng**
```bash
conda create -n rps_env python=3.9
conda activate rps_env
```
Nếu không dùng conda:
```bash
python -m venv .venv
.venv\Scripts\activate
```

**3. Cài thư viện**
```bash
pip install -r requirements.txt
```

**4. Chạy**
```bash
python realtime_static.py
```
Đưa tay vào trước webcam. Cửa sổ hiện nhãn dự đoán kèm độ tin cậy và cử chỉ máy ra. Bấm `Q` để thoát.

> Chạy các lệnh từ bên trong thư mục repo, vì chương trình đọc file bằng đường dẫn tương đối.

### Lưu ý
- **Không cần TensorFlow.** Cài TensorFlow cùng môi trường có thể gây xung đột `numpy`/`protobuf` với MediaPipe.
- Đóng các ứng dụng khác đang dùng webcam (Zoom, trình duyệt...) trước khi chạy.
- Nếu không mở được camera, đổi `cv2.VideoCapture(0)` thành `cv2.VideoCapture(1)`.

## Tái tạo từ đầu (tùy chọn)

Có thể thu dữ liệu và huấn luyện lại model của riêng bạn:

```bash
python collect_sequences.py        # thu dữ liệu
python analyze_extension.py        # (tùy chọn) xem lại ngưỡng độ duỗi
python build_frame_dataset.py      # gán lại nhãn từng frame -> frame_dataset.csv
python train_static.py             # huấn luyện -> static_model.pkl
python realtime_static.py          # chạy thử
```

**Thu dữ liệu (`collect_sequences.py`):** bấm `1` / `2` / `3` để chọn nhãn (kéo / búa / bao), `SPACE` để bắt đầu ghi một lần vung tay và `SPACE` lần nữa để lưu, `Q` để thoát. Mỗi lần bắt đầu từ tay đang nghỉ, vung tay rồi tạo dáng rõ ràng, như khi chơi thật. Chuỗi ngắn hơn 5 frame bị bỏ. Nên thu xen kẽ ba loại và thay đổi tốc độ, góc tay giữa các lần.

Dữ liệu hiện có: 358 chuỗi (120 kéo, 119 búa, 119 bao), tổng cộng 30.818 frame, của một người thu.

## Kết quả đo

Đo trên máy của tác giả (CPU, webcam 640×480).

**Thời gian xử lý mỗi frame** (đo bằng `realtime_profile.py`):

| Công đoạn | Mặc định | `model_complexity=0` |
|---|---|---|
| MediaPipe | ~18,7 ms | ~9,4 ms |
| Vẽ + hiển thị | ~12,8 ms | ~7,6 ms |
| Đọc camera | ~7,9 ms | ~8,3 ms |
| Model RandomForest | ~7,1 ms | ~7,9 ms |
| **Tổng** | **~46,5 ms (~21,5 FPS)** | **~33,3 ms (~30 FPS)** |

Với `model_complexity=0`, tốc độ chạm trần 30 FPS của webcam; tối ưu thêm không tăng được FPS.

**Độ sớm khi đổi nhãn.** Hạ ngưỡng gán nhãn chỉ giúp máy đổi sang kéo/bao sớm hơn khoảng 1 frame, đổi lại một phần frame búa bị nhãn mâu thuẫn (`tune_threshold.py`). Lý do: từ lúc ngón rời dáng nắm đến lúc tay định hình rõ chỉ mất trung vị **3 frame** (35–38% chuỗi chỉ mất 2 frame trở xuống), nên không còn nhiều chỗ để đoán sớm hơn ở 30 FPS (`measure_transition.py`).

**Thử trực tiếp trên webcam** (`live_compare.py`, 26 ván kéo và 24 ván bao): so với lúc tay định hình rõ, model đổi nhãn ở trung vị 0 ms với kéo và 33 ms với bao (khoảng một frame), không ván nào đổi nhãn quá sớm khi tay còn nắm.

> Các số trên đo trên một người, mẫu nhỏ, chỉ nên coi là chỉ báo về xu hướng.

## Hạn chế

- **Dữ liệu của một người**, chưa kiểm tra với người khác, tay khác hoặc điều kiện ánh sáng khác.
- **Trần 30 FPS của webcam.** Khi duỗi ngón nhanh, cả quá trình đổi hình chỉ nằm trong 1–2 khung hình, không model nào đổi sớm hơn được. Muốn nhanh hơn cần camera có tốc độ khung hình cao hơn.
- **Model chỉ nhìn từng frame**, không phân biệt được tay đang duỗi ra hay đang co lại.
- Chia train/test ngẫu nhiên theo frame trong `train_static.py` làm độ chính xác trên tập test **lạc quan hơn thực tế**, vì các frame liền kề của cùng một chuỗi rất giống nhau.
- Dữ liệu thu bằng cấu hình MediaPipe mặc định, còn lúc chạy dùng `model_complexity=0`, nên có thể lệch nhẹ.
- Kết quả hiển thị không làm mượt giữa các frame nên nhãn có thể nhấp nháy khi mốc tay rung.

## Tài liệu tham khảo

- Y. Katsuki, Y. Yamakawa, M. Ishikawa. *High-speed Human/Robot Hand Interaction System.* HRI 2015 Extended Abstracts.
- Ito, Sueishi, Yamakawa, Ishikawa. *Tracking and Recognition of a Human Hand in Dynamic Motion for Janken (rock–paper–scissors) Robot.* IEEE CASE 2016.
- Google MediaPipe Hands: https://developers.google.com/mediapipe


