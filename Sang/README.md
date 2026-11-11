# Nhận diện sớm cử chỉ Kéo – Búa – Bao (Rock-Paper-Scissors)

Hệ thống dự đoán cử chỉ **kéo / búa / bao** của người chơi ngay **trong lúc tay đang ra**, từ webcam hoặc file video, để máy có thể ra cử chỉ khắc chế.

> Thư mục này (`Sang/`, nhánh `Sang`) nằm trong repo `Rock-Paper-Sciccor-Gesture-`.
> Chi tiết quá trình làm, thử nghiệm và kết luận: xem [`NOTES.md`](NOTES.md). **Game hoàn chỉnh (có giao diện, luật chơi): xem [`game_ab/RUN.md`](game_ab/RUN.md).** Bản demo cũ chỉ để kiểm tra mô hình: xem [`demo/RUN.md`](demo/RUN.md).

## 1. Tóm tắt

- **Cách làm:** MediaPipe lấy 21 điểm khớp (landmark) của bàn tay ở mỗi khung hình → dựng đặc trưng → mô hình **TCN** (mạng xử lý chuỗi theo thời gian, chỉ nhìn các khung trước đó) trả về xác suất kéo / búa / bao cho khung hiện tại.
- **Mô hình đang dùng:** cấu hình có thêm đặc trưng khoảng cách giữa các khớp (tên trong repo: `step2`), train với seed 1 trên dữ liệu gốc + `Dataset_bonus` (90 video quay bằng điện thoại). Ngoài ra repo giữ thêm bản ghép 4 mô hình cùng cấu hình (khác seed), lấy trung bình xác suất, để nghiên cứu tiếp. Giải thích tên gọi: mục 6. Lý do chọn: `NOTES.md`, mục 5.6.
- **Kết quả trên tập test (48 video: 33 video cũ + 15 video bonus)**, kiểm tra cả khi xoay bàn tay giả lập ngẫu nhiên ở nhiều góc:

| Góc xoay | 0° | 15° | 30° | 45° | 60° |
|---|---|---|---|---|---|
| Tỉ lệ video có khung vừa đúng vừa tự tin ≥ ngưỡng, trung bình 5 ngưỡng 0.5 → 0.9 (%) | 97.1 | 96.8 | 96.7 | 96.8 | 96.3 |
| Độ chính xác trên mọi khung có tay (%) | 85.9 | 85.8 | 85.9 | 86.6 | 86.0 |

  Cách đọc chỉ số ở hàng đầu: với mỗi ngưỡng tin cậy (0.5, 0.6, 0.7, 0.8, 0.9), đếm số video mà mô hình có ít nhất một khung vừa đoán đúng vừa có độ tin cậy ≥ ngưỡng; lấy tỉ lệ trên tổng số video, rồi trung bình qua 5 ngưỡng.

  Ngoài ra: khi chỉ xét các khung có độ tin cậy ≥ 0.7, độ chính xác là 95.8% và độ phủ (tỉ lệ khung có tay đạt ngưỡng đó) là 81.8% trong `compare_task9.csv`. Báo cáo đếm theo cách khác: ở ngưỡng 0.7 có 18.9% khung không có cử chỉ, tức 81.1% khung có cử chỉ, nên hai số lệch nhau do mẫu số khác nhau. Suy luận khoảng 1 ms mỗi dự đoán (1.0 ms trong `compare_task9.csv`, 1.1 ms trong lần đo của báo cáo). Khi so 3 cấu hình mô hình trên cùng tập test (trung bình mọi góc và 4 seed, cùng chỉ số ở hàng đầu bảng trên): cấu hình có đặc trưng khoảng cách đạt 96.0, cấu hình chỉ xoay góc rộng hơn đạt 89.9, cấu hình cơ bản đạt 85.1. Bảng đầy đủ nằm trong `NOTES.md`, mục 5.6.

  Lưu ý: tập test đã đổi (trước đây 33 video) và mô hình được train lại với dữ liệu bổ sung, nên các số liệu này **không so trực tiếp** được với các bảng ở những bước trước trong `NOTES.md`.
- **Hạn chế chính:**
  - Test chỉ có 48 video (trong đó 15 video bonus), là dữ liệu tự thu cùng nguồn với tập train. Model cuối được chọn sau khi đã xem điểm trên tập test này, nên các số trên có thể lạc quan. Trên 15 video bonus kết quả thấp hơn rõ so với 33 video cũ (xem `NOTES.md`, mục 5.6 và 8).
  - Việc chia dữ liệu theo video chưa tách theo phiên quay: các clip test thuộc cùng nhóm phiên quay với clip train (xem `NOTES.md`, mục 5.7). Bằng chứng với người chơi mới chỉ có một người khác, và kiểm tra đó dùng `tcn_v2`, không phải model cuối.
  - Ở cử chỉ cuối của video, luật ngón tay (không huấn luyện) cho kết quả ngang model: 95.8% ở cả hai (46 trên 48 video). Vì vậy các số này không chứng minh model nhận cử chỉ cuối tốt hơn luật ngón tay. Việc model đoán sớm hơn luật chưa được đo.
  - Model được train trên landmark lấy từ ảnh gốc, và game không dùng bước tiền xử lý ảnh nào. Khi thử offline, không bước tiền xử lý nào cải thiện kết quả.
  - Game mới được đo trong một phiên 50 ván do chính tác giả chơi và biết trước cử chỉ: 47 ván hợp lệ, 3 ván bị hủy, máy đoán đúng cử chỉ theo kế hoạch 50 trên 50 ván. Phiên này không chứng minh việc đoán sớm hay hiệu quả với người chơi khác.

### Giải thích thuật ngữ

| Thuật ngữ | Nghĩa trong dự án này |
|---|---|
| Landmark | 21 điểm khớp của bàn tay (tọa độ x, y, z) mà MediaPipe nhận ra trong mỗi khung hình |
| Khung hình, fps | Một ảnh trong video; fps là số khung hình mỗi giây |
| Nội suy | Chèn thêm khung ở giữa các khung có sẵn để chuỗi đủ 120 khung mỗi giây (video 30 hoặc 60 fps được nâng lên 120 fps) |
| TCN nhân quả | Mạng nơ-ron xử lý chuỗi theo thời gian; "nhân quả" nghĩa là khi đoán một khung chỉ dùng các khung trước nó, không nhìn trước |
| Độ tin cậy, ngưỡng | Xác suất mô hình gán cho nhãn nó chọn; chỉ chấp nhận dự đoán khi độ tin cậy ≥ ngưỡng |
| Độ phủ | Tỉ lệ khung hình mà mô hình đạt độ tin cậy ≥ ngưỡng |
| Data Augmentation | Tạo thêm dữ liệu train từ dữ liệu có sẵn: xoay bàn tay trong không gian 3D, lật gương, thêm nhiễu nhỏ vào tọa độ |
| Seed | Số khởi tạo ngẫu nhiên khi train; cùng cấu hình nhưng khác seed cho các mô hình hơi khác nhau |
| Checkpoint | File lưu trọng số của mô hình đã train (`TCN.pt`) |

### Trạng thái hiện tại và hướng phát triển

Dự án có hai phần chạy được:

- **`game_ab/` (bản hoàn chỉnh, mới):** game chơi trên giao diện web bằng webcam, luật A -> B, máy đoán sớm cử chỉ của người chơi, chốt đáp án và ra cử chỉ khắc chế, có kết quả từng ván và tính điểm (người chơi / máy / hòa). Tự dùng GPU nếu có, không thì chạy CPU. Đã đo trong một phiên 50 ván (xem mục 4.1 và `NOTES.md`, mục 5.7).
- **`demo/` (bản cũ):** chỉ hiển thị khung xương tay, nhãn kèm độ tin cậy và FPS để kiểm tra mô hình. Không có giao diện chơi.

Các hướng đang cải tiến, tối ưu và nâng cấp:

- **Nhận diện nhanh hơn, đoán sớm hơn:** mô hình dự đoán trên từng khung hình và được đánh giá bằng thời điểm đoán đúng đầu tiên (ms). Đã thử làm mượt kết quả webcam để nhãn ít nhảy trong lúc tay đang ra, nhưng bỏ vì độ chính xác giảm, đoán đúng chậm hơn và webcam không thấy khác biệt.
- **Nhận đúng ở mọi góc tay:** đã cải thiện bằng Data Augmentation (xoay 3D, lật gương, nhiễu), bản xoay góc rộng hơn và đặc trưng khoảng cách giữa các khớp (khoảng cách không đổi khi xoay tay). Sau khi thêm dữ liệu bonus, cấu hình có đặc trưng khoảng cách gần như không giảm khi xoay giả lập đến 60°. Khi tay nghiêng mạnh (khoảng 45° trở lên) trên webcam thật vẫn còn nhận sai, nên đây vẫn là hướng cải tiến chính.
- **Mở rộng dữ liệu:** đã thêm `Dataset_bonus` (90 video quay bằng điện thoại) và train lại. Việc tiếp theo: thêm dữ liệu từ nhiều người chơi và camera khác để kiểm tra khả năng tổng quát.
- **Giao diện chơi (máy ra cử chỉ khắc chế):** đã hoàn thành trong `game_ab/` (giao diện web, luật chơi A -> B, tính điểm). Cách chạy: [`game_ab/RUN.md`](game_ab/RUN.md).

## 2. Cấu trúc thư mục

```
Sang/
├── README.md             # file này
├── NOTES.md              # nhật ký: đã làm gì, thử nghiệm, kết luận, việc còn lại
├── requirements.txt      # thư viện cho các notebook trong Data_321 (môi trường train)
├── requirements-lock.txt # danh sách đầy đủ phiên bản thư viện của môi trường train (pip freeze)
├── game_ab/              # GAME HOÀN CHỈNH: giao diện web, luật A -> B (xem game_ab/RUN.md)
│   ├── server.py         # server, mở webcam, nối giao diện web với game
│   ├── detector.py       # MediaPipe + model nhận diện cử chỉ (GPU hoặc CPU)
│   ├── game_logic.py     # luật chơi A -> B
│   ├── test_game_logic.py  # kiểm tra luật chơi
│   ├── requirements.txt  # thư viện của game
│   ├── RUN.md            # hướng dẫn chạy
│   ├── TCN.pt            # checkpoint mô hình đang dùng
│   └── web/              # giao diện web
├── demo/                 # BẢN CŨ: demo webcam / video để kiểm tra mô hình (xem demo/RUN.md)
│   ├── demo.py           # giao diện và vòng lặp webcam / video
│   ├── model.py          # kiến trúc TCN, nạp checkpoint
│   ├── features.py       # nội suy lên 120 fps, dựng 546 đặc trưng
│   ├── requirements.txt  # thư viện của demo
│   ├── RUN.md            # hướng dẫn chạy
│   ├── TCN.pt            # checkpoint mô hình đang dùng
│   └── hand_landmarker.task  # model MediaPipe nhận diện bàn tay
└── Data_321/             # notebook, bảng danh sách video, checkpoint và kết quả trong quá trình nghiên cứu
    ├── 01_PrepareData.ipynb ... 32_Ablation_Preprocess.ipynb   # đánh số theo thứ tự thực hiện; vai trò từng notebook: NOTES.md, mục 4
    ├── hand_landmarker.task  # model MediaPipe dùng cho các notebook
    ├── manifest.csv          # danh sách 215 video đầu (2 nguồn đầu)
    ├── manifest_bonus.csv    # danh sách 90 video Dataset_bonus
    ├── manifest_all.csv      # gộp 305 video, có cột chia train / val / test
    ├── manifest_v2.csv       # chia lại 305 video theo nhóm phiên quay (cột tap, tap_cu, nhom)
    ├── checkpoints_v2/       # các mô hình train ở giai đoạn trước khi thêm dữ liệu bonus
    ├── checkpoints_task9/    # các mô hình train lại sau khi thêm dữ liệu bonus
    │   └── step2/seed0..seed3 (TCN.pt), train_summary.csv, compare_task9.csv, final_model.json, các file gd2_step2_s1_*
    ├── baseline_out/         # kết quả luật ngón tay làm baseline (31a_Baseline_FingerRule)
    ├── ablation_out/         # kết quả ablation tiền xử lý ảnh (32_Ablation_Preprocess)
    ├── preprocess_demo/      # kết quả CLAHE tự viết (24_CLAHE_SelfImpl)
    ├── benchmark_23/         # đo tốc độ từng khâu khi chạy webcam
    └── game_session/         # phiên đo 50 ván của game (25_GameAB_v2, cell 10 đến 12)
```

## 3. Chạy game và demo

### 3.1. Game `game_ab/` (bản hoàn chỉnh)

Game chơi trên trình duyệt, dùng webcam để nhận diện tay. Không cần tải dataset. Máy có card NVIDIA thì tự dùng GPU, không có thì tự chạy bằng CPU (chậm hơn một chút). Bản đầy đủ, kèm bảng xử lý lỗi và mô tả các file: [`game_ab/RUN.md`](game_ab/RUN.md).

**Cần có trước**

- Máy Windows có webcam.
- Đã cài Python (nên dùng 3.13, giống bản demo) và Git.
- Có internet (lần đầu cài thư viện khá nặng vì có PyTorch).
- Hai file model (`TCN.pt` trong `game_ab/` và `hand_landmarker.task` trong `demo/`) đã nằm sẵn trong repo, không cần tải thêm.

**Các bước chạy** (mở Command Prompt rồi làm lần lượt)

Bước 1: tải code về (chỉ làm lần đầu)

```
git clone https://github.com/khangto490-cmd/Rock-Paper-Sciccor-Gesture-.git
cd Rock-Paper-Sciccor-Gesture-
```

Bước 2: chuyển sang nhánh Sang

```
git checkout Sang
```

Bước 3: vào thư mục game

```
cd Sang\game_ab
```

Dòng đầu của cmd phải kết thúc bằng `...\Sang\game_ab>`. Nếu chưa đúng thì các bước sau sẽ báo lỗi không tìm thấy file.

Bước 4: tạo môi trường ảo (chỉ làm lần đầu)

```
python -m venv venv
```

Bước 5: bật môi trường ảo (mỗi lần mở cmd mới đều phải làm)

```
venv\Scripts\activate
```

Thấy chữ `(venv)` ở đầu dòng lệnh là đã bật thành công.

Bước 6: cài thư viện (chỉ làm lần đầu)

```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Bước 7: chạy game

```
python server.py
```

Khi thấy dòng `Sẵn sàng. Mở http://127.0.0.1:8000 để chơi`, mở trình duyệt và vào http://127.0.0.1:8000. Dòng `Đang chạy trên: CPU` hoặc `CUDA` cho biết máy đang dùng CPU hay GPU, cả hai đều chơi được. Muốn tắt game, quay lại cmd và nhấn `Ctrl + C`.

**Điều khiển:** phím `Space` để bắt đầu ván, chơi tiếp, chơi lại; phím `r` để đặt lại điểm.

**Những lần chạy sau:** chỉ cần 3 lệnh (đã có code và đã cài thư viện rồi):

```
cd Rock-Paper-Sciccor-Gesture-\Sang\game_ab
venv\Scripts\activate
python server.py
```

Muốn lấy code mới nhất thì chạy thêm `git pull origin Sang` trước khi `python server.py`. Nếu gặp lỗi (đứng sai thư mục, không mở được webcam, thiếu file...), xem bảng "Gặp lỗi thì làm gì" trong [`game_ab/RUN.md`](game_ab/RUN.md).

Kiểm tra luật chơi bằng dữ liệu giả (không cần webcam): `python test_game_logic.py` trong thư mục `game_ab` khi đã bật `venv`.

### 3.2. Demo cũ `demo/`

Chỉ để kiểm tra mô hình. Không cần tải dataset. Làm theo [`demo/RUN.md`](demo/RUN.md). Tóm tắt:

```
cd Sang/demo
python --version                  # nên là Python 3.13
python -m venv .venv
# kích hoạt .venv (xem RUN.md cho từng hệ điều hành)
python -m pip install -r requirements.txt   # dùng "python -m pip", không dùng "pip" trần
python demo.py                    # webcam
python demo.py --video clip.mp4   # file video
```

Ngưỡng tin cậy hiển thị mặc định là 0.60, đổi bằng phím `[` `]` hoặc tùy chọn `--threshold`.

### 3.3. Luật chơi A -> B (tóm tắt)

Trên hình webcam (đã lật gương) có ô A bên trái, ô B bên phải, một đường nối hai ô và một vạch gần ô A. Luật nằm trong `game_ab/game_logic.py`.

1. Nhấn `Space` để bắt đầu. Nắm tay thành búa, đưa vào ô A và giữ liên tục 0,3 giây (`hold_a`).
2. Đi tay từ A sang B theo đường nối. Lệch ít thì cho qua. Ra khỏi dải cho phép quá 0,5 giây hoặc mất tay quá 0,3 giây thì ván quay về bước chờ ở A.
3. Từ lúc bắt đầu đi tới khi tay chạm vạch (đặt ở 0,25 quãng đường, `gate_f`), tay phải còn là búa. Đổi tay trước vạch quá 0,1 giây (`fist_grace`) thì về ô A. Qua vạch rồi thì ra cử chỉ nào cũng được.
4. Tay chạm ô B: model chốt đáp án ngay tại khung đó, chỉ dùng thông tin tới khung đó. Máy ra cử chỉ khắc chế và không đổi đáp án sau khi chốt.
5. Sau khi chốt, cử chỉ thật của người chơi được xác định trong 0,2 giây (`confirm_sec`) bằng luật ngón tay, độc lập với model. Khớp với đáp án đã chốt thì ra kết quả. Khác thì ván bị hủy, không tính điểm, không hiện đáp án của máy. Mất tay suốt khoảng này thì ván không xác định, không tính điểm.

Vì máy đoán sớm rồi ra cử chỉ khắc chế, kết quả của một ván hợp lệ thường là máy thắng, nên điểm số chỉ để minh họa. Các tham số trên chỉnh ở dòng `g = GameAB(...)` trong `game_ab/server.py`.

## 4. Lưu ý về thư mục `Data_321/`

Các notebook trong `Data_321/` là **notebook nghiên cứu, chạy riêng trên máy cá nhân** (Windows, GPU NVIDIA RTX 3050 Ti, môi trường `train-gpu`). **Người khác không chạy lại được các notebook này từ repo**, vì chúng đọc dữ liệu và file nằm ngoài repo, trong thư mục `~/Downloads/Data_321` của máy train:

- video gốc và file landmark (`.npz`) của dữ liệu tự thu (không nằm trong repo);
- file `hand_landmarker.task` của MediaPipe;
- GPU có CUDA (notebook dừng nếu không có CUDA, không train bằng CPU).

Vì vậy, để xem kết quả hãy đọc **output đã lưu trong notebook** (ví dụ `13_Task8_Step2.ipynb`, `19_Task9_Train.ipynb`), các file kết quả đã đưa lên repo (bảng ở mục 4.1) và các bảng tổng hợp trong `NOTES.md`. Phần có thể chạy lại độc lập là `game_ab/` và `demo/`. Ba notebook `19_Task9_Train` (cell 12), `31a_Baseline_FingerRule` và `32_Ablation_Preprocess` còn cần `game_ab/game_logic.py` đặt cùng thư mục khi chạy.

Danh sách và vai trò từng notebook: `NOTES.md`, mục 4.

### 4.1. Bảng trong báo cáo và nơi lưu kết quả

| Bảng trong báo cáo | Nội dung | Notebook | Kết quả lưu ở |
|---|---|---|---|
| 1 | Nguồn video | `01_PrepareData`, `17_PrepareBonus`, `18_SplitBonus` | `manifest.csv`, `manifest_bonus.csv`, `manifest_all.csv` |
| 2 | Độ chính xác theo ngưỡng, test cũ 33 video | `02_Train` | output notebook |
| 3 đến 6 | So sánh 15 ứng viên theo góc xoay, theo nguồn video và theo ngưỡng | `19_Task9_Train` | `checkpoints_task9/` (`compare_task9.csv`, `train_summary.csv`, `final_model.json`) |
| 7 | Độ chính xác theo mốc thời gian từ khung đầu | `19_Task9_Train` (cell 11) | `checkpoints_task9/` (các file `gd2_step2_s1_*`) |
| 8, 9 | Luật ngón tay và model ở cử chỉ cuối | `31a_Baseline_FingerRule`, `19_Task9_Train` (cell 12) | `baseline_out/` (`finger_rule_per_video.csv`, `finger_rule_summary.csv`, `finger_rule_confusion.csv`), `checkpoints_task9/` |
| 10 | Các thay đổi đã thử và quyết định | `04` đến `19` | tổng hợp ở `NOTES.md`, mục 5 |
| 11, 12 | Ablation tiền xử lý: nhận diện và thời gian | `32_Ablation_Preprocess` | `ablation_out/` (`per_video.csv`, `summary.json` mỗi cấu hình, `timing_interleaved.csv`) |
| 13 | Thời gian từng khâu khi chạy webcam | notebook đo tốc độ | `benchmark_23/20261106_233646/` |
| 14, 15 | Phiên 50 ván: kết quả và thời gian | `25_GameAB_v2` (cell 10 đến 12) | `game_session/20261110_164653_nb25/` |
| 16, 17 | Thời gian từng bước tiền xử lý, độ lệch landmark theo cỡ cắt | `23_Preprocess` | output notebook |
| 18, 19 | CLAHE tự viết so với OpenCV | `24_CLAHE_SelfImpl` | `preprocess_demo/clahe_selfimpl.csv`, `clahe_selfimpl.png` |
| 20 | Nhóm phiên quay và chia lại | `26_LeakCheck_Labeled`, `27_LeakCheck_HandBonus`, `28_SplitV2` | `manifest_v2.csv` |


## 5. Dữ liệu

Toàn bộ dữ liệu là **dữ liệu tự thu** (gồm dữ liệu tự thu của cá nhân và dữ liệu tự thu của nhóm), không dùng dataset công khai. Dữ liệu không nằm trong repo.

| Nguồn | Nội dung | Tốc độ | Dung lượng |
|---|---|---|---|
| `Dataset_hand` | 27 video mỗi lớp (rock / paper / scissors) | 60 fps (nội suy lên 120 fps) | khoảng 133 MB |
| `dataset_labeled/person_01` | paper 48, rock 43, scissors 43 video | 120 fps | khoảng 208 MB |
| `Dataset_bonus` | 90 video (30 mỗi lớp), quay bằng điện thoại | báo 60 fps, một số video đã được làm chậm nên có thể không thật sự đạt 60 fps (nội suy lên 120 fps) | chưa ghi lại |

Nhãn lấy từ tên thư mục (rock = búa, paper = bao, scissors = kéo). Dữ liệu chia **theo video** (không chia theo khung hình, để các khung của cùng một video không nằm ở cả tập train và tập test). Hai nguồn đầu (215 video): train 151 / val 31 / test 33, chia theo nguồn × lớp, seed 42. `Dataset_bonus` cũng chia theo video, tỉ lệ 70/15/15, seed 42: 63 / 12 / 15. Sau khi gộp: 305 video, train 214 / val 43 / test 48. Ngoài ra, 305 video được gán vào 11 nhóm phiên quay và có bản chia lại theo nhóm (`Data_321/manifest_v2.csv`) dùng cho phần phân tích bổ sung (`NOTES.md`, mục 5.7). Model chính được train trên bản chia cũ ở trên.

## 6. Thông tin mô hình AI đang dùng

| Mục | Giá trị |
|---|---|
| Tên trong repo | `step2` seed 1 (viết tắt `step2 s1` trong `NOTES.md`); checkpoint `TCN.pt` |
| Kiến trúc | TCN nhân quả: Linear(546→64) → 5 khối residual Conv1d (kernel 3, dilation 1, 2, 4, 8, 16, dropout 0.1) → Linear(64→3). Khoảng 97 nghìn tham số |
| Đầu vào mỗi khung | 546 số = 126 (tọa độ landmark trừ cổ tay, chia cỡ bàn tay, + vận tốc) + 210 khoảng cách giữa các cặp khớp + 210 vận tốc của khoảng cách |
| Đầu ra | Xác suất 3 lớp: paper = 0, rock = 1, scissors = 2 |
| Tốc độ chuỗi | 120 fps (webcam khoảng 30 fps được nội suy lên 120 fps) |
| Huấn luyện | CUDA, AdamW lr 1e-3, weight decay 1e-2, batch 16, tối đa 120 epoch, dừng sớm sau 20 epoch val loss không giảm (tốt nhất ở epoch 27, dừng ở epoch 47, val loss 0.3224, độ chính xác val 0.8651 trên 43 video val). Dữ liệu train: 214 video (cũ + `Dataset_bonus`) |
| Augmentation | Xoay 3D (±40°, ±45°, ±40° theo 3 trục), lật gương (xác suất 0.5), nhiễu tọa độ (độ lệch chuẩn 0.005) |
| Checkpoint | `Data_321/checkpoints_task9/step2/seed1/TCN.pt`; bản dùng cho game: `game_ab/TCN.pt`; bản dùng cho demo cũ: `demo/TCN.pt` (cùng model `step2` seed 1). Bản train trước khi thêm dữ liệu bonus: `Data_321/checkpoints_v2/task8_step2_distance/TCN.pt` |

### Tên các cấu hình trong notebook, thư mục và NOTES.md

Ba cấu hình TCN đã được so sánh, mỗi cấu hình train với 4 seed (`s0`–`s3`):

| Tên | Cấu hình |
|---|---|
| `tcn_v2` | 126 số mỗi khung (tọa độ + vận tốc), Data Augmentation cơ bản (xoay 3D, lật gương, nhiễu) |
| `wide45_lownoise` | 126 số mỗi khung, xoay góc rộng hơn và nhiễu thấp |
| `step2` (trước khi thêm dữ liệu bonus có tên `task8_step2_distance`) | 546 số mỗi khung: thêm khoảng cách giữa các cặp khớp và vận tốc của chúng. **Cấu hình đang dùng** |

- `s1` là seed 1, ví dụ `step2 s1` là cấu hình `step2` train với seed 1.
- `ens4` là bản ghép 4 model cùng cấu hình (seed 0 đến 3), lấy trung bình xác suất. Bản ghép `step2` (`step2 ens4`) được giữ để nghiên cứu tiếp; chậm hơn `step2 s1` khoảng 3 lần (2.9 ms so với 1.0 ms mỗi dự đoán trong `compare_task9.csv`, báo cáo đo 2.7 ms và 1.1 ms) vì phải chạy 4 model, dùng 4 checkpoint `seed0` – `seed3` trong `Data_321/checkpoints_task9/step2/`.

## 7. Môi trường

- Máy train: Windows, RTX 3050 Ti Laptop GPU (4 GB VRAM), RAM 15,7 GB. Thư viện chính: `torch 2.6.0+cu124`, `mediapipe 1.1.0`, `numpy 2.5.2`, `opencv-contrib-python 5.0.0.93`. Danh sách đầy đủ: [`requirements.txt`](requirements.txt) và [`requirements-lock.txt`](requirements-lock.txt).
- Game (`game_ab/`): Windows, tự dùng GPU nếu PyTorch nhận được CUDA, không thì chạy CPU. Thư viện: `game_ab/requirements.txt`.
- Demo cũ: Python 3.13 (đã chạy thử với 3.13.5), CPU hoặc GPU. Thư viện của demo: `demo/requirements.txt`, cài bằng `python -m pip install -r requirements.txt`.
- MediaPipe cài bằng pip không bật được GPU, nên bước trích landmark và demo luôn chạy MediaPipe bằng CPU.
