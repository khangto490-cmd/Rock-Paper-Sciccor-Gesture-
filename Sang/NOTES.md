# NOTES – Nhật ký thực hiện

Cập nhật: 22/10/2026. Tổng quan dự án: [`README.md`](README.md). Hướng dẫn chạy demo: [`demo/RUN.md`](demo/RUN.md).

## 1. Mục tiêu

Train mô hình có thể dự đoán kết quả (kéo / búa / bao) **khi tay người chơi đang ra**, để máy khắc chế lại. Vì vậy mô hình phải đoán **sớm** và **ổn định**, không chỉ đúng ở cuối clip.

## 2. Dữ liệu

Toàn bộ là dữ liệu tự thu (của cá nhân và của nhóm), không dùng dataset công khai.

| Nguồn | Nội dung | Tốc độ | Dung lượng |
|---|---|---|---|
| `Dataset_hand` | 27 video mỗi lớp (rock / paper / scissors) | 60 fps | khoảng 133 MB |
| `dataset_labeled/person_01` | paper 48, rock 43, scissors 43 | 120 fps | khoảng 208 MB |

- Nhãn lấy từ tên thư mục. Nhãn số theo thứ tự chữ cái: paper = 0, rock = 1, scissors = 2.
- `Dataset_hand` được nội suy từ 60 fps lên 120 fps (có xử lý khung mất tay) để thống nhất với nguồn còn lại.
- Trích landmark bằng MediaPipe (`hand_landmarker.task`, `num_hands=1`); khung mất tay đánh dấu NaN, khi train điền 0 và không tính vào loss.
- `manifest.csv` liệt kê mọi video (đường dẫn file landmark 120 fps, nguồn, lớp, nhãn số, số khung, tập). Các bước chia dữ liệu và train chỉ đọc bảng này.
- Chia **theo video**, không theo khung hình, theo nguồn × lớp, seed 42: train 151 / val 31 / test 33 video (số khung có tay: 19 108 / 3 782 / 3 930).
- Nhãn của video được gán cho mọi khung có tay trong video đó. Mô hình chỉ nhìn các khung trước nó.

## 3. Môi trường

- Máy: RTX 3050 Ti Laptop (4 GB VRAM), RAM 15,7 GB, Windows. Môi trường `train-gpu`.
- Luôn train bằng CUDA, không train bằng CPU. MediaPipe cài bằng pip không bật được GPU (lỗi `GPU processing is disabled in build flags`) nên giữ MediaPipe chạy CPU.
- Thư viện: xem `requirements.txt` và `requirements-lock.txt`.
- Quy ước train để so sánh công bằng: tối đa 60 epoch có dừng sớm (patience 10); riêng Task 8 nâng lên tối đa 100 epoch, patience 18. Luôn lưu checkpoint và tự nạp lại nếu đã có.
- Đầu vào mỗi khung (bản gốc): 126 số = 63 tọa độ landmark (trừ cổ tay, chia cỡ bàn tay) + 63 vận tốc.

## 4. Các notebook trong `Data_321/`

Các notebook chạy riêng trên máy cá nhân, người khác không chạy lại được (xem `README.md`, mục 4). Mỗi notebook chia cell và có chú thích `#cell x: ...` ở đầu mỗi cell.

| Notebook | Nội dung | Trạng thái |
|---|---|---|
| `01_PrepareData` | Trích landmark, nội suy 60 → 120 fps, gộp 2 nguồn, tên lớp chữ thường, tạo `manifest.csv`, chia train/val/test | Xong |
| `02_Train` | Train và đánh giá TCN, GRU nhỏ, Transformer nhỏ | Xong |
| `03_Webcam_Test` | Test webcam 3 mô hình v1 (hiển thị FPS, nhãn kèm ngưỡng, khung xương tay) | Xong |
| `04_Trainv2` | Train lại với Data Augmentation (xoay 3D, nhiễu, lật gương) | Xong |
| `05_Webcam_Test_v2` | Test webcam các mô hình v2 | Xong |
| `06_AngleCheck` | Xoay giả lập tập test ở nhiều góc để tìm góc model sai | Xong |
| `07_TrainWide` | Train 3 bản TCN với góc xoay rộng hơn (wide30, wide45, wide45_lownoise) | Xong |
| `08_WebcamWide` | Test webcam bản `wide45_lownoise` | Xong |
| `09_ImageFeatures`, `09b_ExtractResNet50` | Trích đặc trưng hình ảnh ResNet18 / ResNet50 | Xong, nhánh đã dừng |
| `10_TrainImage` | Train TCN và GRU trên đặc trưng hình ảnh | Xong, nhánh đã dừng |
| `11_Diagnose`, `11_Diagnose(fix)` | Kiểm tra mô hình hình ảnh có học tắt (nền, phiên quay) hay không | Xong, nhánh đã dừng |
| `12_Task8_Step1` | Xoay bàn tay về hướng chuẩn | Đã bỏ |
| `13_Task8_Step2` | Thêm đặc trưng khoảng cách giữa các khớp | **Đang dùng** |
| `14_Task8_Step3a` | Ghép `wide45_lownoise` và `task8_step2_distance` (trung bình xác suất) | Không tốt hơn, không dùng |
| `15_Task8_Step3b` | Ghép 4 bản TCN Step 2 khác seed | Giữ lại để train lại sau |

## 5. Quá trình, thử nghiệm và kết luận

### 5.1. So sánh 3 kiến trúc (v1)

Train TCN, GRU và Transformer nhỏ trên dữ liệu gốc, đánh giá trên 33 video test. Tiêu chí: ở mỗi ngưỡng tin cậy 0.5 → 0.9, tính tỉ lệ video có khung vừa tự tin ≥ ngưỡng vừa đoán đúng. Đạt 80% khi trung bình 5 ngưỡng ≥ 80%; ưu tiên ngưỡng 0.5 và 0.6. Có thêm thời điểm đoán đúng đầu tiên (ms) để tham khảo. Tiêu chí chỉ mang tính tham khảo.

- Cả 3 mô hình đều đạt 80% (con số trung bình ghi nhận: 97,0%).
- Độ chính xác khi tin cậy ≥ ngưỡng 0.5 → 0.9 (cả clip): TCN 93.3 → 98.1, GRU 91.7 → 94.5, Transformer 85.7 → 98.9.
- TCN ổn định nhất ở ngưỡng thấp, Transformer tốt nhất từ ngưỡng 0.8, GRU yếu nhất.
- Test webcam: TCN và GRU tương đối dùng được nhưng chỉ ổn với **tay phải**, tay trái gần như sai hết; Transformer gần như toàn sai.

**Kết luận:** đầu tư vào TCN (webcam tốt nhất). GRU hay tự tin mà sai. Bỏ Transformer.

### 5.2. Data Augmentation (v2)

`04_Trainv2` thêm xoay 3D, nhiễu tọa độ và lật gương (giải quyết tay trái / tay phải và đổi góc nhìn), giữ nguyên val/test, lưu checkpoint riêng (`checkpoints_v2/all/`).

- V2-TCN và V2-Transformer cải thiện rõ so với bản gốc, nhất là ở nửa đầu video; đổi lại độ phủ giảm khi tăng ngưỡng tin cậy.
- Webcam: TCN và GRU v2 tốt hơn v1; Transformer vẫn tệ.

### 5.3. Kiểm tra theo góc và mở rộng augmentation

- `06_AngleCheck`: v2 tốt hơn v1 rõ rệt, chỉ sai khi xoay mạnh (khoảng 45° trở lên). → tập trung vào TCN v2.
- `07_TrainWide`: 3 bản góc xoay rộng hơn đều hơn TCN v2 khi xoay ±45° (khoảng 84% lên 90–91%) mà không mất độ chính xác ở góc thường. `wide45_lownoise` nhỉnh nhất.
- `08_WebcamWide`: `wide45_lownoise` chạy rất tốt trên webcam. Đây là baseline cho Task 8.

### 5.4. Mô hình hình ảnh (không dùng khung xương) – đã dừng

Mục đích: xem mô hình hình ảnh có đoán sớm hơn landmark không, và để giải quyết trường hợp MediaPipe không bắt được khớp.

- Trích đặc trưng ResNet18 (mọi khung, 4 phiên bản mỗi video: gốc, lật gương, augmentation, augmentation + lật) và ResNet50; train TCN và GRU trên chuỗi 30 fps (mô hình học ở tốc độ nào thì phải dùng đúng tốc độ đó).
- Lần chạy đầu: r50-GRU cao nhất trên val (0.97) nhưng **quá cao và quá sớm** (khoảng 17 ms), nghi học tắt từ nền hoặc người quay.
- `11_Diagnose` chưa đủ chặt để kết luận; `11_Diagnose(fix)` bổ sung thống kê số video/khung trước khi thấy tay, cảnh báo dữ liệu ít, làm rõ cách diễn giải. Vẫn chưa xác nhận được mô hình có học tắt hay không.

**Kết luận:** không dùng mô hình hình ảnh (kết quả cao bất thường, không kiểm chứng được); tập trung vào TCN landmark `wide45_lownoise`.

### 5.5. Task 8 – Cải tiến TCN v2

Quy ước: mỗi bước làm riêng, luôn so với bản mới nhất, và kiểm chứng bằng webcam trước khi giữ lại.

**Bước 1 – Xoay bàn tay về hướng chuẩn (`12_Task8_Step1`): bỏ.** Kém hơn `wide45_lownoise`, và khi thử webcam không thấy khác biệt bằng mắt.

**Bước 2 – Thêm đặc trưng ít phụ thuộc góc nhìn (`13_Task8_Step2`): giữ, đang dùng.**

Ý tưởng: thêm khoảng cách giữa các cặp khớp (210 cặp từ 21 khớp) và vận tốc của chúng. Khoảng cách không đổi khi xoay tay (đã kiểm tra: sai số sau xoay 35° chỉ khoảng 2.4e-7), nên model ít phụ thuộc góc. Đặc trưng mỗi khung tăng từ 126 lên 546.

Kết quả trên 33 video test, xoay giả lập ngẫu nhiên trong ±góc (lặp 5 lần mỗi góc, trừ 0°):

Độ chính xác mọi khung có tay (%):

| Mô hình | 0° | 15° | 30° | 45° | 60° |
|---|---|---|---|---|---|
| `task8_step2_distance` | 93.2 | 92.8 | 92.9 | 92.8 | 92.3 |
| `wide45_lownoise` (baseline) | 92.5 | 92.3 | 91.3 | 91.2 | 90.4 |

Độ chính xác khi tin cậy ≥ 0.7 (%):

| Mô hình | 0° | 15° | 30° | 45° | 60° |
|---|---|---|---|---|---|
| `task8_step2_distance` | 95.8 | 95.9 | 95.9 | 95.3 | 95.6 |
| `wide45_lownoise` (baseline) | 96.7 | 96.7 | 97.2 | 97.7 | 96.4 |

Độ phủ: % khung đạt ngưỡng 0.7:

| Mô hình | 0° | 15° | 30° | 45° | 60° |
|---|---|---|---|---|---|
| `task8_step2_distance` | 91.8 | 91.7 | 91.7 | 92.1 | 91.8 |
| `wide45_lownoise` (baseline) | 89.4 | 89.2 | 87.2 | 85.9 | 84.6 |

Nhận xét:
- Độ chính xác mọi khung: Step 2 hơn baseline **+0,6 điểm ở 0°** và **+1,5 điểm ở ±45°**, và giữ gần như phẳng khi tăng góc (93,2 → 92,3) trong khi baseline giảm (92,5 → 90,4). Đạt tiêu chí sơ bộ của notebook (không giảm quá 1 điểm ở 0° và tăng ở ±45°).
- Độ chính xác khi tin cậy ≥ 0.7 của Step 2 **thấp hơn** baseline khoảng 0,8–2,4 điểm. Bù lại **độ phủ cao hơn** rõ (từ +2,4 điểm ở 0° đến +7,2 điểm ở 60°), tức model tự tin trên nhiều khung hơn thay vì chỉ trên một số ít khung.
- Huấn luyện: dừng ở epoch tối ưu 25, val loss 0.2104 (tối đa 100 epoch, patience 18). Baseline là checkpoint epoch 49.
- Webcam: Step 2 cho kết quả khá tốt. Giữ lại và sẽ train lại khi có thêm dataset để so sánh.
- Lưu ý: khoảng cách 3D chỉ bất biến với phép xoay toán học; tọa độ z của MediaPipe có nhiễu, nên cải thiện khi xoay giả lập chưa chắc đúng hoàn toàn với tay thật, cần kiểm chứng bằng webcam.

**Bước 3a – Ghép `wide45_lownoise` + `task8_step2_distance` (trung bình xác suất): không dùng.** So với Step 2: độ chính xác mọi khung thấp hơn 0,1–0,9 điểm, độ phủ ở ngưỡng 0.7 thấp hơn 1,5–4,1 điểm, đoán đúng đầu tiên chậm hơn khoảng 5–15 ms; webcam không thấy khác biệt.

**Bước 3b – Ghép 4 bản TCN Step 2 khác seed (`checkpoints_v2/task8_step3b/seed1–3`): giữ lại, chưa dùng.** Metric test cao hơn Step 2 (độ chính xác mọi khung +0,6 đến +1,2 điểm) nhưng webcam sai ở một số góc, kém Step 2. Sẽ train lại trên dataset lớn hơn ở Bước 5 để so sánh. Đây là ví dụ cho thấy metric test và trải nghiệm webcam có thể không đồng nhất.

## 6. Quyết định chính (tóm tắt)

| Quyết định | Lý do |
|---|---|
| Dùng TCN | Webcam tốt nhất trong 3 kiến trúc; GRU hay tự tin mà sai; Transformer tệ trên webcam |
| Dự đoán trên từng khung hình | Đảm bảo tốc độ; khung nào có tay thì luôn đưa ra dự đoán kèm độ tin cậy |
| Dùng landmark, bỏ mô hình hình ảnh | Mô hình hình ảnh cho kết quả cao bất thường, không loại trừ được học tắt |
| Chọn `task8_step2_distance` | Chính xác hơn và độ phủ cao hơn khi đổi góc; webcam ổn |
| MediaPipe chạy CPU | Bản pip không bật được GPU |
| Chia dữ liệu theo video | Tránh khung của cùng một video nằm ở cả train và test |

## 7. Việc còn lại

Theo kế hoạch Task 8 (chưa đánh dấu hoàn thành):

- **Bước 4:** làm mượt kết quả khi chạy webcam (hiện nhãn có thể đổi trong lúc tay đang ra).
- **Bước 5:** bổ sung thêm dataset, rồi train lại và so sánh `task8_step2_distance` và `task8_step3b` trên dữ liệu lớn hơn.

## 8. Hạn chế đã biết

- Tập test chỉ có 33 video, cùng 2 nguồn quay với tập train. Chưa biết chạy tốt đến đâu với người chơi hoặc camera mới (sẽ kiểm tra sau).
- Khi tay nghiêng mạnh (khoảng 45° trở lên) trên webcam thật, mô hình vẫn có thể nhận sai, dù xoay giả lập cho thấy độ chính xác giảm rất ít (khoảng 92% ở ±60°). Có thể do xoay giả lập không mô phỏng được nhiễu ước lượng của MediaPipe, nhất là trục z.
- MediaPipe đôi lúc không bắt được khớp ở một số góc hoặc khi tay phức tạp; khi đó khung bị coi là mất tay.
- Mô hình dự đoán trên từng khung hình nên nhãn có thể đổi trong lúc tay đang ra. Nhìn kết quả khi tay đã ra gần xong sẽ ổn định nhất.
- Nguồn `dataset_labeled` hiện chỉ có thư mục `person_01`; kết quả webcam chủ yếu được kiểm chứng bằng mắt (chưa có thang đo định lượng cho webcam).

## 9. Khả năng tái hiện

- **Chạy lại được:** `demo/` (cần `TCN.pt` và `hand_landmarker.task` trong `demo/`, xem `demo/RUN.md`).
- **Không chạy lại được từ repo:** notebook trong `Data_321/`, vì cần dữ liệu video, file landmark `.npz` và GPU CUDA trên máy train. Kết quả được lưu kèm trong output của notebook và tổng hợp ở mục 5 của file này.
