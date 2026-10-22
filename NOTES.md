# Ghi chú quá trình làm (Sang)

Tài liệu này ghi lại những gì đã làm trong thư mục `Data_321`: mỗi notebook làm gì, đã thử nghiệm gì, kết quả ra sao, hướng nào giữ và hướng nào bỏ.

> **Lưu ý:** các notebook được viết và chạy trên máy cá nhân (Windows, GPU RTX 3050 Ti 4 GB), đọc dữ liệu video ở máy nên **không chạy lại trực tiếp từ GitHub được**. Output đã chạy được giữ trong notebook để xem kết quả. Dataset không đưa lên GitHub (xem README).

## 1. Mục tiêu

Train model dự đoán cử chỉ oẳn tù tì (kéo / búa / bao) **ngay khi tay người chơi đang ra**, để máy khắc chế lại. Mô hình chỉ nhìn các khung hình trước thời điểm dự đoán.

## 2. Dữ liệu (không đưa lên GitHub)

- `Dataset_hand`: 27 video mỗi lớp (Rock/Paper/Scissors), 60 fps, khoảng 133 MB.
- `dataset_labeled/person_01`: paper 48, rock 43, scissors 43 video, 120 fps, khoảng 208 MB. Nhãn lấy từ tên thư mục.
- Nhãn số theo thứ tự chữ cái: paper=0, rock=1, scissors=2.
- `Dataset_hand` được nội suy lên 120 fps để thống nhất với nguồn còn lại.
- Chia train/val/test theo **video** (không chia theo khung hình): 151 / 31 / 33 video, chia theo nguồn × lớp, seed 42.

## 3. Cách làm

- Trích landmark bàn tay bằng MediaPipe (chạy CPU, vì bản pip không bật được GPU), `num_hands=1`, khung mất tay đánh dấu NaN.
- Đầu vào mô hình mỗi khung: 126 số = 63 tọa độ landmark (đã trừ cổ tay, chia cỡ bàn tay) + 63 vận tốc. Khung mất tay điền 0 và không tính vào loss.
- Dự đoán trên từng khung, khung nào có tay thì luôn đưa ra dự đoán kèm độ tin cậy.
- Train bằng CUDA, tối đa 60 epoch, early stopping 10 epoch, AdamW lr 1e-3, batch 16. Checkpoint lưu trong `checkpoints*`.
- Tiêu chí đánh giá (mang tính tham khảo): ở mỗi ngưỡng tin cậy 0.5 → 0.9, tính tỉ lệ video có khung vừa tự tin ≥ ngưỡng vừa đoán đúng. Đạt 80% = trung bình 5 ngưỡng ≥ 80%. Kèm thời điểm đoán đúng đầu tiên (ms).

## 4. Các notebook

| Notebook | Làm gì | Trạng thái |
|---|---|---|
| `01_PrepareData` | Trích landmark, nội suy 60→120 fps, gộp 2 nguồn, tạo `manifest.csv`, chia train/val/test | Dùng |
| `02_Train` | Train TCN, GRU nhỏ, Transformer nhỏ; đánh giá theo ngưỡng tin cậy và độ phủ | Dùng |
| `04_Trainv2` | Data augmentation khi train: xoay góc nhìn 3D, thêm nhiễu tọa độ, lật gương | Dùng |
| `06_AngleCheck` | Xoay giả lập tập test ở nhiều góc để xem model sai ở góc nào | Dùng |
| `07_TrainWide` | Train 3 bản TCN với góc xoay rộng hơn (wide30, wide45, wide45_lownoise) | Dùng, chọn `wide45_lownoise` |
| `08_WebcamWide` | Demo webcam với `wide45_lownoise` (hiển thị fps, kết quả kèm ngưỡng, khung xương tay) | **Bản demo chính** |
| `09_ImageFeatures`, `09b_ExtractResNet50` | Trích đặc trưng hình ảnh ResNet18 / ResNet50 để so sánh với landmark | Đã bỏ hướng (xem mục 5) |
| `10_TrainImage` | Train TCN và GRU trên đặc trưng ResNet | Đã bỏ hướng |
| `11_Diagnose`, `11_Diagnose_fix` | Kiểm tra model hình ảnh có học tắt (nền, người, phiên quay) hay không | Đã bỏ hướng |
| `12_Task8_Step1` | Xoay bàn tay về một hướng chuẩn | Đã bỏ |
| `13_Task8_Step2` | Thêm đặc trưng hình học (khoảng cách giữa các khớp, góc giữa các ngón) | Giữ lại |
| `[điền tên]` | Ghép `wide45_lownoise` + `task8_step2_distance` (Bước 3a) | Đã thử, không tốt hơn |
| `15_Task8_Step3b` | Ensemble 4 bản TCN Step 2 khác seed | Giữ lại để so sánh sau |
| `[điền nếu có notebook 03, 05, ...]` | | |

## 5. Thử nghiệm và kết quả

**So sánh 3 kiến trúc (`02_Train`, test 33 video):** cả 3 đạt tiêu chí 80% (trung bình 97.0%). Độ chính xác khi tin cậy ≥ ngưỡng 0.5 → 0.9: TCN 93.3 → 98.1, GRU 91.7 → 94.5, Transformer 85.7 → 98.9. TCN ổn định nhất ở ngưỡng thấp, Transformer tốt từ ngưỡng 0.8, GRU yếu nhất.

**Data augmentation (`04_Trainv2`):** V2-TCN và V2-Transformer cải thiện rõ so với bản gốc, nhất là ở nửa đầu video, nhưng độ phủ giảm khi tăng ngưỡng. Test webcam: TCN và GRU v2 tốt hơn v1, Transformer vẫn tệ nên bỏ.

**Kiểm tra góc (`06_AngleCheck`):** v2 chỉ sai khi xoay mạnh (khoảng 45° trở lên), nên tập trung đầu tư TCN v2.

**Mở rộng góc xoay (`07_TrainWide`):** cả 3 bản wide đều hơn TCN v2 khi xoay mạnh (±45°: khoảng 84% lên 90–91%) mà không mất độ chính xác ở góc thường. `wide45_lownoise` nhỉnh nhất, test webcam (`08_WebcamWide`) hoạt động rất tốt.

**Model hình ảnh (09 → 11):** r50-GRU đạt 0.97 trên val nhưng quá cao và quá sớm (~17 ms). Hai notebook chẩn đoán chưa đủ để kết luận model nhìn cử chỉ tay hay học tắt từ nền và phiên quay, và khi so với landmark thì model hình ảnh kém. **Quyết định dừng model hình ảnh**, tập trung TCN landmark `wide45_lownoise`.

**Task 8 (cải tiến TCN v2, mỗi bước so với bản mới nhất và kiểm tra webcam):**
- Bước 1, xoay bàn tay về hướng chuẩn: kém hơn `wide45_lownoise`, webcam không thấy khác biệt bằng mắt. Bỏ.
- Bước 2, đặc trưng hình học (`task8_step2_distance`): kết quả khá tốt. Giữ lại để train lại khi có thêm dữ liệu.
- Bước 3a, ghép `wide45_lownoise` + `task8_step2_distance` (trung bình xác suất): không tốt hơn Step 2. Độ chính xác mọi khung thấp hơn 0.1–0.9 điểm, độ phủ ở ngưỡng 0.7 thấp hơn 1.5–4.1 điểm, đoán đúng đầu tiên chậm hơn khoảng 5–15 ms.
- Bước 3b, ensemble 4 TCN Step 2 khác seed (`checkpoints_v2/task8_step3b`): metric test cao hơn Step 2 (acc_all +0.6 đến +1.2 điểm) nhưng webcam sai ở vài góc, kém Step 2. Giữ lại để train lại trên dataset lớn hơn.
- Bước 4 (làm mượt kết quả webcam) và Bước 5 (bổ sung dataset): chưa làm.

## 6. Checkpoint

- `checkpoints_v2/wide45_lownoise/TCN.pt`: **bản chốt**, dùng cho `08_WebcamWide`.
- `checkpoints_v2/task8_step2_distance/TCN.pt`: bản giữ lại (đặc trưng hình học).
- `checkpoints_v2/task8_step3b/seed1–3`: ensemble, giữ để so sánh.
- Các thư mục khác trong `checkpoints*`: các bản thử nghiệm.

## 7. Hạn chế đã biết

- Test chỉ 33 video, cùng 2 nguồn quay với train, chưa biết chạy tốt với người chơi hoặc camera mới.
- MediaPipe đôi lúc không bắt được khớp ở một số góc hoặc tay phức tạp.
- Dữ liệu có ít người quay, tay trái từng gần như sai hết, đã cải thiện nhờ lật gương nhưng cần thêm dữ liệu thật.
