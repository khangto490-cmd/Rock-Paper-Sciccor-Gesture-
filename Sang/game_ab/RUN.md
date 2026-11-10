Chạy game Oẳn Tù Tì (luật A -> B)

Game chơi trên trình duyệt, dùng webcam để nhận diện tay bạn. Máy có card NVIDIA thì tự dùng GPU, không có thì tự chạy bằng CPU (chậm hơn một chút).

Cần có trước
Máy Windows có webcam
Đã cài Python (nên dùng 3.13, giống bản demo) và Git
Có internet (lần đầu cài thư viện khá nặng vì có PyTorch)

Hai file model (TCN.pt trong game_ab/ và hand_landmarker.task trong demo/) đã nằm sẵn trong repo, không cần tải thêm.

Các bước chạy

Mở Command Prompt (cmd) rồi làm lần lượt.

Bước 1: Tải code về (chỉ làm lần đầu)
git clone https://github.com/khangto490-cmd/Rock-Paper-Sciccor-Gesture-.git
cd Rock-Paper-Sciccor-Gesture-
Bước 2: Chuyển sang nhánh Sang
git checkout Sang
Bước 3: Vào thư mục game
cd Sang\game_ab

Kiểm tra lại: dòng đầu của cmd phải kết thúc bằng ...\Sang\game_ab>. Nếu chưa đúng thì các bước sau sẽ báo lỗi không tìm thấy file (xem bảng lỗi ở cuối).

Bước 4: Tạo môi trường ảo (chỉ làm lần đầu)
python -m venv venv
Bước 5: Bật môi trường ảo (mỗi lần mở cmd mới đều phải làm)
venv\Scripts\activate

Thấy chữ (venv) ở đầu dòng lệnh là đã bật thành công.

Bước 6: Cài thư viện (chỉ làm lần đầu)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Bước 7: Chạy game
python server.py

Khi thấy dòng Sẵn sàng. Mở http://127.0.0.1:8000 để chơi, mở trình duyệt và vào:

http://127.0.0.1:8000

Dòng Đang chạy trên: CPU hoặc CUDA cho biết máy đang dùng CPU hay GPU. Cả hai đều chơi được.

Tắt game

Quay lại cmd, nhấn Ctrl + C.

Điều khiển
Phím Space: bắt đầu ván, chơi tiếp, chơi lại
Phím r: đặt lại điểm
Những lần chạy sau

Chỉ cần 3 lệnh (đã có code và đã cài thư viện rồi):

cd Rock-Paper-Sciccor-Gesture-\Sang\game_ab
venv\Scripts\activate
python server.py

Muốn lấy code mới nhất thì chạy thêm git pull origin Sang trước khi python server.py.

Gặp lỗi thì làm gì
Thông báo	Cách xử lý
destination path ... already exists khi clone	Đã tải code rồi. Bỏ qua bước 1, chỉ cần cd Rock-Paper-Sciccor-Gesture-
The system cannot find the path specified khi cd	Đang đứng sai thư mục. Kiểm tra đường dẫn hiện ở đầu dòng lệnh
Could not open requirements file ... No such file or directory	Đang đứng sai thư mục. Nhìn đầu dòng cmd, gõ cd tới thư mục game_ab rồi chạy lại. Nếu đã lỡ tạo venv ở chỗ sai thì gõ deactivate, cd đúng chỗ, rồi làm lại từ bước 4
No module named 'detector'	Thiếu file code. Chạy git pull origin Sang rồi chạy lại
Không mở được webcam 0	Đóng các app đang dùng camera (Zoom, Teams...). Hoặc thử camera khác: python server.py 1
Không thấy: ...	Thiếu file model. Chạy git pull origin Sang để lấy đủ file, nếu vẫn lỗi thì báo lại cho người làm dự án
Dòng chữ WARNING, W0000 ...	Bình thường, bỏ qua, game vẫn chạy
Các file trong thư mục game_ab
File	Việc nó làm
server.py	Chạy server, mở webcam, nối giao diện web với game
detector.py	Nhận diện tay và đoán cử chỉ bằng model
game_logic.py	Luật chơi A -> B
TCN.pt	File model đã huấn luyện
web/	Giao diện web
requirements.txt	Danh sách thư viện cần cài
test_game_logic.py	Kiểm tra luật chơi
