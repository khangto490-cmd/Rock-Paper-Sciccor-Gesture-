"""
features.py
-----------
Hàm trích đặc trưng cho 1 bàn tay (MediaPipe Hands), dùng CHUNG cho
cả lúc thu dữ liệu (collect_sequences.py) và lúc predict (realtime_predict.py).

QUAN TRỌNG: không copy hàm này sang chỗ khác rồi sửa riêng — nếu lúc thu
dữ liệu và lúc predict tính khác nhau, model sẽ học một kiểu, dùng lại
kiểu khác, và sai mà không báo lỗi gì.
"""

import math

import numpy as np

LABELS = ["keo", "bua", "bao"]  # kéo, búa, bao

# 21 landmark x 3 tọa độ (x, y, z) = 63 chiều mỗi frame
FEATURE_DIM = 63

WRIST = 0             # cổ tay — dùng làm điểm neo
MIDDLE_MCP = 9        # gốc ngón giữa — dùng để tính scale


def extract_features(hand_landmarks):
    """
    Chuyển 21 landmark thô của 1 bàn tay thành vector đặc trưng 63 chiều
    đã chuẩn hóa theo cổ tay (anchor) và kích thước bàn tay (scale).

    Tham số:
        hand_landmarks: results.multi_hand_landmarks[i].landmark
                         (list 21 phần tử, mỗi phần tử có .x .y .z)

    Trả về:
        list 63 số thực, hoặc None nếu không tính được scale hợp lệ
    """
    wrist = hand_landmarks[WRIST]
    mid_mcp = hand_landmarks[MIDDLE_MCP]

    # --- scale: khoảng cách cổ tay -> gốc ngón giữa ---
    # Đây là một đoạn "xương" trong lòng bàn tay, gần như không đổi hình
    # dạng dù nắm hay xòe tay, nên dùng làm đơn vị đo kích thước bàn tay
    # -> bất biến với khoảng cách tay tới camera.
    scale = math.sqrt(
        (mid_mcp.x - wrist.x) ** 2 +
        (mid_mcp.y - wrist.y) ** 2 +
        (mid_mcp.z - wrist.z) ** 2
    )
    if scale < 1e-6:
        return None

    # --- chuẩn hóa toàn bộ 21 điểm theo cổ tay + scale ---
    features = []
    for lm in hand_landmarks:
        features.append((lm.x - wrist.x) / scale)
        features.append((lm.y - wrist.y) / scale)
        features.append((lm.z - wrist.z) / scale)

    return features


# ======================================================================
# ĐẶC TRƯNG TỪNG NGÓN (11 số) -- dùng cho phương pháp M7
# ======================================================================
# Chỉ số các điểm trên bàn tay theo quy ước 21 mốc của MediaPipe Hands:
#   0 cổ tay | 4,8,12,16,20 = đầu ngón cái/trỏ/giữa/áp út/út
#   5,9,13,17 = khớp gốc (MCP) của ngón trỏ/giữa/áp út/út
TIPS5 = [4, 8, 12, 16, 20]
TIPS4 = [8, 12, 16, 20]
MCPS4 = [5, 9, 13, 17]
ENG_DIM = 11


def engineered_features_batch(X):
    """
    Nhận vector 63 chiều ĐÃ CHUẨN HÓA (đầu ra của extract_features), trả về
    11 đặc trưng cho mỗi dòng:

      - 5 số "độ duỗi": khoảng cách từ đầu mỗi ngón tới cổ tay
      - 4 số "độ co":   khoảng cách từ đầu ngón tới khớp gốc của chính ngón đó
      - 2 số "độ xòe":  khoảng cách giữa đầu ngón trỏ-giữa và giữa-áp út

    Tất cả đều là KHOẢNG CÁCH nên không phụ thuộc hướng xoay của bàn tay.

    X có thể là 1 frame (shape (63,) hoặc (1, 63)) hoặc cả chuỗi (n, 63).
    Trả về mảng shape (n, 11).

    QUAN TRỌNG: cùng 1 hàm này được dùng cho cả lúc dựng dữ liệu train và lúc
    chạy realtime, để hai nơi không bao giờ tính khác nhau.
    """
    P = np.asarray(X, dtype=np.float32).reshape(-1, 21, 3)
    ext = np.linalg.norm(P[:, TIPS5], axis=2)                        # (n, 5)
    curl = np.linalg.norm(P[:, TIPS4] - P[:, MCPS4], axis=2)         # (n, 4)
    spread = np.stack([np.linalg.norm(P[:, 8] - P[:, 12], axis=1),
                       np.linalg.norm(P[:, 12] - P[:, 16], axis=1)], axis=1)
    return np.concatenate([ext, curl, spread], axis=1)               # (n, 11)
