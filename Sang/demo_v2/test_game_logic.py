"""Tự kiểm tra luật game A -> B bằng dữ liệu giả (không cần webcam, không cần model). Chạy: python test_game_logic.py"""
import json
import numpy as np
from game_logic import GameAB, hand_point, hand_shape, majority_latest

rock, paper, dt = np.array([.05, .90, .05]), np.array([.90, .05, .05]), 1 / 30
A_, B_ = np.array([0.2, 0.5]), np.array([0.8, 0.5])      # mặc định của GameAB


def go(g, pt, probs=rock, shape=1, n=1):
    for _ in range(n):
        go.t += dt
        g.step(go.t, pt, probs, shape)
        go.locks += g.locked_now


def along(f):                                              # f=0 là ô A, f=1 là ô B
    return A_ + (B_ - A_) * f


def start_go(g, probs=rock):                               # SPACE rồi nắm búa giữ ở A đủ hold_a giây -> GO
    g.start()
    go(g, along(0), probs, 1, n=12)
    assert g.state == "GO", g.state


def walk(g, shape=1, probs=rock, n=20):                    # (còn ở NEED_A thì giữ búa ở A trước) rồi đi A -> B: búa tới vạch, sau vạch ra `shape`
    if g.state == "NEED_A":
        go(g, along(0), probs, 1, n=12)
    for i in range(n + 1):
        go(g, along(i / n), probs, 1 if i / n < g.gate_f else shape)
        if g.state != "GO":
            break


def new(**kw):
    go.t = 0.0
    return GameAB(**kw)


# 1) đi đúng đường: chưa SPACE thì không làm gì; búa giữ ở A đủ giờ mới chạy; chạm B chốt đúng 1 lần; sau 0.2 s ra kết quả
go.locks = 0
g = new(); assert g.state == "WAIT"
go(g, along(0)); assert g.state == "WAIT"
g.start(); assert g.state == "NEED_A"
go(g, along(0.5)); assert g.state == "NEED_A"                # tay chưa vào ô A thì chưa chạy
walk(g); assert g.state == "CONFIRM" and g.guess == 1 and g.ai == 0 and g.locked_now
go(g, along(1), n=9)
assert g.state == "RESULT" and g.outcome == "ai" and g.score["ai"] == 1 and g.stats["ok"] == 1 and go.locks == 1
assert len(g.history) == 1 and g.history[0]["ket_qua"] == "RESULT" and g.history[0]["ai"] == "paper"

# 2) búa nằm yên ở A: vào GO rồi không huỷ; SPACE giữa ván bị bỏ qua
g = new(); g.start(); go(g, along(0), n=60)
assert g.state == "GO"; g.start(); assert g.state == "GO"

# 3) lệch ngắn: đỏ nhưng vẫn chạy; về lại đường thì hết đỏ; lệch quá off_max thì về chờ ở A
g = new(); start_go(g)
go(g, along(0.5) + [0, 0.15], n=6); assert g.state == "GO" and g.off
go(g, along(0.5)); assert g.state == "GO" and not g.off
go(g, along(0.5) + [0, 0.15], n=20); assert g.state == "NEED_A" and "Lệch" in g.msg

# 4) tay lệch nhẹ nhưng còn trong ô A thì không đỏ
g = new(); g.start(); go(g, along(0) + [0.05, 0], n=12); assert g.state == "GO" and not g.off

# 5) mất tay ngắn (0.2 s) vẫn chạy; mất lâu thì về chờ ở A; búa giữ lại ở A đủ giờ thì chạy lại
g = new(); start_go(g)
go(g, None, None, None, n=6); assert g.state == "GO"
go(g, None, None, None, n=10); assert g.state == "NEED_A"
go(g, along(0), n=12); assert g.state == "GO"

# 6) AI chốt tại B rồi không đổi: người chơi đổi sang paper sau đó -> huỷ ván, không tính điểm, không lộ đáp án; SPACE -> chơi lại
g = new(); g.start(); walk(g, shape=1)
assert g.state == "CONFIRM" and g.guess == 1 and g.ai == 0
go(g, along(1), probs=paper, shape=0, n=9)
assert g.state == "CANCEL" and g.guess == 1 and g.ai == 0 and g.outcome is None
assert sum(g.score.values()) == 0 and g.stats["cancel"] == 1
s = g.snapshot(); assert s["ai"] is None and s["player"] is None and s["outcome"] is None and s["cancels"] == 1
g.start(); assert g.state == "NEED_A" and g.outcome is None and g.guess is None

# 7) mất tay suốt cửa sổ xác nhận -> không xác định, không tính điểm
g = new(); g.start(); walk(g); go(g, None, None, None, n=9)
assert g.state == "RESULT" and g.outcome == "unknown" and sum(g.score.values()) == 0 and g.stats["unknown"] == 1

# 8) hoà phiếu trong cửa sổ lấy cử chỉ muộn nhất (AI chốt paper)
for seq, want in (([1, 1, 0, 0], "RESULT"), ([0, 0, 1, 1], "CANCEL")):
    g = new(confirm_sec=0.12); g.start(); walk(g, probs=paper)
    for sh in seq:
        go(g, along(1), probs=paper, shape=sh)
    assert g.state == want, (seq, g.state)
assert majority_latest([]) is None and majority_latest([1, 1, 0, 0]) == 0

# 9) hand_point: lật gương đúng
lm = np.zeros((21, 3), np.float32); lm[:, 0], lm[:, 1] = 0.2, 0.6
assert np.allclose(hand_point(lm, True), [0.8, 0.6]) and np.allclose(hand_point(lm, False), [0.2, 0.6])

# 10) búa ở A giữ đủ hold_a (0.3 s)
for sh in (0, 2):                                           # paper hoặc scissors ở A: không bao giờ bắt đầu được
    g = new(); g.start(); go(g, along(0), shape=sh, n=60)
    assert g.state == "NEED_A" and g.a_hold == 0 and "BÚA" in g.msg, (sh, g.state, g.msg)
g = new(); g.start(); go(g, along(0), shape=1, n=5)        # búa chưa giữ đủ: tiến độ giữa 0 và 1, chưa đi
assert g.state == "NEED_A" and 0 < g.a_hold < 1
go(g, along(0), shape=1, n=7); assert g.state == "GO"
g = new(); g.start(); go(g, along(0), shape=1, n=6)        # đang giữ mà mở tay -> tiến độ về 0, phải giữ lại từ đầu
assert g.a_hold > 0
go(g, along(0), probs=paper, shape=0); assert g.a_hold == 0
go(g, along(0), shape=1, n=6); assert g.state == "NEED_A"
go(g, along(0), shape=1, n=6); assert g.state == "GO"
g = new(); g.start(); go(g, along(0), shape=1, n=6)        # mất tay khi đang giữ -> về 0
go(g, None, None, None); assert g.a_hold == 0 and g.state == "NEED_A"
g = new(); g.start(); go(g, along(0.5), shape=1, n=30)     # búa nhưng ngoài ô A: không tính
assert g.state == "NEED_A" and g.a_hold == 0

# 11) VẠCH: vào GO rồi còn phải giữ búa tới khi chạm vạch (gate_f = 0.25); chạm vạch rồi đổi gì cũng được
g = new(); start_go(g); walk(g, shape=0, probs=paper)      # búa tới vạch, sau vạch ra paper: bình thường
assert g.state == "CONFIRM" and g.guess == 0
g = new(); start_go(g)                                     # đổi tay ngay ở A (chưa chạm vạch)
go(g, along(0), probs=paper, shape=0, n=2); assert g.state == "GO" and g.warn          # mới 2 khung: cảnh báo, chưa huỷ
go(g, along(0), probs=paper, shape=0, n=4); assert g.state == "NEED_A" and "BÚA" in g.msg and not g.passed
go(g, along(0), n=12); assert g.state == "GO"              # giữ búa lại ở A đủ giờ thì đi lại được
g = new(); start_go(g)                                     # nhấp nháy 1 khung sai hình rồi lại búa: bỏ qua
go(g, along(0.1), probs=paper, shape=0); assert g.warn and g.state == "GO"
go(g, along(0.1)); assert g.state == "GO" and not g.warn
g = new(); start_go(g)                                     # đổi tay khi sắp tới vạch (0.2 < 0.25) vẫn bị huỷ
go(g, along(0.2), probs=paper, shape=0, n=6); assert g.state == "NEED_A"
g = new(); start_go(g)                                     # chạm vạch (0.3) là đổi tay được ngay
go(g, along(0.3), probs=paper, shape=0); assert g.state == "GO" and g.passed and not g.warn
go(g, along(0.1), probs=paper, shape=0, n=10); assert g.state == "GO"   # đã qua vạch thì quay lại đổi tay cũng không sao

# 12) snapshot gửi cho giao diện: JSON được, không lộ đáp án trước RESULT
g = new(); json.dumps(g.snapshot())
g.start(); walk(g)
s = g.snapshot(); assert s["phase"] == "CONFIRM" and s["ai"] is None and s["outcome"] is None, s
go(g, along(1), n=9)
s = json.loads(json.dumps(g.snapshot()))
assert s["phase"] == "RESULT" and s["ai"] == "paper" and s["player"] == "rock" and s["outcome"] == "ai" and s["score"]["ai"] == 1
assert s["a"] == {"x": 0.2, "y": 0.5} and s["b"] == {"x": 0.8, "y": 0.5} and s["gate_f"] == 0.25
g.reset_score(); assert sum(g.score.values()) == 0 and g.stats["ok"] == 0

# 13) hand_shape: búa/giấy/kéo dựng từ khoảng cách ngón tay
def fake(ext):                                              # ext: 4 ngón (trỏ, giữa, áp út, út), True = duỗi
    lm = np.zeros((21, 3), np.float32)
    for k, (tip, pip) in enumerate(zip((8, 12, 16, 20), (6, 10, 14, 18))):
        lm[pip, 1] = 1.0
        lm[tip, 1] = 2.0 if ext[k] else 0.5
    return lm
assert hand_shape(fake([1, 1, 1, 1])) == 0 and hand_shape(fake([0, 0, 0, 0])) == 1 and hand_shape(fake([1, 1, 0, 0])) == 2
assert hand_shape(np.full((21, 3), np.nan, np.float32)) == -1

print("Tất cả phép thử đều đạt.")
