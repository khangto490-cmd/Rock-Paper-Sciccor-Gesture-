"""Luật game oẳn tù tì A -> B (bản chốt 06/11/2026, chép từ 22_GameSkeleton_AB; sửa 09/11/2026 theo feedback mục 3, từ 25_GameAB_v2). Không dính webcam hay model,
nên test được riêng bằng test_game_logic.py.

Một ván:
  SPACE -> NEED_A: nắm BÚA đưa vào ô A, giữ hold_a giây (hand_shape đọc là búa)
  -> GO: đi tay từ A tới B; từ lúc bắt đầu đi tới khi tay chạm VẠCH (gate_f của đường A->B) tay phải còn là búa,
         đổi tay trước vạch thì cảnh báo, quá fist_grace giây thì về A; chạm vạch rồi thì ra gì cũng được
  -> tay chạm B: model chốt đáp án (không random, không sửa) -> CONFIRM confirm_sec giây
  -> sau confirm_sec giây so CỬ CHỈ LÚC CHỐT (hình tay đa số trong pre_sec giây tới khung chốt, luật ngón tay)
     với CỬ CHỈ SAU CHỐT, không so với dự đoán của AI:
       giống nhau -> RESULT: chấm thắng/thua/hoà bình thường, kể cả khi AI đoán sai (người giữ nguyên tay)
       khác nhau  -> CANCEL (đổi tay): huỷ ván, không tính điểm, không lộ đáp án AI
       mất tay suốt cửa sổ -> RESULT với outcome "unknown": không tính điểm.
  summary() báo riêng ván hợp lệ / đổi tay / không xác định, tỷ lệ huỷ, độ phủ, độ chính xác AI lúc chốt.
"""
import collections
import numpy as np

CLASSES = ["paper", "rock", "scissors"]          # nhãn 0, 1, 2
COUNTER = {0: 2, 1: 0, 2: 1}                      # người ra paper -> AI ra scissors; rock -> paper; scissors -> rock

# ---------- Đọc hình tay bằng luật ngón tay (độc lập với model) ----------
TIP, PIP = [8, 12, 16, 20], [6, 10, 14, 18]       # trỏ, giữa, áp út, út (bỏ ngón cái)
SIGN = np.array([[1, 1, 1, 1], [-1, -1, -1, -1], [1, 1, -1, -1]], np.float32)   # mẫu duỗi(+)/gập(-): paper, rock, scissors


def finger_scores(lm):
    """lm (...,21,3) -> (...,4): log(đầu ngón / khớp giữa ngón) tính từ cổ tay; > 0 là duỗi, < 0 là gập."""
    d = np.linalg.norm(lm - lm[..., :1, :], axis=-1)
    return np.log((d[..., TIP] + 1e-6) / (d[..., PIP] + 1e-6))


def hand_shape(lm, margin=0.0):
    """0 paper | 1 rock | 2 scissors; -1 chỉ khi mất tay (hoặc 2 lớp đứng đầu cách nhau < margin, mặc định 0)."""
    score = finger_scores(lm) @ SIGN.T
    top = np.sort(score, axis=-1)
    out = np.where(top[..., -1] - top[..., -2] < margin, -1, score.argmax(-1))
    return np.where(np.isnan(lm).any(axis=(-1, -2)), -1, out)


def hand_point(lm, mirror=True, mode="palm"):
    """Điểm đại diện của bàn tay trên màn hình (toạ độ 0-1, lật gương nếu mirror).
    palm: trung bình cổ tay + 4 khớp gốc ngón (ít đổi khi xoè/nắm tay) | box: tâm ô bao quanh tay."""
    xy = lm[:, :2]
    p = (xy.min(0) + xy.max(0)) / 2 if mode == "box" else xy[[0, 5, 9, 13, 17]].mean(0)
    return np.array([1 - p[0] if mirror else p[0], p[1]], np.float32)


def majority_latest(votes):
    """Cử chỉ xuất hiện nhiều nhất; hòa số phiếu thì lấy cái xuất hiện muộn nhất (tay cuối cùng người chơi giữ). Rỗng -> None."""
    if not votes:
        return None
    counts = np.bincount(votes, minlength=3)
    tied = {c for c in range(3) if counts[c] == counts.max()}
    return int(next(v for v in reversed(votes) if v in tied))


# ---------- Máy trạng thái ----------
class GameAB:
    """Trạng thái: WAIT, NEED_A, GO, CONFIRM, RESULT, CANCEL. Gọi step() ở MỖI khung webcam (kể cả khung mất tay)."""

    def __init__(self, A=(0.2, 0.5), B=(0.8, 0.5), ar=4 / 3, box_r=0.13, band=0.09, off_max=0.5, lost_max=0.3,
                 K=1, confirm_sec=0.2, thr=0.6, hold_a=0.3, gate_f=0.25, fist_grace=0.1, pre_sec=0.1):
        self.A_n, self.B_n, self.ar = A, B, ar                       # A, B: toạ độ 0-1 trên màn hình đã lật gương (để vẽ)
        self.A, self.B = np.array([A[0] * ar, A[1]]), np.array([B[0] * ar, B[1]])   # để tính khoảng cách: đơn vị = chiều cao khung hình
        assert gate_f > box_r / np.linalg.norm(self.B - self.A), "Vạch phải nằm ngoài ô A (tăng gate_f hoặc giảm box_r)"
        self.box_r, self.band, self.off_max, self.lost_max = box_r, band, off_max, lost_max
        self.K, self.confirm_sec, self.thr, self.hold_a = K, confirm_sec, thr, hold_a
        self.pre_sec = pre_sec                                       # cử chỉ lúc chốt = hình tay (đa số) trong pre_sec giây tới khung chốt
        self.gate_f, self.fist_grace = gate_f, fist_grace            # gate_f: vị trí vạch trên đường A->B (0 là A, 1 là B) | fist_grace: sai hình tối đa (giây) trước vạch
        self.score = {"ai": 0, "player": 0, "draw": 0}
        self.stats = {"ok": 0, "switch": 0, "unknown": 0}            # đếm ván: hợp lệ (ok) / đổi tay = huỷ (switch) / không xác định (unknown)
        self.history = []                                            # mỗi ván một dòng; đáp án AI + mốc chốt ghi ngay lúc chốt và không bị sửa sau đó
        self.state, self.msg, self.base_msg = "WAIT", "Nhấn SPACE để chơi", ""
        self.off = self.locked_now = self.entered_go = False         # off: tay lệch đường | locked_now: đúng khung chốt | entered_go: đúng khung bắt đầu đi
        self.hold_since, self.a_hold = None, 0.0                     # a_hold: tiến độ giữ búa ở ô A, 0 đến 1
        self.passed = self.warn = False                              # passed: tay đã chạm vạch | warn: đang sai hình trước vạch
        self.bad_since = None
        self.guess = self.ai = self.player = self.outcome = self.lock_shape = None
        self.recent = collections.deque()

    def reset_score(self):
        self.score = {"ai": 0, "player": 0, "draw": 0}
        self.stats = {"ok": 0, "switch": 0, "unknown": 0}

    def start(self, t=None):
        """Bấm SPACE: chơi ván đầu, chơi tiếp sau kết quả, hoặc chơi lại sau khi huỷ. Đang giữa ván thì bỏ qua."""
        if self.state not in ("WAIT", "RESULT", "CANCEL"):
            return
        self.guess = self.ai = self.player = self.outcome = self.lock_shape = None
        self.recent = collections.deque()
        self.off, self.hold_since, self.a_hold = False, None, 0.0
        self.passed = self.warn = False
        self.bad_since = None
        self.state = "NEED_A"
        self.msg = self.base_msg = "Nắm tay thành BÚA, đưa vào ô A"

    def _xy(self, pt):
        return np.array([pt[0] * self.ar, pt[1]])

    def _seg_dist(self, p):
        ab = self.B - self.A
        f = np.clip(np.dot(p - self.A, ab) / np.dot(ab, ab), 0, 1)
        return float(np.linalg.norm(p - (self.A + f * ab)))

    def _prog(self, p):
        """Tay đã đi được bao nhiêu phần đường A->B (0 tại A, 1 tại B)."""
        ab = self.B - self.A
        return float(np.dot(p - self.A, ab) / np.dot(ab, ab))

    def _abort(self, msg):
        self.state, self.off, self.hold_since, self.a_hold = "NEED_A", False, None, 0.0
        self.passed = self.warn = False
        self.bad_since = None
        self.msg = self.base_msg = msg

    def step(self, t, pt, probs, shape):
        """Gọi mỗi khung webcam. pt: điểm tay (None nếu mất tay); probs: 3 xác suất (None nếu mất tay); shape: hình tay đọc bằng luật ngón tay."""
        self.locked_now = self.entered_go = False
        has = pt is not None and probs is not None
        p = self._xy(pt) if has else None
        if self.state == "NEED_A":
            in_a = has and np.linalg.norm(p - self.A) <= self.box_r
            if in_a and shape == 1:                                  # búa trong ô A: tính giờ giữ
                if self.hold_since is None:
                    self.hold_since = t
                self.a_hold = min(1.0, (t - self.hold_since) / self.hold_a)
                self.msg = "Giữ BÚA trong ô A"
                if self.a_hold >= 1.0 - 1e-9:                        # giữ đủ -> bắt đầu đi
                    self.state, self.msg, self.t_go, self.entered_go = "GO", "", t, True
                    self.lost_since = self.off_since = self.bad_since = None
                    self.passed = self.warn = False
                    self.hist = collections.deque([probs], maxlen=self.K); self.recent = collections.deque()
            else:                                                    # sai hình, ra khỏi ô A hoặc mất tay: giữ lại từ đầu
                self.hold_since, self.a_hold = None, 0.0
                self.msg = "Nắm tay thành BÚA (ở ô A)" if in_a else self.base_msg
        elif self.state == "GO":
            self._go(t, p, probs, has, shape)
        elif self.state == "CONFIRM":
            self._confirm(t, shape)

    def _go(self, t, p, probs, has, shape):
        if not has:                                                  # mất tay quá lost_max -> về chờ ở A
            self.lost_since = self.lost_since if self.lost_since is not None else t
            if t - self.lost_since > self.lost_max:
                self._abort("Mất tay! Đưa BÚA về ô A")
            return
        self.lost_since = None
        self.hist.append(probs)
        if shape is not None and shape >= 0:                         # nhớ hình tay (luật ngón tay) vài khung gần nhất để biết cử chỉ lúc chốt
            self.recent.append((t, int(shape)))
        while self.recent and t - self.recent[0][0] > self.pre_sec + 1e-9:
            self.recent.popleft()
        if not self.passed:                                          # chưa chạm vạch: tay phải còn là búa
            if self._prog(p) >= self.gate_f:
                self.passed, self.warn, self.bad_since = True, False, None     # chạm vạch (điểm tay chạm là đủ) -> được đổi tay
            elif shape == 1:
                self.warn, self.bad_since = False, None
            else:
                self.warn = True                                     # cảnh báo; sai hình quá fist_grace giây thì về A
                self.bad_since = self.bad_since if self.bad_since is not None else t
                if t - self.bad_since > self.fist_grace:
                    self._abort("Cảnh báo: phải giữ BÚA tới vạch! Đưa BÚA về ô A")
                    return
        near_a, near_b = np.linalg.norm(p - self.A) <= self.box_r, np.linalg.norm(p - self.B) <= self.box_r
        self.off = not (near_a or near_b or self._seg_dist(p) <= self.band)
        if self.off:                                                 # lệch đường: đỏ lên, đỏ quá off_max thì về chờ ở A
            self.off_since = self.off_since if self.off_since is not None else t
            if t - self.off_since > self.off_max:
                self._abort("Lệch đường! Đưa BÚA về ô A")
                return
        else:
            self.off_since = None
        if near_b:                                                   # chạm B: chốt đáp án ngay, không random, không sửa sau đó
            pm = np.mean(self.hist, axis=0)
            self.guess = int(pm.argmax())
            self.ai = COUNTER[self.guess]
            self.t_lock, self.win = t, []
            self.lock_shape = majority_latest([v for _, v in self.recent])   # cử chỉ người chơi lúc chốt (gồm khung chốt), không dùng dự đoán của AI
            self.state, self.locked_now, self.off = "CONFIRM", True, False
            self.history.append({"round": len(self.history) + 1, "tu_A_den_chot_s": round(t - self.t_go, 2),
                                 "guess": CLASSES[self.guess], "ai": CLASSES[self.ai],
                                 "lock_shape": None if self.lock_shape is None else CLASSES[self.lock_shape]})

    def _confirm(self, t, shape):
        if shape is not None and shape >= 0:                         # khung chốt không tính; bỏ qua khung mất tay
            self.win.append(int(shape))
        if t - self.t_lock < self.confirm_sec:
            return
        self.player = majority_latest(self.win)                      # hoà phiếu: lấy cử chỉ muộn nhất
        if self.player is None or self.lock_shape is None:           # không đọc được cử chỉ (mất tay suốt cửa sổ hoặc lúc chốt) -> không xác định, không tính điểm
            self.outcome, self.state, self.msg = "unknown", "RESULT", "Nhấn SPACE để chơi tiếp"
            self.stats["unknown"] += 1
        elif self.player != self.lock_shape:                         # đổi tay: cử chỉ sau chốt khác cử chỉ lúc chốt (KHÔNG so với dự đoán AI) -> huỷ ván, không lộ đáp án AI
            self.outcome, self.state = None, "CANCEL"
            self.msg = f"Tay lúc chạm B khác tay sau {self.confirm_sec:.1f} giây".replace(".", ",")
            self.stats["switch"] += 1
        else:                                                        # giữ nguyên tay: chấm thắng/thua/hoà bình thường, kể cả khi AI đoán sai
            d = (self.ai - self.player) % 3
            self.outcome = ("draw", "player", "ai")[d]
            self.score[self.outcome] += 1
            self.state, self.msg = "RESULT", "Nhấn SPACE để chơi tiếp"
            self.stats["ok"] += 1
        self.history[-1].update(player=None if self.player is None else CLASSES[self.player],
                                khung_xac_nhan=list(self.win), ket_qua=self.state, outcome=self.outcome,
                                ai_doan_dung=None if self.lock_shape is None else self.guess == self.lock_shape)

    def summary(self):
        """Tổng kết riêng từng loại ván: hợp lệ / đổi tay (huỷ) / không xác định, tỷ lệ huỷ, độ phủ, độ chính xác của AI lúc chốt
        (so dự đoán AI với cử chỉ lúc chốt đọc bằng luật ngón tay, chưa phải nhãn gán tay)."""
        n = sum(self.stats.values())
        dd = [h["ai_doan_dung"] for h in self.history if h.get("ai_doan_dung") is not None]
        r = lambda x: round(x / n, 3) if n else None
        return {"so_van": n, "hop_le": self.stats["ok"], "doi_tay": self.stats["switch"], "khong_xac_dinh": self.stats["unknown"],
                "ty_le_huy": r(self.stats["switch"]), "ty_le_khong_xac_dinh": r(self.stats["unknown"]), "do_phu": r(self.stats["ok"]),
                "ai_doan_dung_luc_chot": round(sum(dd) / len(dd), 3) if dd else None, "so_van_do_ai": len(dd),
                "ai_thang": self.score["ai"], "nguoi_thang": self.score["player"], "hoa": self.score["draw"]}

    def snapshot(self):
        """Trạng thái gửi cho giao diện. Đáp án AI và cử chỉ người chơi CHỈ có ở RESULT (CANCEL thì không lộ gì)."""
        done = self.state == "RESULT"
        name = lambda c: None if c is None else CLASSES[c]
        return {
            "phase": self.state,
            "msg": self.msg,
            "a": {"x": self.A_n[0], "y": self.A_n[1]},
            "b": {"x": self.B_n[0], "y": self.B_n[1]},
            "box_r": self.box_r,
            "band": self.band,
            "gate_f": self.gate_f,
            "a_hold": round(float(self.a_hold), 3),
            "passed": bool(self.passed),
            "warn": bool(self.warn),
            "off_line": bool(self.off),
            "outcome": self.outcome if done else None,
            "ai": name(self.ai) if done else None,
            "player": name(self.player) if done else None,
            "score": dict(self.score),
            "cancels": self.stats["switch"],                           # số ván bị huỷ vì đổi tay
            "unknowns": self.stats["unknown"],
        }
