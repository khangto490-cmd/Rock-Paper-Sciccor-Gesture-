"""
live_compare.py
----------------
So sánh model CŨ (static_model.pkl) và model MỚI (static_model_v2.pkl) trên CÙNG
một luồng webcam, cùng một lúc. Mỗi frame chạy cả hai model, nên cả hai nhìn
đúng cùng một cử động tay -- không bị lệch do mỗi lần Hi làm một kiểu.

Mỗi "ván" đo 2 thứ cho từng model:
  - SỚM: model đổi sang kéo/bao TRƯỚC lúc tay "định hình rõ" bao nhiêu mili-giây
         (số âm = đổi nhầm sau lúc tay đã định hình)
  - ĐỔI NON: model đổi khi tay vẫn còn gần như đang nắm (đây là lỗi kích hoạt nhầm)

Cách dùng (xem hướng dẫn chi tiết trong tin nhắn):
    1 = chọn đo KÉO     2 = chọn đo BAO     SPACE = bắt đầu 1 ván     Q = thoát
Mỗi ván: bấm SPACE -> giữ nắm tay ~1 giây -> duỗi ngón (kéo hoặc bao) và giữ ->
hạ tay xuống nắm lại. Khi thoát, chương trình in bảng tổng kết để Hi gửi lại.

Chỉ đo phía phần mềm (model + xử lý). KHÔNG đo độ trễ camera/màn hình.
"""

import time

import numpy as np

from features import extract_features, engineered_features_batch

OLD_MODEL_PATH = "static_model.pkl"
NEW_MODEL_PATH = "static_model_v2.pkl"

# Ngưỡng độ duỗi lấy từ analyze_extension.py
START_THR = {"keo": 1.18, "bao": 1.02}    # "đã rời dáng nắm"
FORMED_THR = {"keo": 1.76, "bao": 1.66}   # "đã định hình rõ"
RUN = 2             # cần 2 frame liên tiếp cùng nhãn mới tính là "đã đổi"
TIMEOUT_S = 1.0     # sau khi tay định hình, đợi tối đa chừng này giây cho model đổi nhãn
MODELS = ("old", "new")


def ext_value(feats63, gesture):
    """Độ duỗi: kéo = TB(ngón trỏ, giữa); bao = TB(4 ngón). Cùng cách đo ở analyze_extension.py."""
    P = np.asarray(feats63, dtype=np.float32).reshape(21, 3)
    tips = (8, 12) if gesture == "keo" else (8, 12, 16, 20)
    return float(np.mean([np.linalg.norm(P[i]) for i in tips]))


def first_run(labels, target, run=RUN):
    for i in range(len(labels) - run + 1):
        if all(l == target for l in labels[i:i + run]):
            return i
    return None


class RoundTracker:
    """Theo dõi 1 ván: nhận từng frame, trả về kết quả khi ván kết thúc."""

    def __init__(self, gesture):
        self.g = gesture
        self.t, self.ext = [], []
        self.labels = {m: [] for m in MODELS}

    def update(self, t, ext, old_label, new_label):
        self.t.append(t)
        self.ext.append(ext)
        self.labels["old"].append(old_label)
        self.labels["new"].append(new_label)

        formed = next((i for i, e in enumerate(self.ext) if e >= FORMED_THR[self.g]), None)
        if formed is None:
            return None                           # tay chưa định hình -> ván chưa xong

        flips = {m: first_run(self.labels[m], self.g) for m in MODELS}
        t_formed = self.t[formed]
        timed_out = (t - t_formed) > TIMEOUT_S
        if not timed_out and any(f is None for f in flips.values()):
            return None                           # còn model chưa đổi nhãn -> đợi thêm

        result = {}
        for m, f in flips.items():
            if f is None:
                result[m] = {"flip": False}
            else:
                result[m] = {"flip": True,
                             "lead_ms": (t_formed - self.t[f]) * 1000.0,
                             "premature": bool(self.ext[f] < START_THR[self.g])}
        return result


def summarize(rounds):
    """rounds: list các (gesture, result). In bảng tổng kết."""
    print("\n" + "=" * 70)
    print("TONG KET (so voi luc tay dinh hinh ro; ms duong = doi nhan SOM hon)")
    print("=" * 70)
    for g in ("keo", "bao"):
        rs = [r for gg, r in rounds if gg == g]
        if not rs:
            continue
        print(f"\n{g.upper()}: {len(rs)} van")
        for m, name in (("old", "model CU "), ("new", "model MOI")):
            flipped = [r[m] for r in rs if r[m]["flip"]]
            leads = [x["lead_ms"] for x in flipped]
            premature = sum(1 for x in flipped if x["premature"])
            never = len(rs) - len(flipped)
            med = f"{np.median(leads):7.0f} ms" if leads else "    n/a   "
            print(f"  {name}: som (trung vi) = {med} | "
                  f"doi non = {premature}/{len(rs)} van | khong doi = {never}/{len(rs)} van")
    print("\n'doi non' = model doi sang keo/bao khi tay van gan nhu dang nam (kich hoat nham).")


def main():
    import cv2
    import joblib
    import mediapipe as mp

    old = joblib.load(OLD_MODEL_PATH)
    new = joblib.load(NEW_MODEL_PATH)
    old_model, old_cls = old["model"], old["label_encoder"].classes_
    new_model, new_cls = new["model"], new["label_encoder"].classes_

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Khong mo duoc camera.")
        return

    gesture, tracker, rounds = "keo", None, []
    last_msg = "Bam 1 (keo) hoac 2 (bao), roi SPACE de bat dau van"
    mp_hands = mp.solutions.hands

    with mp_hands.Hands(max_num_hands=1, model_complexity=0,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            now = time.perf_counter()

            old_label = new_label = "-"
            feats = None
            if results.multi_hand_landmarks:
                feats = extract_features(results.multi_hand_landmarks[0].landmark)
            if feats is not None:
                x63 = np.array(feats, dtype=np.float32).reshape(1, -1)
                old_label = old_cls[int(old_model.predict(x63)[0])]
                new_label = new_cls[int(new_model.predict(engineered_features_batch(x63))[0])]

            if tracker is not None:
                if feats is None:
                    tracker = None
                    last_msg = "Mat tay -> huy van nay. Bam SPACE de lam lai"
                else:
                    res = tracker.update(now, ext_value(feats, gesture), old_label, new_label)
                    if res is not None:
                        rounds.append((gesture, res))
                        def fmt(r):
                            return (f"{r['lead_ms']:+.0f}ms{' (non)' if r['premature'] else ''}"
                                    if r["flip"] else "khong doi")
                        last_msg = (f"Van {len(rounds)}: cu {fmt(res['old'])} | moi {fmt(res['new'])}"
                                    f"  - SPACE de van tiep")
                        print(last_msg)
                        tracker = None

            state = "DANG DO - nam tay roi duoi ngon" if tracker else "cho SPACE"
            lines = [f"Do: {gesture.upper()}  [{state}]  so van: {len(rounds)}",
                     f"Cu: {old_label}   Moi: {new_label}", last_msg]
            for i, text in enumerate(lines):
                cv2.putText(frame, text, (10, 28 + 26 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("So sanh model cu va moi", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1') and tracker is None:
                gesture = "keo"
            elif key == ord('2') and tracker is None:
                gesture = "bao"
            elif key == ord(' '):
                tracker = RoundTracker(gesture)
                last_msg = "Nam tay ~1 giay, roi duoi ngon va giu"

    cap.release()
    cv2.destroyAllWindows()
    summarize(rounds)


if __name__ == "__main__":
    main()
