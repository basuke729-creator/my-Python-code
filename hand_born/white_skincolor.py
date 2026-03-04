import cv2
import numpy as np

input_path  = r"D:/i-Pro/019_hand/data/3fps/1_bottom_part_3fps.mp4"
output_path = r"D:/i-Pro/019_hand/data/3fps/1_bottom_part_change-hand_3fps.mp4"

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise RuntimeError(f"動画を開けません: {input_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 1e-3:
    fps = 30.0

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

# -----------------------
# ① 白手袋のHSVしきい値（要調整）
# -----------------------
lower_white = np.array([0,   0, 185], dtype=np.uint8)
upper_white = np.array([179, 70, 255], dtype=np.uint8)

# -----------------------
# ② 肌色寄せパラメータ（要調整）
# -----------------------
target_hue = 15          # 10〜20あたりで調整
min_sat    = 35          # 肌っぽさの最低彩度
sat_cap    = 90
val_add    = 15          # 明るさ足す。明るすぎたら下げる/0/マイナスも可

# -----------------------
# ③ 「手袋っぽい塊」だけ残すための条件（ここが効く）
# -----------------------
# 手袋がありそうなROI（y開始を下げるほど“胸の作業着”を拾いにくい）
roi_y0 = int(H * 0.35)      # 0.30〜0.45で調整（おすすめは0.35〜0.40）
roi_y1 = int(H * 0.95)

roi_x0 = int(W * 0.15)      # 中央寄りにしたいなら0.20〜0.30へ
roi_x1 = int(W * 0.85)

# 面積フィルタ（手袋サイズに合わせて調整）
min_area = int(W * H * 0.002)   # 小さすぎる白ノイズ除外
max_area = int(W * H * 0.08)    # 作業着の大きい白塊を除外

# 残す塊の個数（左右手で2が基本。状況で3に）
keep_top_k = 2

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def keep_glove_components(mask_bin: np.ndarray) -> np.ndarray:
    """白マスクから、ROI内にある“手袋っぽい塊”だけ残す"""
    # 連結成分
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    if num <= 1:
        return mask_bin

    candidates = []
    for i in range(1, num):  # 0は背景
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]

        # ROI内
        if not (roi_x0 <= cx <= roi_x1 and roi_y0 <= cy <= roi_y1):
            continue

        # 面積条件
        if area < min_area or area > max_area:
            continue

        candidates.append((area, i))

    if not candidates:
        # 何も残らなかったら「元マスク」を返す（真っ黒回避）
        return mask_bin

    candidates.sort(reverse=True)  # 面積が大きい順
    chosen = candidates[:keep_top_k]

    out_mask = np.zeros_like(mask_bin)
    for _, idx in chosen:
        out_mask[labels == idx] = 255

    return out_mask

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 1) 白領域抽出
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # 2) ノイズ除去
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 3) 手袋っぽい塊だけ残す（重要）
    mask = keep_glove_components(mask)

    m = (mask > 0)

    # 4) マスク部だけ肌色へ寄せる
    hsv[..., 0] = np.where(m, target_hue, hsv[..., 0])

    s = hsv[..., 1].astype(np.int16)
    s_new = np.clip(s, 0, sat_cap).astype(np.uint8)
    s_new = np.where(m, np.maximum(s_new, min_sat), hsv[..., 1])
    hsv[..., 1] = s_new

    v = hsv[..., 2].astype(np.int16)
    v_new = np.clip(v + val_add, 0, 255).astype(np.uint8)
    hsv[..., 2] = np.where(m, v_new, hsv[..., 2])

    new_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # デバッグ表示（ROI枠を描いて確認したいなら）
    # cv2.rectangle(new_frame, (roi_x0, roi_y0), (roi_x1, roi_y1), (0,255,0), 2)

    out.write(new_frame)
    cv2.imshow("frame", new_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("done:", output_path)