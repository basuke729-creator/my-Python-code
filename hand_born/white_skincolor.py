import cv2
import numpy as np


# =========================
# paths
# =========================
input_path  = r"D:/i-Pro/019_hand/data/3fps/1_bottom_part_3fps.mp4"
output_path = r"D:/i-Pro/019_hand/data/3fps/1_bottom_part_change-hand_3fps.mp4"


# =========================
# open video
# =========================
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


# =========================
# ① white-glove candidate (HSV)  ※少し緩め（動きで絞る）
# =========================
lower_white = np.array([0,   0, 140], dtype=np.uint8)
upper_white = np.array([179, 110, 255], dtype=np.uint8)

# =========================
# ② motion mask params
# =========================
diff_thresh = 18          # 小さいほど動きを拾いやすい（12〜30で調整）
use_roi = True

# ROI（手元中心）
roi_y0 = int(H * 0.30)
roi_y1 = int(H * 0.95)
roi_x0 = int(W * 0.05)
roi_x1 = int(W * 0.95)

# =========================
# ③ component filter
# =========================
min_area = int(W * H * 0.0005)
max_area = int(W * H * 0.12)
keep_top_k = 2  # 左右手。分裂するなら3

# =========================
# ④ skin-tone mapping (white glove -> stronger)
# =========================
target_hue = 12
min_sat    = 70
sat_cap    = 150
val_add    = -10

# =========================
# morph kernel
# =========================
kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 再構成用（小さめが安定）


def keep_glove_components(mask_bin: np.ndarray) -> np.ndarray:
    """ROI内＆面積条件で“手袋っぽい塊”だけ残す"""
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    if num <= 1:
        return mask_bin

    cand = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]

        if use_roi:
            if not (roi_x0 <= cx <= roi_x1 and roi_y0 <= cy <= roi_y1):
                continue

        if area < min_area or area > max_area:
            continue

        cand.append((area, i))

    if not cand:
        return mask_bin

    cand.sort(reverse=True)
    chosen = cand[:keep_top_k]

    out_mask = np.zeros_like(mask_bin)
    for _, idx in chosen:
        out_mask[labels == idx] = 255
    return out_mask


def morphological_reconstruction(seed: np.ndarray, mask: np.ndarray, max_iters: int = 120) -> np.ndarray:
    """
    seed を mask の中だけで反復膨張して埋める（形態学的再構成）
    seed, mask は 0/255 の uint8 を想定
    """
    prev = seed.copy()
    for _ in range(max_iters):
        dil = cv2.dilate(prev, kernel3, iterations=1)
        cur = cv2.bitwise_and(dil, mask)
        if cv2.countNonZero(cv2.absdiff(cur, prev)) == 0:
            break
        prev = cur
    return prev


# motion reference
prev_gray = None


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ROI
    if use_roi:
        roi = frame[roi_y0:roi_y1, roi_x0:roi_x1]
    else:
        roi = frame

    # -------------------------
    # motion mask
    # -------------------------
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if prev_gray is None:
        prev_gray = gray.copy()
        out.write(frame)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    diff = cv2.absdiff(gray, prev_gray)
    _, motion = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
    motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN,  kernel5, iterations=1)
    motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, kernel5, iterations=2)
    prev_gray = gray.copy()

    # -------------------------
    # white mask (HSV)
    # -------------------------
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv_roi, lower_white, upper_white)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN,  kernel5, iterations=1)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel5, iterations=2)

    # ==========================================================
    # KEY: seed -> reconstruction inside white (safe fill)
    # ==========================================================
    # seed（動いてる白）＝輪郭になってOK
    motion_d = cv2.dilate(motion, kernel5, iterations=2)
    seed = cv2.bitwise_and(white, motion_d)
    seed = cv2.dilate(seed, kernel5, iterations=1)

    # white側も軽く穴埋めしてから再構成
    white_fill = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel5, iterations=2)

    # 再構成：seed から white_fill の中だけ広げる
    mask_roi = morphological_reconstruction(seed, white_fill, max_iters=140)

    # 仕上げ（小穴を埋める）
    mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, kernel5, iterations=3)
    mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN,  kernel5, iterations=1)

    # “手袋っぽい塊”だけ残す
    mask_roi = keep_glove_components(mask_roi)

    # debug
    cv2.imshow("motion", motion)
    cv2.imshow("white", white)
    cv2.imshow("seed", seed)
    cv2.imshow("mask", mask_roi)

    # -------------------------
    # skin conversion
    # -------------------------
    m = (mask_roi > 0)

    hsv_roi[..., 0] = np.where(m, target_hue, hsv_roi[..., 0])

    s = hsv_roi[..., 1].astype(np.int16)
    s_new = np.clip(s, 0, sat_cap).astype(np.uint8)
    s_new = np.where(m, np.maximum(s_new, min_sat), hsv_roi[..., 1])
    hsv_roi[..., 1] = s_new

    v = hsv_roi[..., 2].astype(np.int16)
    v_new = np.clip(v + val_add, 0, 255).astype(np.uint8)
    hsv_roi[..., 2] = np.where(m, v_new, hsv_roi[..., 2])

    new_roi = cv2.cvtColor(hsv_roi, cv2.COLOR_HSV2BGR)

    new_frame = frame.copy()
    if use_roi:
        new_frame[roi_y0:roi_y1, roi_x0:roi_x1] = new_roi
    else:
        new_frame = new_roi

    out.write(new_frame)
    cv2.imshow("frame", new_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
out.release()
cv2.destroyAllWindows()
print("done:", output_path)