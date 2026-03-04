import cv2
import numpy as np

# -------------------------
# paths
# -------------------------
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

# -------------------------
# white mask in HSV
#   白 = S低い + V高い
# -------------------------
lower_white = np.array([0,   0, 150], dtype=np.uint8)    # V下限（白っぽさ）
upper_white = np.array([179, 90, 255], dtype=np.uint8)   # S上限（彩度が低い）

# -------------------------
# skin-like mapping (HSV)
# -------------------------
target_hue = 12     # 肌色寄りのH（10〜18で調整）
min_sat    = 75     # 肌っぽさ（上げるほど“肌色”が濃くなる）
sat_cap    = 170    # 彩度の上限
val_add    = -10    # 明るさ調整（白っぽさを消すならマイナス推奨）

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 1) 白っぽい領域を抽出
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # 2) ノイズ除去 & 穴埋め（白のムラを整える）
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    m = (mask > 0)

    # 3) Hを肌色に寄せる
    hsv[..., 0] = np.where(m, target_hue, hsv[..., 0])

    # 4) Sを肌色っぽく付ける（白はSが低いので、ここが重要）
    s = hsv[..., 1].astype(np.int16)
    s_new = np.clip(s, 0, sat_cap).astype(np.uint8)
    s_new = np.where(m, np.maximum(s_new, min_sat), hsv[..., 1])
    hsv[..., 1] = s_new

    # 5) Vで白っぽさを調整
    v = hsv[..., 2].astype(np.int16)
    v_new = np.clip(v + val_add, 0, 255).astype(np.uint8)
    hsv[..., 2] = np.where(m, v_new, hsv[..., 2])

    new_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    out.write(new_frame)
    cv2.imshow("mask", mask)
    cv2.imshow("frame", new_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("done:", output_path)