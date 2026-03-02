import cv2
import numpy as np

input_path  = r"D:/i-Pro/019_hand/data/3fps/1_bottom_part_3fps.mp4"
output_path = r"D:/i-Pro/019_hand/data/3fps/1_bottom_part_change-hand_3fps.mp4"

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise RuntimeError(f"動画を開けません: {input_path}")

# 動画情報
fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 1e-3:   # 0になる環境があるので保険
    fps = 30.0

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# -----------------------
# パラメータ（調整ポイント）
# -----------------------

# 白手袋マスク（HSV）
# 白は「Sが低い」「Vが高い」
lower_white = np.array([0,   0, 180], dtype=np.uint8)
upper_white = np.array([179, 70, 255], dtype=np.uint8)

# 肌色っぽいHue（OpenCVのHは0-179）
# 15前後 = オレンジ寄り。人肌はだいたい 5〜25 くらいで調整することが多い
target_hue = 15

# 変換の強さ（値が小さいほど白っぽさが残る / 大きいほど肌色が濃くなる）
sat_cap = 90      # 変換後Sの上限（肌の彩度の上限）
sat_mult = 0.6    # 元のSに掛ける倍率（白はSほぼ0なので実質 cap が効く）
val_add = 20      # Vに加算（明るさ調整）

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 1) 白領域を抽出
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # 2) ノイズ除去（白い点や小さなゴミを減らす）
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 3) 「最大の白領域」だけ残す（背景の白を巻き込みにくくする）
    #    ※背景に大きな白がある場合は逆にそれを拾うので、その場合はこのブロックをOFFにして調整してください
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        max_i = int(np.argmax(areas))
        new_mask = np.zeros_like(mask)
        cv2.drawContours(new_mask, contours, max_i, 255, thickness=-1)
        mask = new_mask

    # マスクを0/1に
    m = (mask > 0)

    # 4) Hueを肌色に寄せる（白手袋領域だけ）
    hsv[..., 0] = np.where(m, target_hue, hsv[..., 0])

    # 5) 彩度Sを「肌っぽく」少し付ける（白はSが0に近いので、ここが重要）
    #    まず元Sに倍率、上限で抑える → さらに最低でも少し彩度を持たせたい場合は下限も入れる
    s_new = np.clip(hsv[..., 1].astype(np.float32) * sat_mult, 0, sat_cap).astype(np.uint8)
    # 白はSがほぼ0なので、マスク領域は「最低彩度」を入れると肌っぽくなりやすい
    min_sat = 35
    s_new = np.where(m, np.maximum(s_new, min_sat), hsv[..., 1])
    hsv[..., 1] = s_new.astype(np.uint8)

    # 6) 明度Vを調整（明るすぎるならval_addを下げる/マイナスにする）
    v_new = np.clip(hsv[..., 2].astype(np.int16) + val_add, 0, 255).astype(np.uint8)
    hsv[..., 2] = np.where(m, v_new, hsv[..., 2])

    new_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    out.write(new_frame)
    cv2.imshow("frame", new_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("done:", output_path)