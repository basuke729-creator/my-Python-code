import cv2
import os

# =========================
# 設定
# =========================
input_video_path = "input.mp4"
output_folder = "frames_png"

# =========================
# フォルダ作成
# =========================
os.makedirs(output_folder, exist_ok=True)

# =========================
# 動画読み込み
# =========================
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print("動画を開けませんでした")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"FPS: {fps}")
print(f"解像度: {width}x{height}")

frame_count = 0

# =========================
# フレーム保存
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    filename = os.path.join(output_folder, f"frame_{frame_count:06d}.png")

    # PNG保存（圧縮レベル0〜9：3がバランス良い）
    cv2.imwrite(filename, frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

    frame_count += 1

cap.release()

print(f"保存完了: {frame_count}枚")