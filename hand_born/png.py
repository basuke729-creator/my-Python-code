import cv2
import os

# =========================
# 設定
# =========================
input_video_path = r"C:/WorkAnalysis/movie_out3/0318_A_1.mp4"
output_folder = r"C:/WorkAnalysis/output_png/作業者A/1"

save_every_n_frames = 3   # ←ここで調整（例：3=10fps、6=5fps、30=1秒ごと）

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
print(f"FPS: {fps}")

frame_count = 0
saved_count = 0

# =========================
# フレーム保存
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 👇 指定間隔で保存
    if frame_count % save_every_n_frames == 0:
        filename = os.path.join(output_folder, f"frame_{saved_count:06d}.png")

        result, encoded_img = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])

        if result:
            encoded_img.tofile(filename)
            saved_count += 1

    frame_count += 1

cap.release()

print("処理が完了しました")
print(f"総フレーム数: {frame_count}")
print(f"保存枚数: {saved_count}")