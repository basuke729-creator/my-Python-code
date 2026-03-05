import cv2
import numpy as np
from pathlib import Path

def side_by_side_half_and_half(
    left_video_path: str,
    right_video_path: str,
    out_path: str = "merged_half_half.mp4",
    out_fps: float | None = None,
    out_size: tuple[int, int] | None = None,   # (W, H). Noneなら「左動画のサイズ」を基準にする
    stop_when_one_ends: bool = True,           # True: どちらかが終わったら終了 / False: 短い方は黒で埋める
    codec: str = "mp4v",
):
    left_path = str(left_video_path)
    right_path = str(right_video_path)
    out_path = str(out_path)

    capL = cv2.VideoCapture(left_path)
    capR = cv2.VideoCapture(right_path)

    if not capL.isOpened():
        raise FileNotFoundError(f"左動画を開けません: {left_path}")
    if not capR.isOpened():
        raise FileNotFoundError(f"右動画を開けません: {right_path}")

    # 基準サイズ（出力の全体サイズ）
    wL = int(capL.get(cv2.CAP_PROP_FRAME_WIDTH))
    hL = int(capL.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fpsL = capL.get(cv2.CAP_PROP_FPS) or 30.0

    if out_size is None:
        out_w, out_h = wL, hL
    else:
        out_w, out_h = out_size

    # 出力FPS
    if out_fps is None:
        out_fps = fpsL if fpsL > 0 else 30.0

    # 左半分／右半分の幅
    half_w = out_w // 2

    # Writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(out_path, fourcc, out_fps, (out_w, out_h))

    if not writer.isOpened():
        raise RuntimeError("VideoWriterを開けません。codecや拡張子を見直してください。")

    # 黒フレーム（片方終了時用）
    black_half = np.zeros((out_h, half_w, 3), dtype=np.uint8)

    try:
        while True:
            retL, frameL = capL.read()
            retR, frameR = capR.read()

            if stop_when_one_ends:
                if (not retL) or (not retR):
                    break
            else:
                # 短い方は黒で埋める
                if not retL and not retR:
                    break
                if not retL:
                    frameL = None
                if not retR:
                    frameR = None

            # 左側
            if frameL is None:
                left_half = black_half
            else:
                # 高さをout_hに合わせ、幅はhalf_wに合わせる
                left_half = cv2.resize(frameL, (half_w, out_h), interpolation=cv2.INTER_AREA)

            # 右側
            if frameR is None:
                right_half = black_half
            else:
                right_half = cv2.resize(frameR, (half_w, out_h), interpolation=cv2.INTER_AREA)

            merged = np.hstack([left_half, right_half])
            writer.write(merged)

    finally:
        capL.release()
        capR.release()
        writer.release()

    print(f"出力しました: {out_path}")


if __name__ == "__main__":
    # ここを書き換えて使ってください
    left_video  = "left.mp4"
    right_video = "right.mp4"
    out_video   = "merged_half_half.mp4"

    side_by_side_half_and_half(
        left_video,
        right_video,
        out_path=out_video,
        out_fps=None,          # Noneなら左動画FPS基準
        out_size=None,         # Noneなら左動画サイズ基準
        stop_when_one_ends=True # True推奨（長さ違いでズレるのを避ける）
    )