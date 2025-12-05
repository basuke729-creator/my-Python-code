#!/usr/bin/env python3
"""
train_effnetv2.py が出した best.ckpt を
ONNX (opset 18) に変換するスクリプト。

ckpt の中身:
{
  "epoch": int,
  "model": state_dict,
  "f1": float,
  "class_names": list[str]
}
"""

import argparse
from pathlib import Path

import torch
import timm


def parse_args():
    ap = argparse.ArgumentParser("Export best.ckpt (EfficientNetV2) to ONNX")

    ap.add_argument(
        "--ckpt",
        required=True,
        help="train_effnetv2.py が出した best.ckpt のパス"
    )
    ap.add_argument(
        "--model",
        required=True,
        help="学習時に使った timm モデル名 (例: tf_efficientnetv2_m)"
    )
    ap.add_argument(
        "--img-size",
        type=int,
        default=384,
        help="入力画像サイズ (train と同じ値)"
    )
    ap.add_argument(
        "--out",
        default="model.onnx",
        help="出力する ONNX ファイルパス"
    )
    ap.add_argument(
        "--drop",
        type=float,
        default=0.0,
        help="モデル構築時の drop/drop_path。学習と合わせたいなら指定"
    )
    ap.add_argument(
        "--no-dynamic",
        action="store_true",
        help="指定すると dynamic_axes 無効 (batch 固定 N=1)"
    )

    return ap.parse_args()


def main():
    args = parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"ckpt が見つかりません: {ckpt_path}")

    print(f"[INFO] load checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # ★ ここがポイント：dict から state_dict と class_names を取り出す
    state_dict = ckpt["model"]
    class_names = ckpt["class_names"]
    num_classes = len(class_names)

    print(f"[INFO] num_classes = {num_classes}")
    print(f"[INFO] model name  = {args.model}")

    # train_effnetv2.py の build_model と同じ構築
    model = timm.create_model(
        args.model,
        pretrained=False,
        num_classes=num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()          # evalモード
    model.to("cpu")       # ONNX export は CPU でOK

    # ダミー入力 (N=1, C=3, H=W=img_size)
    img_size = args.img_size
    dummy_input = torch.randn(1, 3, img_size, img_size, requires_grad=False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    input_names = ["input"]
    output_names = ["output"]

    if args.no_dynamic:
        dynamic_axes = None
        print("[INFO] dynamic_axes = None (batch 固定 N=1)")
    else:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }
        print("[INFO] dynamic_axes あり (batch 次元可変)")

    print(f"[INFO] export ONNX -> {out_path} (opset_version=18)")
    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        export_params=True,
        opset_version=18,   # ★ op=18
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    print("[INFO] done. saved:", out_path.resolve())


if __name__ == "__main__":
    main()
