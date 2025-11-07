# train_patchcore.py
import argparse, os, subprocess, textwrap, datetime, pathlib

YAML_TMPL = """\
model:
  class_path: anomalib.models.image.patchcore.PatchCore
  init_args:
    backbone: resnet50
    layers: [layer2, layer3]
    coreset_sampling_ratio: 0.01
    nn_method:
      method: faiss      # faiss を使わないなら "sklearn"
      n_neighbors: 9

data:
  class_path: anomalib.data.folder.Folder
  init_args:
    root: /path/to/dataset        # ← ここをあなたの実パスに
    normal_dir: normal
    abnormal_dir: abnormal
    task: classification          # 画像レベル（ヒートマップ不要）
    image_size: 256
    test_split_mode: from_dir
    train_batch_size: 32
    eval_batch_size: 32

trainer:
  accelerator: auto
  max_epochs: 1

project:
  path: ./runs/patchcore_cls
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root (train/test 下に normal/abnormal)")
    ap.add_argument("--out",  required=True, help="experiment root output dir")
    args = ap.parse_args()

    data_root = os.path.abspath(args.data)
    exp_root  = os.path.abspath(args.out)
    os.makedirs(exp_root, exist_ok=True)

    # YAMLを書き出し
    cfg_dir = os.path.join(exp_root, "configs")
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = os.path.join(cfg_dir, "patchcore_cls.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(YAML_TMPL.format(data_root=data_root, exp_dir=exp_root))

    print(f"[INFO] config -> {cfg_path}")

    # anomalib train を実行
    # 参考: anomalib CLI ドキュメント（train/test/infer を一貫提供）:contentReference[oaicite:2]{index=2}
    cmd = ["anomalib", "train", "--config", cfg_path]
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    print("\n[OK] Training finished.")
    print(f"[hint] 次は推論・評価: python infer_patchcore_confmat.py --data {data_root} --exp {exp_root}")

if __name__ == "__main__":
    main()
