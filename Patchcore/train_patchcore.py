# train_patchcore_v2.py
import argparse, os, subprocess, textwrap

YAML_TMPL = """\
model:
  class_path: anomalib.models.image.patchcore.Patchcore
  init_args:
    backbone: resnet50
    layers: [layer2, layer3]
    coreset_sampling_ratio: 0.01
    nn_method:
      method: {nn_method}
      n_neighbors: 9

data:
  class_path: anomalib.data.folder.Folder
  init_args:
    root: {data_root}
    normal_dir: normal
    abnormal_dir: abnormal
    task: classification
    image_size: {imgsz}
    train_batch_size: 32
    eval_batch_size: 32
    test_split_mode: from_dir

trainer:
  accelerator: auto
  max_epochs: 1

project:
  path: {exp_dir}
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root")
    ap.add_argument("--out",  required=True, help="experiment dir")
    ap.add_argument("--imgsz", type=int, default=384)
    ap.add_argument("--nn", default="sklearn", choices=["sklearn","faiss"])
    args = ap.parse_args()

    data_root = os.path.abspath(args.data)
    exp_dir   = os.path.abspath(args.out)
    os.makedirs(exp_dir, exist_ok=True)

    cfg_path = os.path.join(exp_dir, "patchcore_cls.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(YAML_TMPL.format(
            data_root=data_root,
            exp_dir=exp_dir,
            imgsz=args.imgsz,
            nn_method=args.nn
        ))
    print("[INFO] config ->", cfg_path)

    subprocess.run(["anomalib", "train", "--config", cfg_path], check=True)
    print("[OK] Training finished at", exp_dir)

if __name__ == "__main__":
    main()

