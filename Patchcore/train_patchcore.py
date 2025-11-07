# train_patchcore_v2.py  (for anomalib 2.2.0)
import argparse, os, subprocess

YAML_TMPL = """\
model:
  class_path: anomalib.models.image.patchcore.Patchcore
  init_args:
    backbone: resnet50
    layers: [layer2, layer3]
    coreset_sampling_ratio: 0.01
{nn_block}
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
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=384)
    ap.add_argument("--faiss", action="store_true")  # --faiss を付けたら faiss を使う
    args = ap.parse_args()

    data_root = os.path.abspath(args.data)
    exp_dir   = os.path.abspath(args.out)
    os.makedirs(exp_dir, exist_ok=True)

    nn_block = ""
    if args.faiss:
        nn_block = "    nn_method:\n      name: faiss\n      n_neighbors: 9\n"

    cfg_path = os.path.join(exp_dir, "patchcore_cls.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(YAML_TMPL.format(
            data_root=data_root,
            exp_dir=exp_dir,
            imgsz=args.imgsz,
            nn_block=nn_block
        ))
    print("[INFO] config ->", cfg_path)

    subprocess.run(["anomalib", "train", "--config", cfg_path], check=True)
    print("[OK] Training finished")

if __name__ == "__main__":
    main()

