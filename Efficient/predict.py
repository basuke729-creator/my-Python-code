# predict.py
import argparse, os
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import timm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def load_model(ckpt_path: str, model_name: str, img_size: int, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    class_names = ckpt.get("class_names", None)
    num_classes = len(class_names) if class_names else ckpt["model"]["classifier.weight"].shape[0]
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model, class_names

def predict_one(model, img_path: Path, tf, device, class_names, topk=5):
    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    k = min(topk, probs.numel())
    conf, idx = torch.topk(probs, k)
    conf = conf.cpu().tolist()
    idx = idx.cpu().tolist()
    names = [class_names[i] if class_names else str(i) for i in idx]
    return list(zip(names, conf))

def iter_images(path: Path):
    if path.is_file():
        yield path
    else:
        for p in sorted(path.rglob("*")):
            if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}:
                yield p

def main():
    ap = argparse.ArgumentParser("Predict with EfficientNet-V2")
    ap.add_argument("--ckpt", required=True, help="best.ckpt のパス")
    ap.add_argument("--model", default="efficientnetv2_s")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--input", required=True, help="画像ファイル or ディレクトリ")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    tf = build_transform(args.img_size)
    model, class_names = load_model(args.ckpt, args.model, args.img_size, device)

    inp = Path(args.input)
    for img_path in iter_images(inp):
        res = predict_one(model, img_path, tf, device, class_names, args.topk)
        line = ", ".join([f"{n}:{c:.3f}" for n,c in res])
        print(f"{img_path.name} -> {line}")

if __name__ == "__main__":
    main()
