# train_effnetv2.py
import argparse, time, math, os, random
from pathlib import Path
from typing import Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import timm
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # 高速化

def build_transforms(img_size: int, aug: str):
    # 代表的な前処理＋軽いAugment
    if aug == "light":
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.1,0.1,0.1,0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    elif aug == "strong":
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf

def build_dataloaders(data_root: str, img_size: int, batch_size: int, workers: int, aug: str,
                      use_weighted: bool) -> Tuple[DataLoader, DataLoader, List[str], torch.Tensor]:
    train_tf, val_tf = build_transforms(img_size, aug)

    train_dir = Path(data_root) / "train"
    val_dir   = Path(data_root) / "val"
    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(val_dir), transform=val_tf)
    class_names = train_ds.classes

    # クラス不均衡が大きい場合、Weighted Samplerを使う
    sampler = None
    class_weights = torch.ones(len(class_names), dtype=torch.float)
    if use_weighted:
        counts = np.bincount([y for _, y in train_ds.samples], minlength=len(class_names))
        # 1 / count を重み化
        weights_per_class = 1.0 / np.clip(counts, 1, None)
        weights_per_sample = [weights_per_class[y] for _, y in train_ds.samples]
        sampler = WeightedRandomSampler(weights_per_sample, num_samples=len(weights_per_sample), replacement=True)
        class_weights = torch.tensor(weights_per_class, dtype=torch.float)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=(sampler is None),
                              sampler=sampler, num_workers=workers, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=True, drop_last=False)
    return train_loader, val_loader, class_names, class_weights

def build_model(model_name: str, num_classes: int, pretrained: bool, drop: float):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, drop=drop)
    return model

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * x.size(0)
            pred = torch.argmax(logits, 1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return correct / max(1,total), loss_sum / max(1,total)

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[Device] {device}")

    train_loader, val_loader, class_names, class_weights = build_dataloaders(
        args.data, args.img_size, args.batch_size, args.workers, args.aug, args.weighted_sampler
    )
    num_classes = len(class_names)
    model = build_model(args.model, num_classes, pretrained=not args.no_pretrained, drop=args.drop)
    model.to(device)

    if args.freeze_backbone:
        # 最終分類層以外を凍結（timmは通常 model.get_classifier() でheadを返す）
        for name, p in model.named_parameters():
            p.requires_grad = False
        # 分類ヘッドだけ学習
        if hasattr(model, "get_classifier"):
            head = model.get_classifier()
            for p in head.parameters():
                p.requires_grad = True
        else:
            # 汎用フォールバック：最後の線形を開放
            for m in reversed(list(model.modules())):
                if isinstance(m, nn.Linear):
                    for p in m.parameters():
                        p.requires_grad = True
                    break

    # Optimizer / Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999))
    total_steps = args.epochs * math.ceil(len(train_loader.dataset) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 損失（クラス重み可）＋Label Smoothing
    weights = class_weights.to(device) if args.weighted_loss else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp))
    best_acc, best_path = 0.0, Path(args.out) / "best.ckpt"
    last_path = Path(args.out) / "last.ckpt"
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # TensorBoard（任意）
    tb = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(log_dir=str(Path(args.out) / "tb"))

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{args.epochs}]")
        running_loss = 0.0
        correct = 0; total = 0

        optimizer.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(pbar, 1):
            x, y = x.to(device), y.to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.amp)):
                logits = model(x)
                loss = criterion(logits, y) / args.accum_steps
            scaler.scale(loss).backward()

            if i % args.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * x.size(0) * args.accum_steps
            pred = torch.argmax(logits, 1)
            correct += (pred == y).sum().item()
            total += x.size(0)
            global_step += 1

            if tb and global_step % 20 == 0:
                tb.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                tb.add_scalar("train/loss", loss.item()*args.accum_steps, global_step)

        train_acc = correct / max(1,total)
        train_loss = running_loss / max(1,total)

        # 検証
        val_acc, val_loss = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={optimizer.param_groups[0]['lr']:.2e}")

        if tb:
            tb.add_scalar("val/acc", val_acc, epoch)
            tb.add_scalar("val/loss", val_loss, epoch)

        # ベスト更新で保存
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"epoch": epoch, "model": model.state_dict(), "acc": best_acc,
                        "class_names": class_names}, best_path)
            print(f"[Best] {best_acc:.4f} -> saved {best_path}")

        # 毎エポック末にラストも保存
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "acc": val_acc, "class_names": class_names}, last_path)

    # 最終レポート（検証セットで詳細）
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"]); model.to(device); model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = torch.argmax(logits, 1)
            y_true += y.cpu().tolist(); y_pred += pred.cpu().tolist()

    print("\n=== Classification Report (best on val) ===")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

def parse_args():
    ap = argparse.ArgumentParser("EfficientNet-V2 Transfer Learning (timm)")
    ap.add_argument("--data", required=True, help="cls_dataset のルート（train/ val/ があるパス）")
    ap.add_argument("--out",  default="./runs/effnetv2", help="出力（チェックポイント/ログ）")
    ap.add_argument("--model", default="efficientnetv2_s",
                    help="timm のモデル名（例: efficientnetv2_s / _m / _l / tf_efficientnetv2_s_in21k など）")
    ap.add_argument("--img-size", type=int, default=384, help="入力画像サイズ（正方）")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4, help="Weight Decay")
    ap.add_argument("--drop", type=float, default=0.0, help="Dropout/DropPath (timm側で有効)")

    ap.add_argument("--aug", choices=["none","light","strong"], default="light", help="データ拡張プリセット")
    ap.add_argument("--freeze-backbone", action="store_true", help="特徴抽出器を凍結してヘッドのみ学習")
    ap.add_argument("--weighted-sampler", action="store_true", help="クラス不均衡対策: サンプラー")
    ap.add_argument("--weighted-loss", action="store_true", help="クラス不均衡対策: 重み付き損失")
    ap.add_argument("--label-smoothing", type=float, default=0.0)

    ap.add_argument("--accum-steps", type=int, default=1, help="勾配累積（実質大きなバッチに）")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", help="自動混合精度 (FP16)")
    ap.add_argument("--tensorboard", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true", help="ImageNet事前学習を使わない")
    ap.add_argument("--cpu", action="store_true", help="強制CPU実行")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    train(args)
