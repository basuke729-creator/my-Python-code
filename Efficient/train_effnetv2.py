# train_effnetv2.py  (manual Phase A/B ready: discriminative LR + warmup+cosine + init-from)
import argparse, time, math, os, random
from pathlib import Path
from typing import Tuple, List, Dict
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

    sampler = None
    class_weights = torch.ones(len(class_names), dtype=torch.float)
    if use_weighted:
        counts = np.bincount([y for _, y in train_ds.samples], minlength=len(class_names))
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
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop,
        drop_path_rate=drop
    )
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

# --------- 追加：判別LR用のパラメタグループ ----------
def make_param_groups(model: nn.Module, base_lr: float, head_lr_mult: float, weight_decay: float):
    head_names = ("classifier", "fc")
    back_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(hn in n for hn in head_names):
            head_params.append(p)
        else:
            back_params.append(p)
    groups = []
    if back_params:
        groups.append({"params": back_params, "lr": base_lr, "weight_decay": weight_decay})
    if head_params:
        groups.append({"params": head_params, "lr": base_lr * head_lr_mult, "weight_decay": weight_decay})
    return groups

# --------- 追加：Warmup + Cosine スケジューラ ----------
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
def build_scheduler(optimizer, total_epochs: int, warmup_epochs: int, min_lr: float):
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - max(0, warmup_epochs)), eta_min=min_lr)
    if warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    return cosine

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

    # 追加：初期重みロード（フェーズBなどで使用）
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu")
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[Init] loaded from {args.init_from}\n  missing={len(missing)} unexpected={len(unexpected)}")

    # フェーズA：バックボーン凍結
    if args.freeze_backbone:
        for name, p in model.named_parameters():
            p.requires_grad = False
        if hasattr(model, "get_classifier"):
            head = model.get_classifier()
            for p in head.parameters():
                p.requires_grad = True
        else:
            for m in reversed(list(model.modules())):
                if isinstance(m, nn.Linear):
                    for p in m.parameters():
                        p.requires_grad = True
                    break

    # Optimizer：フェーズAは単一LR、フェーズBは判別LR
    if args.freeze_backbone:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999))
    else:
        param_groups = make_param_groups(model, base_lr=args.lr, head_lr_mult=args.head_lr_mult, weight_decay=args.wd)
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    # Scheduler：フェーズAは短めWarmupでもOK、フェーズBは指定Warmup
    warmup_epochs = args.warmup_epochs if not args.freeze_backbone else min(args.warmup_epochs, 1)
    scheduler = build_scheduler(optimizer, total_epochs=args.epochs, warmup_epochs=warmup_epochs, min_lr=args.min_lr)

    # Loss（クラス重み/LabelSmoothing）
    weights = class_weights.to(device) if args.weighted_loss else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp))
    best_acc, best_path = 0.0, Path(args.out) / "best.ckpt"
    last_path = Path(args.out) / "last.ckpt"
    Path(args.out).mkdir(parents=True, exist_ok=True)

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
                tb.add_scalar("train/lr_group0", optimizer.param_groups[0]["lr"], global_step)
                if len(optimizer.param_groups) > 1:
                    tb.add_scalar("train/lr_head", optimizer.param_groups[1]["lr"], global_step)
                tb.add_scalar("train/loss", loss.item()*args.accum_steps, global_step)

        train_acc = correct / max(1,total)
        train_loss = running_loss / max(1,total)

        # 検証
        val_acc, val_loss = evaluate(model, val_loader, device)
        scheduler.step()

        # ログ
        lrs = [pg["lr"] for pg in optimizer.param_groups]
        lr_str = ", ".join([f"{lr:.2e}" for lr in lrs])
        print(f"Epoch {epoch}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr=[{lr_str}]")

        if tb:
            tb.add_scalar("val/acc", val_acc, epoch)
            tb.add_scalar("val/loss", val_loss, epoch)

        # ベスト更新で保存
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"epoch": epoch, "model": model.state_dict(), "acc": best_acc,
                        "class_names": class_names}, best_path)
            print(f"[Best] {best_acc:.4f} -> saved {best_path}")

        # ラストも保存
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "acc": val_acc, "class_names": class_names}, last_path)

    # 最終レポート
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
    ap = argparse.ArgumentParser("EfficientNet-V2 Transfer Learning (timm) – manual Phase A/B")
    ap.add_argument("--data", required=True, help="cls_dataset のルート（train/ val/ があるパス）")
    ap.add_argument("--out",  default="./runs/effnetv2", help="出力（チェックポイント/ログ）")
    ap.add_argument("--model", default="tf_efficientnetv2_s_in21k_ft_in1k",
                    help="timm のモデル名（例: tf_efficientnetv2_s_in21k_ft_in1k）")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)

    # LR/WD
    ap.add_argument("--lr", type=float, default=1e-4, help="基準LR（backbone用）。ヘッドは head-lr-mult 倍")
    ap.add_argument("--head-lr-mult", type=float, default=10.0, help="ヘッド層のLR倍率")
    ap.add_argument("--min-lr", type=float, default=1e-6, help="Cosine の到達下限LR")
    ap.add_argument("--warmup-epochs", type=int, default=3, help="ウォームアップエポック数（B推奨:3）")
    ap.add_argument("--wd", type=float, default=1e-4, help="Weight Decay")
    ap.add_argument("--drop", type=float, default=0.0, help="Dropout/DropPath (timm側で有効)")

    # 既存
    ap.add_argument("--aug", choices=["none","light","strong"], default="light")
    ap.add_argument("--freeze-backbone", action="store_true", help="特徴抽出器を凍結してヘッドのみ学習（Phase A）")
    ap.add_argument("--weighted-sampler", action="store_true")
    ap.add_argument("--weighted-loss", action="store_true")
    ap.add_argument("--label-smoothing", type=float, default=0.0)

    ap.add_argument("--accum-steps", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--tensorboard", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # 追加：初期化
    ap.add_argument("--init-from", default="", help="事前の best.ckpt などから重みを初期化（Phase Bで使用）")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    train(args)

