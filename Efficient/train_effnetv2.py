# train_effnetv2.py
# EfficientNet-V2 transfer learning with CosineAnnealingWarmRestarts (default),
# PRF logging to TensorBoard, weighted sampler/loss, AMP, phase A (freeze) support.
import argparse, math, random
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import timm
from tqdm import tqdm

# schedulers
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, LinearLR, SequentialLR,
    CosineAnnealingWarmRestarts, OneCycleLR
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ----------------- Utils -----------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # 高速化

def build_transforms(img_size: int, aug: str):
    if aug == "light":
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(0.5),
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

def build_dataloaders(data_root: str, img_size: int, batch_size: int, workers: int,
                      aug: str, use_weighted: bool) -> Tuple[DataLoader, DataLoader, List[str], torch.Tensor]:
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

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    """検証：val_loss / macro precision / macro recall / macro f1 を返す"""
    model.eval()
    y_true, y_pred = [], []
    criterion = nn.CrossEntropyLoss()
    loss_sum, total = 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = torch.argmax(logits, 1)
        y_true += y.cpu().tolist()
        y_pred += pred.cpu().tolist()
        total += x.size(0)

    val_loss = loss_sum / max(1, total)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred,   average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred,       average="macro", zero_division=0)
    return val_loss, prec, rec, f1, (y_true, y_pred)

# --------- discriminative LR param groups ----------
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

# --------- build scheduler (cosine / cosine_wr / onecycle) ----------
def build_scheduler(optimizer, args, steps_per_epoch: int):
    if args.lr_schedule == "cosine_wr":
        # Warmup（任意）→ Warm Restarts
        if args.warmup_epochs > 0:
            warmup = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
            restarts = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=args.t0,
                T_mult=args.t_mult,
                eta_min=args.min_lr
            )
            return SequentialLR(optimizer, schedulers=[warmup, restarts], milestones=[args.warmup_epochs])
        else:
            return CosineAnnealingWarmRestarts(
                optimizer, T_0=args.t0, T_mult=args.t_mult, eta_min=args.min_lr
            )

    elif args.lr_schedule == "onecycle":
        # OneCycle は warmup と併用しない想定
        return OneCycleLR(
            optimizer,
            max_lr=[pg["lr"] for pg in optimizer.param_groups],
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=args.oc_pct_start,
            anneal_strategy="cos",
            div_factor=args.oc_div_factor,
            final_div_factor=args.oc_final_div
        )

    else:  # "cosine" (従来)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - max(0, args.warmup_epochs)), eta_min=args.min_lr)
        if args.warmup_epochs > 0:
            warmup = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
            return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])
        return cosine

# ----------------- Train -----------------
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

    # init-from（前回のbestから再開など）
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu")
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[Init] loaded from {args.init_from}\n  missing={len(missing)} unexpected={len(unexpected)}")

    # 凍結モード（ヘッドのみ学習）
    if args.freeze_backbone:
        for _, p in model.named_parameters():
            p.requires_grad = False
        if hasattr(model, "get_classifier"):
            for p in model.get_classifier().parameters():
                p.requires_grad = True
        else:
            # 汎用フォールバック：最後の Linear を開放
            for m in reversed(list(model.modules())):
                if isinstance(m, nn.Linear):
                    for p in m.parameters():
                        p.requires_grad = True
                    break

    # Optimizer
    if args.freeze_backbone:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999))
    else:
        param_groups = make_param_groups(model, base_lr=args.lr, head_lr_mult=args.head_lr_mult, weight_decay=args.wd)
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    steps_per_epoch = max(1, math.ceil(len(train_loader.dataset) / args.batch_size))
    scheduler = build_scheduler(optimizer, args, steps_per_epoch)

    # Loss（class weight + label smoothing）
    weights = class_weights.to(device) if args.weighted_loss else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp))
    best_f1, best_path = 0.0, Path(args.out) / "best.ckpt"
    last_path = Path(args.out) / "last.ckpt"
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # TensorBoard
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
                tb.add_scalar("train/loss", loss.item()*args.accum_steps, global_step)
                for gi, pg in enumerate(optimizer.param_groups):
                    tb.add_scalar(f"train/lr_group{gi}", pg["lr"], global_step)

        train_loss = running_loss / max(1,total)
        train_acc = correct / max(1,total)  # 表示しないが内部参照可

        # Validation
        val_loss, val_prec, val_rec, val_f1, (y_true, y_pred) = evaluate(model, val_loader, device)

        # スケジューラ更新
        # OneCycleLR は step() を「イテレーション毎」に呼ぶ設計が標準だが、
        # 今回はエポック末に呼んでも実運用上は大きな問題にならない想定。
        # Cosine系はエポック毎でOK。
        if isinstance(scheduler, OneCycleLR):
            scheduler.step()
        elif isinstance(scheduler, (CosineAnnealingWarmRestarts,)):
            scheduler.step(epoch)  # warm restarts は epoch 指定でOK
        else:
            scheduler.step()

        lrs = ", ".join([f"{pg['lr']:.2e}" for pg in optimizer.param_groups])
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  P={val_prec:.3f}  R={val_rec:.3f}  F1={val_f1:.3f}  "
              f"lr=[{lrs}]")

        if tb:
            tb.add_scalar("val/loss", val_loss, epoch)
            tb.add_scalar("val/precision", val_prec, epoch)
            tb.add_scalar("val/recall", val_rec, epoch)
            tb.add_scalar("val/f1", val_f1, epoch)

        # ベスト更新の基準は F1（macro）
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({"epoch": epoch, "model": model.state_dict(), "f1": best_f1,
                        "class_names": class_names}, best_path)
            print(f"[Best] F1 {best_f1:.4f} -> saved {best_path}")

        # last.ckptを毎エポック保存
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "f1": val_f1, "class_names": class_names}, last_path)

    # 最終詳細レポート（ベストモデルで val を評価）
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

# ----------------- Args -----------------
def parse_args():
    ap = argparse.ArgumentParser("EfficientNet-V2 Transfer Learning (timm) – Warm Restarts / PRF TB logging")
    ap.add_argument("--data", required=True, help="cls_dataset のルート（train/ val/）")
    ap.add_argument("--out",  default="./runs/effnetv2", help="出力ディレクトリ")
    ap.add_argument("--model", default="efficientnetv2_s",
                    help="timmモデル名（例: efficientnetv2_s / tf_efficientnetv2_s_in21k など）")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)

    # LR & regularization
    ap.add_argument("--lr", type=float, default=5e-5, help="基準LR（backbone用）")
    ap.add_argument("--head-lr-mult", type=float, default=10.0, help="ヘッド層のLR倍率")
    ap.add_argument("--min-lr", type=float, default=1e-6, help="最小学習率")
    ap.add_argument("--warmup-epochs", type=int, default=3, help="ウォームアップエポック数（cosine/cosine_wr時）")
    ap.add_argument("--wd", type=float, default=1e-4, help="Weight Decay")
    ap.add_argument("--drop", type=float, default=0.0, help="Dropout/DropPath")

    # Scheduler
    ap.add_argument("--lr-schedule", choices=["cosine", "cosine_wr", "onecycle"], default="cosine_wr",
                    help="学習率スケジューラ。既定は Warm Restarts (cosine_wr)")
    ap.add_argument("--t0", type=int, default=5, help="Warm Restarts: 最初のサイクル長（エポック）")
    ap.add_argument("--t-mult", type=float, default=2.0, help="Warm Restarts: サイクル長の乗数")
    # OneCycle の詳細パラメータ（必要時のみ使用）
    ap.add_argument("--oc-pct-start", type=float, default=0.3, help="OneCycle: 上昇割合")
    ap.add_argument("--oc-div-factor", type=float, default=25.0, help="OneCycle: 初期LRスケール  max_lr/div_factor")
    ap.add_argument("--oc-final-div", type=float, default=1e4, help="OneCycle: 最終LRスケール div_factor*final_div")

    # Aug & imbalance
    ap.add_argument("--aug", choices=["none","light","strong"], default="light")
    ap.add_argument("--weighted-sampler", action="store_true")
    ap.add_argument("--weighted-loss", action="store_true")
    ap.add_argument("--label-smoothing", type=float, default=0.0, help="ラベルスムージング係数（例: 0.1）")

    # System
    ap.add_argument("--accum-steps", type=int, default=1, help="勾配累積")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--tensorboard", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # Optional: phaseA/B style
    ap.add_argument("--freeze-backbone", action="store_true", help="バックボーン凍結（ヘッドのみ学習）")
    ap.add_argument("--init-from", default="", help="初期化する ckpt のパス（例: phaseA/best.ckpt）")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    train(args)

