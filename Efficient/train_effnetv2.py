import argparse, math, os, random
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import timm
from tqdm import tqdm

# ---------- 基本定数 ----------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------- ユーティリティ ----------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # 高速化


def build_transforms(img_size: int, aug: str):
    # 代表的な前処理＋Augment
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


def build_dataloaders(
    data_root: str,
    img_size: int,
    batch_size: int,
    workers: int,
    aug: str,
    use_weighted: bool
) -> Tuple[DataLoader, DataLoader, List[str], torch.Tensor]:

    train_tf, val_tf = build_transforms(img_size, aug)
    train_dir = Path(data_root) / "train"
    val_dir   = Path(data_root) / "val"

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(val_dir), transform=val_tf)
    class_names = train_ds.classes  # これを ckpt に保存する

    # クラス不均衡対応
    sampler = None
    class_weights = torch.ones(len(class_names), dtype=torch.float)
    if use_weighted:
        counts = np.bincount([y for _, y in train_ds.samples], minlength=len(class_names))
        weights_per_class = 1.0 / np.clip(counts, 1, None)
        weights_per_sample = [weights_per_class[y] for _, y in train_ds.samples]
        sampler = WeightedRandomSampler(weights_per_sample, num_samples=len(weights_per_sample), replacement=True)
        class_weights = torch.tensor(weights_per_class, dtype=torch.float)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False
    )
    return train_loader, val_loader, class_names, class_weights


def build_model(model_name: str, num_classes: int, pretrained: bool, drop: float):
    # timm EfficientNetV2
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop,
        drop_path_rate=drop
    )
    return model


# === ここから追加: 部分フリーズ機能 =========================
def freeze_upto(model: nn.Module, max_block: int):
    """
    EfficientNetV2 の shallow 層を段階的にフリーズする。
    max_block の意味:
      -1 : フリーズしない（全層学習）
       0 : stem までフリーズ
       1 : stem + blocks.0 までフリーズ
       2 : stem + blocks.0..1 までフリーズ
       3 : stem + blocks.0..2 までフリーズ
       ...
    """
    # まず全層を学習可に戻す
    for p in model.parameters():
        p.requires_grad = True

    if max_block < 0:
        print("[Freeze] なし（全層学習）")
        return

    # stem をフリーズ
    if hasattr(model, "stem"):
        for p in model.stem.parameters():
            p.requires_grad = False

    # blocks[i] を順番にフリーズ
    if hasattr(model, "blocks"):
        for i, block in enumerate(model.blocks):
            if i <= max_block:
                for p in block.parameters():
                    p.requires_grad = False

    print(f"[Freeze] stem + blocks.0〜{max_block} をフリーズしました。")
# ============================================================


# --------- discriminative LR param groups ----------
# （backboneとheadでLRを変える。headは早く学ばせたい）
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


# --------- warmup + cosine learning rate ----------
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
def build_scheduler(optimizer, total_epochs: int, warmup_epochs: int, min_lr: float, freeze_mode: bool):
    """
    freeze_mode=True のとき(=freezeありのとき)
    → backboneほぼ固めていることが多いので長いウォームアップはあまり意味がない。
      なのでウォームアップを短くする or 0扱いにして扱いやすくする。
    """
    if freeze_mode:
        warm = min(warmup_epochs, 1)  # 凍結ありならウォームアップ短めで十分
    else:
        warm = warmup_epochs

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs - max(0, warm)),
        eta_min=min_lr
    )
    if warm > 0:
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warm)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warm])
    else:
        return cosine


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    """
    val用の精度/損失に加えて precision / recall も返す
    （macro平均ベースでざっくりクオリティを見る）
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_num = 0

    all_true = []
    all_pred = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = torch.argmax(logits, 1)
        total_correct += (pred == y).sum().item()
        total_num += x.size(0)

        all_true.extend(y.cpu().tolist())
        all_pred.extend(pred.cpu().tolist())

    acc = total_correct / max(1, total_num)
    avg_loss = total_loss / max(1, total_num)

    # precision/recall/f1 (macro平均)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_true, all_pred, average="macro", zero_division=0
    )

    return acc, avg_loss, prec, rec, f1


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[Device] {device}")

    # Data
    train_loader, val_loader, class_names, class_weights = build_dataloaders(
        args.data, args.img_size, args.batch_size, args.workers, args.aug, args.weighted_sampler
    )
    num_classes = len(class_names)

    # Model
    model = build_model(args.model, num_classes, pretrained=not args.no_pretrained, drop=args.drop)
    model.to(device)

    # もし init-from が指定されていたら（既存チェックポイントから復元）
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu")
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[Init] loaded from {args.init_from}\n  missing={len(missing)} unexpected={len(unexpected)}")

    # ★★ ここが今回の追加ポイント ★★
    # freeze-upto が -1 以外なら、指定ブロックまでフリーズ
    if args.freeze_upto is not None and args.freeze_upto >= 0:
        freeze_upto(model, args.freeze_upto)
        freeze_mode = True
    else:
        freeze_mode = False

    # もし --freeze-backbone が指定されていたら「ほぼ全部の特徴抽出を停止してheadだけ学習」
    # これは従来仕様そのまま維持
    if args.freeze_backbone:
        for name, p in model.named_parameters():
            p.requires_grad = False
        if hasattr(model, "get_classifier"):
            for p in model.get_classifier().parameters():
                p.requires_grad = True
        else:
            # fallback: 最後に出てくる Linear を開放
            for m in reversed(list(model.modules())):
                if isinstance(m, nn.Linear):
                    for p in m.parameters():
                        p.requires_grad = True
                    break
        freeze_mode = True  # backboneフリーズもfreeze_mode扱いに含める

    # Optimizer:
    #   - freeze系の有無に関わらず、
    #     discriminative LR（ヘッドLR倍率）を維持
    param_groups = make_param_groups(
        model,
        base_lr=args.lr,
        head_lr_mult=args.head_lr_mult,
        weight_decay=args.wd
    )
    # param_groups は requires_grad=True のパラメータだけ含まれるようにしたい
    # → make_param_groups はすでに requires_grad を見てるのでOK

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    # スケジューラ（ウォーム＋コサイン）
    scheduler = build_scheduler(
        optimizer,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        freeze_mode=freeze_mode
    )

    # 損失関数（重み付き + label smoothing）
    weights = class_weights.to(device) if args.weighted_loss else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp))

    # 出力先
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.ckpt"
    last_path = out_dir / "last.ckpt"

    # TensorBoard
    tb = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(log_dir=str(out_dir / "tb"))

    best_acc = 0.0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{args.epochs}]")
        running_loss = 0.0
        correct = 0
        total = 0

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

            # TensorBoard: 学習中のlr, lossの途中経過
            if tb and global_step % 20 == 0:
                for gi, pg in enumerate(optimizer.param_groups):
                    tb.add_scalar(f"train/lr_group{gi}", pg["lr"], global_step)
                tb.add_scalar("train/loss_iter", loss.item()*args.accum_steps, global_step)

        train_acc = correct / max(1, total)
        train_loss = running_loss / max(1, total)

        # 検証
        val_acc, val_loss, val_prec, val_rec, val_f1 = evaluate(model, val_loader, device)
        scheduler.step()

        lrs = ", ".join([f"{pg['lr']:.2e}" for pg in optimizer.param_groups])
        print(
            f"Epoch {epoch}: "
            f"train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"prec={val_prec:.4f} rec={val_rec:.4f} f1={val_f1:.4f} "
            f"lr=[{lrs}]"
        )

        # TensorBoard: エポック単位のまとめ
        if tb:
            tb.add_scalar("train/loss", train_loss, epoch)
            tb.add_scalar("train/acc", train_acc, epoch)
            tb.add_scalar("val/loss",   val_loss, epoch)
            tb.add_scalar("val/acc",    val_acc, epoch)
            tb.add_scalar("val/prec",   val_prec, epoch)
            tb.add_scalar("val/rec",    val_rec, epoch)
            tb.add_scalar("val/f1",     val_f1, epoch)

        # ベストモデル保存
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "acc": best_acc,
                    "class_names": class_names,
                },
                best_path,
            )
            print(f"[Best] {best_acc:.4f} -> saved {best_path}")

        # lastモデルも保存
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "acc": val_acc,
                "class_names": class_names,
            },
            last_path,
        )

    # 最終レポート: ベストモデルで val を詳細に見る
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = torch.argmax(logits, 1)
            y_true += y.cpu().tolist()
            y_pred += pred.cpu().tolist()

    print("\n=== Classification Report (best on val) ===")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


def parse_args():
    ap = argparse.ArgumentParser("EfficientNet-V2 Transfer Learning (timm, freeze対応)")
    ap.add_argument("--data", required=True, help="cls_dataset のルート（train/ val/）")
    ap.add_argument("--out",  default="./runs/effnetv2", help="出力ディレクトリ")
    ap.add_argument("--model", default="efficientnetv2_s",
                    help="timmモデル名（例: efficientnetv2_s / tf_efficientnetv2_s ...）")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)

    # LR & regularization
    ap.add_argument("--lr", type=float, default=1e-4, help="backbone用の基準LR")
    ap.add_argument("--head-lr-mult", type=float, default=10.0, help="分類ヘッドのLR倍率")
    ap.add_argument("--min-lr", type=float, default=1e-6, help="Cosineの最小学習率")
    ap.add_argument("--warmup-epochs", type=int, default=5, help="ウォームアップepoch数")
    ap.add_argument("--wd", type=float, default=1e-4, help="Weight Decay")
    ap.add_argument("--drop", type=float, default=0.0, help="Dropout/DropPath")

    # Aug & imbalance
    ap.add_argument("--aug", choices=["none","light","strong"], default="light")
    ap.add_argument("--weighted-sampler", action="store_true")
    ap.add_argument("--weighted-loss", action="store_true")
    ap.add_argument("--label-smoothing", type=float, default=0.0, help="ラベルスムージング係数（例:0.1）")

    # System
    ap.add_argument("--accum-steps", type=int, default=1, help="勾配累積ステップ数")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--tensorboard", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # 既存機能
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="バックボーンをほぼ固定し、ヘッドだけ学習する従来モード")
    ap.add_argument("--init-from", default="",
                    help="初期化に使う ckpt のパス（例: runs/phaseA/best.ckpt）")

    # ★ 追加機能：段階的フリーズ
    ap.add_argument("--freeze-upto", type=int, default=-1,
                    help="指定ブロックまでをフリーズ (-1で無効, 0=stem, 1=stem+blocks.0, 2=stem+blocks.0..1, ...)")

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    train(args)


