# train_effnetv2.py
# 統合版（2025-10-31版）
# - EfficientNetV2 + 転移学習（timm）
# - 段階的フリーズ (--freeze-upto N)
# - FocalLoss / class-weight / label smoothing
# - WeightedRandomSampler
# - precision / recall / F1 を TensorBoard に記録
# - Warmup + CosineAnnealingWarmRestarts / or plain Cosine
# - AMP / TensorBoard
# - Aug: RandomResizedCrop + RandomAffine(translate) + (オプション)RandomErasing
#   ↑ RandomErasing は今回追加した部分
#
# 使い方(例):
# python train_effnetv2.py \
#   --data "./cls_dataset" \
#   --out "./runs/exp_shiftresize_focal_freeze4" \
#   --model tf_efficientnetv2_s \
#   --img-size 384 \
#   --batch-size 32 \
#   --epochs 50 \
#   --lr 1e-4 \
#   --min-lr 1e-6 \
#   --wd 1e-4 \
#   --warmup-epochs 3 \
#   --lr-schedule cosine_wr --T0 5 --T-mult 2 \
#   --aug light \
#   --amp \
#   --tensorboard \
#   --freeze-upto 4 \
#   --focal-loss --focal-gamma 2.0 \
#   --weighted-loss \
#   --random-erasing-p 0.25 \
#   --random-erasing-scale-min 0.02 \
#   --random-erasing-scale-max 0.10

import argparse, math, random
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import timm
from tqdm import tqdm

# ----------------- const -----------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ----------------- util funcs -----------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # 高速化OK


# ---- Augment builder (ここでRandomErasingも扱う) ----
def build_transforms(img_size: int, aug: str, args):
    """
    aug == "light": 軽い色変換 + ランダムリサイズ + ランダムシフト
    aug == "strong": AutoAugmentベースの強め
    aug == "none": リサイズのみ
    RandomErasing は args.* を見て最後に付ける
    """
    if aug == "light":
        train_tf_list = [
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.8, 1.0),      # 画面の80〜100%を使ってクロップ
                ratio=(0.9, 1.1)       # 少しだけ縦横比を揺らす
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1)   # 最大±10%だけ平行移動 → ランダムシフト
            ),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    elif aug == "strong":
        train_tf_list = [
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    else:  # "none"
        train_tf_list = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]

    # ←今回追加：Random Erasing
    # p>0 のときだけ付ける
    if args.random_erasing_p > 0.0:
        train_tf_list.append(
            transforms.RandomErasing(
                p=args.random_erasing_p,
                scale=(args.random_erasing_scale_min,
                       args.random_erasing_scale_max),
                ratio=(0.3, 3.3),
                value="random"  # ここはお好みで (0) でもOK
            )
        )

    train_tf = transforms.Compose(train_tf_list)

    # val側は絶対に安定した評価（固定リサイズ）
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
    use_weighted_sampler: bool,
    args,
) -> Tuple[DataLoader, DataLoader, List[str], torch.Tensor]:

    train_tf, val_tf = build_transforms(img_size, aug, args)

    train_dir = Path(data_root) / "train"
    val_dir   = Path(data_root) / "val"

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(val_dir),   transform=val_tf)
    class_names = train_ds.classes  # フォルダ順で決まる

    sampler = None
    class_weights = torch.ones(len(class_names), dtype=torch.float)
    if use_weighted_sampler:
        # 各クラスの枚数を数えて逆数で重みづけ
        counts = np.bincount([y for _, y in train_ds.samples], minlength=len(class_names))
        weights_per_class = 1.0 / np.clip(counts, 1, None)
        weights_per_sample = [weights_per_class[y] for _, y in train_ds.samples]

        sampler = WeightedRandomSampler(
            weights_per_sample,
            num_samples=len(weights_per_sample),
            replacement=True
        )
        class_weights = torch.tensor(weights_per_class, dtype=torch.float)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, class_names, class_weights


# ---- model builder ----
def build_model(model_name: str, num_classes: int, pretrained: bool, drop: float):
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop,
        drop_path_rate=drop,
    )
    return model


# ---- layer freezing helpers ----
def group_efficientnet_layers(model: nn.Module) -> List[List[str]]:
    """
    EfficientNetV2系の層を「浅い順」にグルーピングする。
    freeze-upto=N なら groups[0]..groups[N] まで requires_grad=False にする。
    """
    groups: List[List[str]] = []
    names = [n for n, _ in model.named_parameters()]

    # stem
    stem_group = [n for n in names if n.startswith("conv_stem")
                  or n.startswith("bn1") or n.startswith("stem")]
    if stem_group:
        groups.append(stem_group)

    # blocks.x
    block_ids = sorted(
        {n.split(".")[1] for n in names if n.startswith("blocks.") and n.split(".")[1].isdigit()},
        key=lambda x: int(x),
    )
    for bid in block_ids:
        g = [n for n in names if n.startswith(f"blocks.{bid}.")]
        if g:
            groups.append(g)

    # 残り (head / classifier / conv_head / bn2 など)
    used = set()
    for g in groups:
        used.update(g)
    tail = [n for n in names if n not in used]
    if tail:
        groups.append(tail)

    return groups


def freeze_upto(model: nn.Module, upto: int):
    groups = group_efficientnet_layers(model)
    max_idx = min(upto, len(groups) - 1)
    to_freeze = set()
    for gi in range(max_idx + 1):
        for n in groups[gi]:
            to_freeze.add(n)

    for n, p in model.named_parameters():
        if n in to_freeze:
            p.requires_grad = False


# ---- param groups for different LR on head/backbone ----
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


# ---- LR scheduler with warmup + cosine restarts ----
class WarmupThenCosineRestart(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, T_0, T_mult, eta_min, steps_per_epoch, last_epoch=-1):
        self.warmup_epochs = max(0, warmup_epochs)
        self.steps_per_epoch = max(1, steps_per_epoch)
        self.after_warmup = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min,
        )
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        base_lrs = [g.get('initial_lr', g['lr']) for g in self.optimizer.param_groups]

        if epoch < self.warmup_epochs:
            warm_factor = 0.1 + 0.9 * (epoch + 1) / max(1, self.warmup_epochs)
            return [lr * warm_factor for lr in base_lrs]
        else:
            self.after_warmup.last_epoch = epoch - self.warmup_epochs
            self.after_warmup.step(epoch=epoch - self.warmup_epochs)
            return [pg['lr'] for pg in self.after_warmup.optimizer.param_groups]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch


def build_scheduler(optimizer, args, steps_per_epoch):
    if args.lr_schedule == "cosine_wr":
        return WarmupThenCosineRestart(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            T_0=args.T0,
            T_mult=args.T_mult,
            eta_min=args.min_lr,
            steps_per_epoch=steps_per_epoch,
        )
    else:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs),
            eta_min=args.min_lr
        )


# ---- focal loss ----
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, logits, target):
        ce_loss = self.ce(logits, target)  # [B]
        with torch.no_grad():
            pt = torch.softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
            pt = pt.clamp_(1e-8, 1.0)
        focal = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        else:
            return focal


# ---- evaluation helper on val ----
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_logits = []
    all_targets = []
    loss_sum = 0.0
    total = 0

    ce = nn.CrossEntropyLoss()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)

        loss_sum += loss.item() * x.size(0)
        total += x.size(0)

        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    preds = all_logits.argmax(dim=1)

    acc    = (preds == all_targets).float().mean().item()
    prec   = precision_score(all_targets, preds, average="macro", zero_division=0)
    rec    = recall_score(all_targets, preds, average="macro", zero_division=0)
    f1     = f1_score(all_targets, preds, average="macro", zero_division=0)
    vloss  = loss_sum / max(1, total)

    return {
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "loss": vloss,
    }


# ----------------- main train loop -----------------
def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[Device] {device}")

    train_loader, val_loader, class_names, class_weights = build_dataloaders(
        args.data,
        args.img_size,
        args.batch_size,
        args.workers,
        args.aug,
        args.weighted_sampler,
        args,
    )
    num_classes = len(class_names)

    # モデル
    model = build_model(
        args.model,
        num_classes,
        pretrained=not args.no_pretrained,
        drop=args.drop
    ).to(device)

    # 初期重みを別のckptから読み込みたい場合
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu")
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[Init] loaded from {args.init_from} missing={len(missing)} unexpected={len(unexpected)}")

    # freeze指定
    if args.freeze_upto is not None:
        freeze_upto(model, args.freeze_upto)
        print(f"[Freeze] freeze_upto={args.freeze_upto}")

    # Optimizer
    if args.freeze_upto is not None:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999))
    else:
        param_groups = make_param_groups(model, args.lr, args.head_lr_mult, args.wd)
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    steps_per_epoch = max(1, math.ceil(len(train_loader.dataset) / args.batch_size))
    scheduler = build_scheduler(optimizer, args, steps_per_epoch)

    # 損失関数
    weights = class_weights.to(device) if args.weighted_loss else None
    if args.focal_loss:
        criterion = FocalLoss(gamma=args.focal_gamma, weight=weights, reduction='mean')
    else:
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.ckpt"
    last_path = out_dir / "last.ckpt"

    # TensorBoard
    tb = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(log_dir=str(out_dir / "tb"))

    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{args.epochs}]")
        run_loss = 0.0
        all_logits = []
        all_targets = []

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

            run_loss += loss.item() * x.size(0) * args.accum_steps

            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())

        # === epoch終了後 ===
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        train_preds = all_logits.argmax(dim=1)

        train_acc  = (train_preds == all_targets).float().mean().item()
        train_prec = precision_score(all_targets, train_preds, average="macro", zero_division=0)
        train_rec  = recall_score(all_targets, train_preds, average="macro", zero_division=0)
        train_f1   = f1_score(all_targets, train_preds, average="macro", zero_division=0)
        train_loss = run_loss / max(1, len(all_targets))

        # val
        val_stats = evaluate(model, val_loader, device)

        # LRログ
        lrs_now = [pg["lr"] for pg in optimizer.param_groups]
        lrs_str = ", ".join([f"{lr:.2e}" for lr in lrs_now])

        print(
            f"Epoch {epoch}: "
            f"train_acc={train_acc:.4f} val_acc={val_stats['acc']:.4f} "
            f"train_f1={train_f1:.4f} val_f1={val_stats['f1']:.4f} "
            f"train_loss={train_loss:.4f} val_loss={val_stats['loss']:.4f} "
            f"lr=[{lrs_str}]"
        )

        # TensorBoard書き込み
        if tb:
            tb.add_scalar("train/acc",   train_acc,  epoch)
            tb.add_scalar("train/f1",    train_f1,   epoch)
            tb.add_scalar("train/loss_iter", train_loss, epoch)

            tb.add_scalar("val/acc",     val_stats["acc"],  epoch)
            tb.add_scalar("val/f1",      val_stats["f1"],   epoch)
            tb.add_scalar("val/loss",    val_stats["loss"], epoch)
            tb.add_scalar("val/prec",    val_stats["prec"], epoch)
            tb.add_scalar("val/rec",     val_stats["rec"],  epoch)

            for gi, lr in enumerate(lrs_now):
                tb.add_scalar(f"train/lr_group{gi}", lr, epoch)

        # schedulerを1epoch進める
        scheduler.step(epoch)

        # ベストモデル保存 (macro-F1ベース)
        if val_stats["f1"] > best_f1:
            best_f1 = val_stats["f1"]
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "f1": best_f1,
                "class_names": class_names,
            }, best_path)
            print(f"[Best] f1={best_f1:.4f} -> saved {best_path}")

        # lastも毎回更新
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "f1": val_stats["f1"],
            "class_names": class_names,
        }, last_path)

    # 学習終了後レポート
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

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


# ----------------- arg parser -----------------
def parse_args():
    ap = argparse.ArgumentParser("EfficientNet-V2 Transfer Learning (Full Features)")

    # data / io
    ap.add_argument("--data", required=True, help="cls_dataset のルート（train/ と val/ がある）")
    ap.add_argument("--out", default="./runs/effnetv2_run", help="出力ディレクトリ")
    ap.add_argument("--model", default="tf_efficientnetv2_s", help="timmモデル名 (例: tf_efficientnetv2_s)")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)

    # LR / scheduler
    ap.add_argument("--lr", type=float, default=1e-4, help="ベースLR(バックボーン用)")
    ap.add_argument("--head-lr-mult", type=float, default=10.0, help="分類ヘッドLR倍率")
    ap.add_argument("--wd", type=float, default=1e-4, help="Weight Decay")
    ap.add_argument("--min-lr", type=float, default=1e-6, help="cosineでの下限LR")
    ap.add_argument("--warmup-epochs", type=int, default=3, help="ウォームアップエポック数")
    ap.add_argument("--lr-schedule", choices=["cosine", "cosine_wr"], default="cosine_wr",
                    help="cosine:通常CosineAnnealingLR / cosine_wr:Warmup+CosineAnnealingWarmRestarts")
    ap.add_argument("--T0", type=int, default=5, help="CosineAnnealingWarmRestarts 初期間隔")
    ap.add_argument("--T-mult", type=int, default=2, help="CosineAnnealingWarmRestarts のT_mult (>=1)")

    # regularization
    ap.add_argument("--drop", type=float, default=0.0, help="Dropout/DropPath 値")
    ap.add_argument("--label-smoothing", type=float, default=0.0, help="CEのラベルスムージング")

    # imbalance options
    ap.add_argument("--weighted-sampler", action="store_true", help="クラス不均衡対策: サンプラー")
    ap.add_argument("--weighted-loss", action="store_true", help="クラス不均衡対策: class_weightをLossに適用")

    # focal loss
    ap.add_argument("--focal-loss", action="store_true", dest="focal_loss",
                    help="FocalLossを使う（少数クラス重視）")
    ap.add_argument("--focal-gamma", type=float, default=2.0, help="FocalLoss の gamma")

    # aug
    ap.add_argument("--aug", choices=["none", "light", "strong"], default="light",
                    help="データ拡張プリセット")

    # ←今回追加したRandomErasing用の引数
    ap.add_argument("--random-erasing-p", type=float, default=0.0,
                    help="RandomErasing の発動確率 (0で無効)")
    ap.add_argument("--random-erasing-scale-min", type=float, default=0.02,
                    help="RandomErasing の最小面積比")
    ap.add_argument("--random-erasing-scale-max", type=float, default=0.15,
                    help="RandomErasing の最大面積比")

    # grad / system
    ap.add_argument("--accum-steps", type=int, default=1, help="勾配累積 (大きい実質バッチ)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", help="自動混合精度AMP")
    ap.add_argument("--tensorboard", action="store_true", help="TensorBoardログを保存")
    ap.add_argument("--cpu", action="store_true", help="強制CPU")
    ap.add_argument("--seed", type=int, default=42)

    # transfer / freeze
    ap.add_argument("--no-pretrained", action="store_true", help="ImageNetなどの事前学習重みを使わない")
    ap.add_argument("--init-from", default="", help="既存ckptから読み込みたい場合(best.ckptなど)")
    ap.add_argument("--freeze-upto", type=int, default=None,
                    help="0,1,2,... と指定すると浅い層からそのグループまで凍結")

    return ap.parse_args()


# ----------------- entry -----------------
if __name__ == "__main__":
    args = parse_args()

    if args.T_mult < 1:
        raise ValueError("--T-mult は1以上にしてください")

    train(args)




