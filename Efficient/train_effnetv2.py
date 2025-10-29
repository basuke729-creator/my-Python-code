# train_effnetv2.py
# 統合版:
# - EfficientNetV2 + 軺移学習
# - 段階的フリーズ (--freeze-upto)
# - FocalLoss / class-weight / label smoothing
# - WeightedRandomSampler
# - precision / recall / F1 ログ (train/val)
# - Warmup + CosineAnnealingWarmRestarts or plain Cosine
# - AMP / TensorBoard
# - Augment:
#     - RandomResizedCrop + RandomAffine(translateのみ)
#     - ColorJitter / HorizontalFlip
#     - RandomErasing (light / strong のときだけ)
# - 評価時は固定リサイズのみ

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
    torch.backends.cudnn.benchmark = True  # OK: 高速化狙い


# ---- Augment builder (shift & resize + RandomErasing) ----
def build_transforms(img_size: int, aug: str):
    """
    aug == "light":
        - RandomResizedCrop (scale 0.8~1.0, ratioちょいゆらし)
        - RandomAffine (translateのみ ±10%)
        - Flip / ColorJitter
        - Normalize
        - RandomErasing(確率25%)
    aug == "strong":
        - RandomResizedCrop(0.6~1.0)
        - AutoAugment(IMAGENET)
        - Normalize
        - RandomErasing(確率25%)
    aug == "none":
        - Resizeのみ
        - Normalizeのみ
        - RandomErasingはしない
    """
    if aug == "light":
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # 最大±10% 平行移動
            ),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(
                p=0.25,
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3),
                value="random",
            ),
        ])
    elif aug == "strong":
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(
                p=0.25,
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3),
                value="random",
            ),
        ])
    else:  # "none"
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    # val側は固定（評価の安定性を優先）・RandomErasingしない
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
    use_weighted_sampler: bool
) -> Tuple[DataLoader, DataLoader, List[str], torch.Tensor]:

    train_tf, val_tf = build_transforms(img_size, aug)

    train_dir = Path(data_root) / "train"
    val_dir   = Path(data_root) / "val"

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(val_dir),   transform=val_tf)
    class_names = train_ds.classes  # フォルダ名順

    sampler = None
    class_weights = torch.ones(len(class_names), dtype=torch.float)
    if use_weighted_sampler:
        # クラス不均衡対策
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
    """
    timmのEfficientNetV2系モデルを作成
    - pretrained=True なら ImageNet等の事前学習重みをロード
    - num_classes: あなたのクラス数
    - drop_rate/drop_path_rate に同じ値を入れて軽い正則化
    """
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
    EfficientNetV2系の層を「浅い順」にグルーピング。
    freeze-upto=N なら groups[0]..groups[N] を requires_grad=False にする。

    例イメージ:
      0: stem系 (conv_stem/bn1など)
      1: blocks.0
      2: blocks.1
      3: blocks.2
      4: blocks.3
      5: blocks.4以降 + head (残り全部)
    """
    groups: List[List[str]] = []
    names = [n for n, _ in model.named_parameters()]

    # stemらしき層
    stem_group = [
        n for n in names
        if n.startswith("conv_stem")
        or n.startswith("bn1")
        or n.startswith("stem")
    ]
    if stem_group:
        groups.append(stem_group)

    # blocks.{i}.*
    block_ids = sorted(
        {n.split(".")[1] for n in names if n.startswith("blocks.") and n.split(".")[1].isdigit()},
        key=lambda x: int(x)
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
    """
    upto=0 → groups[0]までフリーズ
    upto=1 → groups[0], groups[1]までフリーズ
    ...
    """
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
    """
    ヘッド(classifier, fcなど)には base_lr * head_lr_mult、
    バックボーンには base_lr、
    みたいにLRを分ける。
    """
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
    """
    ざっくり挙動:
      - warmup_epochs までは線形アップ
      - その後 CosineAnnealingWarmRestarts でゆっくり上下を繰り返す
    """
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
        # epoch単位で step() される想定
        epoch = self.last_epoch
        base_lrs = [pg.get('initial_lr', pg['lr']) for pg in self.optimizer.param_groups]

        if epoch < self.warmup_epochs:
            # 線形アップ: 0.1→1.0 みたいにならす
            warm_factor = 0.1 + 0.9 * (epoch + 1) / max(1, self.warmup_epochs)
            return [lr * warm_factor for lr in base_lrs]
        else:
            # warmup後はCosineAnnealingWarmRestartsにお任せ
            self.after_warmup.last_epoch = epoch - self.warmup_epochs
            self.after_warmup.step(epoch=epoch - self.warmup_epochs)
            return [pg['lr'] for pg in self.after_warmup.optimizer.param_groups]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        # get_lr()内で after_warmup も更新済


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
        # fallback: simple cosine
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs),
            eta_min=args.min_lr,
        )


# ---- FocalLoss ----
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
    )
    num_classes = len(class_names)

    # モデル作成
    model = build_model(
        args.model,
        num_classes,
        pretrained=not args.no_pretrained,
        drop=args.drop,
    ).to(device)

    # 既存ckptで初期化したいとき
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
        # いくらかフリーズしてる場合は残りパラメータだけ単一LRで更新
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999))
    else:
        # フルファインチューニング時はheadに大きいLR
        param_groups = make_param_groups(model, args.lr, args.head_lr_mult, args.wd)
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    steps_per_epoch = max(1, math.ceil(len(train_loader.dataset) / args.batch_size))
    scheduler = build_scheduler(optimizer, args, steps_per_epoch)

    # 損失関数選択
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
    global_step = 0

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

            # trainメトリクス用
            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())

            global_step += 1

        # ==== epoch終了後: train側メトリクス ====
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        train_preds = all_logits.argmax(dim=1)

        train_acc  = (train_preds == all_targets).float().mean().item()
        train_prec = precision_score(all_targets, train_preds, average="macro", zero_division=0)
        train_rec  = recall_score(all_targets, train_preds, average="macro", zero_division=0)
        train_f1   = f1_score(all_targets, train_preds, average="macro", zero_division=0)
        train_loss = run_loss / max(1, len(all_targets))

        # ==== valメトリクス ====
        val_stats = evaluate(model, val_loader, device)

        # ==== ログ・表示 ====
        lrs_now = [pg["lr"] for pg in optimizer.param_groups]
        lrs_str = ", ".join([f"{lr:.2e}" for lr in lrs_now])

        print(
            f"Epoch {epoch}: "
            f"train_acc={train_acc:.4f} val_acc={val_stats['acc']:.4f} "
            f"train_f1={train_f1:.4f} val_f1={val_stats['f1']:.4f} "
            f"train_loss={train_loss:.4f} val_loss={val_stats['loss']:.4f} "
            f"lr=[{lrs_str}]"
        )

        if tb:
            tb.add_scalar("train/acc",   train_acc,  epoch)
            tb.add_scalar("train/f1",    train_f1,   epoch)
            tb.add_scalar("train/loss_iter", train_loss, epoch)

            tb.add_scalar("val/acc",     val_stats["acc"],  epoch)
            tb.add_scalar("val/f1",      val_stats["f1"],   epoch)
            tb.add_scalar("val/loss",    val_stats["loss"], epoch)
            tb.add_scalar("val/prec",    val_stats["prec"], epoch)
            tb.add_scalar("val/rec",     val_stats["rec"],  epoch)

            # 複数LRグループの可視化
            for gi, lr in enumerate(lrs_now):
                tb.add_scalar(f"train/lr_group{gi}", lr, epoch)

        # ==== スケジューラ1epoch分進める ====
        scheduler.step(epoch)

        # ==== ベストモデル保存 (macro-F1ベース) ====
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

    # ---- 学習終了後 追加レポート (valセットでの詳細) ----
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
    ap.add_argument("--data", required=True,
                    help="cls_dataset のルート（train/ と val/ があるフォルダ）")
    ap.add_argument("--out", default="./runs/effnetv2_run",
                    help="出力ディレクトリ (ckpt や tb ログが入る)")
    ap.add_argument("--model", default="tf_efficientnetv2_s",
                    help="timmモデル名 (例: tf_efficientnetv2_s, tf_efficientnetv2_m ...)")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)

    # LR / scheduler
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="バックボーン側の基準LR")
    ap.add_argument("--head-lr-mult", type=float, default=10.0,
                    help="分類ヘッド側LR倍率 (freezeなしのとき有効)")
    ap.add_argument("--wd", type=float, default=1e-4,
                    help="Weight Decay (AdamW)")
    ap.add_argument("--min-lr", type=float, default=1e-6,
                    help="Cosineスケジュール到達下限LR")
    ap.add_argument("--warmup-epochs", type=int, default=3,
                    help="ウォームアップ(ゆっくりLR上げる)エポック数")
    ap.add_argument("--lr-schedule", choices=["cosine", "cosine_wr"], default="cosine_wr",
                    help="cosine: CosineAnnealingLR (1回なだらかに下げる)\n"
                         "cosine_wr: Warmup + CosineAnnealingWarmRestarts(山を繰り返す)")
    ap.add_argument("--T0", type=int, default=5,
                    help="CosineAnnealingWarmRestarts の最初の周期長 (エポック数)")
    ap.add_argument("--T-mult", type=int, default=2,
                    help="CosineAnnealingWarmRestarts の周期を伸ばす倍率 (>=1)")

    # regularization
    ap.add_argument("--drop", type=float, default=0.0,
                    help="Dropout/DropPath の強さ")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="CE用のラベルスムージング係数")

    # imbalance options
    ap.add_argument("--weighted-sampler", action="store_true",
                    help="クラス不均衡対策: サンプラー (少数クラスをたくさん引く)")
    ap.add_argument("--weighted-loss", action="store_true",
                    help="クラス不均衡対策: class_weightをLossに適用")

    # focal loss
    ap.add_argument("--focal-loss", action="store_true", dest="focal_loss",
                    help="FocalLossを使用 (少数クラスをより重視)")
    ap.add_argument("--focal-gamma", type=float, default=2.0,
                    help="FocalLossのγ (大きいほど難しい例を重視)")

    # aug
    ap.add_argument("--aug", choices=["none", "light", "strong"], default="light",
                    help="データ拡張プリセット (light 推奨スタート)")

    # grad / system
    ap.add_argument("--accum-steps", type=int, default=1,
                    help="勾配累積で実質バッチサイズを大きくする")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true",
                    help="自動混合精度AMPで高速/省メモリ化")
    ap.add_argument("--tensorboard", action="store_true",
                    help="TensorBoardログを out/tb/ に書き出す")
    ap.add_argument("--cpu", action="store_true",
                    help="CPUで強制実行 (通常は不要)")
    ap.add_argument("--seed", type=int, default=42)

    # transfer / freeze
    ap.add_argument("--no-pretrained", action="store_true",
                    help="ImageNet等の事前学習重みを使わない (ランダム初期化)")
    ap.add_argument("--init-from", default="",
                    help="既存ckptから重みロードしたいとき(best.ckptなど)")
    ap.add_argument("--freeze-upto", type=int, default=None,
                    help="0,1,2,... とすると浅い層から順にそのグループまで凍結。\n"
                         "例: 0=stemだけ止める, 1=stem+blocks.0 など。")

    return ap.parse_args()


# ----------------- entry -----------------
if __name__ == "__main__":
    args = parse_args()

    # PyTorchのCosineAnnealingWarmRestartsはT_mult>=1必須なのでガード
    if args.T_mult < 1:
        raise ValueError("--T-mult は1以上にしてください")

    train(args)



