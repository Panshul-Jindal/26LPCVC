import argparse
import datetime
import os
import time
import json
import warnings
import numpy as np
from types import SimpleNamespace

import yaml
import presets
import torch
import torch.utils.data
import torchvision
import torchvision.datasets.video_utils
import utils
import wandb
from torch import nn
from torch.utils.data.dataloader import default_collate
from datasets import PreprocessedVideoDataset
from model_wrapper import NormalizedModelWrapper
from freeze_utils import (
    apply_freezing, print_trainability, log_freeze_info,
    get_run_tag, VALID_STRATEGIES,
)
from presets import VideoClassificationPresetTrain, VideoClassificationPresetEval

# ─────────────────────────────────────────────────────────────────────────────
#  Train loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader,
                    device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("clips/s", utils.SmoothedValue(window_size=10, fmt="{value:.3f}"))

    header = f"Epoch: [{epoch}]"
    for video, target, _ in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        video, target = video.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(video)
            loss = criterion(output, target)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        acc1 = utils.accuracy(output, target, topk=(1,))[0]
        batch_size = video.shape[0]
        clips_per_sec = batch_size / (time.time() - start_time)

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["clips/s"].update(clips_per_sec)
        lr_scheduler.step()

        # --- W&B: step-level logging ---
        wandb.log({
            "train/loss":      loss.item(),
            "train/acc1":      acc1.item(),
            "train/lr":        optimizer.param_groups[0]["lr"],
            "train/clips_per_sec": clips_per_sec,
            "epoch":           epoch,
        })


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, criterion, data_loader, device, epoch=None, split_prefix="val"):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test ({split_prefix}):"
    
    # Containers to collect all predictions and targets for macro metrics
    all_preds = []
    all_targets = []

    with torch.inference_mode():
        for video, target, _ in metric_logger.log_every(data_loader, 100, header):
            video  = video.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(video)
            loss   = criterion(output, target)

            metric_logger.update(loss=loss.item())
            
            # For macro metrics, collect class predictions
            all_preds.append(output.argmax(dim=1).cpu())
            all_targets.append(target.cpu())

    metric_logger.synchronize_between_processes()
    avg_loss = metric_logger.loss.global_avg

    # Compute macro metrics
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    # Micro Accuracy
    micro_acc = (all_preds == all_targets).mean()
    
    # Macro Accuracy & F1
    classes = np.unique(all_targets)
    per_class_acc = []
    per_class_f1 = []
    
    # Calculate for each class that exists in targets or preds
    unique_labels = sorted(list(set(all_targets).union(set(all_preds))))
    for c in unique_labels:
        tp = ((all_preds == c) & (all_targets == c)).sum()
        fp = ((all_preds == c) & (all_targets != c)).sum()
        fn = ((all_preds != c) & (all_targets == c)).sum()
        
        # Recall (Acc per class)
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        # Precision
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        # Only count classes that actually exist in the targets for macro avg
        if (all_targets == c).any():
            per_class_acc.append(rec)
            per_class_f1.append(f1)

    macro_acc = np.mean(per_class_acc) if per_class_acc else 0
    macro_f1  = np.mean(per_class_f1) if per_class_f1 else 0

    print(f" * [{split_prefix}] Loss: {avg_loss:.4f} Micro-Acc: {micro_acc*100:.2f}% Macro-Acc: {macro_acc*100:.2f}% Macro-F1: {macro_f1:.4f}")

    # --- W&B: logging ---
    log_dict = {
        f"{split_prefix}/loss":      avg_loss,
        f"{split_prefix}/micro_acc": micro_acc,
        f"{split_prefix}/macro_acc": macro_acc,
        f"{split_prefix}/macro_f1":  macro_f1,
    }
    if epoch is not None:
        log_dict["epoch"] = epoch
    wandb.log(log_dict)

    return micro_acc



def _get_cache_path(filepath, args):
    import hashlib
    value = f"{filepath}-{args.clip_len}-{args.kinetics_version}-{args.frame_rate}"
    h = hashlib.sha1(value.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "kinetics", h[:10] + ".pt")
    return os.path.expanduser(cache_path)


def collate_fn(batch):
    return default_collate(batch)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    exp_name = getattr(args, "exp_name", "default_exp")

    # Directories
    checkpoint_dir = os.path.join("checkpoints", exp_name)
    log_dir        = os.path.join("logs", exp_name)
    utils.mkdir(checkpoint_dir)
    utils.mkdir(log_dir)

    # If legacy --output-dir was explicitly set, honour it; otherwise use checkpoint_dir
    if not getattr(args, "output_dir", None) or args.output_dir in (".", ""):
        args.output_dir = checkpoint_dir

    # ── W&B initialisation ──────────────────────────────────────────────────
    if utils.is_main_process():
        freeze_cfg = {
            "freeze_strategy": getattr(args, "freeze_strategy", ""),
            "freeze_ratio":    getattr(args, "frFITeeze_ratio", None),
            "model_name":      getattr(args, "model", "unknown"),
            "lr":              args.lr,
        }
        run_tag  = get_run_tag(freeze_cfg)
        run_name = f"{exp_name}/{run_tag}" if run_tag else exp_name
        wandb.init(
            project="lpcv2026",
            name=run_name,
            config=vars(args),
            dir=log_dir,
            resume="allow",
            tags=[exp_name, run_tag] if run_tag else [exp_name],
        )

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    # ── New Preprocessed Data Loading ──────────────────────────────────────────
    print("Loading data from manifest")
    manifest_path = args.split_manifest
    precomputed_root = args.preprocessed_dir
    
    with open(manifest_path, "r") as f:
        manifest_meta = json.load(f)["metadata"]
        
    class_weights = torch.tensor(manifest_meta["class_weights"], dtype=torch.float32).to(device)
    num_classes = manifest_meta["num_classes"]

    train_crop_size = tuple(args.train_crop_size)
    val_crop_size   = tuple(args.val_crop_size)

    transform_train = presets.VideoClassificationPresetTrain(crop_size=train_crop_size)
    transform_eval  = presets.VideoClassificationPresetEval(crop_size=val_crop_size)

    dataset      = PreprocessedVideoDataset(precomputed_root, "train", manifest_path, transform=transform_train)
    dataset_tv   = PreprocessedVideoDataset(precomputed_root, "val",   manifest_path, transform=transform_eval)
    dataset_test = PreprocessedVideoDataset(precomputed_root, "test",  manifest_path, transform=transform_eval)

    if args.subset_size > 0:
        print(f"[Data] Subsampling {args.subset_size:,} training videos...")
        indices = torch.randperm(len(dataset))[:args.subset_size].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)

    print(f"[Data] New Train: {len(dataset):,} | New Val: {len(dataset_tv):,} | Test (held-out): {len(dataset_test):,}")

    # ── Samplers ───────────────────────────────────────────────────────────
    print("Creating data loaders")

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        train_val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_tv, shuffle=False)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        train_val_sampler = torch.utils.data.SequentialSampler(dataset_tv)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=True,
        )
        data_loader_tv = torch.utils.data.DataLoader(
            dataset_tv,
            batch_size=args.batch_size,
            sampler=train_val_sampler,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=True,
        )


    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
    )

    dataset_test.classes = dataset_tv.classes # match classes
    print(f"[Data] train_train: {len(dataset):,} | train_val: {len(dataset_tv):,} | held-out test: {len(dataset_test):,}")

    # ── Model ───────────────────────────────────────────────────────────────
    print("Creating model")
    model = torchvision.models.get_model(args.model, weights=args.weights)
    
    if getattr(args, "pretrained_path", ""):
        print(f"Loading custom pretrained weights from {args.pretrained_path}")
        state_dict = torch.load(args.pretrained_path, map_location="cpu", weights_only=False)
        # Handle dicts that wrap the model
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "model_state" in state_dict:
            state_dict = state_dict["model_state"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            
        # Optional: remove fc layer keys if they will clash
        keys_to_delete = [k for k in state_dict.keys() if k.startswith("fc.")]
        for k in keys_to_delete:
            del state_dict[k]
            
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Pretrained weights loaded. Missing keys: {msg.missing_keys}")
        
    model.to(device)
    
    # Wrap model with normalization buffer
    model = NormalizedModelWrapper(model)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.to(device)

    # ── Configurable layer freezing ──────────────────────────────────────────
    freeze_cfg = {
        "freeze_strategy": getattr(args, "freeze_strategy", ""),
        "freeze_ratio":    getattr(args, "freeze_ratio", None),
        "model_name":      args.model,
        "lr":              args.lr,
    }
    # Freeze the underlying model so prefixes in freeze_utils (like "layer4", "fc") still work
    freeze_result = apply_freezing(model.model, config=freeze_cfg)
    print(freeze_result)
    print_trainability(model, verbose=True)
    # Log freeze metadata to W&B
    log_freeze_info(freeze_result, freeze_cfg)

    # Note: fc layer is part of the backbone in r2plus1d_18, 
    # but NormalizedModelWrapper proxies attributes.
    model.model.fc = nn.Linear(model.model.fc.in_features, num_classes)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimizer & Differential LR ──────────────────────────────────────────
    optimizer_name = getattr(args, "optimizer", "sgd").lower()
    custom_lr_fc = getattr(args, "lr_fc", None)

    # Differential LR: separate fc params (e.g. classifier) from the backbone
    if custom_lr_fc is not None:
        backbone_params = []
        fc_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # Note: after wrapping, name starts with 'model.'
            if name.startswith("model.fc") or name.startswith("model.classifier") or name.startswith("fc"):
                fc_params.append(p)
            else:
                backbone_params.append(p)
        
        param_groups = [
            {"params": backbone_params, "lr": args.lr},
            {"params": fc_params,      "lr": custom_lr_fc}
        ]
    else:
        # Filter again just in case some are frozen
        param_groups = [p for p in model.parameters() if p.requires_grad]

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            param_groups, lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            param_groups, lr=args.lr, weight_decay=args.weight_decay
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups, lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer '{optimizer_name}'")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # ── LR scheduler ─────────────────────────────────────────────────────────
    if not args.test_only:
        iters_per_epoch = len(data_loader)
        scheduler_name = getattr(args, "lr_scheduler", "multisteplr").lower()

        if scheduler_name == "multisteplr":
            lr_milestones = [
                iters_per_epoch * (m - args.lr_warmup_epochs)
                for m in getattr(args, "lr_milestones", [20, 30, 40])
            ]
            main_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=lr_milestones, gamma=args.lr_gamma
            )
        elif scheduler_name == "cosineannealinglr":
            main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=args.epochs * iters_per_epoch, 
                eta_min=getattr(args, "lr_min", 0.0)
            )
        elif scheduler_name == "reducelronplateau":
            main_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=args.lr_gamma, 
                patience=getattr(args, "lr_patience", 3)
            )
        else:
            raise ValueError(f"Unknown lr_scheduler '{scheduler_name}'")

        if getattr(args, "lr_warmup_epochs", 0) > 0 and scheduler_name != "reducelronplateau":
            warmup_iters = iters_per_epoch * args.lr_warmup_epochs
            args.lr_warmup_method = args.lr_warmup_method.lower()
            if args.lr_warmup_method == "linear":
                warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=args.lr_warmup_decay,
                    total_iters=warmup_iters
                )
            elif args.lr_warmup_method == "constant":
                warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                    optimizer, factor=args.lr_warmup_decay, total_iters=warmup_iters
                )
            else:
                raise RuntimeError(
                    f"Invalid warmup lr method '{args.lr_warmup_method}'."
                )
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_lr_scheduler, main_lr_scheduler],
                milestones=[warmup_iters],
            )
        else:
            lr_scheduler = main_lr_scheduler

    # ── DDP & resume ─────────────────────────────────────────────────────────
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model"]
        
        # Handle prefix mismatch for NormalizedModelWrapper
        first_key = next(iter(state_dict))
        is_wrapped = isinstance(model_without_ddp, NormalizedModelWrapper)
        if is_wrapped and not first_key.startswith("model."):
            print("Legacy checkpoint detected: Prepending 'model.' prefix to keys.")
            state_dict = {"model." + k: v for k, v in state_dict.items()}
            
        msg = model_without_ddp.load_state_dict(state_dict, strict=False)
        print(f"Checkpoint loaded. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            if args.amp:
                scaler.load_state_dict(checkpoint["scaler"])

    # ── Test-only path ────────────────────────────────────────────────────────
    if args.test_only:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        evaluate(model, criterion, data_loader_test, device=device)
        if utils.is_main_process():
            wandb.finish()
        return

    # ── Training loop ─────────────────────────────────────────────────────────
    print("Start training")
    start_time = time.time()

    best_acc1 = 0.0
    current_epoch = args.start_epoch

    try:
        for epoch in range(args.start_epoch, args.epochs):
            current_epoch = epoch
            if args.distributed and hasattr(data_loader.sampler, "set_epoch"):
                data_loader.sampler.set_epoch(epoch)
                
            scheduler_pass = lr_scheduler
            if getattr(args, "lr_scheduler", "").lower() == "reducelronplateau":
                # ReduceLROnPlateau steps after validation, not during training loop batches
                scheduler_pass = utils.SmoothedValue() # Dummy to not fail inside train_one_epoch step()

            train_one_epoch(
                model, criterion, optimizer, scheduler_pass,
                data_loader, device, epoch, args.print_freq, scaler
            )

            # train_val → for hyperparameter tuning and early stopping
            print("\n── Evaluating on train_val (hyper-param split) ──")
            tv_macro_acc = evaluate(model, criterion, data_loader_tv,   device=device, epoch=epoch, split_prefix="train_val")
            
            # held-out test → never used for model selection, just for tracking
            print("── Evaluating on held-out test ──")
            _            = evaluate(model, criterion, data_loader_test, device=device, epoch=epoch, split_prefix="test")

            if getattr(args, "lr_scheduler", "").lower() == "reducelronplateau":
                lr_scheduler.step(tv_macro_acc)

            # ── Checkpoint saving ─────────────────────────────────────────────
            if utils.is_main_process():
                checkpoint = {
                    "model":        model_without_ddp.state_dict(),
                    "optimizer":    optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch":        epoch,
                    "args":         args,
                }
                if args.amp:
                    checkpoint["scaler"] = scaler.state_dict()

                # Always keep a rolling checkpoint
                checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
                utils.save_on_master(checkpoint, checkpoint_path)
                
                # Save a checkpoint for EVERY epoch
                epoch_path = os.path.join(checkpoint_dir, f"model_{epoch}.pth")
                utils.save_on_master(checkpoint, epoch_path)

                # Best model selected on train_val (never on held-out test)
                if tv_macro_acc > best_acc1:
                    best_acc1 = tv_macro_acc
                    best_path = os.path.join(checkpoint_dir, "model.pth")
                    utils.save_on_master(checkpoint, best_path)
                    print(f"  ↑ New best train_val macro_acc = {best_acc1:.3f}  →  saved to {best_path}")
                    wandb.log({"train_val/best_macro_acc": best_acc1, "epoch": epoch})

    except (KeyboardInterrupt, Exception) as e:
        print(f"\n[!] Training interrupted by {type(e).__name__}.")
        if utils.is_main_process():
            print("[!] Saving emergency checkpoint...")
            checkpoint = {
                "model":        model_without_ddp.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch":        current_epoch,
                "args":         args,
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            emergency_path = os.path.join(checkpoint_dir, "checkpoint_emergency.pth")
            utils.save_on_master(checkpoint, emergency_path)
            print(f"[!] Emergency checkpoint safely saved to: {emergency_path}")
            
        if not isinstance(e, KeyboardInterrupt):
            raise e
        else:
            print("[!] Exiting gracefully.")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

    if utils.is_main_process():
        wandb.finish()


# ─────────────────────────────────────────────────────────────────────────────
#  Args: YAML + argparse
# ─────────────────────────────────────────────────────────────────────────────

def load_yaml_config(yaml_path: str) -> dict:
    """Load a YAML config file and return a flat dict."""
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="PyTorch Video Classification Training", add_help=add_help
    )

    # ── Config file (optional) ────────────────────────────────────────────
    parser.add_argument(
        "--config", default=None, type=str,
        help="Path to a YAML config file. Values in the file override the defaults "
             "below; explicit CLI flags override the YAML."
    )

    # ── Experiment identity ───────────────────────────────────────────────
    parser.add_argument("--exp-name", default="default_exp", type=str,
                        help="Experiment name (used for W&B run name and checkpoint/log dirs)")

    # ── Data ──────────────────────────────────────────────────────────────
    parser.add_argument("--data-path", default="./full_dataset/", type=str)
    parser.add_argument("--kinetics-version", default="400", choices=["400", "600"])

    # ── Model ─────────────────────────────────────────────────────────────
    parser.add_argument("--model", default="r2plus1d_18", type=str)
    parser.add_argument("--weights", default=None, type=str)
    parser.add_argument("--pretrained-path", default="", type=str, 
                        help="Path to a custom local .pth checkpoint for pretrained weights")

    # ── Training ──────────────────────────────────────────────────────────
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("-b", "--batch-size", default=24, type=int)
    parser.add_argument("-j", "--workers", default=10, type=int)

    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--lr-fc", default=None, type=float, help="Differential LR for only fc layer")
    parser.add_argument("--optimizer", default="sgd", type=str, choices=["sgd", "adam", "adamw"])
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float,
                        dest="weight_decay")

    parser.add_argument("--lr-scheduler", default="multisteplr", type=str, 
                        choices=["multisteplr", "cosineannealinglr", "reducelronplateau"])
    parser.add_argument("--lr-milestones", nargs="+", default=[20, 30, 40], type=int)
    parser.add_argument("--lr-gamma", default=0.1, type=float)
    parser.add_argument("--lr-warmup-epochs", default=10, type=int)
    parser.add_argument("--lr-warmup-method", default="linear", type=str)
    parser.add_argument("--lr-warmup-decay", default=0.001, type=float)
    parser.add_argument("--lr-min", default=0.0, type=float, help="Min LR for cosineannealinglr")
    parser.add_argument("--lr-patience", default=3, type=int, help="Patience for reducelronplateau")

    # ── Clip / frame ──────────────────────────────────────────────────────
    parser.add_argument("--clip-len", default=8, type=int)
    parser.add_argument("--frame-rate", default=4, type=int)
    parser.add_argument("--clips-per-video", default=1, type=int)

    # ── Augmentation sizes ────────────────────────────────────────────────
    parser.add_argument("--val-resize-size",  default=(128, 171), nargs="+", type=int)
    parser.add_argument("--val-crop-size",    default=(112, 112), nargs="+", type=int)
    parser.add_argument("--train-resize-size", default=(128, 171), nargs="+", type=int)
    parser.add_argument("--train-crop-size",  default=(112, 112), nargs="+", type=int)

    # ── Misc ──────────────────────────────────────────────────────────────
    parser.add_argument("--print-freq", default=10, type=int)
    parser.add_argument("--output-dir", default="", type=str,
                        help="Override checkpoint save directory (default: checkpoints/<exp_name>)")
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("--cache-dataset", dest="cache_dataset", action="store_true")
    parser.add_argument("--sync-bn", dest="sync_bn", action="store_true")
    parser.add_argument("--test-only", dest="test_only", action="store_true")
    parser.add_argument("--use-deterministic-algorithms", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--preprocessed-dir", default="/tmp/lpcvc_preprocessed", type=str)
    parser.add_argument("--split-manifest", default="metadata/split_manifest.json", type=str)

    # ── Freezing ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--freeze-strategy",
        default="",
        dest="freeze_strategy",
        help=(
            "Named layer-freezing strategy.  "
            f"Valid choices: {VALID_STRATEGIES}.  "
            "Overrides --freeze-ratio when both are set.  "
            "Leave empty to use ratio-based freezing or no freezing."
        ),
    )
    parser.add_argument(
        "--freeze-ratio",
        default=None,
        type=float,
        dest="freeze_ratio",
        metavar="R",
        help=(
            "Ratio-based freezing: freeze the first R fraction of parameters "
            "(e.g. 0.7 freezes the first 70%%).  "
            "Used only when --freeze-strategy is not set."
        ),
    )

    # ── Subset / split ────────────────────────────────────────────────────
    parser.add_argument(
        "--subset-size",
        default=50_000,
        type=int,
        dest="subset_size",
        help="Number of training videos to subsample (0 = use all).  Default: 50000.",
    )
    parser.add_argument(
        "--val-fraction",
        default=0.2,
        type=float,
        dest="val_fraction",
        help="Fraction of the subset to use as train_val (stratified).  Default: 0.2.",
    )
    parser.add_argument(
        "--split-seed",
        default=42,
        type=int,
        dest="split_seed",
        help="RNG seed for reproducible stratified split.  Default: 42.",
    )

    # ── Distributed ───────────────────────────────────────────────────────
    parser.add_argument("--world-size", default=1, type=int)
    parser.add_argument("--dist-url", default="env://", type=str)

    return parser


def merge_yaml_into_args(args: argparse.Namespace, yaml_cfg: dict) -> argparse.Namespace:
    """
    Override argparse defaults with values from the YAML config.

    YAML keys use underscores (e.g. `batch_size`).
    Argparse dest names also use underscores, so the mapping is direct.
    """
    for key, value in yaml_cfg.items():
        dest = key  # keys in YAML should already match argparse dest names
        if hasattr(args, dest):
            setattr(args, dest, value)
        else:
            # Allow unknown keys (forward compatibility)
            setattr(args, dest, value)
    return args


if __name__ == "__main__":
    parser = get_args_parser()

    # ── Two-pass parsing: first grab --config, then re-parse with YAML defaults ──
    pre_args, _ = parser.parse_known_args()

    if pre_args.config is not None:
        yaml_cfg = load_yaml_config(pre_args.config)
        # Build a list of YAML values as fake defaults for argparse
        # so that explicit CLI flags still win over YAML
        parser.set_defaults(**{k: v for k, v in yaml_cfg.items() if k != "config"})

    args = parser.parse_args()
    main(args)
