import argparse
import datetime
import os
import time
import warnings
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
from torchvision.datasets.samplers import DistributedSampler, RandomClipSampler, UniformClipSampler
from datasets import KineticsWithVideoId, SubsetVideoDataset, build_stratified_split
from freeze_utils import (
    apply_freezing, print_trainability, log_freeze_info,
    get_run_tag, VALID_STRATEGIES,
)


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
    for video, _, target, _ in metric_logger.log_every(data_loader, print_freq, header):
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

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = video.shape[0]
        clips_per_sec = batch_size / (time.time() - start_time)

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["clips/s"].update(clips_per_sec)
        lr_scheduler.step()

        # --- W&B: step-level logging ---
        wandb.log({
            "train/loss":      loss.item(),
            "train/acc1":      acc1.item(),
            "train/acc5":      acc5.item(),
            "train/lr":        optimizer.param_groups[0]["lr"],
            "train/clips_per_sec": clips_per_sec,
            "epoch":           epoch,
        })


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, criterion, data_loader, device, epoch=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    num_processed_samples = 0

    # Group and aggregate clip predictions per video
    num_videos = len(data_loader.dataset.samples)
    num_classes = len(data_loader.dataset.classes)
    agg_preds   = torch.zeros((num_videos, num_classes), dtype=torch.float32, device=device)
    agg_targets = torch.zeros((num_videos,),             dtype=torch.int32,   device=device)

    with torch.inference_mode():
        for video, _, target, video_idx in metric_logger.log_every(data_loader, 100, header):
            video  = video.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(video)
            loss   = criterion(output, target)

            preds = torch.softmax(output, dim=1)
            for b in range(video.size(0)):
                idx = video_idx[b].item()
                agg_preds[idx]   += preds[b].detach()
                agg_targets[idx]  = target[b].detach().item()

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = video.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size

    # Gather stats from all processes
    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if isinstance(data_loader.sampler, DistributedSampler):
        num_data_from_sampler = len(data_loader.sampler.dataset)
    else:
        num_data_from_sampler = len(data_loader.sampler)

    if (
        hasattr(data_loader.dataset, "__len__")
        and num_data_from_sampler != num_processed_samples
        and utils.get_rank() == 0
    ):
        warnings.warn(
            f"It looks like the sampler has {num_data_from_sampler} samples, but "
            f"{num_processed_samples} samples were used for the validation, which might "
            "bias the results. Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    clip_acc1 = metric_logger.acc1.global_avg
    clip_acc5 = metric_logger.acc5.global_avg
    val_loss  = metric_logger.loss.global_avg
    print(
        " * Clip Acc@1 {top1:.3f} Clip Acc@5 {top5:.3f}".format(
            top1=clip_acc1, top5=clip_acc5
        )
    )

    # Aggregate across GPUs for video-level accuracy
    if utils.is_dist_avail_and_initialized():
        torch.distributed.barrier()
        torch.distributed.all_reduce(agg_preds,   op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(agg_targets, op=torch.distributed.ReduceOp.MAX)
    agg_acc1, agg_acc5 = utils.accuracy(agg_preds, agg_targets, topk=(1, 5))
    print(" * Video Acc@1 {acc1:.3f} Video Acc@5 {acc5:.3f}".format(
        acc1=agg_acc1, acc5=agg_acc5))

    # --- W&B: validation logging ---
    log_dict = {
        "val/loss":       val_loss,
        "val/clip_acc1":  clip_acc1,
        "val/clip_acc5":  clip_acc5,
        "val/video_acc1": agg_acc1.item() if hasattr(agg_acc1, "item") else float(agg_acc1),
        "val/video_acc5": agg_acc5.item() if hasattr(agg_acc5, "item") else float(agg_acc5),
    }
    if epoch is not None:
        log_dict["epoch"] = epoch
    wandb.log(log_dict)

    return clip_acc1


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

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
            "freeze_ratio":    getattr(args, "freeze_ratio", None),
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

    # ── Data loading ──────────────────────────────────────────────────
    print("Loading data")
    val_resize_size   = tuple(args.val_resize_size)
    val_crop_size     = tuple(args.val_crop_size)
    train_resize_size = tuple(args.train_resize_size)
    train_crop_size   = tuple(args.train_crop_size)

    train_dir = os.path.join(args.data_path, "train")
    val_dir   = os.path.join(args.data_path, "val")

    transform_train = presets.VideoClassificationPresetTrain(
        crop_size=train_crop_size, resize_size=train_resize_size
    )
    transform_eval = presets.VideoClassificationPresetEval(
        crop_size=val_crop_size, resize_size=val_resize_size
    )

    # ── Original (held-out) val set ─ NEVER touched during training ────────
    if args.weights and args.test_only:
        transform_eval = torchvision.models.get_weight(args.weights).transforms()

    cache_path_val = _get_cache_path(val_dir, args)
    if args.cache_dataset and os.path.exists(cache_path_val):
        print(f"Loading dataset_test from {cache_path_val}")
        dataset_test, _ = torch.load(cache_path_val, weights_only=False)
        dataset_test.transform = transform_eval
    else:
        if args.distributed:
            print("Recommend pre-computing dataset cache on a single GPU first.")
        dataset_test = KineticsWithVideoId(
            args.data_path,
            frames_per_clip=args.clip_len,
            num_classes=args.kinetics_version,
            split="val",
            step_between_clips=1,
            transform=transform_eval,
            frame_rate=args.frame_rate,
            extensions=("avi", "mp4"),
            output_format="TCHW",
            num_workers=args.workers,
        )
        if args.cache_dataset:
            print(f"Saving dataset_test to {cache_path_val}")
            utils.mkdir(os.path.dirname(cache_path_val))
            utils.save_on_master((dataset_test, val_dir), cache_path_val)

    print(f"[Data] original val (held-out): {len(dataset_test.samples):,} videos")

    if not args.test_only:
        # ── Base training dataset (no copy, just metadata) ──────────────────────
        print("Loading training data (building VideoClips index)...")
        st = time.time()
        cache_path_train = _get_cache_path(train_dir, args)

        if args.cache_dataset and os.path.exists(cache_path_train):
            print(f"Loading dataset_base from {cache_path_train}")
            dataset_base, _ = torch.load(cache_path_train, weights_only=False)
            dataset_base.transform = None          # transforms applied per-split
        else:
            if args.distributed:
                print("Recommend pre-computing dataset cache on a single GPU first.")
            dataset_base = KineticsWithVideoId(
                args.data_path,
                frames_per_clip=args.clip_len,
                num_classes=args.kinetics_version,
                split="train",
                step_between_clips=1,
                transform=None,                    # applied later per-split
                frame_rate=args.frame_rate,
                extensions=("avi", "mp4"),
                output_format="TCHW",
                num_workers=args.workers,
            )
            if args.cache_dataset:
                print(f"Saving dataset_base to {cache_path_train}")
                utils.mkdir(os.path.dirname(cache_path_train))
                utils.save_on_master((dataset_base, train_dir), cache_path_train)

        print(f"  VideoClips index built in {time.time() - st:.1f}s")
        print(f"  Total training videos: {len(dataset_base.samples):,}")

        # ── Stratified split ───────────────────────────────────────────────
        train_clip_idx, val_clip_idx = build_stratified_split(
            dataset_base,
            subset_size  = getattr(args, "subset_size",   50_000),
            val_fraction = getattr(args, "val_fraction",  0.2),
            seed         = getattr(args, "split_seed",    42),
        )

        # Zero-copy views with per-split transforms
        dataset        = SubsetVideoDataset(dataset_base, train_clip_idx, transform=transform_train)
        dataset_tv     = SubsetVideoDataset(dataset_base, val_clip_idx,   transform=transform_eval)

    # ── Samplers ───────────────────────────────────────────────────────────
    print("Creating data loaders")

    # Use simple sequential samplers for the subset views; the clip indices
    # are already shuffled during split construction for train_train.
    test_sampler = UniformClipSampler(dataset_test.video_clips, args.clips_per_video)

    if args.distributed:
        test_sampler = DistributedSampler(test_sampler, shuffle=False)

    if not args.test_only:
        # train_train: shuffle=True via DataLoader, no clip sampler needed
        # (SubsetVideoDataset already maps sequential indices to the right clips)
        train_val_sampler = UniformClipSampler(dataset_tv.video_clips, args.clips_per_video)
        if args.distributed:
            train_val_sampler = DistributedSampler(train_val_sampler, shuffle=False)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        data_loader_tv = torch.utils.data.DataLoader(
            dataset_tv,
            batch_size=args.batch_size,
            sampler=train_val_sampler,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    if args.test_only:
        num_classes = len(dataset_test.classes)
    else:
        num_classes = len(dataset_base.classes)
        print(f"[Data] train_train clips: {len(dataset):,}  "
              f"train_val clips: {len(dataset_tv):,}  "
              f"held-out val videos: {len(dataset_test.samples):,}")

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

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # ── Configurable layer freezing ──────────────────────────────────────────
    freeze_cfg = {
        "freeze_strategy": getattr(args, "freeze_strategy", ""),
        "freeze_ratio":    getattr(args, "freeze_ratio", None),
        "model_name":      args.model,
        "lr":              args.lr,
    }
    freeze_result = apply_freezing(model, config=freeze_cfg)
    print(freeze_result)
    print_trainability(model, verbose=True)
    # Log freeze metadata to W&B (no-op if wandb not initialised yet)
    log_freeze_info(freeze_result, freeze_cfg)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

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
            if name.startswith("fc") or name.startswith("classifier"):
                fc_params.append(p)
            else:
                backbone_params.append(p)
        
        param_groups = [
            {"params": backbone_params, "lr": args.lr},
            {"params": fc_params,      "lr": custom_lr_fc}
        ]
    else:
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
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"])
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
    for epoch in range(args.start_epoch, args.epochs):
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
        tv_acc1 = evaluate(model, criterion, data_loader_tv,   device=device, epoch=epoch)
        # held-out val → never used for model selection, just for tracking
        print("── Evaluating on held-out val ──")
        _        = evaluate(model, criterion, data_loader_test, device=device, epoch=epoch)

        if getattr(args, "lr_scheduler", "").lower() == "reducelronplateau":
            lr_scheduler.step(tv_acc1)

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

            # Best model selected on train_val (never on held-out val)
            if tv_acc1 > best_acc1:
                best_acc1 = tv_acc1
                best_path = os.path.join(checkpoint_dir, "model.pth")
                utils.save_on_master(checkpoint, best_path)
                print(f"  ↑ New best train_val acc1 = {best_acc1:.3f}  →  saved to {best_path}")
                wandb.log({"train_val/best_acc1": best_acc1, "epoch": epoch})

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
