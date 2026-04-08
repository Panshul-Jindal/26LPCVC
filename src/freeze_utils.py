"""
freeze_utils.py
───────────────
Hybrid, model-agnostic freezing system for video classification training.

Two complementary modes
───────────────────────
  1. Named strategy  – arch-aware, explicit control (e.g. "r2plus1d_layer4_only")
  2. Ratio-based     – arch-agnostic, experiment-friendly (e.g. freeze_ratio=0.7)

Priority: freeze_strategy  >  freeze_ratio  >  no freezing

Public API
──────────
  apply_freezing(model, config)  → FreezeResult
  get_run_tag(config)            → str   (W&B run-name component)
  log_freeze_info(result, config)→ None  (wandb.log wrapper)

  # low-level helpers (useful for testing / custom pipelines)
  apply_named_strategy(model, strategy)  → FreezeResult
  apply_ratio_freezing(model, ratio)     → FreezeResult

Extending for new architectures
────────────────────────────────
  Add an entry to _STRATEGY_REGISTRY below:

    "mobilenet_last_only": ("features.18", "classifier"),

  The value is the tuple of name *prefixes* whose parameters stay trainable.
  An empty tuple signals "full fine-tune" (no freezing at all).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
#  Return type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FreezeResult:
    """Summary of what apply_freezing() did."""
    trainable_names:   List[str]
    total_params:      int
    trainable_params:  int
    frozen_params:     int
    trainable_ratio:   float
    method:            str          # "named_strategy" | "ratio" | "none"
    strategy:          Optional[str] = None
    freeze_ratio:      Optional[float] = None
    model_name:        Optional[str] = None

    def __str__(self) -> str:          # pragma: no cover
        lines = [
            f"[Freeze] method        = {self.method}",
        ]
        if self.strategy:
            lines.append(f"         strategy      = {self.strategy}")
        if self.freeze_ratio is not None:
            lines.append(f"         freeze_ratio  = {self.freeze_ratio:.2f}")
        lines += [
            f"         total params  = {self.total_params:,}",
            f"         trainable     = {self.trainable_params:,}  "
            f"({self.trainable_ratio * 100:.1f} %)",
            f"         frozen        = {self.frozen_params:,}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Strategy registry
#
#  Key:   strategy name (lowercase, used in config / CLI)
#  Value: tuple of parameter-name *prefixes* that stay TRAINABLE.
#         Empty tuple  →  full fine-tune (everything trainable).
#
#  Naming convention:  <arch>_<description>
#  "full_finetune" is arch-agnostic and always valid.
# ─────────────────────────────────────────────────────────────────────────────

_STRATEGY_REGISTRY: Dict[str, Tuple[str, ...]] = {
    # ── R(2+1)D ───────────────────────────────────────────────────────────────
    "r2plus1d_layer4_only":  ("layer4", "fc"),
    "r2plus1d_layer3_4":     ("layer3", "layer4", "fc"),
    "r2plus1d_layer2_3_4":   ("layer2", "layer3", "layer4", "fc"),
    # ── ResNet (2D / 3D) ──────────────────────────────────────────────────────
    "resnet_layer4_only":    ("layer4", "fc"),
    "resnet_layer3_4":       ("layer3", "layer4", "fc"),
    # ── MobileNet ─────────────────────────────────────────────────────────────
    "mobilenet_last_blocks": ("features.16", "features.17", "features.18", "classifier"),
    "mobilenet_last_only":   ("features.18", "classifier"),
    # ── Generic ───────────────────────────────────────────────────────────────
    "full_finetune":         (),     # empty → no freezing
}

VALID_STRATEGIES: List[str] = sorted(_STRATEGY_REGISTRY.keys())


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def apply_freezing(model: nn.Module, config: dict) -> FreezeResult:
    """
    Apply freezing to *model* according to *config* (in-place).

    Config keys (all optional)
    --------------------------
    freeze_strategy : str   – named strategy from VALID_STRATEGIES
    freeze_ratio    : float – fraction [0, 1] of params to freeze from the start
    model_name      : str   – used for logging / run-tag generation

    Priority: freeze_strategy  >  freeze_ratio  >  no freezing.
    """
    strategy   = config.get("freeze_strategy") or ""
    ratio      = config.get("freeze_ratio")
    model_name = config.get("model_name") or config.get("model") or "unknown"

    strategy = strategy.strip().lower()

    # ── Reset: unfreeze every parameter first ────────────────────────────────
    _reset_grad(model)

    if strategy and strategy != "none":
        result = apply_named_strategy(model, strategy)
    elif ratio is not None and 0.0 < float(ratio) < 1.0:
        result = apply_ratio_freezing(model, float(ratio))
    else:
        result = _collect_result(model, method="none")

    result.model_name   = model_name
    result.strategy     = strategy or None
    result.freeze_ratio = float(ratio) if ratio is not None else None
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Named-strategy freezing
# ─────────────────────────────────────────────────────────────────────────────

def apply_named_strategy(model: nn.Module, strategy: str) -> FreezeResult:
    """
    Freeze by named strategy.  Modifies *model* in-place.

    Raises ValueError for unknown strategy names.
    """
    key = strategy.strip().lower()
    if key not in _STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown freeze strategy '{strategy}'.\n"
            f"Valid choices: {VALID_STRATEGIES}"
        )

    prefixes = _STRATEGY_REGISTRY[key]

    # Freeze everything that doesn't match a trainable prefix
    if prefixes:                            # empty → full fine-tune
        for name, param in model.named_parameters():
            if not any(name.startswith(p) for p in prefixes):
                param.requires_grad = False

    # BN always in eval mode when we partially freeze
    if key != "full_finetune":
        _freeze_batchnorm(model)

    return _collect_result(model, method="named_strategy", strategy=key)


# ─────────────────────────────────────────────────────────────────────────────
#  Ratio-based freezing
# ─────────────────────────────────────────────────────────────────────────────

def apply_ratio_freezing(model: nn.Module, ratio: float) -> FreezeResult:
    """
    Freeze the first *ratio* fraction of parameters (by iteration order).

    freeze_ratio = 0.7  →  first 70 % of parameters are frozen.

    BatchNorm layers are pinned to eval mode after freezing.
    """
    if not (0.0 <= ratio <= 1.0):
        raise ValueError(f"freeze_ratio must be in [0, 1], got {ratio}")

    all_params = list(model.named_parameters())
    n_freeze   = math.floor(len(all_params) * ratio)

    for i, (_, param) in enumerate(all_params):
        param.requires_grad = (i >= n_freeze)

    if ratio > 0.0:
        _freeze_batchnorm(model)

    return _collect_result(model, method="ratio", freeze_ratio=ratio)


# ─────────────────────────────────────────────────────────────────────────────
#  W&B helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_run_tag(config: dict) -> str:
    """
    Generate a concise, human-readable experiment tag for W&B run names.

    Examples
    --------
    config = {"freeze_strategy": "r2plus1d_layer3_4", "lr": 0.001}
    → "r2plus1d_layer3_4_lr1e-03"

    config = {"freeze_ratio": 0.7, "model_name": "resnet", "lr": 0.01}
    → "resnet_ratio0.70_lr1e-02"
    """
    parts: List[str] = []

    model_name = config.get("model_name") or config.get("model") or ""
    strategy   = (config.get("freeze_strategy") or "").strip()
    ratio      = config.get("freeze_ratio")
    lr         = config.get("lr")

    if strategy and strategy not in ("none", "full_finetune"):
        parts.append(strategy)
    elif model_name:
        prefix = model_name.split("_")[0]  # e.g. "r2plus1d" from "r2plus1d_18"
        if ratio is not None:
            parts.append(f"{prefix}_ratio{float(ratio):.2f}")
        else:
            parts.append(f"{prefix}_fullft")

    if lr is not None:
        parts.append(f"lr{float(lr):.0e}".replace("e-0", "e-").replace("e+0", "e+"))

    return "_".join(parts) if parts else "experiment"


def log_freeze_info(result: FreezeResult, config: dict) -> None:
    """
    Log freeze metadata to W&B.  Safe no-op if wandb is not initialised.
    """
    try:
        import wandb
        if wandb.run is None:
            return
        wandb.log({
            "freeze/method":          result.method,
            "freeze/strategy":        result.strategy or "none",
            "freeze/freeze_ratio":    result.freeze_ratio if result.freeze_ratio is not None else 0.0,
            "freeze/model_name":      result.model_name or "unknown",
            "freeze/total_params":    result.total_params,
            "freeze/trainable_params": result.trainable_params,
            "freeze/frozen_params":   result.frozen_params,
            "freeze/trainable_ratio": result.trainable_ratio,
            "freeze/run_tag":         get_run_tag(config),
        })
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  Debugging
# ─────────────────────────────────────────────────────────────────────────────

def print_trainability(model: nn.Module, *, verbose: bool = True) -> None:
    """
    Print every parameter with a  TRAINABLE ✓  /  FROZEN ✗  label.

    Set verbose=False to print only the summary line.
    """
    all_params = list(model.named_parameters())
    col = max(len(n) for n, _ in all_params) + 2 if all_params else 40

    if verbose:
        sep = "─" * (col + 14)
        print(f"\n{sep}")
        print(f"{'Parameter':<{col}}  Status")
        print(sep)
        for name, param in all_params:
            if param.requires_grad:
                tag = "TRAINABLE ✓"
            else:
                tag = "FROZEN    ✗"
            print(f"{name:<{col}}  {tag}")
        print(sep)

    total     = sum(p.numel() for _, p in all_params)
    trainable = sum(p.numel() for _, p in all_params if p.requires_grad)
    print(
        f"  Summary: {trainable:,} / {total:,} params trainable "
        f"({trainable / total * 100:.1f} %)\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _reset_grad(model: nn.Module) -> None:
    """Set requires_grad=True on every parameter (clean slate)."""
    for param in model.parameters():
        param.requires_grad = True


def _freeze_batchnorm(model: nn.Module) -> None:
    """
    Pin every BN layer to eval mode and freeze its affine parameters.

    Why: frozen upstream conv filters mean the BN running stats would be
    driven by a different distribution than fine-tuning intends.
    Keeping BN in eval() preserves the pre-trained statistics.
    """
    _BN_TYPES = (
        nn.BatchNorm1d, nn.BatchNorm2d,
        nn.BatchNorm3d, nn.SyncBatchNorm,
    )
    for module in model.modules():
        if isinstance(module, _BN_TYPES):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False


def _collect_result(
    model:        nn.Module,
    method:       str,
    strategy:     Optional[str]   = None,
    freeze_ratio: Optional[float] = None,
) -> FreezeResult:
    """Compute parameter counts and return a FreezeResult."""
    all_params  = list(model.named_parameters())
    total       = sum(p.numel() for _, p in all_params)
    trainable   = sum(p.numel() for _, p in all_params if p.requires_grad)
    frozen      = total - trainable
    t_names     = [n for n, p in all_params if p.requires_grad]
    return FreezeResult(
        trainable_names  = t_names,
        total_params     = total,
        trainable_params = trainable,
        frozen_params    = frozen,
        trainable_ratio  = trainable / total if total > 0 else 0.0,
        method           = method,
        strategy         = strategy,
        freeze_ratio     = freeze_ratio,
    )
