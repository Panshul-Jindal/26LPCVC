"""
datasets.py
───────────
Custom dataset classes and index-based subset / split utilities for
video classification training.

Design goals
────────────
* NO data is ever copied on disk — everything is index-based.
* Stratified sampling is done at the *video* level (one label per video),
  so class balance is preserved in both train_train and train_val.
* The same underlying KineticsWithVideoId object is shared between
  train_train and train_val; only the *clip-level* indices differ.

Public API
──────────
  KineticsWithVideoId         – standard Kinetics dataset that also returns
                                the video index for clip-level aggregation.

  SubsetVideoDataset          – wraps any KineticsWithVideoId and exposes
                                only the clip indices that belong to a
                                given subset of video indices.

  build_stratified_split      – given a KineticsWithVideoId, returns
                                (train_clip_indices, val_clip_indices)
                                after stratified video-level sampling.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torchvision
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
#  KineticsWithVideoId
# ─────────────────────────────────────────────────────────────────────────────

class KineticsWithVideoId(torchvision.datasets.Kinetics):
    """
    Identical to torchvision.datasets.Kinetics but __getitem__ also returns
    the integer video index so clip-level predictions can be aggregated per
    video during evaluation.

    Returns
    -------
    (video, audio, label, video_idx)
    """

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int, int]:
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[video_idx][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, audio, label, video_idx


# ─────────────────────────────────────────────────────────────────────────────
#  SubsetVideoDataset
# ─────────────────────────────────────────────────────────────────────────────

class SubsetVideoDataset(torch.utils.data.Dataset):
    """
    A zero-copy view into a KineticsWithVideoId dataset.

    Parameters
    ----------
    base_dataset : KineticsWithVideoId
        The fully-constructed underlying dataset.  This object is *shared*
        between train_train and train_val — no data duplication.
    clip_indices : List[int]
        Which clip-level indices (into base_dataset.video_clips) this
        subset exposes.  Built by build_stratified_split().
    transform : optional
        Override transform.  Useful for applying train vs. eval augmentations
        to the same underlying dataset object.

    Notes
    -----
    * `samples`      is forwarded to satisfy the torchvision sampler API
      (RandomClipSampler / UniformClipSampler inspect .video_clips).
    * `video_clips`  is also forwarded so the clip samplers work out of the box.
    * `classes`      is forwarded so num_classes can be inferred from either split.
    """

    def __init__(
        self,
        base_dataset: KineticsWithVideoId,
        clip_indices: List[int],
        transform=None,
    ):
        self.base      = base_dataset
        self.indices   = clip_indices          # clip-level indices into base
        self.transform = transform             # override; None → use base transform

        # ── Proxy attributes required by torchvision samplers ──────────────
        self.video_clips = _SubsetVideoClips(base_dataset.video_clips, clip_indices)
        self.samples     = base_dataset.samples
        self.classes     = base_dataset.classes

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int, int]:
        real_idx = self.indices[idx]
        video, audio, info, video_idx = self.base.video_clips.get_clip(real_idx)
        label = self.base.samples[video_idx][1]

        transform = self.transform if self.transform is not None else self.base.transform
        if transform is not None:
            video = transform(video)

        return video, audio, label, video_idx


class _SubsetVideoClips:
    """
    Minimal shim so that RandomClipSampler / UniformClipSampler can call
    len() on clip samplers that wrap a SubsetVideoDataset.

    torchvision samplers only need:
        len(dataset.video_clips)   →   number of clips
    """

    def __init__(self, base_clips, clip_indices: List[int]):
        self._base    = base_clips
        self._indices = clip_indices

    def __len__(self) -> int:
        return len(self._indices)

    # Forward everything else to the real VideoClips object
    def __getattr__(self, name: str):
        return getattr(self._base, name)


# ─────────────────────────────────────────────────────────────────────────────
#  build_stratified_split
# ─────────────────────────────────────────────────────────────────────────────

def build_stratified_split(
    dataset: KineticsWithVideoId,
    subset_size: int,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Stratified video-level split → (train_clip_indices, val_clip_indices).

    Steps
    ─────
    1. Build a per-class list of video indices from dataset.samples.
    2. Subsample ``subset_size`` videos using stratified sampling (each class
       contributes proportionally).
    3. Split the subset into train_train / train_val using ``val_fraction``,
       again stratified.
    4. Expand each video index to ALL its clip indices in video_clips.

    Parameters
    ----------
    dataset     : KineticsWithVideoId   – fully constructed dataset
    subset_size : int                   – total number of videos to sample
                                          (0 or negative → use all videos)
    val_fraction: float                 – fraction of subset used for train_val
    seed        : int                   – RNG seed for reproducibility

    Returns
    -------
    train_clip_indices : List[int]  – clip indices for train_train
    val_clip_indices   : List[int]  – clip indices for train_val

    Notes
    ─────
    * Indices are *clip-level* (i.e. indices into VideoClips), NOT video-level.
    * No files are read or copied — only index lists are created.
    """
    rng = random.Random(seed)

    # ── Step 1: group video indices by class ─────────────────────────────────
    class_to_videos: Dict[int, List[int]] = defaultdict(list)
    for video_idx, (_, label) in enumerate(dataset.samples):
        class_to_videos[label].append(video_idx)

    n_total_videos = len(dataset.samples)
    use_subset     = subset_size > 0 and subset_size < n_total_videos
    effective_size = subset_size if use_subset else n_total_videos

    # ── Step 2: stratified subsample ─────────────────────────────────────────
    subset_video_indices: List[int] = []
    classes = sorted(class_to_videos.keys())

    for cls in classes:
        vids = class_to_videos[cls][:]
        rng.shuffle(vids)
        # proportional quota: how many from this class
        quota = max(1, round(effective_size * len(vids) / n_total_videos))
        subset_video_indices.extend(vids[:quota])

    # Trim / shuffle so the total is exactly effective_size
    rng.shuffle(subset_video_indices)
    subset_video_indices = subset_video_indices[:effective_size]

    print(
        f"[Split] subset: {len(subset_video_indices):,} videos "
        f"(requested {effective_size:,} of {n_total_videos:,} total)"
    )

    # ── Step 3: stratified train/val split ───────────────────────────────────
    # Re-group subset by class for per-class stratified split
    subset_by_class: Dict[int, List[int]] = defaultdict(list)
    for vid in subset_video_indices:
        label = dataset.samples[vid][1]
        subset_by_class[label].append(vid)

    train_videos: List[int] = []
    val_videos:   List[int] = []

    for cls in sorted(subset_by_class.keys()):
        vids = subset_by_class[cls][:]
        rng.shuffle(vids)
        n_val = max(1, round(len(vids) * val_fraction))
        val_videos.extend(vids[:n_val])
        train_videos.extend(vids[n_val:])

    print(
        f"[Split] train_train videos: {len(train_videos):,}  |  "
        f"train_val videos: {len(val_videos):,}  |  "
        f"val_fraction={val_fraction:.2f}"
    )

    # ── Step 4: expand video → clip indices ──────────────────────────────────
    video_to_clips = _build_video_to_clips_map(dataset)

    train_clip_indices = _expand_videos_to_clips(train_videos, video_to_clips)
    
    # We want train_val to mimic the test evaluation (e.g. 1 uniform clip per video)
    # instead of wrapping it in UniformClipSampler (which raises TypeErrors on Subsets).
    val_clip_indices = _expand_videos_to_clips(val_videos, video_to_clips, uniform_clips=1)

    print(
        f"[Split] train_train clips: {len(train_clip_indices):,}  |  "
        f"train_val clips:   {len(val_clip_indices):,}"
    )

    return train_clip_indices, val_clip_indices


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_video_to_clips_map(dataset: KineticsWithVideoId) -> Dict[int, List[int]]:
    """
    Build a dict: video_index → [clip_index, clip_index, ...].

    VideoClips.get_clip_location(i) returns (video_idx, clip_within_video).
    We iterate over all clip indices once to build this reverse mapping.
    """
    video_clips = dataset.video_clips
    n_clips     = len(video_clips)                     # total clips in dataset

    video_to_clips: Dict[int, List[int]] = defaultdict(list)
    for clip_idx in range(n_clips):
        video_idx, _ = video_clips.get_clip_location(clip_idx)
        video_to_clips[video_idx].append(clip_idx)

    return dict(video_to_clips)


def _expand_videos_to_clips(
    video_indices:   List[int],
    video_to_clips:  Dict[int, List[int]],
    uniform_clips:   int = 0
) -> List[int]:
    """
    Flatten a list of video indices into their corresponding clip indices.
    If uniform_clips > 0, mimic UniformClipSampler by picking evenly spaced clips.
    """
    clips: List[int] = []
    for vid in video_indices:
        v_clips = video_to_clips.get(vid, [])
        if uniform_clips > 0 and len(v_clips) > 0:
            if uniform_clips >= len(v_clips):
                pass # keep all
            else:
                step = len(v_clips) / uniform_clips
                # match UniformClipSampler math: i * step + step/2
                idxs = [int(i * step + step / 2) for i in range(uniform_clips)]
                v_clips = [v_clips[min(idx, len(v_clips)-1)] for idx in idxs]
        clips.extend(v_clips)
    return clips
