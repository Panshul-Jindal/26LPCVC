"""
datasets.py
───────────
High-performance dataset for loading preprocessed video tensors.
"""

from __future__ import annotations

import os
import json
import torch
import numpy as np


class PreprocessedVideoDataset(torch.utils.data.Dataset):
    """
    Dataset that loads preprocessed uint8 tensors from disk.
    Expects tensors to be saved at {root}/{physical_sub}/{label_dir}/{video_filename}.npy
    The format of the saved tensor is (1, C, T, H, W) to match competition export standards.
    """
    def __init__(self, root, split_name, manifest_path, transform=None):
        self.root = root
        self.split_name = split_name
        self.transform = transform
        
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            
        self.samples = manifest['splits'][split_name]
        self.classes = manifest['metadata']['classes']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Determine physical subfolder
        self.physical_sub = 'train' if split_name in ['train', 'val'] else 'val'

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        v_path = entry['video_path'].lstrip('./')
        label = entry['label']
        label_dir = label.replace(' ', '_')
        
        # Load .npy tensor
        # Format: os.path.join(OUT_ROOT, physical_sub, label_dir, v_path + ".npy")
        npy_path = os.path.join(self.root, self.physical_sub, label_dir, v_path + ".npy")
        
        # Load uint8 tensor. 
        # Load uint8 tensor (organizer format): (1, C, T, H, W)
        video = np.load(npy_path)
        video = torch.from_numpy(video)
        
        # [1, 3, 16, 128, 171] -> [3, 16, 128, 171]
        # (C, T, H, W) is the standard format for torchvision video transforms
        video = video.squeeze(0)
        
        if self.transform:
            video = self.transform(video)
            
        target = self.class_to_idx[label]
        
        # return: video, label, index
        return video, target, idx
