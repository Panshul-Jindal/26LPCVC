import os
import json
import torch
import numpy as np
import torchvision.io as io
import torchvision.transforms.functional as F
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# --- Configuration (Sync with preprocess_to_tensors.py) ---
DATA_ROOT = "full_dataset"
OUT_ROOT = "/tmp/lpcvc_preprocessed"
MANIFEST_PATH = "metadata/split_manifest.json"
TARGET_FRAMES = 16
TARGET_FPS = 4
DUR_LIMIT = TARGET_FRAMES / TARGET_FPS  # 4.0 seconds
RESIZE_SIZE = (128, 171)
EXPECTED_SHAPE = (1, 3, 16, 128, 171)

def process_single_video(item):
    """Mirroring the exact logic from the main preprocessing script, with resize and permute."""
    video_path, rel_path = item
    out_path = os.path.join(OUT_ROOT, rel_path + ".npy")
    
    try:
        # Load video - TCHW format
        v_frames, _, info = io.read_video(video_path, pts_unit='sec', output_format='TCHW')
        if v_frames.shape[0] < 1:
            return None

        total_frames = v_frames.shape[0]
        v_fps = info.get('video_fps', 30.0) or 30.0
        v_duration = total_frames / v_fps

        # Uniform sampling logic
        clip_duration = min(v_duration, DUR_LIMIT)
        frames_in_clip = int(clip_duration * v_fps)
        indices = np.linspace(0, frames_in_clip - 1, TARGET_FRAMES, dtype=int)
        v_frames = v_frames[indices]

        # Resize to 128x171 (spatial)
        v_frames = F.resize(v_frames, RESIZE_SIZE, antialias=True)

        # Permute to (B, C, T, H, W)
        # [16, 3, 128, 171] -> [3, 16, 128, 171] -> [1, 3, 16, 128, 171]
        v_frames = v_frames.permute(1, 0, 2, 3).unsqueeze(0)

        # Save as uint8 to match pipeline
        np.save(out_path, v_frames.numpy().astype(np.uint8))
        return True
    except Exception as e:
        return f"Error processing {video_path}: {str(e)}"

def needs_patch(out_path):
    """Check if the preprocessed file exists and has the correct shape."""
    if not os.path.exists(out_path):
        return True
    try:
        # Check shape using mmap to avoid loading the whole thing
        data = np.load(out_path, mmap_mode='r')
        return data.shape != EXPECTED_SHAPE
    except:
        return True # Corrupted

def main():
    if not os.path.exists(MANIFEST_PATH):
        print(f"Error: Manifest not found at {MANIFEST_PATH}")
        return

    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)

    tasks = []
    print(f"Scanning for missing or inconsistent tensors (target shape {EXPECTED_SHAPE})...")
    
    for split_key in ['train', 'val', 'test']:
        physical_sub = 'train' if split_key in ['train', 'val'] else 'val'
        entries = manifest['splits'].get(split_key, [])
        
        for entry in entries:
            label = entry['label']
            v_path = entry['video_path'].lstrip('./')
            label_dir = label.replace(' ', '_')
            
            # Construct paths
            source_path = os.path.join(DATA_ROOT, physical_sub, label_dir, v_path)
            rel_path = os.path.join(physical_sub, label_dir, v_path)
            out_path = os.path.join(OUT_ROOT, rel_path + ".npy")
            
            # Add to tasks if needs patching and source exists
            if needs_patch(out_path):
                if os.path.exists(source_path):
                    tasks.append((source_path, rel_path))
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if not tasks:
        print("All tensors are consistent! No patching needed.")
        return

    print(f"Found {len(tasks)} tensors to patch. Starting process...")

    with ProcessPoolExecutor(max_workers=20) as executor:
        results = list(tqdm(executor.map(process_single_video, tasks), total=len(tasks)))

    success = [r for r in results if r is True]
    errors = [r for r in results if isinstance(r, str)]
    
    print(f"\nPatch complete.")
    print(f"Success: {len(success)}")
    print(f"Errors:  {len(errors)}")
    
    if errors:
        with open("patch_errors.log", "w") as f:
            for e in errors:
                f.write(e + "\n")
        print("Error details saved to patch_errors.log")

if __name__ == "__main__":
    main()
