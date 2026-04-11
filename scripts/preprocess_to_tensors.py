import os
import json
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import torchvision.io as io
import torchvision.transforms.functional as F
import argparse
import math
# Configuration
MANIFEST_PATH = "metadata/split_manifest.json"
OUT_ROOT = "/tmp/lpcvc_preprocessed"
CLIP_LEN = 16
FRAME_RATE = 4
RESIZE_SIZE = (128, 171) # Height, Width for Resize

def process_single_video(item):
    ##
    # video.clip(min(video_len, target_frames/target_frame_per_sec)).uniformly_sample(target_frames)
    ##
    video_path, rel_path = item
    try:
        # Load video
        # read_video returns (v_frames, a_frames, info)
        # v_frames is (T, H, W, C)
        v_frames, _, info = io.read_video(video_path, pts_unit='sec', output_format="TCHW")
        total_frames = v_frames.shape[0]
        if total_frames < 1:
            return None
            
        fps = info.get('video_fps', 30) or 30
        duration = total_frames / fps
        
        # Simplified logic: 
        # 1. Take at most 4 seconds of video (CLIP_LEN / FRAME_RATE)
        # 2. Sample exactly CLIP_LEN frames uniformly from that window
        target_duration = min(duration, CLIP_LEN / FRAME_RATE)
        num_raw_frames = max(1, min(int(target_duration * fps), total_frames))
        
        indices = np.linspace(0, num_raw_frames - 1, CLIP_LEN).astype(int)
        clip = v_frames[indices] # (T, C, H, W)
        
        # Resize to 128x171 (spatial uint8)
        clip = F.resize(clip, RESIZE_SIZE, antialias=True)
        
        # Permute to (B, C, T, H, W) to match organizers' format
        # [16, 3, 128, 171] -> [3, 16, 128, 171] -> [1, 3, 16, 128, 171]
        clip = clip.permute(1, 0, 2, 3).unsqueeze(0)
        
        # Save as uint8 .npy
        out_path = os.path.join(OUT_ROOT, rel_path + ".npy")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, clip.numpy().astype(np.uint8))
        
        return True
    except Exception as e:
        return str(e)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)
    
    all_videos = []
    DATA_ROOT = "full_dataset"
    
    tasks = []
    # manifest['splits'] has 'train', 'val', 'test'
    # 'train' and 'val' in manifest come from physical 'train' folder
    # 'test' in manifest comes from physical 'val' folder
    for split_key in ['train', 'val', 'test']:
        physical_sub = 'train' if split_key in ['train', 'val'] else 'val'
        for entry in manifest['splits'][split_key]:
            v_path = entry['video_path'].lstrip('./')
            label = entry['label']
            
            # Physical path: full_dataset/{physical_sub}/{label_with_underscores}/{v_path}
            # Note: label in manifest have spaces, while folders have underscores
            label_dir = label.replace(' ', '_')
            candidate = os.path.join(DATA_ROOT, physical_sub, label_dir, v_path)
            
            if os.path.exists(candidate):
                rel = os.path.join(physical_sub, label_dir, v_path)
                tasks.append((candidate, rel))

    
    if args.limit > 0:
        tasks = tasks[:args.limit]

    print(f"Total videos to preprocess: {len(tasks)}")
    
    # Use ProcessPoolExecutor for 20 cores
    with ProcessPoolExecutor(max_workers=20) as executor:
        results = list(tqdm(executor.map(process_single_video, tasks), total=len(tasks)))
    
    success = [r for r in results if r is True]
    errors = [r for r in results if isinstance(r, str)]
    print(f"Preprocessing complete. Success: {len(success)}, Errors: {len(errors)}")
    if errors:
        print(f"Sample error: {errors[0]}")

if __name__ == "__main__":
    main()
