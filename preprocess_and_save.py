import json
import os
from typing import Iterable
import numpy as np
import torch
from video_processing import process_video


# =============================================================================
# USER CONFIGURATION — update these values before running
# =============================================================================

# Root directory of the raw video dataset.
# Expected structure: DATA_ROOT/<class_name>/<video>.mp4
# The class folder name is used as the label in the manifest.
DATA_ROOT = "/home/panshul/AI/projects/Efficient-Video-Understanding/26LPCVC_Track2_Sample_Solution/full_dataset/val"

# Directory where preprocessed .npy tensors and the manifest will be saved.
OUT_ROOT = "/home/panshul/AI/projects/Efficient-Video-Understanding/26LPCVC_Track2_Sample_Solution/processed_val_112"

# Video file extensions to include when scanning DATA_ROOT.
VIDEO_EXTS = {".mp4"}

# Number of frames to sample per clip. Must match the model's expected input.
CLIP_LEN = 16

# Target frame rate used when sampling frames from each video.
FRAME_RATE = 4

# Number of parallel workers (use your 20 cores!)
NUM_WORKERS = 20

# Device for final tensors
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================


def list_videos(root: str) -> list[str]:
    videos: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            _, ext = os.path.splitext(name)
            if ext.lower() in VIDEO_EXTS:
                videos.append(os.path.join(dirpath, name))
    return sorted(videos)


def iter_with_label(videos: Iterable[str], root: str) -> Iterable[tuple[str, str]]:
    for path in videos:
        rel = os.path.relpath(path, root)
        parts = rel.split(os.sep)
        label = parts[0] if len(parts) > 1 else "unknown"
        yield path, label


def save_tensor_npy(tensor: torch.Tensor, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, tensor.detach().cpu().numpy())


def main() -> None:
    if not DATA_ROOT:
        raise ValueError("'DATA_ROOT' is not set. Update it at the top of this script.")
    if not OUT_ROOT:
        raise ValueError("'OUT_ROOT' is not set. Update it at the top of this script.")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm

    videos = list_videos(DATA_ROOT)
    if not videos:
        raise FileNotFoundError(
            f"No {', '.join(VIDEO_EXTS)} files found under '{DATA_ROOT}'. "
            "Check that DATA_ROOT points to the correct directory."
        )

    manifest_path = os.path.join(OUT_ROOT, "manifest.jsonl")
    os.makedirs(OUT_ROOT, exist_ok=True)

    print(f"Starting stable preprocessing with {NUM_WORKERS} processes on CPU...")
    
    video_tasks = list(iter_with_label(videos, DATA_ROOT))
    
    # ProcessPoolExecutor requires global functions or careful initialization
    with open(manifest_path, "w", encoding="utf-8") as manifest:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Submit all tasks
            future_to_video = {executor.submit(process_and_save_single_wrapper, task): task for task in video_tasks}
            
            # Use tqdm to show progress and write result IMMEDIATELY
            for future in tqdm(as_completed(future_to_video), total=len(video_tasks)):
                record = future.result()
                if record:
                    manifest.write(json.dumps(record) + "\n")
                    manifest.flush() # Ensure it actually hits the disk

    print(f"Done! All successful tensors saved to {OUT_ROOT}")
    print(f"Wrote manifest to {manifest_path}")

def process_and_save_single_wrapper(item):
    """Wrapper function that can be pickled for ProcessPoolExecutor"""
    video_path, label = item
    
    # We must redefine or import these inside the worker due to ProcessPool isolation
    import os
    import sys
    import json
    import torch
    import numpy as np
    
    # We use these global-like vars from the parent process (inherited on Linux)
    # But usually it is safer to pass them or hardcode them here for worker stability
    CLIP_LEN = 16
    FRAME_RATE = 4
    DEVICE = "cpu" 
    # Use the same project root
    project_root = os.getcwd()
    OUT_ROOT = os.path.join(project_root, "processed_val_112")
    DATA_ROOT = os.path.join(project_root, "full_dataset/val")

    rel = os.path.relpath(video_path, DATA_ROOT)
    rel_no_ext = os.path.splitext(rel)[0]
    out_path = os.path.join(OUT_ROOT, f"{rel_no_ext}.npy")

    try:
        from video_processing import process_video

        clip = process_video(
            video_path=video_path,
            batch_size=1,
            clip_len=CLIP_LEN,
            frame_rate=FRAME_RATE,
            clip_strategy="uniform",
            device=torch.device(DEVICE),
            output_dtype=torch.float32,
        )

        # Save to disk
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, clip.detach().cpu().numpy())

        return {
            "video_path": video_path,
            "label": label,
            "tensor_path": out_path,
            "shape": list(clip.shape),
            "dtype": str(clip.dtype),
        }
    except Exception as e:
        # If one fails, we just skip it
        return None


if __name__ == "__main__":
    main()
