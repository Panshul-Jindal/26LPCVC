import argparse
import sys
import os
import torch
import torchvision.io

# Import the existing preprocessing logic
from video_processing import process_video

def main():
    parser = argparse.ArgumentParser(description="Visualize the impact of video preprocessing.")
    parser.add_argument("video_path", type=str, help="Path to the input video.")
    parser.add_argument("--output", type=str, default="preprocessed_output.mp4", help="Path for the output .mp4 file.")
    parser.add_argument("--fps", type=int, default=4, help="Frames per second for output video (should match frame_rate of extraction).")
    
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: Could not find '{args.video_path}'")
        sys.exit(1)

    print(f"Reading and preprocessing video: '{args.video_path}'...")
    try:
        # returns shape: (Batch=1, Channels=3, Time=16, Height=112, Width=112)
        # value range: float32 [0, 1]
        batch = process_video(
            video_path=args.video_path,
            batch_size=1,
            clip_len=16,
            frame_rate=args.fps,
            clip_strategy="uniform"
        )
    except Exception as e:
        print(f"Error during video processing: {e}")
        sys.exit(1)

    # 1. Extract the single clip from the batch -> (C, T, H, W)
    clip = batch[0]
    
    # 2. Convert from (C, T, H, W) to (T, H, W, C) as required by torchvision.io.write_video
    clip_thwc = clip.permute(1, 2, 3, 0)
    
    # 3. Convert float32 [0, 1] to uint8 [0, 255]
    clip_uint8 = (clip_thwc * 255.0).clamp(0, 255).to(torch.uint8)

    print(f"Preprocessed tensor shape (T, H, W, C): {clip_uint8.shape}")
    print(f"Saving resulting video to '{args.output}' at {args.fps} FPS...")
    
    # Note: write_video requires moviepy or pyav backend. PyAV is what torchvision usually relies on.
    torchvision.io.write_video(
        filename=args.output,
        video_array=clip_uint8,
        fps=args.fps,
        video_codec="libx264"
    )
    
    print(f"Success! You can now view '{args.output}'.")

if __name__ == "__main__":
    main()
