import argparse
import sys
import os
import json

# Fix for local 'torchvision' directory shadowing the system installation
_current_dir = os.path.abspath(os.path.dirname(__file__))
if '' in sys.path: sys.path.remove('')
if _current_dir in sys.path: sys.path.remove(_current_dir)

import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18

# Add back current directory for local imports
sys.path.insert(0, _current_dir)
from video_processing import process_video

"""

Sets up the device checking for CUDA.
Loads the 92 class labels directly from class_labels.json.
Initializes a r2plus1d_18 model and points its output linear layer to output logits for 92 classes.
Loads your saved weights from model/model_29.pth.
Uses process_video() functionality to uniform-sample 16 frames from the specified video, formats it to PyTorch tensors, and standardizes to 4 fps.
Outputs the model's top 5 predictions along with their respective percentages.


python infer_single_video.py full_dataset/train/buttkickers/00029585.mp4 --model_path model/model_29.pth


"""
def main():
    parser = argparse.ArgumentParser(description="Run inference on a single video")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("--model_path", type=str, default="model/model_29.pth", help="Path to the model weights")
    parser.add_argument("--labels_path", type=str, default="class_map.json", help="Path to class labels json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load class labels
    try:
        with open(args.labels_path, "r") as f:
            class_map = json.load(f)
        # Assuming class_map is {"class_name": int_id, ...}
        # Invert it so we can index by output node id
        reversed_map = {v: k for k, v in class_map.items()}
        # Make a list sorted by id internally for safety
        max_id = max(reversed_map.keys())
        class_labels = [reversed_map.get(i, f"Class_{i}") for i in range(max_id + 1)]
    except Exception as e:
        print(f"Error loading labels: {e}")
        class_labels = [f"Class {i}" for i in range(92)]

    num_classes = len(class_labels)

    # Initialize model architecture
    print("Initializing r2plus1d_18 model...")
    model = r2plus1d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Load weights
    try:
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        # Check if it contains just state dict or the whole checkpoint
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict, strict=True)
        print(f"Successfully loaded model weights from {args.model_path}")
    except Exception as e:
        print(f"Failed to load model weights: {e}")
        sys.exit(1)

    model.to(device)
    model.eval()

    print(f"Processing video: {args.video_path}")
    # Process video using the provided video_processing.py utilities.
    # Uses 16 frames and 4 fps uniformly sampled (defaults).
    try:
        video_tensor = process_video(
            video_path=args.video_path, 
            batch_size=1, 
            clip_len=16, 
            frame_rate=4, 
            device=device
        )
    except Exception as e:
        print(f"Error processing video: {e}")
        sys.exit(1)

    print("Running inference...")
    with torch.no_grad():
        outputs = model(video_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_classes = torch.topk(probabilities, k=5, dim=1)

    print("\n--------------------------")
    print("--- Prediction Results ---")
    print("--------------------------")
    for i in range(5):
        prob = top_probs[0][i].item() * 100
        class_idx = top_classes[0][i].item()
        class_name = class_labels[class_idx]
        print(f"{i+1}. {class_name:<40} ({prob:>6.2f}%)")

if __name__ == "__main__":
    main()
