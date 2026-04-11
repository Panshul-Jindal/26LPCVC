import pandas as pd
import json
import os
import numpy as np

def generate_splits(csv_path, val_fraction=0.2, seed=42):
    np.random.seed(seed)
    df = pd.read_csv(csv_path)
    
    # 1. Separate current 'train' and 'val'
    df_train_pool = df[df['split'] == 'train'].reset_index(drop=True)
    df_test = df[df['split'] == 'val'].reset_index(drop=True)
    
    print(f"Original Train Pool: {len(df_train_pool)} videos")
    print(f"Original Val Pool (Test set): {len(df_test)} videos")

    # 2. Split df_train_pool into New Train and New Val by worker_id
    all_workers = df_train_pool['worker_id'].unique()
    np.random.shuffle(all_workers)
    
    val_size = int(len(all_workers) * val_fraction)
    val_workers = set(all_workers[:val_size])
    train_workers = set(all_workers[val_size:])
    
    df_new_train = df_train_pool[df_train_pool['worker_id'].isin(train_workers)]
    df_new_val = df_train_pool[df_train_pool['worker_id'].isin(val_workers)]
    
    print(f"New Train: {len(df_new_train)} videos, {len(train_workers)} workers")
    print(f"New Val: {len(df_new_val)} videos, {len(val_workers)} workers")

    # Success Check
    intersect = train_workers.intersection(val_workers)
    if intersect:
        print(f"CRITICAL ERROR: Leakage detected!")
    else:
        print("Success: Zero worker overlap between New Train and New Val.")

    # 3. Compute Inverse Frequency Weights based on New Train
    all_labels = sorted(df['label'].unique())
    class_counts = df_new_train['label'].value_counts().to_dict()
    
    weights = {}
    for label in all_labels:
        count = class_counts.get(label, 0)
        # Standard inverse frequency: w_c = 1/count
        weights[label] = 1.0 / count if count > 0 else 0.0
    
    # 4. Save Manifest
    manifest = {
        "metadata": {
            "val_fraction": val_fraction,
            "seed": seed,
            "num_classes": len(all_labels),
            "classes": all_labels,
            "class_weights": [weights[l] for l in all_labels]
        },
        "splits": {
            "train": df_new_train[['video_path', 'label']].to_dict(orient='records'),
            "val": df_new_val[['video_path', 'label']].to_dict(orient='records'),
            "test": df_test[['video_path', 'label']].to_dict(orient='records')
        }
    }
    
    os.makedirs("metadata", exist_ok=True)
    with open("metadata/split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Saved manifest to metadata/split_manifest.json")

if __name__ == "__main__":
    csv_path = "EDA_Results/competition_dataset_with_worker_id.csv"
    generate_splits(csv_path)
