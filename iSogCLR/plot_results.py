import matplotlib.pyplot as plt
import json
import os
import pandas as pd

# --- CONFIGURATION ---
# Update this list with your actual output folder names!
experiments = {
    "Baseline (AdamW)": "output/exp_A_adamw",
    "Challenger (SNAG)": "output/exp_B_snag",
}
# ---------------------

def load_log(log_path):
    data = []
    if not os.path.exists(log_path):
        print(f"Skipping {log_path} (Not found)")
        return pd.DataFrame()
    
    with open(log_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                # Ensure epoch is present
                if 'epoch' in entry:
                    data.append(entry)
            except:
                continue
    return pd.DataFrame(data)

def plot_metrics(experiments):
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Training Loss
    plt.subplot(1, 2, 1)
    for label, path in experiments.items():
        df = load_log(os.path.join(path, "coco_log.txt"))
        if df.empty: continue
        
        # Filter only rows that have training loss
        train_df = df.dropna(subset=['train_loss_ita'])
        if not train_df.empty:
            plt.plot(train_df['epoch'], train_df['train_loss_ita'], label=label, marker='.')
            
    plt.title("Training Loss (Lower is Better)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Subplot 2: Validation Recall
    plt.subplot(1, 2, 2)
    for label, path in experiments.items():
        df = load_log(os.path.join(path, "coco_log.txt"))
        if df.empty: continue

        # Filter only rows that have validation recall
        val_df = df.dropna(subset=['val_r_mean'])
        if not val_df.empty:
            plt.plot(val_df['epoch'], val_df['val_r_mean'], label=label, marker='o')

    plt.title("Validation Recall (Higher is Better)")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Recall")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("experiment_results.png")
    print("Graph saved to experiment_results.png")
    plt.show()

if __name__ == "__main__":
    plot_metrics(experiments)