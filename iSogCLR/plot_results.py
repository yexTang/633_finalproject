import matplotlib.pyplot as plt
import json
import os
import pandas as pd

# --- CONFIGURATION ---
# Add your experiments here. 
# Format: "Label Name": "Path/to/output_directory"
experiments = {
    "Baseline (AdamW)": "output/exp_A_adamw",
    "Challenger (SNAG)": "output/exp_B_snag",
    # Add more experiments here later...
}
# ---------------------

def load_log(log_path):
    """Reads the line-by-line JSON log file."""
    data = []
    if not os.path.exists(log_path):
        print(f"Warning: File not found {log_path}")
        return pd.DataFrame()
    
    with open(log_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    return pd.DataFrame(data)

def plot_training_curves(experiments):
    plt.figure(figsize=(10, 6))
    
    for label, path in experiments.items():
        df = load_log(os.path.join(path, "coco_log.txt"))
        if df.empty or 'train_loss_ita' not in df.columns:
            continue
            
        # Plot Loss
        plt.plot(df['epoch'], df['train_loss_ita'], label=f"{label} (Loss)", marker='.')
        
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Contrastive Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("comparison_loss_curve.png")
    plt.show()
    print("Saved training curve to comparison_loss_curve.png")

def plot_final_metrics(experiments):
    metrics = {'Experiment': [], 'Recall@1 (Mean)': [], 'Zero-Shot Acc': []}
    
    for label, path in experiments.items():
        # 1. Get Validation Recall (from coco_log.txt)
        df_coco = load_log(os.path.join(path, "coco_log.txt"))
        recall = 0
        if not df_coco.empty and 'val_r_mean' in df_coco.columns:
            recall = df_coco['val_r_mean'].iloc[-1] # Get last epoch
        
        # 2. Get Zero-Shot Accuracy (from zeroshot_imagenet_log.txt)
        # Note: This file might be separate depending on how you ran it
        zs_path = os.path.join(path, "zeroshot_imagenet_log.txt")
        acc = 0
        if os.path.exists(zs_path):
            df_zs = load_log(zs_path)
            if not df_zs.empty and 'zeroshot_top1' in df_zs.columns:
                acc = df_zs['zeroshot_top1'].iloc[-1]

        metrics['Experiment'].append(label)
        metrics['Recall@1 (Mean)'].append(recall)
        metrics['Zero-Shot Acc'].append(acc)
        
    df_metrics = pd.DataFrame(metrics)
    
    # Plotting Bar Chart
    df_metrics.plot(x='Experiment', y=['Recall@1 (Mean)', 'Zero-Shot Acc'], kind='bar', figsize=(10, 6))
    plt.title("Final Performance Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("comparison_bar_chart.png")
    plt.show()
    print("Saved bar chart to comparison_bar_chart.png")

# --- RUN ---
print("Generating visualizations...")
plot_training_curves(experiments)
plot_final_metrics(experiments)