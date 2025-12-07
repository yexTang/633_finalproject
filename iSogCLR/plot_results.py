import matplotlib.pyplot as plt
import numpy as np
import json
import os
import re
import pandas as pd
import seaborn as sns

# ============================================
# CONFIGURATION
# ============================================

OUTPUT_ROOT = "/content/drive/MyDrive/Final Project CSCE Hu Zhao Yexiang Tang/outputs"
SAVE_DIR = "/content/drive/MyDrive/Final Project CSCE Hu Zhao Yexiang Tang/visualizations"

# Define your 12 experiments: folder_name -> (loss, optimizer)
FOLDER_MAPPING = {
    # Baseline (InfoNCE) - ita_type: clip
    "infonce_adamw_e30": ("InfoNCE", "AdamW"),
    "infonce_sgd_e30": ("InfoNCE", "SGD"),
    "exp_infonce_lion_e30": ("InfoNCE", "Lion"),
    
    # SogCLR - ita_type: sogclr
    "sogclr_adamw_e30": ("SogCLR", "AdamW"),
    "sogclr_sgd_e30": ("SogCLR", "SGD"),
    "sogclr_lion_e30": ("SogCLR", "Lion"),
    
    # iSogCLR (no TempNet) - ita_type: isogclr_new, isogclr_temp_net: false
    "isogclr_adamw_e30": ("iSogCLR", "AdamW"),
    "isogclr_sgd_e30": ("iSogCLR", "SGD"),
    "isogclr_lion_e30": ("iSogCLR", "Lion"),
    
    # iSogCLR + TempNet - ita_type: isogclr_new, isogclr_temp_net: true
    "iSogCLR_TempNET_adamw_e30": ("iSogCLR+TempNet", "AdamW"),
    "iSogCLR_TempNET_sgd_e30": ("iSogCLR+TempNet", "SGD"),
    "iSogCLR_TempNET_lion_e30": ("iSogCLR+TempNet", "Lion"),
}

LOSSES = ["InfoNCE", "SogCLR", "iSogCLR", "iSogCLR+TempNet"]
OPTIMIZERS = ["AdamW", "SGD", "Lion"]
COLORS = {"AdamW": "#4285F4", "SGD": "#EA4335", "Lion": "#34A853"}

# ============================================
# PARSING FUNCTIONS
# ============================================

def parse_eval_log(folder_path):
    """Parse eval_unified_log.txt to extract metrics"""
    eval_path = os.path.join(folder_path, "eval_unified_log.txt")
    metrics = {"txt_r1": None, "img_r1": None, "zeroshot_top1": None}
    
    if not os.path.exists(eval_path):
        return metrics
    
    with open(eval_path, 'r') as f:
        content = f.read()
    
    # Parse: coco val: {'txt_r1': 1.54, ... 'img_r1': 1.247...}
    coco_match = re.search(r"coco val:\s*\{([^}]+)\}", content)
    if coco_match:
        txt_match = re.search(r"'txt_r1':\s*([\d.]+)", coco_match.group(1))
        img_match = re.search(r"'img_r1':\s*([\d.]+)", coco_match.group(1))
        if txt_match:
            metrics["txt_r1"] = float(txt_match.group(1))
        if img_match:
            metrics["img_r1"] = float(img_match.group(1))
    
    # Parse: zeroshot: {'zeroshot_top1': 0.148, ...}
    zs_match = re.search(r"'zeroshot_top1':\s*([\d.]+)", content)
    if zs_match:
        metrics["zeroshot_top1"] = float(zs_match.group(1))
    
    return metrics


def parse_train_log(folder_path):
    """Parse train_log.txt to extract per-epoch metrics"""
    train_path = os.path.join(folder_path, "train_log.txt")
    
    if not os.path.exists(train_path):
        return None
    
    with open(train_path, 'r') as f:
        content = f.read()
    
    epochs_data = {}
    
    # Find: "Train Epoch: [X] Total time:..." followed by "Averaged stats: ..."
    pattern = r"Train Epoch: \[(\d+)\] Total time:.*?\nAveraged stats:([^\n]+)"
    matches = re.findall(pattern, content)
    
    for epoch_str, stats_str in matches:
        epoch = int(epoch_str)
        epochs_data[epoch] = {}
        
        # Parse each stat: "key: value"
        for match in re.finditer(r"(\w+):\s*([-\d.]+|nan)", stats_str):
            key, value = match.groups()
            try:
                epochs_data[epoch][key] = float(value) if value != 'nan' else float('nan')
            except:
                pass
    
    return epochs_data


def calculate_average(metrics):
    """Calculate average of three metrics (final evaluation metric)"""
    vals = [metrics.get("txt_r1"), metrics.get("img_r1"), metrics.get("zeroshot_top1")]
    if None in vals:
        return None
    return sum(vals) / 3


# ============================================
# LOAD ALL RESULTS
# ============================================

def load_results():
    """Load results from all experiment folders"""
    results = {loss: {opt: {} for opt in OPTIMIZERS} for loss in LOSSES}
    
    print("Loading results...")
    print("-" * 60)
    
    for folder, (loss, opt) in FOLDER_MAPPING.items():
        folder_path = os.path.join(OUTPUT_ROOT, folder)
        
        if not os.path.exists(folder_path):
            print(f"❌ {loss} + {opt}: folder not found ({folder})")
            continue
        
        # Load evaluation metrics
        metrics = parse_eval_log(folder_path)
        metrics["folder_path"] = folder_path
        
        avg = calculate_average(metrics)
        results[loss][opt] = metrics
        
        if metrics["txt_r1"] is not None:
            print(f"✅ {loss} + {opt}: txt_r1={metrics['txt_r1']:.2f}, img_r1={metrics['img_r1']:.2f}, zs={metrics['zeroshot_top1']:.3f}, avg={avg:.3f}")
        else:
            print(f"⚠️  {loss} + {opt}: no eval metrics found")
    
    print("-" * 60)
    return results


# ============================================
# VISUALIZATION 1: BAR CHARTS (2x2)
# ============================================

def plot_bar_charts(results, save_path=None):
    """Create 2x2 grouped bar charts"""
    
    metrics_config = [
        ("txt_r1", "Image-to-Text Recall@1 (%)"),
        ("img_r1", "Text-to-Image Recall@1 (%)"),
        ("zeroshot_top1", "Zero-Shot Top-1 Accuracy (%)"),
        ("average", "Average Score (%)"),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    x = np.arange(len(LOSSES))
    width = 0.25
    
    for idx, (metric_key, title) in enumerate(metrics_config):
        ax = axes[idx]
        
        for i, opt in enumerate(OPTIMIZERS):
            values = []
            for loss in LOSSES:
                if metric_key == "average":
                    val = calculate_average(results[loss][opt])
                else:
                    val = results[loss][opt].get(metric_key)
                values.append(val if val is not None else 0)
            
            offset = (i - 1) * width
            bars = ax.bar(x + offset, values, width, label=opt, color=COLORS[opt], edgecolor='black', linewidth=0.5)
            
            # Value labels
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)
        
        ax.set_xlabel('Loss Function', fontweight='bold')
        ax.set_ylabel(title, fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(LOSSES, rotation=15, ha='right')
        ax.legend(title='Optimizer')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    plt.show()


# ============================================
# VISUALIZATION 2: HEATMAPS
# ============================================

def plot_heatmaps(results, save_path=None):
    """Create 2x2 heatmaps for each metric"""
    
    metrics_config = [
        ("txt_r1", "Image-to-Text Recall@1 (%)"),
        ("img_r1", "Text-to-Image Recall@1 (%)"),
        ("zeroshot_top1", "Zero-Shot Top-1 Accuracy (%)"),
        ("average", "Average Score (Final Metric)"),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (metric_key, title) in enumerate(metrics_config):
        ax = axes[idx]
        
        # Build data matrix: rows=losses, cols=optimizers
        data = []
        for loss in LOSSES:
            row = []
            for opt in OPTIMIZERS:
                if metric_key == "average":
                    val = calculate_average(results[loss][opt])
                else:
                    val = results[loss][opt].get(metric_key)
                row.append(val if val is not None else np.nan)
            data.append(row)
        
        df = pd.DataFrame(data, index=LOSSES, columns=OPTIMIZERS)
        
        sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                    linewidths=0.5, annot_kws={'size': 11, 'weight': 'bold'})
        
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xlabel('Optimizer', fontweight='bold')
        ax.set_ylabel('Loss Function', fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    plt.show()


def plot_single_heatmap(results, save_path=None):
    """Create single heatmap for the final metric (average)"""
    
    data = []
    for loss in LOSSES:
        row = []
        for opt in OPTIMIZERS:
            val = calculate_average(results[loss][opt])
            row.append(val if val is not None else np.nan)
        data.append(row)
    
    df = pd.DataFrame(data, index=LOSSES, columns=OPTIMIZERS)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                linewidths=1, annot_kws={'size': 14, 'weight': 'bold'})
    
    ax.set_title('Final Metric: Average of (I→T R@1, T→I R@1, ZS Top-1)', fontweight='bold', fontsize=13)
    ax.set_xlabel('Optimizer', fontweight='bold', fontsize=12)
    ax.set_ylabel('Loss Function', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    plt.show()


# ============================================
# VISUALIZATION 3: TRAINING CURVES
# ============================================

def plot_training_curves_by_loss(results, metric='loss_ita', save_path=None):
    """Training curves: one subplot per loss function"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, loss in enumerate(LOSSES):
        ax = axes[idx]
        
        for opt in OPTIMIZERS:
            folder_path = results[loss][opt].get("folder_path")
            if not folder_path:
                continue
            
            epochs_data = parse_train_log(folder_path)
            if not epochs_data:
                continue
            
            epochs = sorted(epochs_data.keys())
            values = [epochs_data[e].get(metric, np.nan) for e in epochs]
            
            if not all(np.isnan(v) for v in values):
                ax.plot(epochs, values, label=opt, color=COLORS[opt], linewidth=2, marker='o', markersize=3)
        
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel(metric, fontweight='bold')
        ax.set_title(f'{loss}', fontweight='bold', fontsize=12)
        ax.legend(title='Optimizer')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Training Curves: {metric}', fontweight='bold', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    plt.show()


def plot_training_curves_all(results, metric='loss_ita', save_path=None):
    """All 12 experiments on one plot"""
    
    markers = {"InfoNCE": "o", "SogCLR": "s", "iSogCLR": "^", "iSogCLR+TempNet": "D"}
    linestyles = {"AdamW": "-", "SGD": "--", "Lion": "-."}
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for loss in LOSSES:
        for opt in OPTIMIZERS:
            folder_path = results[loss][opt].get("folder_path")
            if not folder_path:
                continue
            
            epochs_data = parse_train_log(folder_path)
            if not epochs_data:
                continue
            
            epochs = sorted(epochs_data.keys())
            values = [epochs_data[e].get(metric, np.nan) for e in epochs]
            
            if not all(np.isnan(v) for v in values):
                ax.plot(epochs, values, label=f"{loss} + {opt}", 
                       color=COLORS[opt], linestyle=linestyles[opt],
                       marker=markers[loss], markersize=4, linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontweight='bold', fontsize=12)
    ax.set_ylabel(metric, fontweight='bold', fontsize=12)
    ax.set_title(f'Training Curves: {metric} (All Experiments)', fontweight='bold', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    plt.show()


# ============================================
# SUMMARY TABLE
# ============================================

def print_summary_table(results):
    """Print formatted summary table"""
    
    print("\n" + "=" * 90)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Loss':<18} {'Optimizer':<10} {'I→T R@1':<10} {'T→I R@1':<10} {'ZS Top-1':<10} {'Average':<10}")
    print("-" * 90)
    
    best_avg, best_config = 0, ""
    
    for loss in LOSSES:
        for opt in OPTIMIZERS:
            m = results[loss][opt]
            txt = f"{m['txt_r1']:.3f}" if m.get('txt_r1') else "-"
            img = f"{m['img_r1']:.3f}" if m.get('img_r1') else "-"
            zs = f"{m['zeroshot_top1']:.3f}" if m.get('zeroshot_top1') else "-"
            avg = calculate_average(m)
            avg_str = f"{avg:.3f}" if avg else "-"
            
            if avg and avg > best_avg:
                best_avg, best_config = avg, f"{loss} + {opt}"
            
            print(f"{loss:<18} {opt:<10} {txt:<10} {img:<10} {zs:<10} {avg_str:<10}")
    
    print("-" * 90)
    print(f"🏆 Best: {best_config} (avg={best_avg:.3f})")
    print("=" * 90)


def export_to_csv(results, save_path):
    """Export results to CSV"""
    rows = []
    for loss in LOSSES:
        for opt in OPTIMIZERS:
            m = results[loss][opt]
            rows.append({
                "Loss": loss,
                "Optimizer": opt,
                "I→T R@1": m.get("txt_r1"),
                "T→I R@1": m.get("img_r1"),
                "ZS Top-1": m.get("zeroshot_top1"),
                "Average": calculate_average(m)
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"✅ Saved: {save_path}")
    return df


# ============================================
# MAIN: RUN EVERYTHING
# ============================================

def main():
    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Load results
    results = load_results()
    
    # Summary table
    print_summary_table(results)
    export_to_csv(results, os.path.join(SAVE_DIR, "results.csv"))
    
    # Bar charts
    plot_bar_charts(results, os.path.join(SAVE_DIR, "bar_charts.png"))
    
    # Heatmaps
    plot_heatmaps(results, os.path.join(SAVE_DIR, "heatmaps_all.png"))
    plot_single_heatmap(results, os.path.join(SAVE_DIR, "heatmap_final.png"))
    
    # Training curves
    plot_training_curves_by_loss(results, metric='loss_ita', save_path=os.path.join(SAVE_DIR, "training_loss_by_loss.png"))
    plot_training_curves_all(results, metric='loss_ita', save_path=os.path.join(SAVE_DIR, "training_loss_all.png"))
    
    print("\n✅ All visualizations complete!")


# Run
if __name__ == "__main__":
    main()