import matplotlib.pyplot as plt
import numpy as np
import json
import os
import re

# ============================================
# STEP 1: Define your experiment results
# ============================================

# Structure: results[loss][optimizer] = {metrics}
# You'll need to fill these in from your log files

results = {
    # Baseline (InfoNCE/CLIP)
    "InfoNCE": {
        "AdamW": {"txt_r1": None, "img_r1": None, "zeroshot_top1": None},
        "SGD":   {"txt_r1": None, "img_r1": None, "zeroshot_top1": None},
        "Lion":  {"txt_r1": None, "img_r1": None, "zeroshot_top1": None},
    },
    # SogCLR
    "SogCLR": {
        "AdamW": {"txt_r1": None, "img_r1": None, "zeroshot_top1": None},
        "SGD":   {"txt_r1": None, "img_r1": None, "zeroshot_top1": None},
        "Lion":  {"txt_r1": None, "img_r1": None, "zeroshot_top1": None},
    },
    # iSogCLR (without TempNet)
    "iSogCLR": {
        "AdamW": {"txt_r1": None, "img_r1": None, "zeroshot_top1": None},
        "SGD":   {"txt_r1": None, "img_r1": None, "zeroshot_top1": None},
        "Lion":  {"txt_r1": None, "img_r1": None, "zeroshot_top1": None},
    },
    # iSogCLR + TempNet
    "iSogCLR+TempNet": {
        "AdamW": {"txt_r1": None, "img_r1": None, "zeroshot_top1": None},
        "SGD":   {"txt_r1": None, "img_r1": None, "zeroshot_top1": None},
        "Lion":  {"txt_r1": None, "img_r1": None, "zeroshot_top1": None},
    },
}

# ============================================
# STEP 2: Helper function to parse log files
# ============================================

def parse_eval_log(log_path):
    """
    Parse eval_unified_log.txt or coco_log.txt to extract metrics.
    Returns dict with txt_r1, img_r1, zeroshot_top1
    """
    metrics = {"txt_r1": None, "img_r1": None, "zeroshot_top1": None}
    
    if not os.path.exists(log_path):
        print(f"Warning: {log_path} not found")
        return metrics
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Parse COCO results: coco val: {'txt_r1': 1.52, 'img_r1': 1.2475...}
    coco_match = re.search(r"coco val:\s*(\{[^}]+\})", content)
    if coco_match:
        try:
            # Clean up the dict string and parse
            coco_str = coco_match.group(1).replace("'", '"')
            coco_dict = json.loads(coco_str)
            metrics["txt_r1"] = coco_dict.get("txt_r1")
            metrics["img_r1"] = coco_dict.get("img_r1")
        except:
            pass
    
    # Parse zeroshot results: zeroshot: {'zeroshot_top1': 0.054, ...}
    zs_match = re.search(r"zeroshot:\s*(\{[^}]+\})", content)
    if zs_match:
        try:
            zs_str = zs_match.group(1).replace("'", '"')
            zs_dict = json.loads(zs_str)
            metrics["zeroshot_top1"] = zs_dict.get("zeroshot_top1")
        except:
            pass
    
    return metrics

# ============================================
# STEP 3: Load results from experiment folders
# ============================================

# Define mapping from folder names to (loss, optimizer)
OUTPUT_ROOT = "/content/drive/MyDrive/Final Project CSCE Hu Zhao Yexiang Tang/outputs"

folder_mapping = {
    # InfoNCE (Baseline)
    "exp_infonce_adamw_e30": ("InfoNCE", "AdamW"),
    "exp_infonce_sgd_e30": ("InfoNCE", "SGD"),
    "exp_infonce_lion_e30": ("InfoNCE", "Lion"),
    
    # SogCLR
    "sogclr_adamw_e30": ("SogCLR", "AdamW"),
    "sogclr_sgd_e30": ("SogCLR", "SGD"),
    "sogclr_lion_e30": ("SogCLR", "Lion"),
    
    # iSogCLR (without TempNet)
    "isogclr_adamw_e30": ("iSogCLR", "AdamW"),
    "isogclr_sgd_e30": ("iSogCLR", "SGD"),
    "isogclr_lion_e30": ("iSogCLR", "Lion"),
    
    # iSogCLR + TempNet
    "iSogCLR_TempNET_adamw_e30": ("iSogCLR+TempNet", "AdamW"),
    "iSogCLR_TempNET_sgd_e30": ("iSogCLR+TempNet", "SGD"),
    "iSogCLR_TempNET_lion_e30": ("iSogCLR+TempNet", "Lion"),
}

def load_all_results():
    """Load results from all experiment folders"""
    for folder, (loss, opt) in folder_mapping.items():
        log_path = os.path.join(OUTPUT_ROOT, folder, "eval_unified_log.txt")
        metrics = parse_eval_log(log_path)
        
        if metrics["txt_r1"] is not None:
            results[loss][opt] = metrics
            print(f"✓ Loaded {loss} + {opt}")
        else:
            print(f"✗ Missing {loss} + {opt}")

# Uncomment to auto-load from files:
# load_all_results()

# ============================================
# STEP 4: Manual data entry (fill in your results)
# ============================================

# Example: Fill in from your actual results
# results["InfoNCE"]["Lion"] = {"txt_r1": 1.52, "img_r1": 1.25, "zeroshot_top1": 0.054}
# results["SogCLR"]["Lion"] = {"txt_r1": 1.56, "img_r1": 1.46, "zeroshot_top1": 2.756}
# ... etc

# ============================================
# STEP 5: Calculate average metric
# ============================================

def calculate_average(metrics):
    """Calculate average of the three metrics (the final evaluation metric)"""
    if None in [metrics["txt_r1"], metrics["img_r1"], metrics["zeroshot_top1"]]:
        return None
    return (metrics["txt_r1"] + metrics["img_r1"] + metrics["zeroshot_top1"]) / 3

# ============================================
# STEP 6: Create visualizations
# ============================================

def create_visualizations(results, save_path=None):
    """Create 4 grouped bar charts for the metrics"""
    
    losses = ["InfoNCE", "SogCLR", "iSogCLR", "iSogCLR+TempNet"]
    optimizers = ["AdamW", "SGD", "Lion"]
    
    # Colors for each optimizer
    colors = {
        "AdamW": "#4285F4",  # Blue
        "SGD": "#EA4335",    # Red
        "Lion": "#34A853",   # Green
    }
    
    # Metric configurations
    metrics_config = [
        ("txt_r1", "Image-to-Text Recall@1 (%)", "Image-to-Text Retrieval Performance"),
        ("img_r1", "Text-to-Image Recall@1 (%)", "Text-to-Image Retrieval Performance"),
        ("zeroshot_top1", "Top-1 Accuracy (%)", "ImageNet Zero-Shot Classification"),
        ("average", "Average Score (%)", "Overall Performance (Final Metric)"),
    ]
    
    # Create figure with 4 subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    x = np.arange(len(losses))
    width = 0.25
    
    for idx, (metric_key, ylabel, title) in enumerate(metrics_config):
        ax = axes[idx]
        
        for i, opt in enumerate(optimizers):
            values = []
            for loss in losses:
                if metric_key == "average":
                    val = calculate_average(results[loss][opt])
                else:
                    val = results[loss][opt].get(metric_key)
                values.append(val if val is not None else 0)
            
            offset = (i - 1) * width
            bars = ax.bar(x + offset, values, width, label=opt, color=colors[opt], edgecolor='black', linewidth=0.5)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.annotate(f'{val:.2f}',
                                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=7, rotation=0)
        
        ax.set_xlabel('Loss Function', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(losses, rotation=15, ha='right')
        ax.legend(title='Optimizer', loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()

# ============================================
# STEP 7: Create individual plots (optional)
# ============================================

def create_individual_plots(results, save_dir=None):
    """Create separate figures for each metric"""
    
    losses = ["InfoNCE", "SogCLR", "iSogCLR", "iSogCLR+TempNet"]
    optimizers = ["AdamW", "SGD", "Lion"]
    
    colors = {"AdamW": "#4285F4", "SGD": "#EA4335", "Lion": "#34A853"}
    
    metrics_config = [
        ("txt_r1", "Image-to-Text Recall@1 (%)", "Image-to-Text Retrieval Performance"),
        ("img_r1", "Text-to-Image Recall@1 (%)", "Text-to-Image Retrieval Performance"),
        ("zeroshot_top1", "Top-1 Accuracy (%)", "ImageNet Zero-Shot Classification"),
        ("average", "Average Score (%)", "Overall Performance (Final Metric)"),
    ]
    
    x = np.arange(len(losses))
    width = 0.25
    
    for metric_key, ylabel, title in metrics_config:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, opt in enumerate(optimizers):
            values = []
            for loss in losses:
                if metric_key == "average":
                    val = calculate_average(results[loss][opt])
                else:
                    val = results[loss][opt].get(metric_key)
                values.append(val if val is not None else 0)
            
            offset = (i - 1) * width
            bars = ax.bar(x + offset, values, width, label=opt, color=colors[opt], edgecolor='black', linewidth=0.5)
            
            # Add value labels
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.annotate(f'{val:.2f}',
                                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Loss Function', fontweight='bold', fontsize=12)
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=12)
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(losses)
        ax.legend(title='Optimizer')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        if save_dir:
            filename = f"{metric_key}_comparison.png"
            plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Saved {filename}")
        
        plt.show()

# ============================================
# STEP 8: Create summary table
# ============================================

def create_summary_table(results):
    """Print a formatted summary table of all results"""
    
    losses = ["InfoNCE", "SogCLR", "iSogCLR", "iSogCLR+TempNet"]
    optimizers = ["AdamW", "SGD", "Lion"]
    
    print("\n" + "="*100)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*100)
    print(f"{'Loss':<18} {'Optimizer':<10} {'I→T R@1':<10} {'T→I R@1':<10} {'ZS Top-1':<10} {'Average':<10}")
    print("-"*100)
    
    best_avg = 0
    best_config = ""
    
    for loss in losses:
        for opt in optimizers:
            m = results[loss][opt]
            txt_r1 = m["txt_r1"] if m["txt_r1"] else "-"
            img_r1 = m["img_r1"] if m["img_r1"] else "-"
            zs = m["zeroshot_top1"] if m["zeroshot_top1"] else "-"
            avg = calculate_average(m)
            avg_str = f"{avg:.3f}" if avg else "-"
            
            if avg and avg > best_avg:
                best_avg = avg
                best_config = f"{loss} + {opt}"
            
            print(f"{loss:<18} {opt:<10} {txt_r1:<10} {img_r1:<10} {zs:<10} {avg_str:<10}")
    
    print("-"*100)
    print(f"Best configuration: {best_config} with average score: {best_avg:.3f}")
    print("="*100)

# ============================================
# MAIN: Run the visualization
# ============================================

if __name__ == "__main__":
    # Option 1: Auto-load from files (uncomment if folder structure matches)
    # load_all_results()
    
    # Option 2: Manually fill in results (example)
    # Fill in your actual results here:
    
    # Example data (replace with your actual results):
    results["InfoNCE"]["Lion"] = {"txt_r1": 1.52, "img_r1": 1.25, "zeroshot_top1": 0.054}
    results["SogCLR"]["Lion"] = {"txt_r1": 1.56, "img_r1": 1.46, "zeroshot_top1": 2.756}
    # ... fill in all 12 experiments
    
    # Print summary table
    create_summary_table(results)
    
    # Create 2x2 subplot figure
    create_visualizations(results, save_path="experiment_results.png")
    
    # Create individual plots (optional)
    # create_individual_plots(results, save_dir="./plots")