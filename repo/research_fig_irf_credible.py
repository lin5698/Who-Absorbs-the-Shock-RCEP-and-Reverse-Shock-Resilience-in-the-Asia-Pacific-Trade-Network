import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Nature style setup
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'lines.linewidth': 1.0,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'pdf.fonttype': 42
})

COLORS = {
    'blue': '#0072B2',
    'red': '#D55E00',
    'gray': '#999999',
    'light_blue': '#56B4E9'
}

def plot_with_intervals(ax, x, boot_data, label, color):
    # boot_data is a list of arrays
    boot_data = np.array(boot_data)
    mean = np.mean(boot_data, axis=0)
    p025 = np.percentile(boot_data, 2.5, axis=0)
    p16 = np.percentile(boot_data, 16, axis=0)
    p84 = np.percentile(boot_data, 84, axis=0)
    p975 = np.percentile(boot_data, 97.5, axis=0)
    
    ax.plot(x, mean, label=label, color=color, linewidth=1.5)
    ax.fill_between(x, p025, p975, color=color, alpha=0.1, label=f"{label} (95% CI)")
    ax.fill_between(x, p16, p84, color=color, alpha=0.2, label=f"{label} (68% CI)")

def main():
    REPO_DIR = Path(__file__).parent
    OUTPUT_DIR = REPO_DIR / "research_output/nature_figures"
    DATA_PATH = REPO_DIR / "research_output/girf_bootstrap_data.json"
    
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found.")
        return

    with open(DATA_PATH, 'r') as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    
    horizons = np.arange(len(data['2018-12-31'][0]['total']))
    
    # Panel a: Pre-period (2018Q4)
    ax1 = axes[0]
    total_boot_pre = [d['total'] for d in data['2018-12-31']]
    direct_boot_pre = [d['direct'] for d in data['2018-12-31']]
    
    plot_with_intervals(ax1, horizons, total_boot_pre, "Total Effect", COLORS['blue'])
    plot_with_intervals(ax1, horizons, direct_boot_pre, "Direct Effect", COLORS['gray'])
    
    ax1.set_title("a. Pre-RCEP Response (2018Q4)", loc='left', fontweight='bold')
    ax1.set_ylabel("Response of China VAX Growth")
    ax1.set_xlabel("Horizon (Quarters)")
    ax1.axhline(0, color='black', linewidth=0.5, linestyle=':')
    ax1.legend(frameon=False, loc='upper right')
    
    # Panel b: Implementation period (2022Q4)
    ax2 = axes[1]
    total_boot_post = [d['total'] for d in data['2022-12-31']]
    direct_boot_post = [d['direct'] for d in data['2022-12-31']]
    
    plot_with_intervals(ax2, horizons, total_boot_post, "Total Effect", COLORS['red'])
    plot_with_intervals(ax2, horizons, direct_boot_post, "Direct Effect", COLORS['gray'])
    
    ax2.set_title("b. RCEP Implementation (2022Q4)", loc='left', fontweight='bold')
    ax2.set_xlabel("Horizon (Quarters)")
    ax2.axhline(0, color='black', linewidth=0.5, linestyle=':')
    ax2.legend(frameon=False, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig_TVP_GIRFs_Intervals.pdf")
    plt.savefig(OUTPUT_DIR / "Fig_TVP_GIRFs_Intervals.png", dpi=300)
    print(f"Figure saved to {OUTPUT_DIR / 'Fig_TVP_GIRFs_Intervals.pdf'}")

if __name__ == "__main__":
    main()
