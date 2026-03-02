import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TEMP_DIR = Path("research_output")
OUTPUT_DIR = Path("research_output/nature_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Nature Style
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
    'up': '#009E73',     # Green
    'down': '#D55E00',   # Red
    'stable': '#999999', # Gray
    'pre': '#56B4E9',    # Light Blue
    'post': '#0072B2'    # Dark Blue
}

def set_nature_style(ax):
    ax.tick_params(axis='both', which='major', width=0.8, length=3)

def calc_net_absorber_pos(df, h_val, w_type):
    """
    Calculate Pos_i = sum_j (A_{j<-i} - A_{i<-j})
    """
    sub = df[(df['H'] == h_val) & (df['W_type'] == w_type)].copy()
    if sub.empty:
        return pd.DataFrame()
        
    # Pre-period: 2016-2019
    # Post-period: 2022-2024
    sub['period'] = 'Other'
    sub.loc[sub['date'] < '2020-01-01', 'period'] = 'Pre'
    sub.loc[sub['date'] >= '2022-01-01', 'period'] = 'Post'
    
    # Filter to periods of interest
    sub = sub[sub['period'].isin(['Pre', 'Post'])]
    
    # A_ji is amplification of i to j (outgoing from i)
    # A_ij is amplification of j to i (incoming to i)
    # We aggregate by period and reporter/partner
    
    # Outgoing: sum of A for reporter_iso=i (this is A_ji if reporter=partner in some logic, 
    # but in our CSV: reporter_iso=receiver, partner_iso=sender.
    # To follow Pos_i = sum_j (A_{j<-i} - A_{i<-j}):
    # A_{j<-i} is when partner_iso=i (sender) and reporter_iso=j (receiver)
    # A_{i<-j} is when reporter_iso=i (receiver) and partner_iso=j (sender)
    
    # Outgoing (transmitted): sum(A) where partner_iso = i
    outgoing = sub.groupby(['period', 'partner_iso'])['A'].mean().unstack('period')
    # Incoming (received): sum(A) where reporter_iso = i
    incoming = sub.groupby(['period', 'reporter_iso'])['A'].mean().unstack('period')
    
    # Pos = Outgoing - Incoming
    pos = outgoing - incoming
    return pos

def plot_slope_graph(pos_df, title, filename):
    """
    Create a slope graph for re-ranking.
    """
    fig, ax = plt.subplots(figsize=(3, 5))
    
    # Sort by Post value for ranking
    pos_df = pos_df.sort_values(by='Post', ascending=True)
    
    countries = pos_df.index
    y_pre = pos_df['Pre']
    y_post = pos_df['Post']
    
    # Normalize for display if needed, but here we use absolute values
    # Rank them
    rank_pre = pos_df['Pre'].rank(ascending=True)
    rank_post = pos_df['Post'].rank(ascending=True)
    
    # Staggering logic for labels
    def get_staggered_y(y_values, min_dist=0.015):
        sorted_idx = np.argsort(y_values)
        staggered = y_values.copy()
        for i in range(1, len(sorted_idx)):
            curr = sorted_idx[i]
            prev = sorted_idx[i-1]
            if staggered[curr] - staggered[prev] < min_dist:
                staggered[curr] = staggered[prev] + min_dist
        return staggered

    y_pre_staggered = get_staggered_y(y_pre.values)
    y_post_staggered = get_staggered_y(y_post.values)
    
    # Map back to index
    pre_map = dict(zip(y_pre.index, y_pre_staggered))
    post_map = dict(zip(y_post.index, y_post_staggered))

    for i, country in enumerate(countries):
        val_pre = y_pre[country]
        val_post = y_post[country]
        
        # Staggered label positions
        label_pre = pre_map[country]
        label_post = post_map[country]
        
        r_pre = rank_pre[country]
        r_post = rank_post[country]
        
        # Color based on rank change
        if r_post > r_pre + 0.5:
            color = COLORS['up']
        elif r_post < r_pre - 0.5:
            color = COLORS['down']
        else:
            color = COLORS['stable']
            
        # Draw line (to actual values)
        ax.plot([0, 1], [val_pre, val_post], color=color, alpha=0.6, marker='o', markersize=3)
        
        # Labels (at staggered positions)
        ax.text(-0.05, label_pre, f"{country} ({val_pre:.2f})", ha='right', va='center', fontsize=6)
        ax.text(1.05, label_post, f"{country} ({val_post:.2f})", ha='left', va='center', fontsize=6)
        
        # Optional: draw tiny light lines from marker to staggered label if distance is large
        if abs(label_pre - val_pre) > 0.001:
            ax.plot([-0.02, 0], [label_pre, val_pre], color='gray', alpha=0.3, linewidth=0.5)
        if abs(label_post - val_post) > 0.001:
            ax.plot([1.0, 1.02], [val_post, label_post], color='gray', alpha=0.3, linewidth=0.5)
        
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Pre-RCEP\n(2016-19)', 'Implementation\n(2022-24)'])
    
    ax.set_ylabel("Net Absorber Position")
    ax.set_title(title, loc='left', fontweight='bold', pad=15)
    
    # Remove spines
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{filename}.pdf", bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_absorber_figures():
    logger.info("Loading regression data...")
    try:
        df = pd.read_csv(TEMP_DIR / "pairwise_regression_data_full.csv")
        df['date'] = pd.to_datetime(df['date'])
    except FileNotFoundError:
        logger.error("Data file not found. Please run research_table_resilience_final.py first.")
        return

    # 1. Primary: H=8, Time-Varying
    logger.info("Generating Figure 1: H=8, Time-Varying")
    pos_h8 = calc_net_absorber_pos(df, 8, 'Time-Varying')
    if not pos_h8.empty:
        plot_slope_graph(pos_h8, "Directional re-ranking in resilience (H=8)", "Fig_Absorber_Reranking_H8_TV")

    # 2. Robustness: H=12, Time-Varying
    logger.info("Generating Figure 2: H=12")
    pos_h12 = calc_net_absorber_pos(df, 12, 'Time-Varying')
    if not pos_h12.empty:
        plot_slope_graph(pos_h12, "Robustness: Directional re-ranking (H=12)", "Fig_Absorber_Reranking_H12_TV")

    # 3. Robustness: H=8, Fixed-Pre
    logger.info("Generating Figure 3: Fixed-Pre Network")
    pos_fixed = calc_net_absorber_pos(df, 8, 'Fixed-Pre')
    if not pos_fixed.empty:
        plot_slope_graph(pos_fixed, "Robustness: Fixed-Pre Network (H=8)", "Fig_Absorber_Reranking_H8_FixedPre")

if __name__ == "__main__":
    generate_absorber_figures()
