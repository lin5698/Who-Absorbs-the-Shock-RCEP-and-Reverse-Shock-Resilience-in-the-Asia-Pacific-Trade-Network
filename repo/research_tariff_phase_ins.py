import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up styles for Nature
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
    'orange': '#E69F00',
    'green': '#009E73',
    'purple': '#CC79A7',
    'skyblue': '#56B4E9',
    'red': '#D55E00',
    'yellow': '#F0E442',
    'gray': '#999999',
    'black': '#000000'
}
NATURE_PALETTE = list(COLORS.values())
sns.set_palette(NATURE_PALETTE)

def set_nature_style(ax):
    ax.tick_params(axis='both', which='major', width=0.8, length=3)
    ax.tick_params(axis='both', which='minor', width=0.6, length=2)

def main():
    OUTPUT_DIR = Path("research_output/nature_figures")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv("research_output/tariff_relief_TC_quarterly.csv")
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    
    # Reduction in magnitude (positive numbers)
    df['reduction_mag'] = df['TC'].abs() 
    
    # We only care about pairs starting from 2022 for the actual phase-in table, 
    # but "全样本" (full sample) could mean all years or 2022-2023. RCEP started 2022.
    df_post = df[df['year'] >= 2022].copy()
    
    # Group pairs by max reduction to define High/Medium/Low
    pair_max = df_post.groupby(['reporter_iso', 'partner_iso'])['reduction_mag'].max().reset_index()
    
    # Filter only those that actually reduced tariffs for tertiles
    reduced_pairs = pair_max[pair_max['reduction_mag'] > 0].copy()
    reduced_pairs['group'] = pd.qcut(reduced_pairs['reduction_mag'], 3, labels=['Low', 'Medium', 'High'])
    
    df_post = df_post.merge(reduced_pairs[['reporter_iso', 'partner_iso', 'group']], on=['reporter_iso', 'partner_iso'], how='left')
    df_post['group'] = df_post['group'].astype(str).replace('nan', 'Zero')
    
    # --- Table Calculation ---
    rows = []
    
    def calc_stats(sub_df, name):
        vals = sub_df['reduction_mag']
        non_zero = (vals > 0).mean() * 100
        row = {
            'Group': name,
            'Mean': vals.mean(),
            'Std': vals.std(),
            'P25': vals.quantile(0.25),
            'P50': vals.median(),
            'P75': vals.quantile(0.75),
            'Min': vals.min(),
            'Max': vals.max(),
            'Non_Zero_Pct': non_zero,
            'N': len(vals)
        }
        return row
    
    # 1. Full Sample (all quarters)
    rows.append(calc_stats(df, "Full Sample (2010-2023)"))
    # 2. 2022
    df_2022 = df_post[df_post['year'] == 2022]
    rows.append(calc_stats(df_2022, "2022"))
    # 3. 2023
    df_2023 = df_post[df_post['year'] == 2023]
    rows.append(calc_stats(df_2023, "2023"))
    # 4. High
    rows.append(calc_stats(df_post[df_post['group'] == 'High'], "High Reduction Pairs"))
    # 5. Medium
    rows.append(calc_stats(df_post[df_post['group'] == 'Medium'], "Medium Reduction Pairs"))
    # 6. Low
    rows.append(calc_stats(df_post[df_post['group'] == 'Low'], "Low Reduction Pairs"))
    
    table_df = pd.DataFrame(rows)
    table_df.to_csv(OUTPUT_DIR / "Table_TC_Summary_Statistics.csv", index=False)
    print("Table saved to Table_TC_Summary_Statistics.csv")
    print(table_df)
    
    # Stats for figure caption
    # Let's calculate the pair-level max reduction in 2022 for the caption
    pair_max_2022 = df_2022.groupby(['reporter_iso', 'partner_iso'])['reduction_mag'].max()
    pair_max_2022_nz = pair_max_2022[pair_max_2022 > 0] * 100  # in basis points
    
    if len(pair_max_2022_nz) > 0:
        bp_2022 = pair_max_2022_nz.mean()
        p25 = pair_max_2022_nz.quantile(0.25)
        p50 = pair_max_2022_nz.median()
        p75 = pair_max_2022_nz.quantile(0.75)
        max_2022 = pair_max_2022_nz.max()
        min_2022 = pair_max_2022_nz.min()
    else:
        bp_2022 = p25 = p50 = p75 = max_2022 = min_2022 = 0
    
    caption = (f"2022年第一年平均减让{bp_2022:.1f}个基点（非零承诺对）、"
               f"P25/P50/P75={p25:.1f}/{p50:.1f}/{p75:.1f}基点、"
               f"最大/最小={max_2022:.1f}/{min_2022:.1f}基点")
    print("\nCaption to use in the paper:")
    print(caption)
    
    # --- Figure Generation ---
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    
    # Left: Line plot of 6 typical pairs
    ax1 = axes[0]
    key_pairs = [("CHN", "JPN", COLORS['red']), 
                 ("CHN", "KOR", COLORS['blue']), 
                 ("CHN", "AUS", COLORS['green']),
                 ("CHN", "SGP", COLORS['orange']),
                 ("JPN", "KOR", COLORS['purple']),
                 ("KOR", "AUS", COLORS['skyblue'])]
    
    for (i, j, color) in key_pairs:
        sub = df[(df['reporter_iso'] == i) & (df['partner_iso'] == j)].sort_values('date')
        if not sub.empty and sub['reduction_mag'].max() > 0:
            ax1.plot(sub['date'], sub['reduction_mag'], label=f"{i}-{j}", color=color, linewidth=1.5)
        else:
            # try reverse
            sub2 = df[(df['reporter_iso'] == j) & (df['partner_iso'] == i)].sort_values('date')
            if not sub2.empty and sub2['reduction_mag'].max() > 0:
                ax1.plot(sub2['date'], sub2['reduction_mag'], label=f"{j}-{i}", color=color, linewidth=1.5)
    
    rcep_date = pd.Timestamp("2022-01-01")
    ax1.axvline(rcep_date, color='black', linestyle='--', linewidth=0.8, zorder=0)
    ax1.text(rcep_date + pd.Timedelta(days=60), ax1.get_ylim()[1]*0.9 if ax1.get_ylim()[1] > 0 else 1.0, 'RCEP effective\n(Jan 2022)', 
            fontsize=6, color='black')
    
    ax1.set_ylabel("Tariff Reduction (Percentage Points)")
    ax1.set_xlabel("Year")
    ax1.legend(frameon=False, loc='upper left')
    ax1.set_title("a. Phase-in paths of selected pairs", loc='left', fontweight='bold', fontsize=8)
    set_nature_style(ax1)
    
    # Right: Full sample distribution
    ax2 = axes[1]
    # To show a meaningful distribution, we plot the maximum reduction achieved by each pair post-RCEP
    val_nz = pair_max[pair_max['reduction_mag'] > 0]['reduction_mag']
    
    if len(val_nz) > 1:
        sns.kdeplot(val_nz, ax=ax2, color=COLORS['blue'], fill=True, alpha=0.3, linewidth=1.5, bw_adjust=0.5)
    elif len(val_nz) > 0:
        sns.histplot(val_nz, ax=ax2, color=COLORS['blue'], bins=10)
        
    if len(val_nz) > 0:
        ax2.axvline(val_nz.mean(), color=COLORS['red'], linestyle=':', linewidth=1)
        ax2.text(val_nz.mean() + 0.05, ax2.get_ylim()[1]*0.8, f'Mean:\n{val_nz.mean():.2f}', color=COLORS['red'])
    
    ax2.set_xlabel("Maximum Tariff Reduction per pair (Percentage Points)")
    ax2.set_ylabel("Density / Count")
    ax2.set_title("b. Distribution of Tariff Reduction (Non-zero pairs)", loc='left', fontweight='bold', fontsize=8)
    if len(val_nz) > 0 and val_nz.max() > 0:
        ax2.set_xlim(-0.1, val_nz.max() * 1.1)
    set_nature_style(ax2)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig_Tariff_Phase_ins.pdf")
    plt.savefig(OUTPUT_DIR / "Fig_Tariff_Phase_ins.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
