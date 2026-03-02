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
    'lines.linewidth': 0.8,
    'axes.linewidth': 0.5,
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
    'black': '#000000',
    'gray': '#808080'
}

def set_nature_style(ax):
    ax.tick_params(axis='both', which='major', width=0.5, length=3)
    ax.tick_params(axis='both', which='minor', width=0.4, length=2)

def main():
    REPO_DIR = Path(__file__).parent
    OUTPUT_DIR = REPO_DIR / "research_output/nature_figures"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    data_path = REPO_DIR / "research_output/tariff_relief_TC_quarterly.csv"
    if not data_path.exists():
        print(f"Error: {data_path} not found. Run repo/research_data_construction.py first.")
        return

    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    
    # TC is usually reduction magnitude. If negative, make it positive for relief magnitude.
    df['reduction_mag'] = df['TC'].abs() * 100 # In basis points
    
    # Filter for relevant period: 2018Q1 - 2023Q4
    df_sample = df[(df['date'] >= '2018-01-01') & (df['date'] <= '2023-12-31')].copy()
    
    # --- Table Generation ---
    def get_stats(sub_df, label):
        vals = sub_df['reduction_mag']
        nonzero = (vals > 0).mean()
        stats = {
            'Panel': label,
            'Mean': vals.mean(),
            'SD': vals.std(),
            'P10': vals.quantile(0.1),
            'P25': vals.quantile(0.25),
            'P50 (Median)': vals.median(),
            'P75': vals.quantile(0.75),
            'P90': vals.quantile(0.9),
            'Min': vals.min(),
            'Max': vals.max(),
            'Share Non-zero': nonzero
        }
        return stats

    table_data = []
    # Full Sample
    table_data.append(get_stats(df_sample, "Full Sample (2018-2023)"))
    # 2022
    table_data.append(get_stats(df_sample[df_sample['year'] == 2022], "2022"))
    # 2023
    table_data.append(get_stats(df_sample[df_sample['year'] == 2023], "2023"))
    
    table_df = pd.DataFrame(table_data)
    table_df.to_csv(OUTPUT_DIR / "Table_TC_Summary_Stats_Final.csv", index=False)
    print("Table saved to Table_TC_Summary_Stats_Final.csv")

    # --- Figure Generation ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    
    # Panel A: Representative Pairs
    representative_pairs = [
        ("CHN", "JPN", COLORS['red']),
        ("CHN", "KOR", COLORS['blue']),
        ("JPN", "KOR", COLORS['green']),
        ("AUS", "CHN", COLORS['orange']),
        ("IDN", "CHN", COLORS['purple']),
        ("CHN", "VNM", COLORS['skyblue'])
    ]
    
    for (i, j, color) in representative_pairs:
        pair_df = df_sample[(df_sample['reporter_iso'] == i) & (df_sample['partner_iso'] == j)].sort_values('date')
        if not pair_df.empty:
            ax1.plot(pair_df['date'], pair_df['reduction_mag'], label=f"{i}→{j}", color=color, alpha=0.9)
        else:
            # Try reverse
            pair_df = df_sample[(df_sample['reporter_iso'] == j) & (df_sample['partner_iso'] == i)].sort_values('date')
            if not pair_df.empty:
                ax1.plot(pair_df['date'], pair_df['reduction_mag'], label=f"{j}→{i}", color=color, alpha=0.9)

    ax1.axvline(pd.Timestamp("2022-01-01"), color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.text(pd.Timestamp("2022-01-15"), ax1.get_ylim()[1]*0.9, "RCEP", fontsize=6, fontweight='bold')
    
    ax1.set_title("a. Representative tariff phase-ins", loc='left', fontweight='bold')
    ax1.set_ylabel("Tariff Relief (Basis Points)")
    ax1.set_xlabel("Year")
    ax1.legend(loc='upper left', ncol=2, frameon=False, columnspacing=0.5)
    set_nature_style(ax1)
    
    # Panel B: Distribution
    df_2022 = df_sample[df_sample['year'] == 2022]
    df_2023 = df_sample[df_sample['year'] == 2023]
    
    # Group by pair to get cross-pair distribution (max reduction in that year)
    dist_2022 = df_2022.groupby(['reporter_iso', 'partner_iso'])['reduction_mag'].max()
    dist_2023 = df_2023.groupby(['reporter_iso', 'partner_iso'])['reduction_mag'].max()
    
    # Only non-zero pairs for distribution plot clarity if requested, but request says "cross-pair distribution"
    # Typically we show all pairs but maybe zoom in or use density
    sns.kdeplot(dist_2022, ax=ax2, label="2022", color=COLORS['blue'], fill=True, alpha=0.2)
    sns.kdeplot(dist_2023, ax=ax2, label="2023", color=COLORS['red'], fill=True, alpha=0.2)
    
    # Report 2022 stats in legend or text
    m22 = dist_2022.mean()
    p50_22 = dist_2022.median()
    p75_22 = dist_2022.quantile(0.75)
    
    ax2.annotate(f"2022 Mean: {m22:.1f} bp\nMedian: {p50_22:.1f} bp", 
                xy=(0.6, 0.7), xycoords='axes fraction', fontsize=6, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    ax2.set_title("b. Distribution of tariff relief", loc='left', fontweight='bold')
    ax2.set_xlabel("Reduction Magnitude (Basis Points)")
    ax2.set_ylabel("Density")
    ax2.legend(loc='upper right', frameon=False)
    set_nature_style(ax2)
    
    plt.tight_layout()
    output_fig = OUTPUT_DIR / "Fig_Tariff_Phase_ins_Final.pdf"
    plt.savefig(output_fig)
    plt.savefig(OUTPUT_DIR / "Fig_Tariff_Phase_ins_Final.png", dpi=300)
    print(f"Figure saved to {output_fig}")

if __name__ == "__main__":
    main()
