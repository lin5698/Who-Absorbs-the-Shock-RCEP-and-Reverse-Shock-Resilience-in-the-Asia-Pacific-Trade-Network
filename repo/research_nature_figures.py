import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Directory Setup ---
DATA_DIR = Path(__file__).parent / "research_output"
OUTPUT_DIR = DATA_DIR / "nature_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Nature Style Setup ---
# 1. Dimensions (Nature standard)
# Single column width: 89 mm = 3.5 inches
# Double column width: 183 mm = 7.2 inches
SINGLE_COL = 3.5
DOUBLE_COL = 7.2

# 2. Fonts and Lineweights
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,          # Nature uses 5-7 pt for body
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'lines.linewidth': 1.0,  # Standard line thickness
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,       # High resolution
    'pdf.fonttype': 42       # Ensures text is editable in Illustrator/Inkscape
})

# 3. Colorblind-Friendly Palette (Okabe-Ito modified)
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
    """Apply final polish to axes"""
    ax.tick_params(axis='both', which='major', width=0.8, length=3)
    ax.tick_params(axis='both', which='minor', width=0.6, length=2)

# --- Data Loading ---
def load_data():
    tc_path = DATA_DIR / "tariff_relief_TC_quarterly.csv"
    a_path = DATA_DIR / "network_amplification_share_A.csv"
    pos_path = DATA_DIR / "net_absorber_position.csv"
    
    df_tc = pd.read_csv(tc_path) if tc_path.exists() else pd.DataFrame()
    df_a = pd.read_csv(a_path, index_col=0) if a_path.exists() else pd.DataFrame()
    df_pos = pd.read_csv(pos_path) if pos_path.exists() else pd.DataFrame()
    
    # Load panel for combined info if needed
    panel_path = DATA_DIR / "step0_aligned_panel.csv"
    df_panel = pd.read_csv(panel_path) if panel_path.exists() else pd.DataFrame()
    
    return df_tc, df_a, df_pos, df_panel

# --- Figures ---
def figure1_refined_tariff_path(df_tc):
    """
    Figure 1: Tariff relief paths.
    Single column size.
    """
    if df_tc.empty: return
    logger.info("Generating Figure 1...")
    
    df_tc['date'] = pd.to_datetime(df_tc['date'])
    key_pairs = [("CHN", "JPN", COLORS['red']), 
                 ("CHN", "KOR", COLORS['blue']), 
                 ("JPN", "KOR", COLORS['green'])]
    
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.75))
    
    for (i, j, color) in key_pairs:
        sub = df_tc[(df_tc['reporter_iso'] == i) & (df_tc['partner_iso'] == j)]
        if sub.empty:
            sub = df_tc[(df_tc['reporter_iso'] == j) & (df_tc['partner_iso'] == i)]
            
        if not sub.empty:
            sub = sub.sort_values('date')
            ax.plot(sub['date'], sub['TC'], label=f"{i}-{j}", color=color, linewidth=1.5)
            
    # Add intervention line (RCEP)
    rcep_date = pd.Timestamp("2022-01-01")
    ax.axvline(rcep_date, color='black', linestyle='--', linewidth=0.8, zorder=0)
    ax.text(rcep_date + pd.Timedelta(days=15), ax.get_ylim()[1]*0.9, 'RCEP effective', 
            fontsize=6, color='black')
            
    ax.set_ylabel("Tariff Relief (TC)")
    ax.set_xlabel("Year")
    ax.legend(frameon=False, loc='upper left')
    set_nature_style(ax)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig1_Tariff_Path.pdf")
    plt.savefig(OUTPUT_DIR / "Fig1_Tariff_Path.png", dpi=300)
    plt.close()

def figure2_amplification_joyplot(df_a):
    """
    Figure 2: Distribution of Network Amplification
    Since we only have A matrix (reporters x partners), we plot a simple density
    with strict Nature styling.
    """
    if df_a.empty: return
    logger.info("Generating Figure 2...")
    
    vals = df_a.values.flatten()
    vals = vals[np.isfinite(vals) & (vals >= 0) & (vals <= 1)]
    
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.75))
    
    sns.kdeplot(vals, ax=ax, color=COLORS['blue'], fill=True, alpha=0.3, linewidth=1.5)
    
    # Annotate Median
    med = np.median(vals)
    ax.axvline(med, color=COLORS['red'], linestyle=':', linewidth=1)
    ax.text(med + 0.02, ax.get_ylim()[1]*0.5, f'Median:\n{med:.2f}', color=COLORS['red'])
    
    ax.set_xlabel("Network Amplification Share (A)")
    ax.set_ylabel("Density")
    set_nature_style(ax)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig2_Amplification.pdf")
    plt.savefig(OUTPUT_DIR / "Fig2_Amplification.png", dpi=300)
    plt.close()

def figure3_absorber_divergent(df_pos):
    """
    Figure 3: Net Absorber Position.
    Divergent bar chart, sorted by magnitude.
    """
    if df_pos.empty: return
    logger.info("Generating Figure 3...")
    
    df_pos = df_pos.sort_values(by='net_absorber_position')
    
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 1.2)) # Taller for countries
    
    # Divergent colors
    # Net senders: Orange, Net absorbers: Blue
    colors = [COLORS['blue'] if x > 0 else COLORS['orange'] for x in df_pos['net_absorber_position']]
    
    y_pos = np.arange(len(df_pos))
    ax.barh(y_pos, df_pos['net_absorber_position'], color=colors, height=0.7)
    
    ax.axvline(0, color='black', linewidth=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_pos['iso3'])
    ax.set_xlabel("Net Absorber Position")
    
    # Custom legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=COLORS['blue'], lw=4),
                    Line2D([0], [0], color=COLORS['orange'], lw=4)]
    ax.legend(custom_lines, ['Net Absorber (>0)', 'Net Spiller (<0)'], frameon=False, loc='lower right')
    
    set_nature_style(ax)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig3_Absorber.pdf")
    plt.savefig(OUTPUT_DIR / "Fig3_Absorber.png", dpi=300)
    plt.close()

def main():
    df_tc, df_a, df_pos, df_panel = load_data()
    figure1_refined_tariff_path(df_tc)
    figure2_amplification_joyplot(df_a)
    figure3_absorber_divergent(df_pos)
    logger.info("Nature figures generated successfully.")

if __name__ == "__main__":
    main()
