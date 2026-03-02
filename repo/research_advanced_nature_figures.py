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
SINGLE_COL = 3.5
DOUBLE_COL = 7.2

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
    'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73',
    'purple': '#CC79A7', 'skyblue': '#56B4E9', 'red': '#D55E00',
    'yellow': '#F0E442', 'gray': '#999999', 'black': '#000000'
}

def set_nature_style(ax):
    """Apply final polish to axes"""
    ax.tick_params(axis='both', which='major', width=0.8, length=3)
    ax.tick_params(axis='both', which='minor', width=0.6, length=2)

# --- Data Check ---
def load_data():
    panel_path = DATA_DIR / "step0_aligned_panel.csv"
    if panel_path.exists():
        df_panel = pd.read_csv(panel_path)
    else:
        df_panel = pd.DataFrame()
    return df_panel

# --- Figures ---
def figure5_gravity_diagnostic(df_panel):
    """
    Figure 5: Gravity Diagnostic
    Scatter of Trade Volume vs (GDP_i * GDP_j) / Distance
    Since we only have 'total_exports', we plot Exports vs Reporter GDP (simplified proxy for gravity mass)
    """
    if df_panel.empty: return
    logger.info("Generating Figure 5 (Gravity Proxy)...")
    
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.8))
    
    # Very simple diagnostic: log(exports) vs log(GDP)
    valid = df_panel.dropna(subset=['gdp_current_usd', 'total_exports_usd_k'])
    
    # Sample a manageable subset for the scatter
    if len(valid) > 2000:
        valid = valid.sample(2000, random_state=42)
        
    x = np.log(valid['gdp_current_usd'] + 1)
    y = np.log(valid['total_exports_usd_k'] + 1)
    
    ax.scatter(x, y, alpha=0.3, s=2, color=COLORS['blue'], edgecolors='none')
    
    # Trendline
    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m*x + b, color=COLORS['red'], linewidth=1.5, linestyle='--')
    
    ax.set_xlabel("Log(Reporter GDP)")
    ax.set_ylabel("Log(Total Exports)")
    
    # Annotate correlation
    corr = np.corrcoef(x, y)[0,1]
    ax.text(0.05, 0.9, f"r = {corr:.2f}", transform=ax.transAxes, fontsize=7, color=COLORS['black'])
    
    set_nature_style(ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig5_Gravity_Proxy.pdf")
    plt.savefig(OUTPUT_DIR / "Fig5_Gravity_Proxy.png", dpi=300)
    plt.close()

def figure6_macro_correlation_heatmap(df_panel):
    """
    Figure 6: Strict Nature Style Heatmap for Macro variables
    """
    if df_panel.empty: return
    logger.info("Generating Figure 6 (Macro Corrs)...")
    
    cols = ['gdp_current_usd', 'total_exports_usd_k', 'total_imports_usd_k', 
            'population', 'tariff_ahs_weighted', 'tariff_mfn_weighted']
    
    # Filter available
    cols = [c for c in cols if c in df_panel.columns]
    
    if len(cols) < 2: return
    
    corr = df_panel[cols].corr()
    
    # Clean labels
    labels = [c.replace('_', ' ').title() for c in cols]
    
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
    
    # Heatmap setup
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, 
                annot=True, fmt=".1f", annot_kws={"size": 6},
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar_kws={'shrink': 0.8})
                
    ax.tick_params(axis='x', rotation=45)
    
    set_nature_style(ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig6_Macro_Corr.pdf")
    plt.savefig(OUTPUT_DIR / "Fig6_Macro_Corr.png", dpi=300)
    plt.close()

def main():
    df_panel = load_data()
    figure5_gravity_diagnostic(df_panel)
    figure6_macro_correlation_heatmap(df_panel)
    logger.info("Nature advanced figures generated successfully.")

if __name__ == "__main__":
    main()
