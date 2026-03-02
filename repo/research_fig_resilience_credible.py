import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    'green': '#009E73',
    'red': '#D55E00',
    'gray': '#999999'
}

def main():
    REPO_DIR = Path(__file__).parent
    OUTPUT_DIR = REPO_DIR / "research_output/nature_figures"
    DATA_PATH = REPO_DIR / "research_output/rolling_resilience_bootstrapped.csv"
    
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)
    rcep_date = pd.Timestamp("2022-01-01")
    
    # Panel a: Network Amplification Share A_t
    ax1 = axes[0]
    ax1.plot(df['date'], df['A_p50'], label="Median", color=COLORS['blue'], linewidth=1.5)
    ax1.fill_between(df['date'], df['A_p025'], df['A_p975'], color=COLORS['blue'], alpha=0.1, label="95% CI")
    ax1.fill_between(df['date'], df['A_p16'], df['A_p84'], color=COLORS['blue'], alpha=0.2, label="68% CI")
    
    ax1.axvline(rcep_date, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.text(rcep_date + pd.Timedelta(days=15), ax1.get_ylim()[1]*0.9, 'RCEP effective', fontsize=6)
    
    ax1.set_title("a. Aggregated Network Amplification Share $A_t$ (H=12)", loc='left', fontweight='bold')
    ax1.set_ylabel("Share")
    ax1.legend(frameon=False, loc='upper left')
    
    # Panel b: Half-life HL_t
    ax2 = axes[1]
    ax2.plot(df['date'], df['HL_p50'], label="Median", color=COLORS['green'], linewidth=1.5)
    ax2.fill_between(df['date'], df['HL_p025'], df['HL_p975'], color=COLORS['green'], alpha=0.1, label="95% CI")
    ax2.fill_between(df['date'], df['HL_p16'], df['HL_p84'], color=COLORS['green'], alpha=0.2, label="68% CI")
    
    ax2.axvline(rcep_date, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    ax2.set_title("b. Aggregated Half-life $HL_t$ (H=12)", loc='left', fontweight='bold')
    ax2.set_ylabel("Quarters")
    ax2.set_xlabel("Year")
    ax2.legend(frameon=False, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig_Resilience_Measures_Intervals.pdf")
    plt.savefig(OUTPUT_DIR / "Fig_Resilience_Measures_Intervals.png", dpi=300)
    print(f"Figure saved to {OUTPUT_DIR / 'Fig_Resilience_Measures_Intervals.pdf'}")

if __name__ == "__main__":
    main()
