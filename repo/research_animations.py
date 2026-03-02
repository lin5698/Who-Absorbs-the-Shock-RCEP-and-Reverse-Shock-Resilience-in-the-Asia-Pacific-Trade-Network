import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import networkx as nx
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directories
DATA_DIR = Path(__file__).parent / "research_output"
OUTPUT_DIR = DATA_DIR / "animations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Shared Styling Constants (Nature Style approx)
NATURE_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#7f7f7f", "#e377c2", "#17becf"]
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 10,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150
})

RCEP_COUNTRIES = {
    'CHN': 'China', 'JPN': 'Japan', 'KOR': 'South Korea', 'AUS': 'Australia',
    'NZL': 'New Zealand', 'IDN': 'Indonesia', 'MYS': 'Malaysia', 'PHL': 'Philippines',
    'SGP': 'Singapore', 'THA': 'Thailand', 'VNM': 'Vietnam', 'BRN': 'Brunei',
    'KHM': 'Cambodia', 'LAO': 'Laos', 'MMR': 'Myanmar'
}
RCEP_LIST = list(RCEP_COUNTRIES.keys())

# Simple coordinate approximation for layout
RCEP_COORDS = {
    'CHN': (116, 40), 'JPN': (139, 35), 'KOR': (127, 37), 'AUS': (133, -25),
    'NZL': (174, -40), 'IDN': (113, -0.7), 'MYS': (101, 3.1), 'PHL': (121, 14),
    'SGP': (103, 1.3), 'THA': (100, 15), 'VNM': (105, 14), 'BRN': (114, 4.5),
    'KHM': (104, 12), 'LAO': (102, 18), 'MMR': (96, 21)
}


def load_data():
    """Load core datasets."""
    logger.info("Loading datasets...")
    
    # 1. Trade Network Weights (Not strictly needed if we use master bilateral, but keeping)
    w_path = DATA_DIR / "trade_network_W_quarterly.csv"
    if w_path.exists():
        df_w = pd.read_csv(w_path)
    else:
        df_w = pd.DataFrame()
        
    # 2. Tariff Relief TC
    tc_path = DATA_DIR / "tariff_relief_TC_quarterly.csv"
    if tc_path.exists():
        df_tc = pd.read_csv(tc_path)
    else:
        df_tc = pd.DataFrame()
        
    # 3. Quarterly Master Bilateral Data (for trading volume)
    # The step0_aligned_panel does not have reporter/partner breakdown.
    # Using data/master_bilateral_quarterly.csv
    panel_path = Path(__file__).parent / "data" / "master_bilateral_quarterly.csv"
    if panel_path.exists():
        df_panel = pd.read_csv(panel_path, usecols=['reporter_iso', 'partner_iso', 'date', 'export_usd'])
    else:
        logger.warning(f"File not found: {panel_path}")
        df_panel = pd.DataFrame()

    return df_w, df_tc, df_panel

def create_network_animation(df_panel):
    """
    Creates an MP4 animation showing trade volume growth.
    Nodes = Countries, Edges = Trade volume.
    """
    if df_panel.empty:
        logger.warning("No panel data for network animation.")
        return
        
    logger.info("Generating Network Animation...")
    
    # Ensure date format
    # The actual column is 'date', let's drop nan
    df_panel = df_panel.dropna(subset=['date'])
    df_panel['date'] = pd.to_datetime(df_panel['date'], errors='coerce')
    df_panel = df_panel.dropna(subset=['date'])
    
    quarters = sorted(df_panel['date'].unique())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Pre-compute positions
    pos = {k: np.array([v[0], v[1]]) for k, v in RCEP_COORDS.items()}

    def update(frame):
        ax.clear()
        date_val = quarters[frame]
        
        subset = df_panel[df_panel['date'] == date_val]
        
        G = nx.DiGraph()
        for node in RCEP_LIST:
            G.add_node(node)
            
        # Add edges and track max weight for scaling
        max_trade = subset['export_usd'].max() if not subset.empty else 1
        if max_trade == 0 or pd.isna(max_trade): max_trade = 1
        
        for _, row in subset.iterrows():
            i, j, val = row['reporter_iso'], row['partner_iso'], row['export_usd']
            if i in RCEP_LIST and j in RCEP_LIST and pd.notna(val) and val > 0:
                # Only show significant edges to reduce clutter
                if val > max_trade * 0.05: 
                    G.add_edge(i, j, weight=val)
                
        # Draw base nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='#1f77b4', node_size=300, alpha=0.9)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color='white', font_weight='bold')
        
        edges = G.edges(data=True)
        if edges:
            weights = [d['weight'] for (u, v, d) in edges]
            # Normalize edge widths between 0.5 and 4.0
            norm_weights = [0.5 + (w / max_trade) * 3.5 for w in weights]
            nx.draw_networkx_edges(G, pos, ax=ax, width=norm_weights, 
                                   edge_color='gray', alpha=0.5, arrows=True, arrowsize=10, connectionstyle='arc3,rad=0.1')
        
        ax.set_title(f"RCEP Intra-Regional Trade Network\nQuarter: {pd.to_datetime(date_val).strftime('%Y-Q%q')}", fontsize=12, pad=10)
        ax.axis('off')
        
        return ax,
        
    ani = animation.FuncAnimation(fig, update, frames=len(quarters), interval=300, blit=False)
    
    out_path = OUTPUT_DIR / "rcep_network_evolution.mp4"
    try:
        ani.save(out_path, writer='ffmpeg', fps=3, dpi=200)
        logger.info(f"Saved Network Animation to {out_path}")
    except Exception as e:
        logger.error(f"Failed to save MP4 (is ffmpeg installed?): {e}")
        # Fallback to GIF
        out_path = OUTPUT_DIR / "rcep_network_evolution.gif"
        ani.save(out_path, writer='pillow', fps=3, dpi=120)
        logger.info(f"Saved Network Animation to {out_path} (GIF fallback)")
        
    plt.close()


def create_tariff_relief_heatmap_animation(df_tc):
    """
    Creates an animated heatmap showing the tariff relief (TC) increasing over time.
    X-axis: Partner, Y-axis: Reporter
    """
    if df_tc.empty:
        logger.warning("No TC data for Heatmap animation.")
        return
        
    logger.info("Generating Tariff Relief Heatmap Animation...")
    df_tc['date'] = pd.to_datetime(df_tc['date'])
    quarters = sorted(df_tc['date'].unique())
    
    # Pre-pivoting data to ensure structure stability
    max_tc = min(df_tc['TC'].max(), 5.0) # Cap at 5% or 100% depending on scale. Assuming max is < 5 (it's 0-1 or 0-100)
    if 'TC' in df_tc.columns and df_tc['TC'].max() > 10:
         max_tc = 100
         
    fig, ax = plt.subplots(figsize=(8, 6))

    def update(frame):
        ax.clear()
        date_val = quarters[frame]
        subset = df_tc[df_tc['date'] == date_val]
        
        # Pivot
        pivot_tc = subset.pivot(index='reporter_iso', columns='partner_iso', values='TC')
        # Reindex to ensure all RCEP countries exist
        pivot_tc = pivot_tc.reindex(index=RCEP_LIST, columns=RCEP_LIST).fillna(0)
        
        sns.heatmap(pivot_tc, cmap="YlOrRd", vmin=0, vmax=max_tc, ax=ax, cbar=False, 
                    annot=True, fmt=".1f", annot_kws={"size": 6})
        
        ax.set_title(f"Cumulative Tariff Relief Phase-in\nQuarter: {pd.to_datetime(date_val).strftime('%Y-Q%q')}", fontsize=12)
        ax.set_xlabel("Partner Country")
        ax.set_ylabel("Reporter Country")
        
        # Manually clear tick marks for clean look
        ax.tick_params(axis='both', which='both', length=0)
        
        return ax,
        
    ani = animation.FuncAnimation(fig, update, frames=len(quarters), interval=400, blit=False)
    
    out_path = OUTPUT_DIR / "tariff_relief_heatmap.mp4"
    try:
        ani.save(out_path, writer='ffmpeg', fps=2, dpi=200)
        logger.info(f"Saved Heatmap Animation to {out_path}")
    except Exception as e:
        logger.error(f"Failed to save MP4: {e}")
        out_path = OUTPUT_DIR / "tariff_relief_heatmap.gif"
        ani.save(out_path, writer='pillow', fps=2, dpi=120)
        logger.info(f"Saved Heatmap Animation to {out_path} (GIF fallback)")
        
    plt.close()


def generate_all_animations():
    df_w, df_tc, df_panel = load_data()
    create_network_animation(df_panel)
    create_tariff_relief_heatmap_animation(df_tc)
    logger.info("All dynamic animations generation complete.")

if __name__ == "__main__":
    generate_all_animations()
