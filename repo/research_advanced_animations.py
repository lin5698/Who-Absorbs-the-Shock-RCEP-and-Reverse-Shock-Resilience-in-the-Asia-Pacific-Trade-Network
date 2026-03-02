import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import seaborn as sns
import networkx as nx
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Directories ---
DATA_DIR = Path(__file__).parent / "research_output"
OUTPUT_DIR = DATA_DIR / "animations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Nature Inspired Basic Styling ---
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150
})

RCEP_COUNTRIES = ['CHN','JPN','KOR','AUS','NZL','IDN','MYS','PHL','SGP','THA','VNM','BRN','KHM','LAO','MMR']

def get_country_colors():
    """Consistent color map for racing bars"""
    base_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
                    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    # Repeat colors if more than 10 countries
    colors = base_palette * 2 
    return {country: colors[i] for i, country in enumerate(RCEP_COUNTRIES)}

def calculate_centrality(df_w):
    """Calculate PageRank/Eigenvector equivalent from W matrix panels"""
    logger.info("Calculating Network Centrality...")
    # df_w has rows: reporter (index 0), then columns for partners, then 'date'
    # Actually df_w format: index=reporter, cols=partners, 'date'
    
    records = []
    quarters = sorted(df_w['date'].unique())
    
    for q in quarters:
        sub = df_w[df_w['date'] == q]
        
        # Build adjacency matrix
        G = nx.DiGraph()
        for _, row in sub.iterrows():
            reporter = row.iloc[0] # assuming first column or index is reporter
            if reporter not in RCEP_COUNTRIES: continue
            for partner in RCEP_COUNTRIES:
                if partner in sub.columns:
                    val = row[partner]
                    if pd.notna(val) and val > 0:
                        G.add_edge(reporter, partner, weight=val)
        
        if len(G.nodes) > 0:
            try:
                # PageRank is stable for directed trade networks
                centrality = nx.pagerank(G, weight='weight')
                for node, score in centrality.items():
                    records.append({'date': q, 'country': node, 'score': score})
            except Exception as e:
                logger.warning(f"Centrality error for {q}: {e}")
                
    return pd.DataFrame(records)

def create_centrality_race(df_cent):
    if df_cent.empty:
        logger.warning("No centrality data to animate.")
        return
        
    logger.info("Generating Centrality Race Animation...")
    df_cent['date'] = pd.to_datetime(df_cent['date'])
    quarters = sorted(df_cent['date'].unique())
    country_colors = get_country_colors()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def update(frame):
        ax.clear()
        date_val = quarters[frame]
        
        sub = df_cent[df_cent['date'] == date_val].sort_values(by='score', ascending=True)
        # Keep top 12 for readability
        sub = sub.tail(12)
        
        colors = [country_colors.get(c, '#cccccc') for c in sub['country']]
        
        bars = ax.barh(sub['country'], sub['score'], color=colors, alpha=0.8)
        
        # Label values
        for i, (value, name) in enumerate(zip(sub['score'], sub['country'])):
            ax.text(value + 0.005, i, f'{value:.3f}', va='center', ha='left', fontsize=8)
            
        date_str = pd.to_datetime(date_val).strftime('%Y Q%q')
        ax.text(0.95, 0.2, date_str, transform=ax.transAxes, color='#777777', 
                size=36, ha='right', weight=800)
                
        ax.set_title("RCEP Network Centrality (PageRank)", fontsize=14, pad=15)
        ax.set_xlim(0, max(0.4, df_cent['score'].max() * 1.1))
        
        # Nature styling touches
        ax.tick_params(axis='both', which='both', length=0)
        ax.xaxis.grid(True, linestyle='--', alpha=0.3)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        return ax,
        
    ani = animation.FuncAnimation(fig, update, frames=len(quarters), interval=300, blit=False)
    
    out_path = OUTPUT_DIR / "network_centrality_race.mp4"
    try:
        ani.save(out_path, writer='ffmpeg', fps=4, dpi=150)
        logger.info(f"Saved to {out_path}")
    except Exception:
        out_path = OUTPUT_DIR / "network_centrality_race.gif"
        ani.save(out_path, writer='pillow', fps=4, dpi=120)
        logger.info(f"Saved to {out_path} (GIF fallback)")
        
    plt.close()

def main():
    w_path = DATA_DIR / "trade_network_W_quarterly.csv"
    if not w_path.exists():
        logger.error("Weight matrix not found.")
        return
        
    # Read W matrix, assuming first col is reporter name
    # The actual W matrix format: '', 'AUS', 'BRN'... 'date'
    df_w = pd.read_csv(w_path)
    if 'Unnamed: 0' in df_w.columns:
        df_w = df_w.rename(columns={'Unnamed: 0': 'reporter'})
        
    df_cent = calculate_centrality(df_w)
    create_centrality_race(df_cent)

if __name__ == "__main__":
    main()
