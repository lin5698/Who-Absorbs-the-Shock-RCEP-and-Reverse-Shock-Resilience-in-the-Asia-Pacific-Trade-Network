"""
Fig: Reverse-shock resilience over time
Calculates and plots the time-varying aggregated Network Amplification Share and Half-life metrics
using a rolling Network VAR estimation.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent))
from research_network_tvp_var import (
    load_constructed_data, var_ols, moving_average_coefficients, 
    girf_one, network_amplification_share, half_life_approx
)
from config import RCEP_COUNTRIES
RCEP_LIST = list(RCEP_COUNTRIES.keys())
from research_nature_figures import set_nature_style, COLORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("research_output/nature_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR = Path("research_output")

# Weights (Pre RCEP 2021 average)
def get_trade_weights_2021(df_bilat):
    df_21 = df_bilat[df_bilat['year'] == 2021]
    w_mat = np.zeros((len(RCEP_LIST), len(RCEP_LIST)))
    
    if df_21.empty:
        return np.ones((len(RCEP_LIST), len(RCEP_LIST))) / (len(RCEP_LIST)*(len(RCEP_LIST)-1))
        
    for i, rep in enumerate(RCEP_LIST):
        for j, par in enumerate(RCEP_LIST):
            if i == j: continue
            mask = (df_21['reporter_iso'] == rep) & (df_21['partner_iso'] == par)
            if not df_21[mask].empty:
                w_mat[i, j] = df_21[mask]['total_exports_usd_k'].mean() if 'total_exports_usd_k' in df_21.columns else 1.0
                
    w_mat = w_mat / (w_mat.sum() + 1e-10)
    return w_mat

def compute_rolling_metrics():
    logger.info("Loading constructed data...")
    Y, dates, W_t, df_bilat, X_exog = load_constructed_data()
    T, N = Y.shape
    
    trade_weights = get_trade_weights_2021(df_bilat)
    
    dates_arr = list(dates) if hasattr(dates, "index") else list(dates)
    W_list = []
    for d in dates_arr:
        W = W_t.get(d, np.zeros((N, N)))
        if isinstance(W, pd.DataFrame):
            W = W.reindex(index=RCEP_LIST, columns=RCEP_LIST).fillna(0).values
        W_list.append(W)
        
    window_min = 40  # Rolling window size (40 quarters = 10 years)
    horizon = 12    # User requested 8/12 quarters
    p = 2
    
    aggregated_results = []
    pairwise_results = []
    
    logger.info("Computing rolling Network VAR metrics...")
    for t in range(window_min, T):
        date_t = dates_arr[t]
        
        # Extracted window
        Y_w = Y[t - window_min : t]
        W_w = W_list[t - window_min : t]
        X_w = X_exog[t - window_min : t] if X_exog is not None else None
        
        try:
            c, A_list, B_list, Pi, Sigma, _ = var_ols(Y_w, W_w, X_exog=X_w, p=p)
        except Exception as e:
            logger.warning(f"Estimation failed at {date_t}: {e}")
            continue

        B_list_zero = [np.zeros_like(B) for B in B_list]
        
        Psi_total = moving_average_coefficients(A_list, B_list, W_w, horizon, W_fixed=W_w[-1])
        Psi_direct = moving_average_coefficients(A_list, B_list_zero, W_w, horizon, W_fixed=W_w[-1])
        
        S_tot_mat = np.zeros((N, N))
        S_dir_mat = np.zeros((N, N))
        HL_mat = np.zeros((N, N))
        
        for j in range(N):
            irf_tot_j_all = [girf_one(Psi_total[h], Sigma, j) for h in range(horizon + 1)]
            irf_dir_j_all = [girf_one(Psi_direct[h], Sigma, j) for h in range(horizon + 1)]
            
            # Simple peak detection for HL
            peak_idx = 0 
            
            for i in range(N):
                if i == j: continue
                
                # S = sum of absolute responses up to horizon
                S_tot = sum(np.abs(irf_tot_j_all[h][i]) for h in range(horizon + 1))
                S_dir = sum(np.abs(irf_dir_j_all[h][i]) for h in range(horizon + 1))
                
                # Pairwise Half-life
                irf_pair = [irf_tot_j_all[h][i] for h in range(horizon + 1)]
                hl = half_life_approx_pairwise(irf_pair)
                
                S_tot_mat[i, j] = S_tot
                S_dir_mat[i, j] = S_dir
                HL_mat[i, j] = hl
                
                # Save pairwise
                a_ij = (S_tot - S_dir) / np.maximum(S_tot, 1e-10)
                pairwise_results.append({
                    'date': date_t,
                    'reporter_iso': RCEP_LIST[i],
                    'partner_iso': RCEP_LIST[j],
                    'A': a_ij,
                    'S_total': S_tot,
                    'HL': hl
                })
                
        # Aggregate for the main figure
        eq_weight = np.ones((N, N)) / (N * (N - 1))
        np.fill_diagonal(eq_weight, 0)
        
        # Better ratio aggregate: sum of diffs / sum of totals
        a_eq = np.sum((S_tot_mat - S_dir_mat) * eq_weight) / np.maximum(np.sum(S_tot_mat * eq_weight), 1e-10)
        a_trade = np.sum((S_tot_mat - S_dir_mat) * trade_weights) / np.maximum(np.sum(S_tot_mat * trade_weights), 1e-10)
        s_eq = np.sum(S_tot_mat * eq_weight)
        hl_eq = np.sum(HL_mat * eq_weight)
        
        aggregated_results.append({
            'date': date_t,
            'A_equal': a_eq,
            'A_trade': a_trade,
            'S_equal': s_eq,
            'HL_equal': hl_eq
        })
        
    df_res = pd.DataFrame(aggregated_results)
    df_res.to_csv(TEMP_DIR / "rolling_resilience_metrics.csv", index=False)
    
    df_pair = pd.DataFrame(pairwise_results)
    df_pair.to_csv(TEMP_DIR / "pairwise_rolling_metrics.csv", index=False)
    
    return df_res

def half_life_approx_pairwise(irf_pair):
    """Pairwise half-life: first index where |irf| <= 0.5 * max(|irf|)"""
    vals = np.abs(irf_pair)
    peak = np.max(vals)
    if peak <= 1e-12: return 0.0
    for h, v in enumerate(vals):
        if v <= 0.5 * peak:
            return float(h)
    return float(len(irf_pair) - 1)

def plot_reverse_shock_resilience(df_res):
    logger.info("Plotting Reverse-shock resilience over time...")
    df_res['date'] = pd.to_datetime(df_res['date'])
    
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5), sharex=True)
    
    rcep_date = pd.Timestamp('2022-01-01')
    
    # Panel A: Network Amplification Share
    ax1 = axes[0]
    ax1.plot(df_res['date'], df_res['A_equal'], label="Baseline (Equal-weight)", 
             color=COLORS['blue'], linewidth=1.5, zorder=3)
    ax1.plot(df_res['date'], df_res['A_trade'], label="Robustness (Trade-weighted)", 
             color=COLORS['gray'], linestyle='--', linewidth=1.2, zorder=2)
    
    ax1.axvline(rcep_date, color=COLORS['red'], linestyle=':', linewidth=1.2)
    ax1.text(rcep_date + pd.Timedelta(days=60), 0.85, '2022Q1', 
             transform=ax1.get_xaxis_transform(),
             color=COLORS['red'], fontsize=8, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.5))
    
    ax1.set_ylabel("Network Amp Share $A_{i\\leftarrow j,t}$")
    ax1.set_title("A. Aggregated Network Amplification Share", loc='left', fontweight='bold', fontsize=8)
    ax1.legend(frameon=False, loc='upper left')
    set_nature_style(ax1)
    
    # Panel B: Cumulative Response
    ax2 = axes[1]
    ax2.plot(df_res['date'], df_res['S_equal'], label="Cumulative Response $S^{Total}_{i\\leftarrow j,t}(H)$", 
             color=COLORS['green'], linewidth=1.5, zorder=3)
             
    ax2.axvline(rcep_date, color=COLORS['red'], linestyle=':', linewidth=1.2)
    
    ax2.set_ylabel("Cumulative Response")
    ax2.set_xlabel("Year")
    ax2.set_title("B. Aggregated Cumulative Response Strength", loc='left', fontweight='bold', fontsize=8)
    ax2.legend(frameon=False, loc='upper left')
    set_nature_style(ax2)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig_Reverse_Shock_Resilience.pdf")
    plt.savefig(OUTPUT_DIR / "Fig_Reverse_Shock_Resilience.png", dpi=300)
    plt.close()
    logger.info("Figure saved successfully.")

if __name__ == "__main__":
    df = compute_rolling_metrics()
    plot_reverse_shock_resilience(df)
