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

def compute_resilience_comparison():
    logger.info("Loading constructed data...")
    Y, dates, W_t, df_bilat, X_exog = load_constructed_data()
    T, N = Y.shape
    
    dates_arr = list(dates) if hasattr(dates, "index") else list(dates)
    
    # 1. Time-varying W_t
    W_tv = []
    for d in dates_arr:
        W = W_t.get(d, np.zeros((N, N)))
        if isinstance(W, pd.DataFrame):
            W = W.reindex(index=RCEP_LIST, columns=RCEP_LIST).fillna(0).values
        W_tv.append(W)
        
    # 2. Fixed W_pre (2021Q4)
    try:
        idx_2021Q4 = dates_arr.index(pd.Timestamp('2021-12-31'))
        W_pre = W_tv[idx_2021Q4]
    except:
        # Fallback to the last available before 2022
        w_idx = [i for i, d in enumerate(dates_arr) if pd.Timestamp(d).year < 2022][-1]
        W_pre = W_tv[w_idx]
        
    W_fixed = [W_pre] * T
        
    window_min = 40  # Rolling window size
    horizon = 12    # Requested horizon
    p = 2
    
    aggregated_results = []
    
    logger.info("Computing rolling Network VAR metrics for both TV and Fixed networks...")
    for t in range(window_min, T):
        date_t = dates_arr[t]
        
        # Extracted window data
        Y_w = Y[t - window_min : t]
        X_w = X_exog[t - window_min : t] if X_exog is not None else None
        
        # 1. TV Calculation
        W_tv_w = W_tv[t - window_min : t]
        try:
            c, A_list, B_list, Pi, Sigma, _ = var_ols(Y_w, W_tv_w, X_exog=X_w, p=p)
            B_list_zero = [np.zeros_like(B) for B in B_list]
            Psi_total_tv = moving_average_coefficients(A_list, B_list, W_tv_w, horizon, W_fixed=W_tv_w[-1])
            Psi_direct_tv = moving_average_coefficients(A_list, B_list_zero, W_tv_w, horizon, W_fixed=W_tv_w[-1])
            
            S_tot_mat_tv = np.zeros((N, N))
            S_dir_mat_tv = np.zeros((N, N))
            for j in range(N):
                irf_tot_j = [girf_one(Psi_total_tv[h], Sigma, j) for h in range(horizon + 1)]
                irf_dir_j = [girf_one(Psi_direct_tv[h], Sigma, j) for h in range(horizon + 1)]
                for i in range(N):
                    if i == j: continue
                    S_tot_mat_tv[i, j] = sum(np.abs(irf_tot_j[h][i]) for h in range(horizon + 1))
                    S_dir_mat_tv[i, j] = sum(np.abs(irf_dir_j[h][i]) for h in range(horizon + 1))
            
            a_tv = np.sum(S_tot_mat_tv - S_dir_mat_tv) / np.maximum(np.sum(S_tot_mat_tv), 1e-10)
        except Exception as e:
            logger.warning(f"TV estimation failed at {date_t}: {e}")
            a_tv = np.nan
            
        # 2. Fixed Calculation
        W_fix_w = W_fixed[t - window_min : t]
        try:
            c_f, A_list_f, B_list_f, Pi_f, Sigma_f, _ = var_ols(Y_w, W_fix_w, X_exog=X_w, p=p)
            B_list_zero_f = [np.zeros_like(B) for B in B_list_f]
            Psi_total_fix = moving_average_coefficients(A_list_f, B_list_f, W_fix_w, horizon, W_fixed=W_pre)
            Psi_direct_fix = moving_average_coefficients(A_list_f, B_list_zero_f, W_fix_w, horizon, W_fixed=W_pre)
            
            S_tot_mat_fix = np.zeros((N, N))
            S_dir_mat_fix = np.zeros((N, N))
            for j in range(N):
                irf_tot_j = [girf_one(Psi_total_fix[h], Sigma_f, j) for h in range(horizon + 1)]
                irf_dir_j = [girf_one(Psi_direct_fix[h], Sigma_f, j) for h in range(horizon + 1)]
                for i in range(N):
                    if i == j: continue
                    S_tot_mat_fix[i, j] = sum(np.abs(irf_tot_j[h][i]) for h in range(horizon + 1))
                    S_dir_mat_fix[i, j] = sum(np.abs(irf_dir_j[h][i]) for h in range(horizon + 1))
            
            a_fix = np.sum(S_tot_mat_fix - S_dir_mat_fix) / np.maximum(np.sum(S_tot_mat_fix), 1e-10)
        except Exception as e:
            logger.warning(f"Fixed estimation failed at {date_t}: {e}")
            a_fix = np.nan
            
        aggregated_results.append({
            'date': date_t,
            'A_tv': a_tv,
            'A_fixed': a_fix
        })
        
    df_res = pd.DataFrame(aggregated_results)
    df_res.to_csv(TEMP_DIR / "resilience_tv_vs_fixed.csv", index=False)
    
    return df_res

def plot_resilience_comparison(df_res):
    logger.info("Plotting Resilience Trends Comparison...")
    df_res['date'] = pd.to_datetime(df_res['date'])
    
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    rcep_date = pd.Timestamp('2022-01-01')
    
    ax.plot(df_res['date'], df_res['A_tv'], label="Time-Varying Network ($W_t$)", 
             color=COLORS['blue'], linewidth=1.5, zorder=3)
    ax.plot(df_res['date'], df_res['A_fixed'], label="Fixed Pre-RCEP Network ($W^{pre}$)", 
             color=COLORS['red'], linestyle='--', linewidth=1.5, zorder=2)
    
    ax.axvline(rcep_date, color='black', linestyle=':', linewidth=1.2)
    ax.text(rcep_date + pd.Timedelta(days=60), ax.get_ylim()[1]*0.8, 'RCEP Implementation', 
             transform=ax.get_xaxis_transform(),
             color='black', fontsize=8, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.5))
    
    ax.set_ylabel("Aggregated Network Amplification Share $A_t(12)$")
    ax.set_xlabel("Year")
    ax.set_title("Resilience Trends: Fixed-Weight vs Time-Varying Networks", loc='left', fontweight='bold', fontsize=8)
    ax.legend(frameon=False, loc='upper left')
    set_nature_style(ax)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig_Resilience_Trends_Fixed_vs_TV.pdf")
    plt.savefig(OUTPUT_DIR / "Fig_Resilience_Trends_Fixed_vs_TV.png", dpi=300)
    plt.close()
    logger.info("Figure saved successfully.")

if __name__ == "__main__":
    df = compute_resilience_comparison()
    plot_resilience_comparison(df)
