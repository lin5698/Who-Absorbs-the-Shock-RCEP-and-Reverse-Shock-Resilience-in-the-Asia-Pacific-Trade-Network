import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from research_network_tvp_var import (
    load_constructed_data, var_ols, moving_average_coefficients, 
    girf_one, network_amplification_share, half_life_approx
)
from config import RCEP_COUNTRIES
RCEP_LIST = list(RCEP_COUNTRIES.keys())

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TEMP_DIR = Path("research_output")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

def bootstrap_rolling_metrics(n_boot=100, window_min=40, horizon=12, p=2):
    logger.info("Loading constructed data...")
    Y, dates, W_t, df_bilat, X_exog = load_constructed_data()
    T, N = Y.shape
    dates_arr = list(dates) if hasattr(dates, "index") else list(dates)
    
    W_list = []
    for d in dates_arr:
        W = W_t.get(d, np.zeros((N, N)))
        if isinstance(W, pd.DataFrame):
            W = W.reindex(index=RCEP_LIST, columns=RCEP_LIST).fillna(0).values
        W_list.append(W)
        
    bootstrap_results = []
    
    # We want to track:
    # 1. Aggregated A and HL over time
    # 2. GIRFs for CHN-partner at specific dates (e.g. 2018Q4 and 2022Q4)
    target_dates = ['2018-12-31', '2022-12-31']
    target_dates_ts = [pd.Timestamp(d) for d in target_dates]
    
    girf_boot_data = {d: [] for d in target_dates}

    logger.info(f"Starting rolling bootstrap (n_boot={n_boot})...")
    for t in range(window_min, T):
        date_t = dates_arr[t]
        logger.info(f"Processing date: {date_t}")
        
        # Original estimation
        Y_w = Y[t - window_min : t]
        W_w = W_list[t - window_min : t]
        X_w = X_exog[t - window_min : t] if X_exog is not None else None
        
        try:
            c, A_list, B_list, Pi, Sigma, residuals = var_ols(Y_w, W_w, X_exog=X_w, p=p)
        except Exception as e:
            logger.warning(f"Initial estimation failed at {date_t}: {e}")
            continue

        # Initial point estimates
        def calc_metrics(A_l, B_l, S, W_curr):
            B_zero = [np.zeros_like(B) for B in B_l]
            Psi_tot = moving_average_coefficients(A_l, B_l, [W_curr]* (horizon+p), horizon, W_fixed=W_curr)
            Psi_dir = moving_average_coefficients(A_l, B_zero, [W_curr]* (horizon+p), horizon, W_fixed=W_curr)
            
            S_tot_mat = np.zeros((N, N))
            S_dir_mat = np.zeros((N, N))
            HL_mat = np.zeros((N, N))
            
            # Focused on CHN (index 0 usually if RCEP_LIST starts with CHN)
            chn_idx = RCEP_LIST.index('CHN') if 'CHN' in RCEP_LIST else 0
            
            for j in range(N):
                irf_tot_j = [girf_one(Psi_tot[h], S, j) for h in range(horizon + 1)]
                irf_dir_j = [girf_one(Psi_dir[h], S, j) for h in range(horizon + 1)]
                for i in range(N):
                    if i == j: continue
                    S_tot_mat[i, j] = sum(np.abs(irf_tot_j[h][i]) for h in range(horizon + 1))
                    S_dir_mat[i, j] = sum(np.abs(irf_dir_j[h][i]) for h in range(horizon + 1))
                    # Half-life for pairwise
                    irf_pair = [irf_tot_j[h][i] for h in range(horizon + 1)]
                    HL_mat[i, j] = half_life_approx_pairwise(irf_pair)
            
            # Aggregate
            eq_weight = np.ones((N, N))
            np.fill_diagonal(eq_weight, 0)
            a_agg = np.sum(S_tot_mat - S_dir_mat) / np.maximum(np.sum(S_tot_mat), 1e-10)
            hl_agg = np.sum(HL_mat * eq_weight) / (N * (N - 1))
            
            return a_agg, hl_agg, Psi_tot, Psi_dir

        # Residual Bootstrap
        boot_a = []
        boot_hl = []
        
        # For specific target dates, we save the full IRF vector
        is_target = any(abs((date_t - ts).days) < 45 for ts in target_dates_ts)
        target_str = None
        if is_target:
            target_str = next(d for d, ts in zip(target_dates, target_dates_ts) if abs((date_t - ts).days) < 45)

        for b in range(n_boot):
            # 1. Resample residuals
            res_idx = np.random.choice(len(residuals), size=len(residuals), replace=True)
            u_star = residuals[res_idx]
            
            # 2. Reconstruct Y_star
            Y_star = np.zeros_like(Y_w)
            Y_star[:p] = Y_w[:p] # Start with same lags
            for s in range(p, len(Y_w)):
                y_hat = c.copy()
                for l in range(1, p + 1):
                    y_hat += A_list[l-1] @ Y_star[s-l]
                    Wy = W_w[s-l] @ Y_star[s-l]
                    y_hat += B_list[l-1] @ Wy
                if X_w is not None:
                    y_hat += Pi @ X_w[s]
                Y_star[s] = y_hat + u_star[s-p]
            
            # 3. Re-estimate
            try:
                c_b, A_b, B_b, Pi_b, Sigma_b, _ = var_ols(Y_star, W_w, X_exog=X_w, p=p)
                a_b, hl_b, Psi_tot_b, Psi_dir_b = calc_metrics(A_b, B_b, Sigma_b, W_w[-1])
                boot_a.append(a_b)
                boot_hl.append(hl_b)
                
                if target_str:
                    # Save CHN <- JPN (index 1 usually) GIRF
                    chn_idx = RCEP_LIST.index('CHN')
                    jpn_idx = RCEP_LIST.index('JPN')
                    irf_tot_b = [girf_one(Psi_tot_b[h], Sigma_b, jpn_idx)[chn_idx] for h in range(horizon + 1)]
                    irf_dir_b = [girf_one(Psi_dir_b[h], Sigma_b, jpn_idx)[chn_idx] for h in range(horizon + 1)]
                    girf_boot_data[target_str].append({
                        'b': b,
                        'total': irf_tot_b,
                        'direct': irf_dir_b
                    })
            except:
                continue
        
        # Percentiles
        if boot_a:
            bootstrap_results.append({
                'date': date_t,
                'A_mean': np.mean(boot_a),
                'A_p025': np.percentile(boot_a, 2.5),
                'A_p16': np.percentile(boot_a, 16),
                'A_p50': np.percentile(boot_a, 50),
                'A_p84': np.percentile(boot_a, 84),
                'A_p975': np.percentile(boot_a, 97.5),
                'HL_mean': np.mean(boot_hl),
                'HL_p025': np.percentile(boot_hl, 2.5),
                'HL_p16': np.percentile(boot_hl, 16),
                'HL_p50': np.percentile(boot_hl, 50),
                'HL_p84': np.percentile(boot_hl, 84),
                'HL_p975': np.percentile(boot_hl, 97.5)
            })

    # Save outputs
    df_boot = pd.DataFrame(bootstrap_results)
    df_boot.to_csv(TEMP_DIR / "rolling_resilience_bootstrapped.csv", index=False)
    
    # Save GIRF boot data to JSON or pickle for plotting
    import json
    with open(TEMP_DIR / "girf_bootstrap_data.json", 'w') as f:
        json.dump(girf_boot_data, f)
    
    logger.info("Bootstrap complete.")

def half_life_approx_pairwise(irf_pair):
    vals = np.abs(irf_pair)
    peak = np.max(vals)
    if peak <= 1e-12: return 0.0
    for h, v in enumerate(vals):
        if v <= 0.5 * peak:
            return float(h)
    return float(len(irf_pair) - 1)

if __name__ == "__main__":
    # Use n_boot = 50 for speed during testing, or 100-200 for final
    n_sum = 50 
    bootstrap_rolling_metrics(n_boot=n_sum)
