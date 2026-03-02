import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
from linearmodels.panel import PanelOLS

# Add current path to import local modules
sys.path.insert(0, str(Path(__file__).parent))
from research_data_construction import (
    load_quarterly_macro_and_bilateral, 
    chow_lin_quarterly_vax, 
    build_trade_network_w,
    build_tariff_relief_tc,
    quality_control_missing,
    quality_control_outliers,
    RCEP_LIST
)
from research_network_tvp_var import (
    var_ols, moving_average_coefficients, girf_one, 
    network_amplification_share
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("research_output/nature_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR = Path("research_output")

def estimate_metrics_rolling_single_h(Y, W_list, dates, window_min=40, horizon=8):
    """
    Rolling Estimation for A share for a specific horizon.
    """
    T, N = Y.shape
    results_list = []
    
    logger.info(f"Starting rolling estimation (T={T}, window={window_min})...")
    for t in range(window_min, T):
        date_t = dates[t]
        Y_w = Y[t - window_min : t]
        W_w = W_list[t - window_min : t]
        
        try:
            c, A_list, B_list, Pi, Sigma, _ = var_ols(Y_w, W_w, p=2)
            B_list_zero = [np.zeros_like(B) for B in B_list]
        except:
            continue

        Psi_total = moving_average_coefficients(A_list, B_list, W_w, horizon, W_fixed=W_w[-1])
        Psi_direct = moving_average_coefficients(A_list, B_list_zero, W_w, horizon, W_fixed=W_w[-1])
        
        for j in range(N):
            irf_tot_j = [girf_one(Psi_total[step], Sigma, j) for step in range(horizon + 1)]
            irf_dir_j = [girf_one(Psi_direct[step], Sigma, j) for step in range(horizon + 1)]
            
            for i in range(N):
                if i == j: continue
                s_tot = sum(np.abs(irf_tot_j[step][i]) for step in range(horizon + 1))
                s_dir = sum(np.abs(irf_dir_j[step][i]) for step in range(horizon + 1))
                a_ij = network_amplification_share(s_tot, s_dir)
                
                results_list.append({
                    'date': date_t,
                    'reporter_iso': RCEP_LIST[i],
                    'partner_iso': RCEP_LIST[j],
                    'A': a_ij
                })
                    
    return pd.DataFrame(results_list)

def run_network_robustness():
    logger.info("Loading Data...")
    df_macro, df_bilateral = load_quarterly_macro_and_bilateral()
    df_tc = build_tariff_relief_tc(df_bilateral)
    
    # Prep Y
    vax = chow_lin_quarterly_vax(df_macro)
    vax_p = vax.set_index(["date", "iso3"])["vax_q"].unstack("iso3")
    for c in vax_p.columns:
        vax_p[c] = quality_control_outliers(quality_control_missing(vax_p[c]))
    Y = np.log(vax_p + 1e-6).diff().dropna(how="all")
    dates_y = Y.index
    Y_vals = Y.reindex(columns=RCEP_LIST).fillna(0).values
    
    # Pre-RCEP fixed weight matrix (e.g. 2021Q4)
    fixed_date = pd.Timestamp('2021-12-31')
    W_fixed_pre = build_trade_network_w(df_bilateral, fixed_date, window_quarters=4, mode='import')
    if isinstance(W_fixed_pre, pd.DataFrame):
        W_fixed_pre_val = W_fixed_pre.reindex(index=RCEP_LIST, columns=RCEP_LIST).fillna(0).values
    else:
        W_fixed_pre_val = W_fixed_pre
    
    configs = [
        ('import', 4, 'Col 1: Import-based (Baseline)'),
        ('export', 4, 'Col 2: Export-based'),
        ('symmetric', 4, 'Col 3: Symmetric'),
        ('import', 8, 'Col 4: Window m=8'),
        ('fixed', 4, 'Col 5: Fixed Pre-RCEP (2021Q4)')
    ]
    
    summary_results = []
    
    for mode, win, label in configs:
        logger.info(f"Processing variant: {label}")
        
        # Build W_list for this config
        W_list = []
        for d in dates_y:
            if mode == 'fixed':
                W_list.append(W_fixed_pre_val)
            else:
                W = build_trade_network_w(df_bilateral, d, window_quarters=win, mode=mode)
                if isinstance(W, pd.DataFrame):
                    W = W.reindex(index=RCEP_LIST, columns=RCEP_LIST).fillna(0).values
                W_list.append(W)
            
        # Estimate A
        df_metrics = estimate_metrics_rolling_single_h(Y_vals, W_list, dates_y, horizon=8)
        
        # Merge with TC
        df_tc['date'] = pd.to_datetime(df_tc['date'])
        df_metrics['date'] = pd.to_datetime(df_metrics['date'])
        df_reg_data = pd.merge(df_metrics, df_tc, on=['date', 'reporter_iso', 'partner_iso'])
        
        # Prep
        df_reg_data['A'] = df_reg_data['A'].clip(0, 1)
        df_reg_data['TC_relief'] = df_reg_data['TC'].abs() * 100
        df_reg_data['pair'] = df_reg_data['reporter_iso'] + "_" + df_reg_data['partner_iso']
        
        # PanelOLS
        df_reg = df_reg_data.set_index(['pair', 'date'])
        try:
            mod = PanelOLS.from_formula('A ~ TC_relief + EntityEffects + TimeEffects', data=df_reg)
            res = mod.fit(cov_type='clustered', cluster_entity=True)
            
            summary_results.append({
                'Variant': label,
                'Coefficient': res.params['TC_relief'],
                'Std.Err': res.std_errors['TC_relief'],
                'p-value': res.pvalues['TC_relief'],
                'N': res.nobs,
                'R-squared': res.rsquared
            })
            logger.info(f"Done: {label}")
        except Exception as e:
            logger.error(f"Failed: {label} - {e}")
            
    res_df = pd.DataFrame(summary_results)
    res_df.to_csv(OUTPUT_DIR / "Table_Network_Robustness_Final.csv", index=False)
    print("\n--- Network Robustness Table ---")
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    run_network_robustness()
