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

def estimate_metrics_rolling_custom(Y, W_list, dates, window_min=40, horizon=8):
    T, N = Y.shape
    results_list = []
    
    for t in range(window_min, T):
        date_t = dates[t]
        Y_w = Y[t - window_min : t]
        W_w = W_list[t - window_min : t]
        
        try:
            c, A_list, B_list, Pi, Sigma, _ = var_ols(Y_w, W_w, p=2)
            B_list_zero = [np.zeros_like(B) for B in B_list]
            
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
        except:
            continue
                    
    return pd.DataFrame(results_list)

def run_sensitivity_hm():
    logger.info("Loading Data...")
    df_macro, df_bilateral = load_quarterly_macro_and_bilateral()
    df_tc = build_tariff_relief_tc(df_bilateral)
    df_tc['date'] = pd.to_datetime(df_tc['date'])
    
    # Prep Y
    vax = chow_lin_quarterly_vax(df_macro)
    vax_p = vax.set_index(["date", "iso3"])["vax_q"].unstack("iso3")
    for c in vax_p.columns:
        vax_p[c] = quality_control_outliers(quality_control_missing(vax_p[c]))
    Y = np.log(vax_p + 1e-6).diff().dropna(how="all")
    dates_y = Y.index
    Y_vals = Y.reindex(columns=RCEP_LIST).fillna(0).values
    
    m_values = [32, 40, 48]
    h_values = [4, 8, 12, 16]
    
    results = []
    
    for m in m_values:
        # Optimization: Pre-calculate W_list for fixed m (baseline window=4)
        W_list_m = []
        for d in dates_y:
            W = build_trade_network_w(df_bilateral, d, window_quarters=4, mode='import')
            if isinstance(W, pd.DataFrame):
                W = W.reindex(index=RCEP_LIST, columns=RCEP_LIST).fillna(0).values
            W_list_m.append(W)
            
        for h in h_values:
            logger.info(f"Processing Sensitivity: m={m}, H={h}...")
            
            df_metrics = estimate_metrics_rolling_custom(Y_vals, W_list_m, dates_y, window_min=m, horizon=h)
            if df_metrics.empty: continue
            
            df_metrics['date'] = pd.to_datetime(df_metrics['date'])
            df_reg_data = pd.merge(df_metrics, df_tc, on=['date', 'reporter_iso', 'partner_iso'])
            df_reg_data['A'] = df_reg_data['A'].clip(0, 1)
            df_reg_data['TC_relief'] = df_reg_data['TC'].abs() * 100
            df_reg_data['pair'] = df_reg_data['reporter_iso'] + "_" + df_reg_data['partner_iso']
            
            df_reg = df_reg_data.set_index(['pair', 'date'])
            try:
                mod = PanelOLS.from_formula('A ~ TC_relief + EntityEffects + TimeEffects', data=df_reg)
                res = mod.fit(cov_type='clustered', cluster_entity=True)
                
                results.append({
                    'm': m,
                    'H': h,
                    'beta': res.params['TC_relief'],
                    'se': res.std_errors['TC_relief'],
                    'p_val': res.pvalues['TC_relief']
                })
            except Exception as e:
                logger.error(f"Reg failed for m={m}, H={h}: {e}")
                
    # Format into Table: Columns vary H, Rows vary m
    df_res = pd.DataFrame(results)
    
    # Create pivot tables for presentation
    pivot_beta = df_res.pivot(index='m', columns='H', values='beta')
    pivot_se = df_res.pivot(index='m', columns='H', values='se')
    
    # Combined string table
    def format_cell(m, h):
        b = pivot_beta.loc[m, h]
        s = pivot_se.loc[m, h]
        return f"{b:.4f}\n({s:.4f})"
    
    final_table = pivot_beta.copy().astype(str)
    for m in m_values:
        for h in h_values:
            final_table.loc[m, h] = format_cell(m, h)
            
    final_table.to_csv(OUTPUT_DIR / "Table_Sensitivity_HM.csv")
    print("\n--- Sensitivity Analysis for H and m ---")
    print(final_table)
    
    # Also save raw data
    df_res.to_csv(OUTPUT_DIR / "Table_Sensitivity_HM_Raw.csv", index=False)

if __name__ == "__main__":
    run_sensitivity_hm()
