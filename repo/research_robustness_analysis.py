import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
from linearmodels.panel import PanelOLS

sys.path.insert(0, str(Path(__file__).parent))
from research_data_construction import (
    load_quarterly_macro_and_bilateral, 
    chow_lin_quarterly_vax, 
    build_trade_network_w,
    quality_control_missing,
    quality_control_outliers,
    RCEP_LIST
)
from research_network_tvp_var import (
    var_ols, moving_average_coefficients, girf_one, 
    network_amplification_share, half_life_approx
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("research_output/nature_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR = Path("research_output")

# --- Helper: Estimation Engine ---
def estimate_metrics(Y, W_list, dates, window_min=40, horizon=12, p=2):
    T, N = Y.shape
    pairwise_results = []
    
    for t in range(window_min, T):
        date_t = dates[t]
        Y_w = Y[t - window_min : t]
        W_w = W_list[t - window_min : t]
        
        try:
            c, A_list, B_list, Pi, Sigma, _ = var_ols(Y_w, W_w, p=p)
        except:
            continue

        B_list_zero = [np.zeros_like(B) for B in B_list]
        Psi_total = moving_average_coefficients(A_list, B_list, W_w, horizon, W_fixed=W_w[-1])
        Psi_direct = moving_average_coefficients(A_list, B_list_zero, W_w, horizon, W_fixed=W_w[-1])
        
        for j in range(N):
            irf_tot_j = [girf_one(Psi_total[h], Sigma, j) for h in range(horizon + 1)]
            irf_dir_j = [girf_one(Psi_direct[h], Sigma, j) for h in range(horizon + 1)]
            for i in range(N):
                if i == j: continue
                s_tot = sum(np.abs(irf_tot_j[h][i]) for h in range(horizon + 1))
                s_dir = sum(np.abs(irf_dir_j[h][i]) for h in range(horizon + 1))
                a_ij = (s_tot - s_dir) / np.maximum(s_tot, 1e-10)
                pairwise_results.append({
                    'date': date_t,
                    'reporter_iso': RCEP_LIST[i],
                    'partner_iso': RCEP_LIST[j],
                    'A': a_ij
                })
    return pd.DataFrame(pairwise_results)

# --- Task 1: Alternative Networks ---
def run_network_robustness(df_macro, df_bilateral, df_tc):
    logger.info("Running Network Robustness...")
    dates = pd.to_datetime(df_macro["date"].unique())
    dates = pd.Series(dates).sort_values().values
    # Baseline VAX
    vax_base = chow_lin_quarterly_vax(df_macro)
    vax_pivot = vax_base.set_index(["date", "iso3"])["vax_q"].unstack("iso3")
    for c in vax_pivot.columns:
        vax_pivot[c] = quality_control_outliers(quality_control_missing(vax_pivot[c]))
    Y = np.log(vax_pivot + 1e-6).diff().dropna(how="all")
    dates_y = Y.index
    Y_vals = Y.reindex(columns=RCEP_LIST).fillna(0).values

    def get_w_list(mode, window):
        return [build_trade_network_w(df_bilateral, d, window_quarters=window, mode=mode) for d in dates_y]

    configs = [
        ('import', 4, 'Import (Baseline)'),
        ('export', 4, 'Export'),
        ('symmetric', 4, 'Symmetric'),
        ('import', 8, 'Window=8'),
    ]
    
    table_rows = []
    for mode, win, label in configs:
        logger.info(f"  Estimating {label}...")
        W_list = get_w_list(mode, win)
        df_pair = estimate_metrics(Y_vals, W_list, dates_y)
        
        # Regression
        df_reg = pd.merge(df_pair, df_tc, on=['date', 'reporter_iso', 'partner_iso'])
        df_reg['A'] = df_reg['A'].clip(0, 1)
        df_reg['TC_relief'] = df_reg['TC'].abs() * 100
        # Post-RCEP Focus as established in previous Turn
        df_sub = df_reg[df_reg['date'] >= '2021-01-01'].copy()
        df_sub['pair'] = df_sub['reporter_iso'] + "_" + df_sub['partner_iso']
        df_sub = df_sub.set_index(['pair', 'date'])
        
        try:
            mod = PanelOLS.from_formula('A ~ TC_relief + EntityEffects + TimeEffects', data=df_sub)
            res = mod.fit(cov_type='clustered', cluster_entity=True)
            table_rows.append({
                'Case': label,
                'Coef': res.params['TC_relief'],
                'p-val': res.pvalues['TC_relief'],
                'N': res.nobs
            })
        except:
            continue
            
    pd.DataFrame(table_rows).to_csv(OUTPUT_DIR / "Table_Robustness_Network.csv", index=False)

# --- Task 2: VAX Quarterlyization ---
def run_vax_robustness(df_macro, df_bilateral, df_tc):
    logger.info("Running VAX Robustness...")
    dates = pd.to_datetime(df_macro["date"].unique())
    dates = pd.Series(dates).sort_values().values
    
    # baseline W_list
    vax_base = chow_lin_quarterly_vax(df_macro)
    vax_pivot = vax_base.set_index(["date", "iso3"])["vax_q"].unstack("iso3")
    dates_y = vax_pivot.dropna(how='all').index
    W_list = [build_trade_network_w(df_bilateral, d, window_quarters=4, mode='import') for d in dates_y]

    configs = [
        (['total_exports_usd_k'], 'Exports indicator (Baseline)'),
        (['GDP-VOL'], 'GDP volume indicator'),
        (['total_imports_usd_k'], 'Imports indicator'),
    ]
    
    table_rows = []
    for indicators, label in configs:
        logger.info(f"  Estimating with {label}...")
        vax_alt = chow_lin_quarterly_vax(df_macro, indicator_cols=indicators)
        vax_pivot_alt = vax_alt.set_index(["date", "iso3"])["vax_q"].unstack("iso3")
        for c in vax_pivot_alt.columns:
            vax_pivot_alt[c] = quality_control_outliers(quality_control_missing(vax_pivot_alt[c]))
        
        Y = np.log(vax_pivot_alt + 1e-6).diff().dropna(how="all")
        Y_vals = Y.reindex(index=dates_y, columns=RCEP_LIST).fillna(0).values
        
        df_pair = estimate_metrics(Y_vals, W_list, dates_y)
        
        # Regression
        df_reg = pd.merge(df_pair, df_tc, on=['date', 'reporter_iso', 'partner_iso'])
        df_reg['A'] = df_reg['A'].clip(0, 1)
        df_reg['TC_relief'] = df_reg['TC'].abs() * 100
        df_sub = df_reg[df_reg['date'] >= '2021-01-01'].copy()
        df_sub['pair'] = df_sub['reporter_iso'] + "_" + df_sub['partner_iso']
        df_sub = df_sub.set_index(['pair', 'date'])
        
        try:
            mod = PanelOLS.from_formula('A ~ TC_relief + EntityEffects + TimeEffects', data=df_sub)
            res = mod.fit(cov_type='clustered', cluster_entity=True)
            table_rows.append({
                'Indicator': label,
                'Coef': res.params['TC_relief'],
                'p-val': res.pvalues['TC_relief'],
                'N': res.nobs
            })
        except:
            continue
            
    pd.DataFrame(table_rows).to_csv(OUTPUT_DIR / "Table_Robustness_VAX.csv", index=False)

# --- Task 3: Placebo ---
def run_placebo_tests(df_tc):
    logger.info("Running Placebo Tests...")
    # Load baseline A
    df_pair = pd.read_csv(TEMP_DIR / "pairwise_rolling_metrics.csv")
    df_pair['date'] = pd.to_datetime(df_pair['date'])
    
    shifts = [0, 4, 8, -4, -8]
    table_rows = []
    
    for s in shifts:
        label = f"Shift {s}q" if s != 0 else "Baseline"
        df_tc_s = df_tc.copy()
        df_tc_s['date'] = pd.to_datetime(df_tc_s['date']) + pd.DateOffset(months=3*s)
        
        df_reg = pd.merge(df_pair, df_tc_s, on=['date', 'reporter_iso', 'partner_iso'])
        df_reg['A'] = df_reg['A'].clip(0, 1)
        df_reg['TC_relief'] = df_reg['TC'].abs() * 100
        df_sub = df_reg[df_reg['date'] >= '2021-01-01'].copy()
        df_sub['pair'] = df_sub['reporter_iso'] + "_" + df_sub['partner_iso']
        df_sub = df_sub.set_index(['pair', 'date'])
        
        try:
            mod = PanelOLS.from_formula('A ~ TC_relief + EntityEffects + TimeEffects', data=df_sub)
            res = mod.fit(cov_type='clustered', cluster_entity=True)
            table_rows.append({
                'Placebo': label,
                'Coef': res.params['TC_relief'],
                'p-val': res.pvalues['TC_relief'],
                'N': res.nobs
            })
        except:
            continue
            
    pd.DataFrame(table_rows).to_csv(OUTPUT_DIR / "Table_Placebo_Timing.csv", index=False)

if __name__ == "__main__":
    df_macro, df_bilateral = load_quarterly_macro_and_bilateral()
    from research_data_construction import build_tariff_relief_tc
    df_tc = build_tariff_relief_tc(df_bilateral)
    
    run_network_robustness(df_macro, df_bilateral, df_tc)
    run_vax_robustness(df_macro, df_bilateral, df_tc)
    run_placebo_tests(df_tc)
    logger.info("Robustness analysis complete.")
