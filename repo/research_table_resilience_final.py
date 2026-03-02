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

def compute_average_pre_weight(df_bilateral):
    """Compute average trade network W for pre-RCEP period (2016-2019)."""
    pre_dates = pd.date_range("2016-01-01", "2019-12-31", freq="QS")
    w_list = []
    for d in pre_dates:
        W = build_trade_network_w(df_bilateral, d, window_quarters=4, mode='import')
        if isinstance(W, pd.DataFrame):
             W = W.reindex(index=RCEP_LIST, columns=RCEP_LIST).fillna(0).values
        w_list.append(W)
    return np.mean(w_list, axis=0)

def estimate_metrics_rolling(Y, W_list, dates, window_min=40, horizons=[8, 12], W_pre=None):
    """
    Rolling Estimation for A share. 
    Returns a dataframe with A for each horizon and variant.
    """
    T, N = Y.shape
    results_list = []
    
    # Pre-RCEP static W matrix for variant
    W_fixed_pre = W_pre if W_pre is not None else None
    
    logger.info(f"Starting rolling estimation (T={T}, window={window_min})...")
    for t in range(window_min, T):
        date_t = dates[t]
        Y_w = Y[t - window_min : t]
        W_w = W_list[t - window_min : t]
        
        try:
            # Estimate Network VAR on this window
            c, A_list, B_list, Pi, Sigma, _ = var_ols(Y_w, W_w, p=2)
            B_list_zero = [np.zeros_like(B) for B in B_list]
        except Exception as e:
            continue

        res_t = {'date': date_t}
        
        # Variants: H=8, H=12 (Time-varying W)
        for h in horizons:
            # Psi_total using current W sequence
            Psi_total = moving_average_coefficients(A_list, B_list, W_w, h, W_fixed=W_w[-1])
            Psi_direct = moving_average_coefficients(A_list, B_list_zero, W_w, h, W_fixed=W_w[-1])
            
            for j in range(N):
                # GIRF from node j
                irf_tot_j = [girf_one(Psi_total[step], Sigma, j) for step in range(h + 1)]
                irf_dir_j = [girf_one(Psi_direct[step], Sigma, j) for step in range(h + 1)]
                
                for i in range(N):
                    if i == j: continue
                    s_tot = sum(np.abs(irf_tot_j[step][i]) for step in range(h + 1))
                    s_dir = sum(np.abs(irf_dir_j[step][i]) for step in range(h + 1))
                    a_ij = network_amplification_share(s_tot, s_dir)
                    
                    results_list.append({
                        'date': date_t,
                        'reporter_iso': RCEP_LIST[i],
                        'partner_iso': RCEP_LIST[j],
                        'H': h,
                        'W_type': 'Time-Varying',
                        'A': a_ij
                    })

        # Variant: H=8, Pre-RCEP Fixed W
        if W_fixed_pre is not None:
            Psi_total_p = moving_average_coefficients(A_list, B_list, [W_fixed_pre]*window_min, 8, W_fixed=W_fixed_pre)
            Psi_direct_p = moving_average_coefficients(A_list, B_list_zero, [W_fixed_pre]*window_min, 8, W_fixed=W_fixed_pre)
            for j in range(N):
                irf_tot_j = [girf_one(Psi_total_p[step], Sigma, j) for step in range(9)]
                irf_dir_j = [girf_one(Psi_direct_p[step], Sigma, j) for step in range(9)]
                for i in range(N):
                    if i == j: continue
                    s_tot = sum(np.abs(irf_tot_j[step][i]) for step in range(9))
                    s_dir = sum(np.abs(irf_dir_j[step][i]) for step in range(9))
                    a_ij = network_amplification_share(s_tot, s_dir)
                    results_list.append({
                        'date': date_t,
                        'reporter_iso': RCEP_LIST[i],
                        'partner_iso': RCEP_LIST[j],
                        'H': 8,
                        'W_type': 'Fixed-Pre',
                        'A': a_ij
                    })
                    
    return pd.DataFrame(results_list)

def run_all_regressions():
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
    
    # Compute W_pre
    W_pre = compute_average_pre_weight(df_bilateral)
    
    # Prep W_list
    W_list = []
    for d in dates_y:
        W = build_trade_network_w(df_bilateral, d, window_quarters=4, mode='import')
        if isinstance(W, pd.DataFrame):
            W = W.reindex(index=RCEP_LIST, columns=RCEP_LIST).fillna(0).values
        W_list.append(W)
        
    # Estimation
    df_metrics = estimate_metrics_rolling(Y_vals, W_list, dates_y, horizons=[8, 12], W_pre=W_pre)
    
    # Merge with TC
    df_tc['date'] = pd.to_datetime(df_tc['date'])
    df_metrics['date'] = pd.to_datetime(df_metrics['date'])
    df_all = pd.merge(df_metrics, df_tc, on=['date', 'reporter_iso', 'partner_iso'])
    
    # Prep for regression
    df_all['A'] = df_all['A'].clip(0, 1)
    df_all['TC_relief'] = df_all['TC'].abs() * 100
    df_all['qtr'] = df_all['date'].dt.strftime('%YQ%q')
    df_all['pair'] = df_all['reporter_iso'] + "_" + df_all['partner_iso']
    df_all['origin_time'] = df_all['reporter_iso'] + "_" + df_all['qtr']
    df_all['dest_time'] = df_all['partner_iso'] + "_" + df_all['qtr']
    
    # Results collector
    summary_rows = []
    
    specs = [
        # (Label, DV_filter_args, Cluster_type, FE)
        ("Col 1: Baseline H=8", {'H': 8, 'W_type': 'Time-Varying'}, 'pair', 'pair + qtr'),
        ("Col 2: Baseline H=12", {'H': 12, 'W_type': 'Time-Varying'}, 'pair', 'pair + qtr'),
        ("Col 3: 2-Way Cluster H=8", {'H': 8, 'W_type': 'Time-Varying'}, 'origin_dest', 'pair + qtr'),
        ("Col 4: Robust FE H=8", {'H': 8, 'W_type': 'Time-Varying'}, 'pair', 'o_t + d_t'),
        ("Col 5: Fixed Weight H=8", {'H': 8, 'W_type': 'Fixed-Pre'}, 'pair', 'pair + qtr'),
    ]
    
    for label, filters, cluster, fe in specs:
        df_sub = df_all.copy()
        for k, v in filters.items():
            df_sub = df_sub[df_sub[k] == v]
        
        if df_sub.empty:
            logger.warning(f"Empty data for {label}")
            continue
            
        # Set indices based on FE
        if fe == 'pair + qtr':
            df_reg = df_sub.set_index(['pair', 'date'])
            formula = 'A ~ TC_relief + EntityEffects + TimeEffects'
        else: # o_t + d_t
            # Using O_T and D_T as fixed effects
            df_reg = df_sub.set_index(['pair', 'date'])
            # We can't easily do 3-way/4-way FE in PanelOLS directly without manual absorption.
            # But the user asked for origin*time and dest*time.
            # Let's use the 'other_effects' parameter in PanelOLS or just manual dummies if small.
            # For 14 countries + time, it's roughly 14 * 60 = 840 dummies.
            # We'll use the 'AbsorbingOLS' or just PanelOLS with OtherEffects.
            formula = 'A ~ TC_relief + EntityEffects' # We'll handle other FE via 'other_effects'
            
        try:
            # Type conversion to be safe
            df_sub['A'] = df_sub['A'].astype(float)
            df_sub['TC_relief'] = df_sub['TC_relief'].astype(float)
            
            if fe == 'o_t + d_t':
                # Custom fit for high-dim FE
                from linearmodels.iv.absorbing import AbsorbingLS
                # AbsorbingLS needs (dependent, exog, absorb=...)
                # Ensure absorbing columns are treated as categories and have no NaNs
                absorb_df = df_sub[['origin_time', 'dest_time']].copy()
                absorb_df['origin_time'] = absorb_df['origin_time'].astype('category')
                absorb_df['dest_time'] = absorb_df['dest_time'].astype('category')
                
                mod = AbsorbingLS(df_sub['A'], df_sub[['TC_relief']], 
                                 absorb=absorb_df)
                res = mod.fit(cov_type='clustered', clusters=df_sub['pair'])
            else:
                mod = PanelOLS.from_formula(formula, data=df_reg)
                if cluster == 'pair':
                    res = mod.fit(cov_type='clustered', cluster_entity=True)
                elif cluster == 'origin_dest':
                    # Cluster by reporter and partner
                    # Use the actual model index to filter clusters
                    # mod.dependent.index is the MultiIndex after dropping NaNs
                    clusters = df_sub.set_index(['pair', 'date']).loc[mod.dependent.index, ['reporter_iso', 'partner_iso']]
                    res = mod.fit(cov_type='clustered', clusters=clusters)
                else:
                    res = mod.fit(cov_type='clustered', cluster_entity=True)
                
            summary_rows.append({
                'Specification': label,
                'Coefficient': res.params['TC_relief'],
                'Std.Err': res.std_errors['TC_relief'],
                't-stat': res.tstats['TC_relief'],
                'p-value': res.pvalues['TC_relief'],
                'N': res.nobs,
                'R-squared': res.rsquared
            })
            logger.info(f"Completed {label}")
        except Exception as e:
            logger.error(f"Failed {label}: {e}")
            
    res_df = pd.DataFrame(summary_rows)
    res_df.to_csv(OUTPUT_DIR / "Table_Tariff_Resilience_Full.csv", index=False)
    print("\n--- Final Regression Table ---")
    print(res_df.to_string(index=False))
    
    # Save raw results for potential plotting
    df_all.to_csv(TEMP_DIR / "pairwise_regression_data_full.csv", index=False)

if __name__ == "__main__":
    run_all_regressions()
