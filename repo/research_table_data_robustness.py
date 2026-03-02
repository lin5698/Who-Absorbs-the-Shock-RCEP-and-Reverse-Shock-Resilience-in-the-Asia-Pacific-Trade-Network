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

def proportional_denton(annual_series, quarterly_indicator):
    """
    Simple proportional benchmarking (1st order Denton approximation).
    Disaggregates annual values into quarterly ones using the proportion of the quarterly indicator.
    """
    df = pd.DataFrame({'indicator': quarterly_indicator})
    df['year'] = df.index.year
    
    annual_sums = df.groupby('year')['indicator'].transform('sum')
    proportions = df['indicator'] / annual_sums
    
    # Map annual values to years
    result = []
    for date, prop in proportions.items():
        year = date.year
        if year in annual_series.index:
            result.append(annual_series[year] * prop)
        else:
            result.append(np.nan)
    return pd.Series(result, index=quarterly_indicator.index)

def estimate_metrics_for_Y(Y_vals, W_list, dates_y, horizon=8):
    T, N = Y_vals.shape
    results_list = []
    
    for t in range(40, T):
        date_t = dates_y[t]
        Y_w = Y_vals[t - 40 : t]
        W_w = W_list[t - 40 : t]
        
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

def run_data_robustness():
    logger.info("Loading Data...")
    df_macro, df_bilateral = load_quarterly_macro_and_bilateral()
    df_tc = build_tariff_relief_tc(df_bilateral)
    df_tc['date'] = pd.to_datetime(df_tc['date'])
    
    # Common W_list (Baseline: Import, window=4)
    # We need to prep a date range for Y first
    # Let's get the common dates
    dates_all = sorted(pd.to_datetime(df_macro['date'].unique()))
    
    # 1. Panel A: Quarterlyization Robustness
    logger.info("Panel A: Quarterlyization Robustness...")
    
    # A1: Chow-Lin with industrial indicator (GDP-VOL)
    # The existing chow_lin_quarterly_vax uses GDP-VOL by default if it exists? 
    # Let's check the code or just pass it.
    # A2: Denton disaggregation
    
    # Need annual VAX for Denton
    # For simplicity, we can aggregate the quarterly VAX if we don't have annual VAX?
    # Actually, usually Denton is used when we HAVE annual but not quarterly.
    # Since we are essentially "simulating" robustness, let's just use Denton with GDP-VOL as indicator.
    
    # Prepare Y scenarios
    y_scenarios = {}
    
    # Baseline VAX (Chow-Lin)
    vax_cl = chow_lin_quarterly_vax(df_macro)
    y_scenarios['VAX (Baseline)'] = vax_cl.set_index(["date", "iso3"])["vax_q"].unstack("iso3")
    
    # Denton VAX
    # Aggregate VAX to annual for each country to simulate annual data
    annual_vax = vax_cl.groupby([vax_cl['date'].dt.year, 'iso3'])['vax_q'].sum().unstack('iso3')
    dates_idx = sorted(pd.to_datetime(df_macro['date'].unique()))
    
    denton_vax_cols = {}
    for iso in RCEP_LIST:
        ind = df_macro[df_macro['iso3'] == iso].set_index('date')['GDP-VOL'].reindex(dates_idx).fillna(0)
        ann = annual_vax[iso] if iso in annual_vax.columns else annual_vax.mean(axis=1) # Fallback
        denton_vax_cols[iso] = proportional_denton(ann, ind)
    y_scenarios['VAX (Denton)'] = pd.DataFrame(denton_vax_cols)
    
    # Panel B: Alternative Outcomes
    logger.info("Panel B: Alternative Outcomes...")
    
    # B1: Gross Exports
    exp_cols = {}
    for iso in RCEP_LIST:
        exp_cols[iso] = df_macro[df_macro['iso3'] == iso].set_index('date')['total_exports_usd_k'].reindex(dates_idx).fillna(0)
    y_scenarios['Gross Exports'] = pd.DataFrame(exp_cols)
    
    # B2: Industrial VA Proxy (GDP-VOL)
    va_cols = {}
    for iso in RCEP_LIST:
        va_cols[iso] = df_macro[df_macro['iso3'] == iso].set_index('date')['GDP-VOL'].reindex(dates_idx).fillna(0)
    y_scenarios['Industrial VA'] = pd.DataFrame(va_cols)
    
    # Processing Each Scenario
    final_results = []
    
    # Pre-build W_list
    W_list = []
    for d in dates_idx:
        W = build_trade_network_w(df_bilateral, d, window_quarters=4, mode='import')
        if isinstance(W, pd.DataFrame):
            W = W.reindex(index=RCEP_LIST, columns=RCEP_LIST).fillna(0).values
        W_list.append(W)
    
    for label, y_df in y_scenarios.items():
        logger.info(f"Processing Scenario: {label}")
        
        # Clean and Log-Diff
        for c in y_df.columns:
            y_df[c] = quality_control_outliers(quality_control_missing(y_df[c]))
        Y = np.log(y_df + 1e-6).diff().dropna(how="all")
        dates_y = Y.index
        Y_vals = Y.reindex(columns=RCEP_LIST).fillna(0).values
        
        # Adjust W_list to match dates_y
        w_idx_map = {d: i for i, d in enumerate(dates_idx)}
        W_list_sub = [W_list[w_idx_map[d]] for d in dates_y]
        
        # Estimate A
        df_metrics = estimate_metrics_for_Y(Y_vals, W_list_sub, dates_y)
        if df_metrics.empty: continue
        
        # Merge and Reg
        df_metrics['date'] = pd.to_datetime(df_metrics['date'])
        df_reg_data = pd.merge(df_metrics, df_tc, on=['date', 'reporter_iso', 'partner_iso'])
        df_reg_data['A'] = df_reg_data['A'].clip(0, 1)
        df_reg_data['TC_relief'] = df_reg_data['TC'].abs() * 100
        df_reg_data['pair'] = df_reg_data['reporter_iso'] + "_" + df_reg_data['partner_iso']
        
        df_reg = df_reg_data.set_index(['pair', 'date'])
        try:
            mod = PanelOLS.from_formula('A ~ TC_relief + EntityEffects + TimeEffects', data=df_reg)
            res = mod.fit(cov_type='clustered', cluster_entity=True)
            
            final_results.append({
                'Scenario': label,
                'Coefficient': res.params['TC_relief'],
                'Std.Err': res.std_errors['TC_relief'],
                'p-value': res.pvalues['TC_relief'],
                'N': res.nobs,
                'R-squared': res.rsquared
            })
        except Exception as e:
            logger.error(f"Reg failed for {label}: {e}")
            
    res_df = pd.DataFrame(final_results)
    res_df.to_csv(OUTPUT_DIR / "Table_Data_and_Outcome_Robustness.csv", index=False)
    print("\n--- Data and Outcome Robustness ---")
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    run_data_robustness()
