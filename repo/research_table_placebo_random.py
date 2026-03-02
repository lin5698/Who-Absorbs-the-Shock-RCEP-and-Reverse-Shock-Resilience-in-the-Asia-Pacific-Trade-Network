import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
from linearmodels.panel import PanelOLS
from tqdm import tqdm

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

def estimate_A_rolling(Y_vals, W_list, dates_y, window=40, horizon=8):
    T, N = Y_vals.shape
    results = []
    for t in range(window, T):
        date_t = dates_y[t]
        try:
            c, A_l, B_l, Pi, Sigma, _ = var_ols(Y_vals[t-window:t], W_list[t-window:t], p=2)
            B_l_z = [np.zeros_like(B) for B in B_l]
            Psi_t = moving_average_coefficients(A_l, B_l, W_list[t-window:t], horizon, W_fixed=W_list[t-1])
            Psi_d = moving_average_coefficients(A_l, B_l_z, W_list[t-window:t], horizon, W_fixed=W_list[t-1])
            
            for j in range(N):
                irf_t_j = [girf_one(Psi_t[s], Sigma, j) for s in range(horizon+1)]
                irf_d_j = [girf_one(Psi_d[s], Sigma, j) for s in range(horizon+1)]
                for i in range(N):
                    if i == j: continue
                    a_ij = network_amplification_share(sum(np.abs(irf_t_j[s][i]) for s in range(horizon+1)),
                                                      sum(np.abs(irf_d_j[s][i]) for s in range(horizon+1)))
                    results.append({'date': date_t, 'reporter_iso': RCEP_LIST[i], 'partner_iso': RCEP_LIST[j], 'A': a_ij})
        except: continue
    return pd.DataFrame(results)

def run_placebo_random():
    logger.info("Loading Data...")
    df_macro, df_bilat = load_quarterly_macro_and_bilateral()
    df_tc_real = build_tariff_relief_tc(df_bilat)
    df_tc_real['date'] = pd.to_datetime(df_tc_real['date'])
    
    # Get A (Baseline: Import, m=40, H=8)
    vax = chow_lin_quarterly_vax(df_macro)
    vax_p = vax.set_index(["date", "iso3"])["vax_q"].unstack("iso3")
    for c in vax_p.columns: vax_p[c] = quality_control_outliers(quality_control_missing(vax_p[c]))
    Y = np.log(vax_p + 1e-6).diff().dropna(how="all")
    dates_y = Y.index
    Y_vals = Y.reindex(columns=RCEP_LIST).fillna(0).values
    
    W_list = []
    for d in dates_y:
        W = build_trade_network_w(df_bilat, d, window_quarters=4, mode='import')
        W_list.append(W.reindex(index=RCEP_LIST, columns=RCEP_LIST).fillna(0).values if isinstance(W, pd.DataFrame) else W)
    
    logger.info("Estimating baseline A series...")
    df_a = estimate_A_rolling(Y_vals, W_list, dates_y)
    df_a['date'] = pd.to_datetime(df_a['date'])
    
    # Baseline Result
    df_reg_base = pd.merge(df_a, df_tc_real, on=['date', 'reporter_iso', 'partner_iso'])
    df_reg_base['A'] = df_reg_base['A'].clip(0, 1)
    df_reg_base['TC_relief'] = df_reg_base['TC'].abs() * 100
    df_reg_base['pair'] = df_reg_base['reporter_iso'] + "_" + df_reg_base['partner_iso']
    mod_base = PanelOLS.from_formula('A ~ TC_relief + EntityEffects + TimeEffects', data=df_reg_base.set_index(['pair', 'date']))
    res_base = mod_base.fit(cov_type='clustered', cluster_entity=True)
    beta_base = res_base.params['TC_relief']
    
    results_placebo = []
    results_placebo.append({'Variant': 'True Baseline', 'Coefficient': beta_base, 'p-value': res_base.pvalues['TC_relief']})
    
    # Panel A: Time-Shifted Placebo
    logger.info("Panel A: Time-Shifted Placebo...")
    for shift_year in [2018, 2019]:
        logger.info(f"Shift RCEP start to {shift_year}Q1...")
        df_tc_shift = df_tc_real.copy()
        # Original starts at 2022-01-01. Move it to shift_year
        # Simple shift: if pair has treatment, shift the whole series
        # Actually, let's just shift the dates of the TC relief.
        offset = pd.Timestamp('2022-01-01') - pd.Timestamp(f'{shift_year}-01-01')
        df_tc_shift['date'] = df_tc_shift['date'] - offset
        
        df_reg_shift = pd.merge(df_a, df_tc_shift, on=['date', 'reporter_iso', 'partner_iso'])
        df_reg_shift['A'] = df_reg_shift['A'].clip(0, 1)
        df_reg_shift['TC_relief'] = df_reg_shift['TC'].abs() * 100
        df_reg_shift['pair'] = df_reg_shift['reporter_iso'] + "_" + df_reg_shift['partner_iso']
        
        try:
            mod = PanelOLS.from_formula('A ~ TC_relief + EntityEffects + TimeEffects', data=df_reg_shift.set_index(['pair', 'date']))
            res = mod.fit(cov_type='clustered', cluster_entity=True)
            results_placebo.append({'Variant': f'Placebo Start {shift_year}', 'Coefficient': res.params['TC_relief'], 'p-value': res.pvalues['TC_relief']})
        except: pass
        
    # Panel B: Randomized Path Assignment
    logger.info("Panel B: Randomized Path Assignment (n=100)...")
    boot_betas = []
    unique_pairs = df_tc_real[['reporter_iso', 'partner_iso']].drop_duplicates()
    
    for _ in tqdm(range(100)):
        # Randomly shuffle TC series across pairs
        # For each pair i-j, assign the TC series of a random pair k-l
        shuffled_pairs = unique_pairs.sample(frac=1).reset_index(drop=True)
        shuffled_pairs.columns = ['shuf_rep', 'shuf_part']
        pair_map = pd.concat([unique_pairs.reset_index(drop=True), shuffled_pairs], axis=1)
        
        # Build fake TC mapping
        df_tc_fake = pd.merge(df_tc_real, pair_map, on=['reporter_iso', 'partner_iso'])
        # Now rename back so we can merge with real A
        df_tc_fake = df_tc_fake.drop(columns=['reporter_iso', 'partner_iso']).rename(columns={'shuf_rep': 'reporter_iso', 'shuf_part': 'partner_iso'})
        
        df_reg_fake = pd.merge(df_a, df_tc_fake, on=['date', 'reporter_iso', 'partner_iso'])
        df_reg_fake['A'] = df_reg_fake['A'].clip(0, 1)
        df_reg_fake['TC_relief'] = df_reg_fake['TC'].abs() * 100
        df_reg_fake['pair'] = df_reg_fake['reporter_iso'] + "_" + df_reg_fake['partner_iso']
        
        try:
            mod = PanelOLS.from_formula('A ~ TC_relief + EntityEffects + TimeEffects', data=df_reg_fake.set_index(['pair', 'date']))
            res = mod.fit() # No need for cluster for speed in boot
            boot_betas.append(res.params['TC_relief'])
        except: continue
        
    if boot_betas:
        results_placebo.append({'Variant': 'Randomized Mean', 'Coefficient': np.mean(boot_betas), 'p-value': np.nan})
        results_placebo.append({'Variant': 'Randomized SD', 'Coefficient': np.std(boot_betas), 'p-value': np.nan})
        
    res_df = pd.DataFrame(results_placebo)
    res_df.to_csv(OUTPUT_DIR / "Table_Placebo_Randomization.csv", index=False)
    print("\n--- Placebo and Randomization Results ---")
    print(res_df)

if __name__ == "__main__":
    run_placebo_random()
