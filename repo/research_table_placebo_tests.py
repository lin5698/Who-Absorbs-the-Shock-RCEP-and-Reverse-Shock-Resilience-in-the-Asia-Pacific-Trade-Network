import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
from linearmodels.panel import PanelOLS

# Add current path to import local modules
sys.path.insert(0, str(Path(__file__).parent))
from research_data_construction import build_tariff_relief_tc

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("research_output/nature_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR = Path("research_output")

def run_placebo_timing_tests():
    logger.info("Loading regression data for consistency...")
    # Load consolidated regression data (H=8, Time-Varying) from previous script
    try:
        df_all_orig = pd.read_csv(TEMP_DIR / "pairwise_regression_data_full.csv")
        df_all_orig['date'] = pd.to_datetime(df_all_orig['date'])
        # Filter to Baseline (H=8, Time-Varying)
        df_base = df_all_orig[(df_all_orig['H'] == 8) & (df_all_orig['W_type'] == 'Time-Varying')].copy()
        
        # We need the original A and TC components
        df_a = df_base[['date', 'reporter_iso', 'partner_iso', 'A', 'pair']].copy()
        df_tc_orig = df_base[['date', 'reporter_iso', 'partner_iso', 'TC']].drop_duplicates().copy()
    except FileNotFoundError:
        logger.error("Main regression data not found. Please run research_table_resilience_final.py first.")
        return
    
    shifts = [
        (0, 'Baseline'),
        (4, 'Shift +4q (Placebo)'),
        (8, 'Shift +8q (Placebo)'),
        (-4, 'Shift -4q (Placebo)'),
        (-8, 'Shift -8q (Placebo)')
    ]
    
    summary_results = []
    
    for s_val, label in shifts:
        logger.info(f"Running specification: {label}")
        
        # Shift TC dates
        # Positive shift means we apply the tariff relief later
        # Negative shift means we apply it earlier
        df_tc_s = df_tc_orig.copy()
        df_tc_s['date'] = df_tc_s['date'] + pd.DateOffset(months=3 * s_val)
        
        # Merge shifted TC with original A
        df_reg_data = pd.merge(df_a, df_tc_s, on=['date', 'reporter_iso', 'partner_iso'])
        
        if df_reg_data.empty:
            logger.warning(f"Empty data for shift {s_val}")
            continue
            
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
                'Specification': label,
                'Shift (Q)': s_val,
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
    res_df.to_csv(OUTPUT_DIR / "Table_Placebo_Timing_Final.csv", index=False)
    print("\n--- Placebo Timing Test Table ---")
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    run_placebo_timing_tests()
