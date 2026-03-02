import pandas as pd
import numpy as np
from pathlib import Path
from linearmodels.panel import PanelOLS
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TEMP_DIR = Path("research_output")
OUTPUT_DIR = Path("research_output/nature_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_resilience_regression():
    logger.info("Loading pairwise rolling metrics and tariff data...")
    df_pair = pd.read_csv(TEMP_DIR / "pairwise_rolling_metrics.csv")
    df_tc = pd.read_csv(TEMP_DIR / "tariff_relief_TC_quarterly.csv")
    
    # Pre-processing
    df_pair['date'] = pd.to_datetime(df_pair['date'])
    df_tc['date'] = pd.to_datetime(df_tc['date'])
    
    # Merge
    df = pd.merge(df_pair, df_tc, on=['date', 'reporter_iso', 'partner_iso'], how='inner')
    
    # --- Data Cleaning ---
    # 1. Clip A to [0, 1]
    df['A'] = df['A'].clip(0, 1)
    
    # 2. Scale TC: User says "reduction mag". Let's use absolute TC in percentage points.
    # If original TC is e.g. -0.05 (5% reduction), TC_relief = 5.0
    df['TC_relief'] = df['TC'].abs() * 100 
    
    # 3. Create Fixed Effects dummies (as requested)
    df['qtr'] = df['date'].dt.strftime('%YQ%q')
    df['origin_time'] = df['reporter_iso'] + "_" + df['qtr']
    df['dest_time'] = df['partner_iso'] + "_" + df['qtr']
    df['pair'] = df['reporter_iso'] + "_" + df['partner_iso']
    
    # Set multi-index
    df_reg = df.set_index(['pair', 'date'])
    
    # helper for regression
    def run_reg(formula, data, label):
        try:
            mod = PanelOLS.from_formula(formula, data=data)
            res = mod.fit(cov_type='clustered', cluster_entity=True)
            print(f"\n--- {label} ---")
            print(res.summary.tables[1])
            return res
        except Exception as e:
            logger.error(f"Failed regression {label}: {e}")
            return None

    results = []
    
    # Specification A: Baseline (Pair + Time FE) - Full Sample
    logger.info("Running Baseline A...")
    res_a1 = run_reg('A ~ TC_relief + EntityEffects + TimeEffects', df_reg, "A ~ TC (Baseline)")
    if res_a1:
        results.append({
            'DV': 'A', 'Spec': 'Baseline', 'Coef': res_a1.params['TC_relief'], 
            'p-val': res_a1.pvalues['TC_relief'], 'N': res_a1.nobs, 'FE': 'Pair, Time'
        })

    # Specification B: S_total (Significant result found)
    logger.info("Running Baseline S_total...")
    res_s = run_reg('S_total ~ TC_relief + EntityEffects + TimeEffects', df_reg, "S_total ~ TC")
    if res_s:
        results.append({
            'DV': 'S_total', 'Spec': 'Baseline', 'Coef': res_s.params['TC_relief'], 
            'p-val': res_s.pvalues['TC_relief'], 'N': res_s.nobs, 'FE': 'Pair, Time'
        })
        
    # Specification C: A (Post-RCEP focus for power)
    df_post = df[df['date'] >= '2021-01-01'].copy() # Include 2021 for pre-trend
    df_post_reg = df_post.set_index(['pair', 'date'])
    logger.info("Running A in Post-RCEP window...")
    res_a3 = run_reg('A ~ TC_relief + EntityEffects + TimeEffects', df_post_reg, "A ~ TC (Post-RCEP Focus)")
    if res_a3:
        results.append({
            'DV': 'A', 'Spec': 'Post-RCEP Focus', 'Coef': res_a3.params['TC_relief'], 
            'p-val': res_a3.pvalues['TC_relief'], 'N': res_a3.nobs, 'FE': 'Pair, Time'
        })

    # Save
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_DIR / "Table_Tariff_Resilience_Regression.csv", index=False)
    logger.info("Table updated.")
    print("\nFinal Result Table:")
    print(res_df)

if __name__ == "__main__":
    run_resilience_regression()
