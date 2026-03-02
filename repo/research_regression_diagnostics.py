import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import logging

logging.basicConfig(level=logging.INFO)

df_pair = pd.read_csv("research_output/pairwise_rolling_metrics.csv")
df_tc = pd.read_csv("research_output/tariff_relief_TC_quarterly.csv")
df_pair['date'] = pd.to_datetime(df_pair['date'])
df_tc['date'] = pd.to_datetime(df_tc['date'])

df = pd.merge(df_pair, df_tc, on=['date', 'reporter_iso', 'partner_iso'], how='inner')
df['A'] = df['A'].clip(0, 1)
df['TC_relief'] = df['TC'].abs() * 100
df['pair'] = df['reporter_iso'] + "_" + df['partner_iso']

# Additional cleaning
df['A_win'] = df['A'].clip(df['A'].quantile(0.01), df['A'].quantile(0.99))

# Trade Weights (using 2021 pre-RCEP)
trade_21 = df[df['date'].dt.year == 2021].groupby('pair')['S_total'].mean() # placeholder for trade weight
# Actually let's use the actual trade data if available, but S_total is a proxy for impact
df['weight'] = df['pair'].map(trade_21).fillna(1e-6)

df_reg = df.set_index(['pair', 'date'])

print("\n--- Testing Weights and Winsorization ---")
# Weighted A
mod_w = PanelOLS.from_formula('A_win ~ TC_relief + EntityEffects + TimeEffects', data=df_reg, weights=df_reg['weight'])
res_w = mod_w.fit(cov_type='clustered', cluster_entity=True)
print(f"Weighted A: Coef={res_w.params['TC_relief']:.6f}, p-val={res_w.pvalues['TC_relief']:.4f}")

# First Differences
df = df.sort_values(['pair', 'date'])
df['dA'] = df.groupby('pair')['A'].diff()
df['dTC'] = df.groupby('pair')['TC_relief'].diff()

df_diff = df.dropna(subset=['dA', 'dTC']).set_index(['pair', 'date'])
print("\n--- First Differences (dA ~ dTC + Time FE) ---")
mod_diff = PanelOLS.from_formula('dA ~ dTC + TimeEffects', data=df_diff)
res_diff = mod_diff.fit(cov_type='clustered', cluster_entity=True)
print(f"First Differences: Coef={res_diff.params['dTC']:.6f}, p-val={res_diff.pvalues['dTC']:.4f}")

# Try 2022 onwards specifically for differences
df_diff_post = df_diff[df_diff.index.get_level_values('date') >= '2022-01-01']
mod_diff_post = PanelOLS.from_formula('dA ~ dTC + TimeEffects', data=df_diff_post)
res_diff_post = mod_diff_post.fit(cov_type='clustered', cluster_entity=True)
print(f"First Differences Post-2022: Coef={res_diff_post.params['dTC']:.6f}, p-val={res_diff_post.pvalues['dTC']:.4f}")
