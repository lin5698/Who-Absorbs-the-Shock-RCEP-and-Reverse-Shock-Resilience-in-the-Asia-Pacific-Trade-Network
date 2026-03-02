"""
Step 0: 统一频率与对齐（Quarterly panel alignment）
产出：15国 × 2005Q1–2024Q4 × 季度面板，同一套季度时间索引 t。
（实际有效区间将通过参数切出）

可直接写进论文的规则（写死、可复现）：
- A 主时间轴：PeriodIndex 范围，国家统一为 ISO3。
- B 这是核心：由于原始数据只有年度频次，我们没有真实的季度指示变量。为避免被审稿人质疑季度总和与年度总和不匹配：
    - B_flow) 年度流量型（flow，如出口额、GDP）：使用基于累积和（Cumulative sum）的三次样条插值。这保证了 Q1+Q2+Q3+Q4 精确等于原始年度流量。
    - B_stock) 年度存量/指数/价格型（stock/index/price，如CPI/人口/汇率）：使用标准PCHIP插值平滑到季度节点。
- C 财年/断点：目前未定义明确宏观断点，预留了记录表。
- D 主骨架：生成外连接骨架，并输出缺失矩阵。
- F 验收：检查 Flow 类季度加总是否精准等于年度数据。
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, str(Path(__file__).parent))
from config import RCEP_COUNTRIES, RCEP_ISO2

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

Q_START = "2005Q1"
Q_END = "2024Q4"
DATA_DIR_LOCAL = Path(__file__).parent / "data_acquisition" / "data"
OUTPUT_DIR = Path(__file__).parent / "research_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RCEP_ISO3_LIST = list(RCEP_COUNTRIES.keys())

# ============== A. 主时间轴与国家编码表 ==============

def build_master_quarter_index():
    periods = pd.period_range(start=Q_START, end=Q_END, freq="Q")
    df = pd.DataFrame({
        "quarter_t": periods,
        "year": periods.year,
        "quarter": periods.quarter,
        "quarter_start": periods.to_timestamp(how="start"),
        "quarter_end": periods.to_timestamp(how="end"),
    })
    df["quarter_str"] = df["quarter_t"].astype(str)
    return df, periods

def build_country_code_table():
    rows = []
    for iso3, name in RCEP_COUNTRIES.items():
        iso2 = next((k for k, v in RCEP_ISO2.items() if v == name), None)
        rows.append({
            "iso3": iso3,
            "country_name": name,
            "iso2": iso2,
        })
    df = pd.DataFrame(rows)
    return df

# ============== B. 核心：年度到季度严格降频算法 ==============

def disaggregate_flow_cumulative(df_annual, group_cols, year_col, value_col):
    """
    B_flow: 年度流量转季度。采用累积和插值法。
    保证同一年 4 个季度的求和 等于 原年度值 (硬约束)。
    """
    results = []
    df_annual = df_annual.sort_values(group_cols + [year_col]).dropna(subset=[value_col])
    
    for name, group in df_annual.groupby(group_cols):
        # Aggregate duplicates for the same year
        group_orig = group
        group = group.groupby(year_col, as_index=False).mean(numeric_only=True)
        group = group.sort_values(year_col)
        years = group[year_col].values
        # Ensure value_col is present after mean()
        if value_col in group.columns:
            values = group[value_col].values
        else:
            values = group_orig[group_orig[year_col].isin(years)].groupby(year_col)[value_col].mean().values
        
        name_tuple = name if isinstance(name, tuple) else (name,)
        
        if len(years) < 2:
            for y, v in zip(years, values):
                for q in range(1, 5):
                    row = {k: v for k, v in zip(group_cols, name_tuple)}
                    row['year'] = int(y)
                    row['quarter'] = q
                    row[value_col] = v / 4.0
                    results.append(row)
            continue
            
        C = np.cumsum(values)
        x_nodes = np.array([years[0] - 1.0] + list(years))
        y_nodes = np.array([0.0] + list(C))
        
        interp = PchipInterpolator(x_nodes, y_nodes)
        
        for y in years:
            # 季度末时间点
            xq = np.array([y - 0.75, y - 0.50, y - 0.25, y])
            yq = interp(xq)
            
            prev_C = interp(y - 1.0)
            q1 = yq[0] - prev_C
            q2 = yq[1] - yq[0]
            q3 = yq[2] - yq[1]
            q4 = yq[3] - yq[2]
            
            flows = [max(0, f) for f in [q1, q2, q3, q4]] # 防止极小负数
            tot = sum(flows)
            annual_val = interp(y) - prev_C
            if tot > 0:
                flows = [f * (annual_val / tot) for f in flows]  # 严格对齐
                
            for q, flow in zip(range(1, 5), flows):
                row = {k: v for k, v in zip(group_cols, name_tuple)}
                row['year'] = int(y)
                row['quarter'] = q
                row[value_col] = flow
                results.append(row)
                
    return pd.DataFrame(results)

def disaggregate_stock_spline(df_annual, group_cols, year_col, value_col):
    """
    B_stock: 年度存量/指数转季度。
    采用中点 PCHIP 样条插值。
    """
    results = []
    df_annual = df_annual.sort_values(group_cols + [year_col]).dropna(subset=[value_col])
    
    for name, group in df_annual.groupby(group_cols):
        # Aggregate duplicates for the same year
        group_orig = group
        group = group.groupby(year_col, as_index=False).mean(numeric_only=True)
        group = group.sort_values(year_col)
        years = group[year_col].values
        # Ensure value_col is present after mean()
        if value_col in group.columns:
            values = group[value_col].values
        else:
            values = group_orig[group_orig[year_col].isin(years)].groupby(year_col)[value_col].mean().values
        
        name_tuple = name if isinstance(name, tuple) else (name,)
        
        if len(years) < 2:
            for y, v in zip(years, values):
                for q in range(1, 5):
                    row = {k: v for k, v in zip(group_cols, name_tuple)}
                    row['year'] = int(y)
                    row['quarter'] = q
                    row[value_col] = v
                    results.append(row)
            continue
            
        x_nodes = np.array(years) - 0.5  # 假设年度值是期中
        y_nodes = values
        
        interp = PchipInterpolator(x_nodes, y_nodes)
        
        for y in years:
            # 季度中点
            xq = np.array([y - 0.875, y - 0.625, y - 0.375, y - 0.125])
            yq = interp(xq)
            for q, val in zip(range(1, 5), yq):
                row = {k: v for k, v in zip(group_cols, name_tuple)}
                row['year'] = int(y)
                row['quarter'] = q
                row[value_col] = val
                results.append(row)
                
    return pd.DataFrame(results)

# ============== 主流程 ==============

def run_step0_alignment():
    logger.info("=" * 60)
    logger.info("Step 0: Quarterly panel alignment (Strict Rules)")
    logger.info("=" * 60)

    # A1
    df_master, periods = build_master_quarter_index()
    df_master.to_csv(OUTPUT_DIR / "step0_master_quarter_index.csv", index=False)
    
    # 骨架
    skeleton_macro = pd.MultiIndex.from_product(
        [RCEP_ISO3_LIST, periods], names=["iso3", "quarter_t"]
    ).to_frame(index=False)

    df_annual_macro = pd.read_csv(DATA_DIR_LOCAL / "master_macro_panel_2005_2024.csv")
    df_annual_bilat = pd.read_csv(DATA_DIR_LOCAL / "master_bilateral_trade_2005_2024.csv")

    # 定义变量属性
    macro_flows = ["gdp_current_usd", "gdp_ppp_current_usd", "total_exports_usd_k", "total_imports_usd_k"]
    macro_stocks = ["population", "exchange_rate", "tariff_ahs_weighted", "tariff_mfn_weighted"]
    
    bilat_flows = ["export_usd", "import_usd"]
    bilat_stocks = ["bilateral_tariff"]

    macro_q_dfs = []
    
    for c in macro_flows:
        if c in df_annual_macro.columns:
            logger.info(f"Disaggregating Flow: {c}")
            dq = disaggregate_flow_cumulative(df_annual_macro, ["iso3"], "year", c)
            macro_q_dfs.append(dq)
            
    for c in macro_stocks:
        if c in df_annual_macro.columns:
            logger.info(f"Disaggregating Stock: {c}")
            dq = disaggregate_stock_spline(df_annual_macro, ["iso3"], "year", c)
            macro_q_dfs.append(dq)

    # 合并宏观
    df_macro_q = skeleton_macro.copy()
    for dq in macro_q_dfs:
        dq["quarter_t"] = pd.PeriodIndex(dq["year"].astype(str) + "Q" + dq["quarter"].astype(str), freq="Q")
        df_macro_q = df_macro_q.merge(dq.drop(columns=["year", "quarter"]), on=["iso3", "quarter_t"], how="left")
        
    df_macro_q.to_csv(OUTPUT_DIR / "step0_aligned_macro.csv", index=False)

    # 同样的逻辑应用到双边矩阵
    pairs = df_annual_bilat[["reporter_iso", "partner_iso"]].drop_duplicates()
    skeleton_bilat = pd.MultiIndex.from_product(
        [pairs["reporter_iso"].unique(), pairs["partner_iso"].unique(), periods], 
        names=["reporter_iso", "partner_iso", "quarter_t"]
    ).to_frame(index=False)
    # filter to actual pairs in RCEP
    skeleton_bilat = skeleton_bilat.merge(pairs, on=["reporter_iso", "partner_iso"], how="inner")
    
    bilat_q_dfs = []
    
    for c in bilat_flows:
        if c in df_annual_bilat.columns:
            logger.info(f"Disaggregating Bilateral Flow: {c}")
            dq = disaggregate_flow_cumulative(df_annual_bilat, ["reporter_iso", "partner_iso"], "year", c)
            bilat_q_dfs.append(dq)
            
    for c in bilat_stocks:
        if c in df_annual_bilat.columns:
            logger.info(f"Disaggregating Bilateral Stock: {c}")
            dq = disaggregate_stock_spline(df_annual_bilat, ["reporter_iso", "partner_iso"], "year", c)
            bilat_q_dfs.append(dq)
            
    df_bilat_q = skeleton_bilat.copy()
    for dq in bilat_q_dfs:
        dq["quarter_t"] = pd.PeriodIndex(dq["year"].astype(str) + "Q" + dq["quarter"].astype(str), freq="Q")
        df_bilat_q = df_bilat_q.merge(dq.drop(columns=["year", "quarter"]), on=["reporter_iso", "partner_iso", "quarter_t"], how="left")
        
    df_bilat_q.to_csv(OUTPUT_DIR / "step0_aligned_bilateral.csv", index=False)
    
    # 替换原本直接复制年度数据的09_script产物位置，方便下游使用
    df_macro_q.to_csv(DATA_DIR_LOCAL / "master_quarterly_macro_2005_2024.csv", index=False)
    df_bilat_q.to_csv(DATA_DIR_LOCAL / "master_quarterly_bilateral_2005_2024.csv", index=False)

    # 验证报告 (验证 flow 相加等于年)
    logger.info("Running Verifications...")
    report_lines = ["Validation Report"]
    
    if "total_exports_usd_k" in df_macro_q.columns:
        df_macro_q["year"] = df_macro_q["quarter_t"].dt.year
        chk = df_macro_q.groupby(["iso3", "year"])["total_exports_usd_k"].sum().reset_index()
        ann = df_annual_macro[["iso3", "year", "total_exports_usd_k"]].rename(columns={"total_exports_usd_k": "ann_val"})
        chk = chk.merge(ann, on=["iso3", "year"])
        chk["diff"] = (chk["total_exports_usd_k"] - chk["ann_val"]).abs()
        max_err = chk["diff"].max()
        report_lines.append(f"Max error mapping Quarters to Annual Total Exports: {max_err}")
        
    with open(OUTPUT_DIR / "step0_validation_report.txt", "w") as f:
        f.write("\n".join(report_lines))

    logger.info("Step 0 complete. Aligned Macro: %s, Aligned Bilateral: %s", df_macro_q.shape, df_bilat_q.shape)


if __name__ == "__main__":
    run_step0_alignment()
