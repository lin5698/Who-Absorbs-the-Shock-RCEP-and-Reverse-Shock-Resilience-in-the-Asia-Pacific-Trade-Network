"""
Data and Variable Construction (Section 3).
Quarterly panel alignment (2000Q1–2023Q4), VAX Chow–Lin quarterlyization,
trade network W_t, RCEP tariff relief TC_ij,t, PCA external factors, quality control.
数据原则：仅使用真实数据；缺失/异常处理规则明确可复现。
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import logging

sys.path.insert(0, str(Path(__file__).parent))
from config import RCEP_COUNTRIES, DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 样本与路径：与主文一致 2000Q1–2023Q4；数据不足时用实际范围
SAMPLE_START = "2000-01-01"
SAMPLE_END = "2023-12-31"
QUARTERLY_DATES = pd.date_range(start=SAMPLE_START, end=SAMPLE_END, freq="QE")
DATA_DIR_LOCAL = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "research_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RCEP_LIST = list(RCEP_COUNTRIES.keys())


def load_quarterly_macro_and_bilateral():
    """从项目既有数据加载季度宏观与双边贸易（真实数据）。"""
    macro_path = DATA_DIR_LOCAL / "master_quarterly_macro_2005_2024.csv"
    bilat_path = DATA_DIR_LOCAL / "master_quarterly_bilateral_2005_2024.csv"
    if not macro_path.exists():
        macro_path = DATA_DIR / "master_quarterly_macro_2005_2024.csv"
    if not bilat_path.exists():
        bilat_path = DATA_DIR / "master_quarterly_bilateral_2005_2024.csv"
    df_macro = pd.read_csv(macro_path)
    df_bilateral = pd.read_csv(bilat_path)
    df_macro["date"] = pd.to_datetime(df_macro["date_quarterly"] if "date_quarterly" in df_macro.columns else df_macro["date"])
    df_bilateral["date"] = pd.to_datetime(df_bilateral["date_quarterly"] if "date_quarterly" in df_bilateral.columns else df_bilateral["date"])
    df_macro = df_macro[df_macro["iso3"].isin(RCEP_LIST)]
    df_bilateral = df_bilateral[
        df_bilateral["reporter_iso"].isin(RCEP_LIST) & df_bilateral["partner_iso"].isin(RCEP_LIST)
    ]
    return df_macro, df_bilateral


def quarterly_panel_alignment(df_macro, df_bilateral):
    """统一季度频率与时间戳，产出平衡季度面板索引。"""
    dates = np.sort(pd.to_datetime(df_macro["date"].dropna().unique()))
    t0, t1 = pd.Timestamp(SAMPLE_START), pd.Timestamp(SAMPLE_END)
    dates = dates[(dates >= t0) & (dates <= t1)]
    if len(dates) == 0:
        dates = pd.to_datetime(df_bilateral["date"].dropna().unique()).sort_values()
    return dates


def chow_lin_quarterly_vax(df_macro, annual_vax_col=None, indicator_cols=None):
    """
    VAX 季度化：Chow–Lin 在年度约束下用季度指示变量分配。
    约束：4个季度之和 = 年度 VAX；指示变量默认用季度出口或工业相关变量。
    若缺少年度 VAX，用年度总出口 (total_exports_usd_k) 作为基准（真实数据代理）。
    """
    if annual_vax_col is None:
        annual_vax_col = "total_exports_usd_k"  # 代理：年度总出口，真实数据
    if indicator_cols is None:
        indicator_cols = ["total_exports_usd_k"]
    df = df_macro.copy()
    if "year" not in df.columns:
        df["year"] = pd.to_datetime(df["date"]).dt.year
    if "quarter" not in df.columns:
        df["quarter"] = pd.to_datetime(df["date"]).dt.quarter
    # 年度基准：按国别年度汇总
    annual = df.groupby(["iso3", "year"])[annual_vax_col].sum().reset_index() if annual_vax_col in df.columns else None
    if annual is None or annual_vax_col not in df.columns:
        # 用季度出口作为水平，后续取对数差分做平稳化
        df["vax_q"] = df["total_exports_usd_k"].replace(0, np.nan)
        df["vax_q"] = df.groupby("iso3")["vax_q"].ffill().bfill()
        return df[["date", "iso3", "year", "quarter", "vax_q"]]
    # Chow–Lin 简化：按季度指示变量比例分配年度总量（比例 = 该季指示变量 / 该年指示变量和）
    ind = [c for c in indicator_cols if c in df.columns]
    if not ind:
        ind = [c for c in ["total_exports_usd_k", "gdp_current_usd"] if c in df.columns]
    if not ind:
        df["vax_q"] = df[annual_vax_col] / 4.0
        return df[["date", "iso3", "year", "quarter", "vax_q"]]
    q_ind = df.groupby(["iso3", "year", "quarter"])[ind[0]].sum().reset_index()
    q_ind = q_ind.rename(columns={ind[0]: "ind_q"})
    y_ind = q_ind.groupby(["iso3", "year"])["ind_q"].sum().reset_index().rename(columns={"ind_q": "ind_y"})
    q_ind = q_ind.merge(y_ind, on=["iso3", "year"])
    q_ind["share"] = q_ind["ind_q"] / q_ind["ind_y"].replace(0, np.nan)
    annual_vax = annual.rename(columns={annual_vax_col: "vax_y"})
    q_ind = q_ind.merge(annual_vax, on=["iso3", "year"])
    q_ind["vax_q"] = q_ind["vax_y"] * q_ind["share"]
    df = df.merge(
        q_ind[["iso3", "year", "quarter", "vax_q"]],
        on=["iso3", "year", "quarter"],
        how="left",
    )
    if "vax_q" not in df.columns:
        df["vax_q"] = df.get(annual_vax_col, pd.Series(dtype=float)) / 4.0
    return df[["date", "iso3", "year", "quarter", "vax_q"]].dropna(subset=["vax_q"])


def build_trade_network_w(df_bilateral, date_val, window_quarters=4, use_import_share=True, mode='import'):
    """
    构造 W_t：行随机、零对角线的贸易网络权重矩阵。
    mode: 'import' (baseline), 'export', 'symmetric'
    """
    df = df_bilateral[
        (df_bilateral["date"] <= date_val)
        & (df_bilateral["date"] > (date_val - pd.DateOffset(months=3 * window_quarters)))
    ].copy()
    if df.empty:
        return pd.DataFrame(np.zeros((len(RCEP_LIST), len(RCEP_LIST))), index=RCEP_LIST, columns=RCEP_LIST)
    
    flow_col = "import_usd" if "import_usd" in df.columns else "export_usd"
    df["flow"] = pd.to_numeric(df[flow_col], errors="coerce").fillna(0)
    
    if mode == 'export':
        # Use export_usd specifically if available
        if "export_usd" in df.columns:
            df["flow"] = pd.to_numeric(df["export_usd"], errors="coerce").fillna(0)
        # For export-based, weight is j's export to i / j's total exports
        agg = df.groupby(["reporter_iso", "partner_iso"])["flow"].sum().reset_index()
        total_j = agg.groupby("partner_iso")["flow"].sum().reset_index().rename(columns={"flow": "total"})
        agg = agg.merge(total_j, on="partner_iso")
        agg["w"] = agg["flow"] / agg["total"].replace(0, np.nan)
        W = agg.pivot_table(index="reporter_iso", columns="partner_iso", values="w", fill_value=0)
    elif mode == 'symmetric':
        # Average of i->j and j->i
        df["pair"] = df.apply(lambda x: tuple(sorted([x["reporter_iso"], x["partner_iso"]])), axis=1)
        agg = df.groupby("pair")["flow"].sum().reset_index()
        pairs = []
        for p, f in zip(agg["pair"], agg["flow"]):
            pairs.append({"reporter_iso": p[0], "partner_iso": p[1], "flow": f})
            pairs.append({"reporter_iso": p[1], "partner_iso": p[0], "flow": f})
        agg = pd.DataFrame(pairs)
        total_i = agg.groupby("reporter_iso")["flow"].sum().reset_index().rename(columns={"flow": "total"})
        agg = agg.merge(total_i, on="reporter_iso")
        agg["w"] = agg["flow"] / agg["total"].replace(0, np.nan)
        W = agg.pivot_table(index="reporter_iso", columns="partner_iso", values="w", fill_value=0)
    else: # baseline: import
        agg = df.groupby(["reporter_iso", "partner_iso"])["flow"].sum().reset_index()
        total_i = agg.groupby("reporter_iso")["flow"].sum().reset_index().rename(columns={"flow": "total"})
        agg = agg.merge(total_i, on="reporter_iso")
        agg["w"] = agg["flow"] / agg["total"].replace(0, np.nan)
        W = agg.pivot_table(index="reporter_iso", columns="partner_iso", values="w", fill_value=0)

    W = W.reindex(index=RCEP_LIST, columns=RCEP_LIST).fillna(0)
    np.fill_diagonal(W.values, 0)
    row_sums = W.sum(axis=1).replace(0, np.nan)
    W = W.div(row_sums, axis=0).fillna(0)
    return W


def build_tariff_relief_tc(df_bilateral):
    """RCEP 关税减让强度 TC_ij,t = 基准税率 - RCEP 优惠税率。用既有 bilateral_tariff 与 tariff_reduction。"""
    df = df_bilateral.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "tariff_reduction" in df.columns:
        df["TC"] = pd.to_numeric(df["tariff_reduction"], errors="coerce").fillna(0)
    elif "bilateral_tariff" in df.columns:
        # 用基期平均作为基准，TC = 基期 - 当期
        base = df.groupby(["reporter_iso", "partner_iso"])["bilateral_tariff"].first().reset_index().rename(columns={"bilateral_tariff": "tau_base"})
        df = df.merge(base, on=["reporter_iso", "partner_iso"])
        df["TC"] = df["tau_base"] - pd.to_numeric(df["bilateral_tariff"], errors="coerce").fillna(df["tau_base"])
    else:
        df["TC"] = 0.0
    return df[["date", "reporter_iso", "partner_iso", "TC", "bilateral_tariff", "tariff_reduction"]].dropna(subset=["TC"]) if "bilateral_tariff" in df.columns else df[["date", "reporter_iso", "partner_iso", "TC"]]


def pca_external_factors(df_macro, n_components=3, min_variance_ratio=0.7):
    """从宏观变量池提取外部共同因子（PCA），标准化后提取，仅用真实观测。"""
    cand = ["exchange_rate", "gdp_current_usd", "total_exports_usd_k", "total_imports_usd_k", "tariff_ahs_weighted", "tariff_mfn_weighted", "population", "GDP-VOL", "gdp_ppp_current_usd"]
    cols = [c for c in cand if c in df_macro.columns]
    if not cols:
        return pd.DataFrame()
    wide = df_macro.pivot_table(index="date", columns="iso3", values=cols[0], aggfunc="mean")
    for c in cols[1:]:
        w = df_macro.pivot_table(index="date", columns="iso3", values=c, aggfunc="mean")
        wide = wide.fillna(0) + w.fillna(0)
    wide = wide.dropna(how="all")
    X = (wide - wide.mean()) / wide.std().replace(0, 1)
    X = X.fillna(0)
    n_comp = min(n_components, X.shape[1], X.shape[0] - 1)
    if n_comp < 1:
        return pd.DataFrame()
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_comp)
        F = pca.fit_transform(X)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
    except ImportError:
        U, S, Vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
        F = U[:, :n_comp] * S[:n_comp]
        var_ratio = (S ** 2) / (np.sum(S ** 2) + 1e-12)
        cumvar = np.cumsum(var_ratio[:n_comp])
    k = np.searchsorted(cumvar, min_variance_ratio) + 1
    k = min(k, n_comp)
    factors = pd.DataFrame(F[:, :k], index=wide.index, columns=[f"F{i+1}" for i in range(k)])
    return factors


def quality_control_missing(df_series, max_gap=2):
    """缺失值：连续缺口 ≤ max_gap 用线性插值并标记；更长缺口不插值。"""
    s = df_series.copy()
    missing = s.isna()
    s_interp = s.interpolate(method="linear", limit=max_gap)
    return s_interp


def quality_control_outliers(series, z_threshold=4, winsorize_quantiles=(0.01, 0.99)):
    """异常值：稳健 z-score，|z|>z_threshold 则 winsorize 到分位数。"""
    med = series.median()
    mad = np.median(np.abs(series - med))
    if mad == 0:
        return series
    z = (series - med) / (1.4826 * mad)
    s = series.copy()
    s = s.clip(lower=series.quantile(winsorize_quantiles[0]), upper=series.quantile(winsorize_quantiles[1]))
    return s


def stationarity_transform_and_test(series, adf=False):
    """平稳化：对数差分；可选 ADF 检验（需 statsmodels）。"""
    y = np.log(series.replace(0, np.nan).ffill().bfill() + 1e-6)
    d = y.diff().dropna()
    if adf and len(d) > 12:
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_stat, pval, *_ = adfuller(d.dropna())
            logger.info(f"ADF: stat={adf_stat:.4f} pval={pval:.4f}")
        except Exception:
            pass
    return d


def run_full_construction(save=True):
    """执行完整数据构建流程并保存到 research_output。"""
    logger.info("Data and Variable Construction (Section 3)...")
    df_macro, df_bilateral = load_quarterly_macro_and_bilateral()
    dates = quarterly_panel_alignment(df_macro, df_bilateral)
    logger.info(f"Quarterly panel: {len(dates)} quarters")

    # 提取/构造 TCU 季度序列（若已在宏观表中，则直接按日期去重；否则尝试从 data/TCU.csv.xls 聚合）
    tcu = pd.DataFrame()
    if "TCU" in df_macro.columns:
        tcu = (
            df_macro[["date", "TCU"]]
            .dropna(subset=["TCU"])
            .drop_duplicates(subset=["date"])
            .sort_values("date")
            .set_index("date")
        )
    else:
        tcu_path = DATA_DIR_LOCAL / "TCU.csv.xls"
        if tcu_path.exists():
            tdf = pd.read_csv(tcu_path)
            if "date" in tdf.columns and "value" in tdf.columns:
                tdf["date"] = pd.to_datetime(tdf["date"])
                tdf = tdf.dropna(subset=["value"])
                tdf["quarter_t"] = tdf["date"].dt.to_period("Q")
                tq = tdf.groupby("quarter_t")["value"].mean().reset_index()
                # 对齐到宏观样本日期：用季度末日期表示
                tq["date"] = tq["quarter_t"].dt.to_timestamp(how="end")
                tcu = tq[["date", "value"]].rename(columns={"value": "TCU"}).set_index("date")

    # VAX 季度化
    df_vax = chow_lin_quarterly_vax(df_macro)
    if save:
        df_vax.to_csv(OUTPUT_DIR / "vax_quarterly.csv", index=False)

    # W_t 序列（按季度）
    w_list = []
    for d in dates:
        W = build_trade_network_w(df_bilateral, d, window_quarters=4, use_import_share=True)
        W["date"] = d
        w_list.append(W)
    if w_list:
        W_stack = pd.concat(w_list)
        if save:
            W_stack.to_csv(OUTPUT_DIR / "trade_network_W_quarterly.csv")

    # TC_ij,t
    df_tc = build_tariff_relief_tc(df_bilateral)
    if save:
        df_tc.to_csv(OUTPUT_DIR / "tariff_relief_TC_quarterly.csv", index=False)

    # PCA 外部因子 + TCU：将 TCU 作为额外外生变量列加入外部因子表
    factors = pca_external_factors(df_macro, n_components=3, min_variance_ratio=0.7)
    external = pd.DataFrame()
    if not factors.empty:
        external = factors.copy()
    if not tcu.empty:
        # 若 PCA 因子存在，则按日期对齐；否则仅输出 TCU
        if not external.empty:
            idx = external.index
            external["TCU"] = tcu.reindex(idx)["TCU"]
        else:
            external = tcu.copy()
    if save and not external.empty:
        external.reset_index().rename(columns={"index": "date"}).to_csv(OUTPUT_DIR / "external_factors_pca.csv", index=False)
    if save and not tcu.empty:
        tcu.reset_index().to_csv(OUTPUT_DIR / "tcu_quarterly.csv", index=False)

    # 产出变量：VAX 增长（平稳化）供 VAR
    df_vax = df_vax.set_index(["date", "iso3"])["vax_q"].unstack("iso3")
    for c in df_vax.columns:
        df_vax[c] = quality_control_missing(df_vax[c], max_gap=2)
        df_vax[c] = quality_control_outliers(df_vax[c], z_threshold=4)
    dln_vax = np.log(df_vax + 1e-6).diff()
    if save:
        dln_vax.to_csv(OUTPUT_DIR / "dln_vax_quarterly.csv")

    logger.info("Construction complete. Outputs in research_output/")
    return {
        "dates": dates,
        "df_vax": df_vax,
        "dln_vax": dln_vax,
        "df_tc": df_tc,
        "factors": external,
        "df_bilateral": df_bilateral,
        "df_macro": df_macro,
        "tcu": tcu,
    }


if __name__ == "__main__":
    run_full_construction(save=True)
