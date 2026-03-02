"""
Figures and Tables for main text (Section 3 / Section 4).
Fig1: RCEP tariff phase-in path (TC by quarter, key pairs).
Fig2: Network amplification share (total vs direct, over time or distribution).
Fig3: Who absorbs the shock (net absorber position).
Table1: Baseline regression A_ij,t ~ TC_ij,t + bilateral FE + time FE.
Table2: Forecast evaluation (from research_validation).
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import logging

sys.path.insert(0, str(Path(__file__).parent))
from config import RCEP_COUNTRIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "research_output"
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RCEP_LIST = list(RCEP_COUNTRIES.keys())


def load_tc_and_amplification():
    """Load tariff relief and amplification share from construction + model output."""
    tc_path = OUTPUT_DIR / "tariff_relief_TC_quarterly.csv"
    a_path = OUTPUT_DIR / "network_amplification_share_A.csv"
    pos_path = OUTPUT_DIR / "net_absorber_position.csv"
    df_tc = pd.read_csv(tc_path) if tc_path.exists() else pd.DataFrame()
    A_df = pd.read_csv(a_path, index_col=0) if a_path.exists() else pd.DataFrame()
    pos_df = pd.read_csv(pos_path) if pos_path.exists() else pd.DataFrame()
    return df_tc, A_df, pos_df


def figure1_tariff_path(df_tc, key_pairs=None):
    """Figure 1: RCEP tariff phase-in path (TC by quarter)."""
    if df_tc.empty:
        logger.warning("No TC data for Figure 1.")
        return
    df_tc["date"] = pd.to_datetime(df_tc["date"])
    if key_pairs is None:
        key_pairs = [("CHN", "JPN"), ("CHN", "KOR"), ("CHN", "VNM")]
    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 5))
    for (i, j) in key_pairs:
        sub = df_tc[(df_tc["reporter_iso"] == i) & (df_tc["partner_iso"] == j)]
        if sub.empty:
            sub = df_tc[(df_tc["reporter_iso"] == j) & (df_tc["partner_iso"] == i)]
        if not sub.empty:
            sub = sub.sort_values("date")
            ax.plot(sub["date"], sub["TC"], label=f"{i}-{j}", alpha=0.9)
    ax.axvline(pd.Timestamp("2022-01-01"), color="gray", linestyle="--", alpha=0.8, label="RCEP effective")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Tariff relief (TC)")
    ax.set_title("Figure 1: RCEP Tariff Phase-in Path (Key Bilateral Pairs)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure1_tariff_path.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Figure 1 saved.")


def figure2_amplification_share(A_df):
    """Figure 2: Network amplification share (distribution or time series)."""
    if A_df.empty:
        logger.warning("No amplification share data for Figure 2.")
        return
    plt.close("all")
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = A_df.values.flatten()
    vals = vals[np.isfinite(vals) & (vals >= 0) & (vals <= 1)]
    if len(vals) > 0:
        ax.hist(vals, bins=30, density=True, alpha=0.7, color="steelblue", edgecolor="white")
        ax.axvline(np.median(vals), color="red", linestyle="--", label=f"Median = {np.median(vals):.3f}")
    ax.set_xlabel("Network amplification share A")
    ax.set_ylabel("Density")
    ax.set_title("Figure 2: Distribution of Network Amplification Share (Total vs Direct)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure2_amplification_share.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Figure 2 saved.")


def figure3_absorber_position(pos_df):
    """Figure 3: Who absorbs the shock (net absorber position by country)."""
    if pos_df.empty:
        logger.warning("No net absorber position data for Figure 3.")
        return
    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 6))
    pos_df = pos_df.sort_values("net_absorber_position", ascending=True)
    colors = ["#2ecc71" if x >= 0 else "#e74c3c" for x in pos_df["net_absorber_position"]]
    ax.barh(pos_df["iso3"], pos_df["net_absorber_position"], color=colors, alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Net absorber position (Pos_i)")
    ax.set_ylabel("Country (ISO3)")
    ax.set_title("Figure 3: Who Absorbs the Shock? (Net Absorber Position)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure3_absorber_position.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Figure 3 saved.")


def table1_baseline_regression(df_tc, A_df):
    """Table 1: A_ij ~ TC_ij + bilateral FE + time FE. (Simplified: cross-section or panel)."""
    if A_df.empty or df_tc.empty:
        logger.warning("Insufficient data for Table 1.")
        return pd.DataFrame()
    # Flatten A to (i,j) pairs; merge TC by (reporter, partner) and time
    A_flat = []
    for i in A_df.index:
        for j in A_df.columns:
            if i != j:
                A_flat.append({"reporter_iso": i, "partner_iso": j, "A": A_df.loc[i, j]})
    df_a = pd.DataFrame(A_flat)
    df_tc["date"] = pd.to_datetime(df_tc["date"])
    tc_agg = df_tc.groupby(["reporter_iso", "partner_iso"])["TC"].mean().reset_index()
    df = df_a.merge(tc_agg, on=["reporter_iso", "partner_iso"], how="left")
    df["TC"] = df["TC"].fillna(0)
    df["pair"] = df["reporter_iso"] + "_" + df["partner_iso"]
    X = np.column_stack([np.ones(len(df)), df["TC"].fillna(0).values])
    y = df["A"].values
    beta, res, rank, s = np.linalg.lstsq(X, y, rcond=None)
    n, k = X.shape
    mse = np.sum((y - X @ beta) ** 2) / max(n - k, 1)
    var_beta = mse * np.linalg.inv(X.T @ X + 1e-10 * np.eye(k))
    se = np.sqrt(np.diag(var_beta)) + 1e-12
    t = beta / se
    from scipy import stats as scipy_stats
    p = 2 * (1 - scipy_stats.t.cdf(np.abs(t), n - k))
    table1 = pd.DataFrame({
        "coef": beta,
        "std_err": se,
        "t": t,
        "p": p,
    }, index=["const", "TC"])
    table1.to_csv(OUTPUT_DIR / "table1_baseline_regression.csv")
    logger.info("Table 1 saved.")
    return table1


def run_all_figures_and_tables():
    """Generate all main-text figures and tables."""
    df_tc, A_df, pos_df = load_tc_and_amplification()
    figure1_tariff_path(df_tc)
    figure2_amplification_share(A_df)
    figure3_absorber_position(pos_df)
    table1_baseline_regression(df_tc, A_df)
    # Table 2 is produced by research_validation
    logger.info("Figures and tables done.")


if __name__ == "__main__":
    run_all_figures_and_tables()
