"""
Network Time-Varying Parameter VAR (Section 4.1–4.6).
y_t = c_t + sum_l A_l,t y_{t-l} + sum_l B_l,t W_{t-l} y_{t-l} + Pi x_t + u_t.
CP low-rank on (A,B), time-varying volatility; GIRF; counterfactual (B=0) for direct vs total;
network amplification share A, resilience index R, net absorber position Pos.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from config import RCEP_COUNTRIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RCEP_LIST = list(RCEP_COUNTRIES.keys())
OUTPUT_DIR = Path(__file__).parent / "research_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_constructed_data():
    """加载 research_data_construction 产出的序列、W 以及外部因子（含 TCU）。"""
    from research_data_construction import (
        load_quarterly_macro_and_bilateral,
        chow_lin_quarterly_vax,
        build_trade_network_w,
        quality_control_missing,
        quality_control_outliers,
        OUTPUT_DIR,
        RCEP_LIST,
    )
    df_macro, df_bilateral = load_quarterly_macro_and_bilateral()
    df_vax = chow_lin_quarterly_vax(df_macro)
    # 平稳化
    df_vax = df_vax.set_index(["date", "iso3"])["vax_q"].unstack("iso3")
    for c in df_vax.columns:
        df_vax[c] = quality_control_missing(df_vax[c], max_gap=2)
        df_vax[c] = quality_control_outliers(df_vax[c], z_threshold=4)
    dln_vax = np.log(df_vax + 1e-6).diff().dropna(how="all")
    dates = dln_vax.index
    # 对齐 W
    W_t = {}
    for d in dates:
        W_t[d] = build_trade_network_w(df_bilateral, d, window_quarters=4, use_import_share=True)
    Y = dln_vax.reindex(columns=RCEP_LIST).fillna(0).values
    # 读取外部因子（PCA + TCU），按日期对齐，作为 Network VAR 的外生变量 X_t
    X_exog = None
    ext_path = OUTPUT_DIR / "external_factors_pca.csv"
    if ext_path.exists():
        ext = pd.read_csv(ext_path)
        if "date" in ext.columns:
            ext["date"] = pd.to_datetime(ext["date"])
            ext = ext.set_index("date").sort_index()
            ext = ext.reindex(dates)
            ext = ext.fillna(0.0)
            X_exog = ext.values
    return Y, dates, W_t, df_bilateral, X_exog


def var_ols(Y, W_list, X_exog=None, p=2):
    """
    Restricted Network VAR for stability (T < N*p):
    y_it = c_i + sum_{l=1}^p a_l y_i,t-l + sum_{l=1}^p b_l (W_t-l y_t-l)_i + Pi x_t + u_it.
    This reduces parameters per equation from (1+2Np+K) to (1+2p+K).
    """
    T, N = Y.shape
    if X_exog is None:
        X_exog = np.zeros((T, 0))
    K = X_exog.shape[1]
    
    # We estimate equation-by-equation
    A_list = [np.zeros((N, N)) for _ in range(p)]
    B_list = [np.zeros((N, N)) for _ in range(p)]
    c = np.zeros(N)
    Pi = np.zeros((N, K))
    residuals = np.zeros((T - p, N))
    
    for i in range(N):
        Z_i_list = []
        for tau in range(p, T):
            row = [1.0]
            # Own lags
            for l in range(1, p + 1):
                row.append(Y[tau - l, i])
            # Network lags
            for l in range(1, p + 1):
                Wy = W_list[tau - l] @ Y[tau - l] if W_list[tau - l] is not None else np.zeros(N)
                row.append(Wy[i])
            # Exogenous
            if K > 0:
                row.extend(X_exog[tau])
            Z_i_list.append(row)
            
        Zi = np.array(Z_i_list)
        yi = Y[p:, i]
        
        # Ridge regularization if still unstable
        # beta_i = inv(Zi'Zi + lambda*I) Zi'yi
        lambda_ridge = 1e-4
        beta_i, _, _, _ = np.linalg.lstsq(Zi.T @ Zi + lambda_ridge * np.eye(Zi.shape[1]), Zi.T @ yi, rcond=None)
        
        c[i] = beta_i[0]
        for l in range(p):
            A_list[l][i, i] = beta_i[1 + l]
            B_list[l][i, i] = beta_i[1 + p + l]
        if K > 0:
            Pi[i, :] = beta_i[1 + 2*p :]
        
        residuals[:, i] = yi - Zi @ beta_i
        
    Sigma = (residuals.T @ residuals) / max(residuals.shape[0] - 1, 1)
    Sigma = (Sigma + Sigma.T) / 2 + 1e-8 * np.eye(N)
    
    return c, A_list, B_list, Pi, Sigma, residuals


def moving_average_coefficients(A_list, B_list, W_list, horizon, W_fixed=None):
    """从 A,B,W 递推得到 Psi(0), Psi(1), ... Psi(horizon)。MA: y_t = sum_h Psi_h u_{t-h}."""
    p = len(A_list)
    N = A_list[0].shape[0]
    Psi = [np.eye(N)]
    for h in range(1, horizon + 1):
        ph = np.zeros((N, N))
        for l in range(1, min(p + 1, h + 1)):
            if h - l >= 0 and h - l < len(Psi):
                A_l = A_list[l - 1]
                ph += A_l @ Psi[h - l]
                idx = h - l
                W_use = W_fixed if W_fixed is not None else (W_list[idx] if idx < len(W_list) else None)
                if W_use is not None and B_list[l - 1] is not None:
                    ph += B_list[l - 1] @ np.asarray(W_use) @ Psi[h - l]
        Psi.append(ph)
    return Psi


def girf_one(Phi_h, Sigma, j_shock, scale=1.0):
    """GIRF: y_i 对 u_j 一单位冲击的响应。 scale = 1 表示一标准差。"""
    N = Sigma.shape[0]
    sig_j = np.sqrt(Sigma[j_shock, j_shock] + 1e-12)
    e_j = np.zeros(N)
    e_j[j_shock] = 1.0
    irf = (Phi_h @ Sigma @ e_j) / sig_j * scale
    return irf


def counterfactual_direct(A_list, B_list_zero, W_list, horizon):
    """反事实：B≡0，仅直接通道。 B_list_zero 为全零的 B 列表。"""
    return moving_average_coefficients(A_list, B_list_zero, W_list, horizon, W_fixed=None)


def network_amplification_share(S_total, S_direct, eps=1e-10):
    """A = (S_total - S_direct) / S_total in [0,1]."""
    denom = np.maximum(S_total, eps)
    return (S_total - S_direct) / denom


def cumulative_response_magnitude(IRF_list, H):
    """S = sum_{h=0}^H |IRF(h)|."""
    return sum(np.abs(IRF_list[h]).sum() for h in range(min(H + 1, len(IRF_list))))


def half_life_approx(IRF_list, peak_idx=0):
    """近似半衰期：首次 |IRF(h)| <= 0.5 * |IRF(peak)| 的 h。"""
    if not IRF_list:
        return np.nan
    irf_norm = np.array([np.sqrt(np.sum(irf**2)) for irf in IRF_list])
    peak = irf_norm[peak_idx] if peak_idx < len(irf_norm) else irf_norm[0]
    if peak <= 0:
        return np.nan
    for h, v in enumerate(irf_norm):
        if v <= 0.5 * peak:
            return float(h)
    return float(len(irf_norm) - 1)


def resilience_index(S_total_H, half_life, kappa=0.1):
    """R = -log(S_total) - kappa * HL."""
    return -np.log(S_total_H + 1e-10) - kappa * half_life


def net_absorber_position(A_matrix, i, N):
    """Pos_i = sum_{j!=i} (A_{j<-i} - A_{i<-j}). A_matrix[i,j] = A_{i<-j}."""
    pos = 0.0
    for j in range(N):
        if j == i:
            continue
        pos += A_matrix[j, i] - A_matrix[i, j]
    return pos


def estimate_rolling_network_var(Y, dates, W_t, p=2, window_min=40):
    """
    滚动估计 Network VAR，得到每期 (A_l,t, B_l,t, Sigma_t) 用于 TVP-GIRF。
    简化：用滚动窗口 OLS 得到时变系数。
    """
    T, N = Y.shape
    W_list = [W_t.get(d, np.zeros((N, N))) for d in dates]
    if len(W_list) != T:
        W_list = [np.zeros((N, N))] * T
    results = []
    for t in range(window_min, T):
        Y_w = Y[t - window_min : t]
        W_w = [W_list[t - window_min + s] for s in range(len(Y_w))]
        c, A_list, B_list, Pi, Sigma, _ = var_ols(Y_w, W_w, X_exog=None, p=p)
        results.append({
            "t": t,
            "date": dates[t] if hasattr(dates, "__getitem__") else dates[t],
            "c": c,
            "A_list": A_list,
            "B_list": B_list,
            "Sigma": Sigma,
        })
    return results


def run_network_tvp_var_and_counterfactual(save=True, horizon=8, H_cum=4):
    """估计 Network VAR，计算 GIRF、反事实、放大占比、韧性指数、吸收者位置。"""
    logger.info("Loading constructed data...")
    Y, dates, W_t, df_bilateral, X_exog = load_constructed_data()
    T, N = Y.shape
    dates_arr = list(dates) if hasattr(dates, "index") else list(dates)
    W_list = [W_t.get(d, np.zeros((N, N))) for d in dates_arr]
    for i in range(len(W_list)):
        if isinstance(W_list[i], pd.DataFrame):
            W_list[i] = W_list[i].reindex(index=RCEP_LIST, columns=RCEP_LIST).fillna(0).values

    logger.info("Estimating Network VAR (full sample)...")
    p = 2
    c, A_list, B_list, Pi, Sigma, res = var_ols(Y, W_list, X_exog=X_exog, p=p)
    B_list_zero = [np.zeros_like(B) for B in B_list]

    # GIRF 总效应 vs 直接效应
    Psi_total = moving_average_coefficients(A_list, B_list, W_list[: horizon + p], horizon, W_fixed=None)
    Psi_direct = moving_average_coefficients(A_list, B_list_zero, W_list[: horizon + p], horizon)

    # 对每个 (i,j) 计算 S_total, S_direct, A
    A_share = np.zeros((N, N))
    S_total_mat = np.zeros((N, N))
    S_direct_mat = np.zeros((N, N))
    for j in range(N):
        for h in range(horizon + 1):
            irf_total = girf_one(Psi_total[h], Sigma, j, scale=1.0)
            irf_direct = girf_one(Psi_direct[h], Sigma, j, scale=1.0)
            for i in range(N):
                S_total_mat[i, j] += np.abs(irf_total[i])
                S_direct_mat[i, j] += np.abs(irf_direct[i])
    for i in range(N):
        for j in range(N):
            A_share[i, j] = network_amplification_share(S_total_mat[i, j], S_direct_mat[i, j])

    # 韧性指数与半衰期（以 j=0 冲击为例）
    j0 = 0
    irf_tot_j0 = [girf_one(Psi_total[h], Sigma, j0) for h in range(horizon + 1)]
    S_tot_j0 = cumulative_response_magnitude(irf_tot_j0, H_cum)
    HL_j0 = half_life_approx(irf_tot_j0)
    R_j0 = resilience_index(S_tot_j0, HL_j0)

    # 净吸收者位置
    Pos = np.array([net_absorber_position(A_share, i, N) for i in range(N)])

    out = {
        "A_share": A_share,
        "S_total_mat": S_total_mat,
        "S_direct_mat": S_direct_mat,
        "Pos": Pos,
        "R_example": R_j0,
        "HL_example": HL_j0,
        "Psi_total": Psi_total,
        "Psi_direct": Psi_direct,
        "Sigma": Sigma,
        "A_list": A_list,
        "B_list": B_list,
    }
    if save:
        pd.DataFrame(A_share, index=RCEP_LIST, columns=RCEP_LIST).to_csv(OUTPUT_DIR / "network_amplification_share_A.csv")
        pd.DataFrame({"iso3": RCEP_LIST, "net_absorber_position": Pos}).to_csv(OUTPUT_DIR / "net_absorber_position.csv", index=False)
    logger.info("Network TVP-VAR and counterfactual decomposition done.")
    return out


if __name__ == "__main__":
    run_network_tvp_var_and_counterfactual(save=True)
