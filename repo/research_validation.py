"""
Model Validation and Evaluation (Section 4.7).
Rolling/expanding CV, h=1 and h=4; RMSE, MAE, R²_oos vs AR(1), static VAR, no-network VAR.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent))
from research_data_construction import (
    load_quarterly_macro_and_bilateral,
    chow_lin_quarterly_vax,
    build_trade_network_w,
    quality_control_missing,
    quality_control_outliers,
    RCEP_LIST,
)
from research_network_tvp_var import var_ols, load_constructed_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "research_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def forecast_ar1(Y, t, h):
    """AR(1) h-step forecast at origin t (recursive)."""
    rho = np.corrcoef(Y[: t, :].flat[1:], Y[: t, :].flat[:-1])[0, 1] if t > 2 else 0
    if np.isnan(rho):
        rho = 0
    y_t = Y[t - 1]
    return y_t * (rho ** h)


def forecast_var_static(Y, W_list, t, h, p=2):
    """Static VAR (no TVP) h-step forecast at origin t."""
    Y_w = Y[:t]
    W_w = [W_list[s] for s in range(t)]
    c, A_list, B_list, _, _, _ = var_ols(Y_w, W_w, X_exog=None, p=p)
    y_pred = Y[t - 1].copy()
    for _ in range(h):
        y_new = c.copy()
        for l in range(p):
            idx = t - 1 - l
            if idx >= 0:
                y_new += A_list[l] @ Y_w[idx]
                if idx < len(W_w):
                    y_new += B_list[l] @ W_w[idx] @ Y_w[idx]
        y_pred = y_new
        Y_w = np.vstack([Y_w, y_pred])
        W_w.append(W_w[-1] if W_w else np.zeros((Y.shape[1], Y.shape[1])))
    return y_pred


def forecast_var_no_network(Y, t, h, p=2):
    """VAR without network term (B=0) h-step forecast."""
    N = Y.shape[1]
    W_zero = [np.zeros((N, N)) for _ in range(t)]
    return forecast_var_static(Y, W_zero, t, h, p=p)


def rolling_cv(Y, dates, W_list, h_steps=(1, 4), min_train=40, p=2):
    """
    Rolling CV: at each origin tau, fit on [0, tau], forecast h-step; collect errors.
    Returns dict model_name -> (h -> list of (y_true, y_pred) per country).
    """
    T, N = Y.shape
    results = {m: {h: {"y_true": [], "y_pred": []} for h in h_steps} for m in ["AR1", "VAR_static", "VAR_no_net"]}
    for tau in range(min_train + max(h_steps), T):
        for h in h_steps:
            if tau + h >= T:
                continue
            y_true = Y[tau + h - 1]
            # AR(1) per country
            pred_ar1 = np.array([forecast_ar1(Y[:, i : i + 1], tau, h) for i in range(N)]).flatten()
            results["AR1"][h]["y_true"].append(y_true)
            results["AR1"][h]["y_pred"].append(pred_ar1)
            # VAR static
            try:
                pred_var = forecast_var_static(Y, W_list, tau, h, p=p)
                results["VAR_static"][h]["y_true"].append(y_true)
                results["VAR_static"][h]["y_pred"].append(pred_var)
            except Exception:
                results["VAR_static"][h]["y_true"].append(y_true)
                results["VAR_static"][h]["y_pred"].append(np.nan * y_true)
            # VAR no network
            try:
                pred_non = forecast_var_no_network(Y, tau, h, p=p)
                results["VAR_no_net"][h]["y_true"].append(y_true)
                results["VAR_no_net"][h]["y_pred"].append(pred_non)
            except Exception:
                results["VAR_no_net"][h]["y_true"].append(y_true)
                results["VAR_no_net"][h]["y_pred"].append(np.nan * y_true)
    return results


def compute_metrics(results, benchmark="AR1"):
    """RMSE, MAE, R²_oos per model and h."""
    metrics = []
    for model in results:
        for h in results[model]:
            yt = np.array(results[model][h]["y_true"])
            yp = np.array(results[model][h]["y_pred"])
            yp = np.nan_to_num(yp, nan=0.0)
            n = yt.size
            if n == 0:
                continue
            rmse = np.sqrt(np.mean((yt - yp) ** 2))
            mae = np.mean(np.abs(yt - yp))
            ss_res = np.sum((yt - yp) ** 2)
            ss_tot = np.sum((yt - np.mean(yt)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-12)
            yt_bench = np.array(results[benchmark][h]["y_true"])
            yp_bench = np.array(results[benchmark][h]["y_pred"])
            yp_bench = np.nan_to_num(yp_bench, nan=0.0)
            ss_bench = np.sum((yt_bench - yp_bench) ** 2)
            r2_oos = 1 - ss_res / (ss_bench + 1e-12)
            metrics.append({"model": model, "h": h, "RMSE": rmse, "MAE": mae, "R2": r2, "R2_oos": r2_oos})
    return pd.DataFrame(metrics)


def run_validation(save=True):
    """Run rolling CV and save Table 2 (forecast evaluation)."""
    logger.info("Loading data for validation...")
    # load_constructed_data 现在返回 (Y, dates, W_t, df_bilateral, X_exog)
    Y, dates, W_t, _, _ = load_constructed_data()
    dates_arr = list(dates) if hasattr(dates, "index") else list(dates)
    W_list = [W_t.get(d, np.zeros((Y.shape[1], Y.shape[1]))) for d in dates_arr]
    for i in range(len(W_list)):
        if hasattr(W_list[i], "values"):
            W_list[i] = W_list[i].values
        W_list[i] = np.nan_to_num(W_list[i], nan=0.0)

    logger.info("Running rolling CV (h=1, 4)...")
    results = rolling_cv(Y, dates_arr, W_list, h_steps=(1, 4), min_train=40, p=2)
    df_metrics = compute_metrics(results, benchmark="AR1")
    if save:
        df_metrics.to_csv(OUTPUT_DIR / "table2_forecast_evaluation.csv", index=False)
    logger.info("Validation complete. Table 2 saved.")
    return df_metrics


if __name__ == "__main__":
    run_validation(save=True)
