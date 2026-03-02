"""
数据整理：将多个数据集合并为统一的两张主表，并移除未整理数据。
产出：
  - data/master_macro_quarterly.csv  一国一期一行，仅 RCEP 15 国、统一季度
  - data/master_bilateral_quarterly.csv  双边一期一行，仅 RCEP 对、统一季度
数据原则：仅使用真实数据；重复列去重；国家统一 ISO3；时间统一 quarter_t。
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
from config import RCEP_COUNTRIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RCEP_LIST = list(RCEP_COUNTRIES.keys())
OUTPUT_DIR = REPO_ROOT / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR = REPO_ROOT / "data" / "archive"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def _find_file(*names):
    """优先 data/（合并后），再 repo 根、data_acquisition/data。"""
    for name in names:
        for base in [REPO_ROOT / "data", REPO_ROOT, REPO_ROOT.parent / "data_acquisition" / "data"]:
            p = base / name
            if p.exists():
                return p
    return None


def _date_to_quarter(ser):
    return pd.to_datetime(ser).dt.to_period("Q")


def load_quarterly_macro():
    """加载季度宏观，统一列名与 ISO3。"""
    path = _find_file("master_quarterly_macro_2005_2024.csv")
    if path is None:
        logger.warning("master_quarterly_macro not found")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df.get("date_quarterly", df.get("date", pd.NaT)))
    df["quarter_t"] = _date_to_quarter(df["date"])
    df["iso3"] = df["iso3"].astype(str).str.upper().str.strip()
    df = df[df["iso3"].isin(RCEP_LIST)]
    # 去掉纯重复列
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def load_quarterly_bilateral():
    """加载季度双边，统一列名与 ISO3。"""
    path = _find_file("master_quarterly_bilateral_2005_2024.csv")
    if path is None:
        logger.warning("master_quarterly_bilateral not found")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df.get("date_quarterly", df.get("date", pd.NaT)))
    df["quarter_t"] = _date_to_quarter(df["date"])
    for c in ["reporter_iso", "partner_iso"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper().str.strip()
    df = df[df["reporter_iso"].isin(RCEP_LIST) & df["partner_iso"].isin(RCEP_LIST)]
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def load_annual_macro():
    """加载年度宏观，用于补充或校验。"""
    path = _find_file("master_macro_panel_2005_2024.csv")
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["iso3"] = df["iso3"].astype(str).str.upper().str.strip()
    df = df[df["iso3"].isin(RCEP_LIST)]
    return df.loc[:, ~df.columns.duplicated()]


def load_annual_bilateral():
    """加载年度双边。"""
    path = _find_file("master_bilateral_trade_2005_2024.csv")
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    for c in ["reporter_iso", "partner_iso"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper().str.strip()
    df = df[df["reporter_iso"].isin(RCEP_LIST) & df["partner_iso"].isin(RCEP_LIST)]
    return df.loc[:, ~df.columns.duplicated()]


def load_macro_wdi_long():
    """WDI 长表：country_code, year, indicator_name, value → 仅作补充时按需合并。"""
    path = _find_file("macro_wdi_2005_2024_20260110.csv")
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["iso3"] = df.get("country_code", df.get("iso3", "")).astype(str).str.upper().str.strip()
    df = df[df["iso3"].isin(RCEP_LIST)]
    return df


def load_tcu_quarterly_for_merge():
    """
    从 data/TCU.csv.xls 读取月度 TCU 序列，并按季度平均聚合为 (quarter_t, TCU)，用于合并到宏观主表。
    TCU 视为全球指数，对所有国家相同，仅按季度对齐。
    """
    path = OUTPUT_DIR / "TCU.csv.xls"
    if not path.exists():
        # 兼容从 repo 根或 data_acquisition/data 放置的情况
        path = _find_file("TCU.csv.xls")
        if path is None:
            return pd.DataFrame()
    df = pd.read_csv(path)
    if "date" not in df.columns or "value" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["value"])
    df["quarter_t"] = df["date"].dt.to_period("Q")
    tcu_q = df.groupby("quarter_t")["value"].mean().reset_index()
    tcu_q = tcu_q.rename(columns={"value": "TCU"})
    return tcu_q


def merge_macro_quarterly():
    """合并为一张季度宏观主表：iso3, quarter_t, date_quarterly, year, quarter, + 宏观变量 + 全局 TCU。"""
    df = load_quarterly_macro()
    if df.empty:
        return pd.DataFrame()
    # 统一列名
    if "date_quarterly" not in df.columns and "date" in df.columns:
        df["date_quarterly"] = df["date"]
    cols_order = ["iso3", "quarter_t", "date_quarterly", "year", "quarter"]
    extra = [c for c in df.columns if c not in cols_order]
    df = df[cols_order + extra]
    # 合并 TCU（若存在）：按 quarter_t 加在所有国家行上
    tcu_q = load_tcu_quarterly_for_merge()
    if not tcu_q.empty:
        df = df.merge(tcu_q, on="quarter_t", how="left")
    df = df.sort_values(["iso3", "quarter_t"]).drop_duplicates(subset=["iso3", "quarter_t"], keep="first")
    return df


def merge_bilateral_quarterly():
    """合并为一张季度双边主表：reporter_iso, partner_iso, quarter_t, date_quarterly, + 贸易与关税变量。"""
    df = load_quarterly_bilateral()
    if df.empty:
        return pd.DataFrame()
    if "date_quarterly" not in df.columns and "date" in df.columns:
        df["date_quarterly"] = df["date"]
    if "year" not in df.columns:
        df["year"] = df["quarter_t"].astype(str).str[:4].astype(int)
    if "quarter" not in df.columns:
        df["quarter"] = df["quarter_t"].astype(str).str[-1].astype(int)
    cols_order = ["reporter_iso", "partner_iso", "quarter_t", "date_quarterly", "year", "quarter"]
    extra = [c for c in df.columns if c not in cols_order]
    df = df[cols_order + extra]
    df = df.sort_values(["reporter_iso", "partner_iso", "quarter_t"]).drop_duplicates(
        subset=["reporter_iso", "partner_iso", "quarter_t"], keep="first"
    )
    return df


def run_merge(save=True):
    """执行合并并写入 data/。"""
    logger.info("Merging datasets...")
    macro = merge_macro_quarterly()
    bilateral = merge_bilateral_quarterly()
    if macro.empty and bilateral.empty:
        logger.error("No data to merge")
        return
    if save:
        if not macro.empty:
            for name in ["master_macro_quarterly.csv", "master_quarterly_macro_2005_2024.csv"]:
                out_m = OUTPUT_DIR / name
                macro.to_csv(out_m, index=False)
            logger.info("Saved macro: %s", macro.shape)
        if not bilateral.empty:
            for name in ["master_bilateral_quarterly.csv", "master_quarterly_bilateral_2005_2024.csv"]:
                out_b = OUTPUT_DIR / name
                bilateral.to_csv(out_b, index=False)
            logger.info("Saved bilateral: %s", bilateral.shape)
        with open(OUTPUT_DIR / "data_manifest.txt", "w") as f:
            f.write("RCEP data (merged, single source)\n")
            f.write("master_macro_quarterly.csv / master_quarterly_macro_2005_2024.csv: (iso3, quarter_t, ...)\n")
            f.write("master_bilateral_quarterly.csv / master_quarterly_bilateral_2005_2024.csv: (reporter_iso, partner_iso, quarter_t, ...)\n")
    return {"macro": macro, "bilateral": bilateral}


def archive_and_remove_duplicates(dry_run=False):
    """
    将 repo 根目录及 data_acquisition/data 下的原始/重复 CSV 移入 data/archive，
    合并后的主表仅保留在 data/。dry_run=True 只列不删。
    """
    import shutil
    to_archive = [
        REPO_ROOT / "master_quarterly_macro_2005_2024.csv",
        REPO_ROOT / "master_quarterly_bilateral_2005_2024.csv",
        REPO_ROOT / "master_macro_panel_2005_2024.csv",
        REPO_ROOT / "master_bilateral_trade_2005_2024.csv",
        REPO_ROOT / "34_years_world_export_import_dataset.csv",
        REPO_ROOT / "macro_wdi_2005_2024_20260110.csv",
        REPO_ROOT / "bri_macro_trade_tariff_2013_2023.csv",
        REPO_ROOT / "global_supply_chain_disruption_v1.csv",
    ]
    # 仅从 repo 根目录移除重复；data_acquisition/data 仅做归档备份不删，避免影响依赖 DATA_DIR 的脚本
    data_acq = REPO_ROOT.parent / "data_acquisition" / "data"
    if data_acq.exists():
        for f in data_acq.glob("*.csv"):
            to_archive.append(f)
    moved = 0
    for path in to_archive:
        if not path.exists():
            continue
        dest = ARCHIVE_DIR / path.name
        if dest.resolve() == path.resolve():
            continue
        if dry_run:
            logger.info("Would archive: %s", path)
            moved += 1
            continue
        try:
            shutil.copy2(path, dest)
            # 仅删除 repo 根目录下的重复文件；data_acquisition 下只备份不删
            if path.parent == REPO_ROOT:
                path.unlink()
                logger.info("Archived and removed: %s", path.name)
            else:
                logger.info("Archived (kept original): %s", path.name)
            moved += 1
        except Exception as e:
            logger.warning("Skip %s: %s", path.name, e)
    logger.info("Archived/removed %d files. Originals in data/archive/", moved)
    return moved


if __name__ == "__main__":
    run_merge(save=True)
    archive_and_remove_duplicates(dry_run=False)

