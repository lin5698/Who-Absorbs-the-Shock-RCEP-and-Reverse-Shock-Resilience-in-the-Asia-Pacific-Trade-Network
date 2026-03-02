# 数据目录（合并后唯一来源）

本目录为整理后的**唯一数据存放位置**：多个原始数据集已合并为两张主表，未整理数据已归档。

## 主表（合并后）

| 文件 | 说明 | 行 |
|------|------|---|
| **master_macro_quarterly.csv** | 季度宏观面板：`iso3`, `quarter_t`, `date_quarterly`, `year`, `quarter` + 宏观变量（含全球外部因子 `TCU`）；仅 RCEP 15 国 | 一国一期一行 |
| **master_bilateral_quarterly.csv** | 季度双边贸易：`reporter_iso`, `partner_iso`, `quarter_t`, 贸易/关税等；仅 RCEP 对 | 双边一期一行 |

兼容别名（内容相同，供旧脚本使用）：

- `master_quarterly_macro_2005_2024.csv` ≡ master_macro_quarterly.csv  
- `master_quarterly_bilateral_2005_2024.csv` ≡ master_bilateral_quarterly.csv  

## 归档

- **data/archive/**：原始/未整理 CSV 的备份（repo 根目录下的重复文件已移除，此处仅作保留）。

## 重新整理

```bash
cd repo && python3 scripts/merge_and_clean_data.py
```

会从 `data/` 或 `data_acquisition/data` 读取现有 CSV，合并后写回本目录并更新归档。
