# Research Model Outputs (Section 3 & 4)

本目录存放「方法与数据」章节对应研究模型的输出，可直接用于主文图表与表格。

## Step 0：统一频率与对齐 (Quarterly panel alignment)

- `step0_master_quarter_index.csv` — 主时间轴 2000Q1–2023Q4（year, quarter, quarter_start, quarter_end）
- `step0_country_code_table.csv` — 国家编码表（ISO3 + iso2/comtrade 等映射）
- `step0_aligned_panel.csv` — 对齐后的季度面板（iso3, quarter_t, 各变量）；以骨架 outer join 得到
- `step0_missing_matrix.csv` — 缺失矩阵（每国每变量缺多少期）
- `step0_break_record.csv` — 断点记录表（国家、季度、变量、处理方式）
- `step0_validation_report.txt` — 验收结果（每国季度数、VAX 四季度和=年度等）

聚合规则（写死）：Q→Q 直接映射；M→Q 流量=和、指数/价格=平均、期末=季末；A→Q Chow-Lin 且四季度和=年度。缺失 ≤2 期线性插值，>2 期不插值。

## 数据与变量构建 (Section 3)

- `vax_quarterly.csv` — 季度 VAX（Chow–Lin 季度化，真实年度约束）
- `trade_network_W_quarterly.csv` — 贸易网络权重矩阵 W_t（行随机、零对角）
- `tariff_relief_TC_quarterly.csv` — RCEP 关税减让强度 TC_ij,t
- `external_factors_pca.csv` — 外部共同因子（PCA）
- `dln_vax_quarterly.csv` — 平稳化后的 VAX 增长率（供 VAR）

## Network TVP-VAR 与反事实 (Section 4.1–4.6)

- `network_amplification_share_A.csv` — 网络放大占比 A_{i←j}(H)
- `net_absorber_position.csv` — 各国净吸收者位置 Pos_i,t

## 模型验证 (Section 4.7)

- `table2_forecast_evaluation.csv` — 滚动 CV：RMSE / MAE / R²_oos（h=1, 4），基准 AR(1)、静态 VAR、无网络 VAR

## 主文图表与表格

- `figures/figure1_tariff_path.png` — Figure 1：RCEP 关税分期减让路径
- `figures/figure2_amplification_share.png` — Figure 2：网络放大占比分布
- `figures/figure3_absorber_position.png` — Figure 3：Who absorbs the shock（净吸收者位置）
- `table1_baseline_regression.csv` — Table 1：基准回归 A ~ TC + FE

## 运行方式

在 `repo` 目录下执行：

```bash
python3 run_research_pipeline.py
```

依赖：本项目既有 `config`、`data/` 下季度宏观与双边数据；可选 `sklearn`（PCA）、`scipy`（统计量）。
