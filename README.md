# Who Absorbs the Shock?
## RCEP and Reverse Shock Resilience in the Asia-Pacific Trade Network

本仓库用于复现与扩展 RCEP 区域贸易网络韧性研究，覆盖数据采集、数据清洗、网络构建、计量分析和图表产出全流程。

## Repository Structure

- `repo/`: 研究主代码（建模、回归、图表、稳健性分析）
- `data_acquisition/`: 原始/中间数据采集与整理
- `research_output/`: 关键结果文件（表格、图、bootstrap 结果）

## Quick Start

```bash
# 1) 安装依赖
pip install -r repo/requirements.txt

# 2) 运行主流程
python repo/run_research_pipeline.py
```

如需分步骤运行，可按 `repo/` 与 `data_acquisition/` 下脚本编号顺序执行。

## Data Notes

- 仓库已提交大部分研究相关数据与代码。
- GitHub 单文件限制为 100MB，因此 `world_development_indicators.csv`（约 218MB）无法以原始单文件直接提交。
- 为尽可能完整保留数据，仓库提供了分片文件：
  - `data_acquisition/data/large_files/world_development_indicators.csv.part-000`
  - `data_acquisition/data/large_files/world_development_indicators.csv.part-001`
  - `data_acquisition/data/large_files/world_development_indicators.csv.part-002`

可在本地还原为原始 CSV：

```bash
cat data_acquisition/data/large_files/world_development_indicators.csv.part-* \
  > data_acquisition/data/world_development_indicators.csv
```

## Reproducibility

- 建议 Python 3.10+。
- 运行前请检查 `repo/config.py` 中路径配置。
- 主要结果位于 `research_output/` 与 `repo/research_output/`。
