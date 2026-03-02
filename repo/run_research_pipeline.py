"""
主流程：按 Section 3 & 4 顺序执行数据构建 → Network TVP-VAR 与反事实 → 验证 → 图表与表格。
对应方法与数据章节：15 个 RCEP 成员、季度面板、VAX 季度化、W 与 TC、Network-TVP-VAR + 反事实分解。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    print("=" * 60)
    print("Research pipeline: Step0 → Data → Model → Validation → Figures/Tables")
    print("=" * 60)

    # 0) Step 0: 统一频率与对齐 (2000Q1–2023Q4)
    print("\n[0/5] Step 0: Quarterly panel alignment (master index, country codes, Q/M/A rules, missing matrix)...")
    from research_step0_alignment import run_step0_alignment
    run_step0_alignment(save=True)

    # 1) Data and Variable Construction (Section 3)
    print("\n[1/5] Data construction (quarterly panel, VAX, W, TC, PCA, QC)...")
    from research_data_construction import run_full_construction
    run_full_construction(save=True)

    # 2) Network TVP-VAR and counterfactual (Section 4.1–4.6)
    print("\n[2/5] Network TVP-VAR and counterfactual decomposition...")
    from research_network_tvp_var import run_network_tvp_var_and_counterfactual
    run_network_tvp_var_and_counterfactual(save=True)

    # 3) Model validation (Section 4.7)
    print("\n[3/5] Rolling CV and forecast evaluation (Table 2)...")
    from research_validation import run_validation
    run_validation(save=True)

    # 4) Figures and Tables
    print("\n[4/5] Figures 1–3 and Table 1...")
    from research_figures_tables import run_all_figures_and_tables
    run_all_figures_and_tables()

    print("\n" + "=" * 60)
    print("Pipeline complete. Outputs in repo/research_output/")
    print("=" * 60)


if __name__ == "__main__":
    main()
