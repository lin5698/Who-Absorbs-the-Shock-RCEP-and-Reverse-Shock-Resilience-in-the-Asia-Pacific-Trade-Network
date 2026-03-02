## Summary
This PR adds missing analysis deliverables requested for the RCEP reverse-shock resilience paper pipeline:
1. Event-study / leads-lags figure around tariff phase-ins.
2. Structural break test table for aggregated and selected pair-level resilience series.
3. Baseline coefficient stability table under fixed-weight network (`W_pre` vs `W_t`).
4. Alternative absorber-position figure comparing amplification-share vs cumulative-response constructions.

It also ensures figure outputs include both PDF and PNG formats for submission and web display.

## What Changed
- Added a new script:
  - `research_additional_requested_outputs.py`
- Added generated outputs:
  - `research_output/nature_figures/Fig_Event_Study_LeadsLags.pdf`
  - `research_output/nature_figures/Fig_Event_Study_LeadsLags.png`
  - `research_output/nature_figures/Fig_Alternative_Absorber_Position_Measures.pdf`
  - `research_output/nature_figures/Fig_Alternative_Absorber_Position_Measures.png`
  - `research_output/nature_figures/Table_Baseline_FixedWeight_Stability.csv`
  - `research_output/nature_figures/Table_Structural_Break_Tests.csv`

## Methods/Implementation Notes
- Event-study:
  - Pair and time fixed effects with clustered SE by pair.
  - Event time defined using pair-specific first major tariff step-down.
  - Leads/lags window: `[-8, +8]` quarters (baseline omitted at `k=-1`).
  - `drop_absorbed=True` is used to handle absorbed terms under FE.
- Structural breaks:
  - Dynamic programming segmentation with BIC-based break-count selection.
  - Approximate break confidence intervals reported as neighboring-quarter windows.
  - Includes aggregated resilience and selected CHN-related pair-level series.
- Fixed-weight stability:
  - Compares baseline regression coefficient under `W_t` vs `W_pre`.
  - Reports coefficient, SE, t-stat, p-value, N, R², and relative difference/ratio.

## Validation
- Script executed successfully and produced all listed outputs.
- Existing `nature_figures` PDF files were checked and all have PNG counterparts.

## Diff Scope
- Commit: `b5ebf5d`
- Files changed: `7`
- Insertions: `354`
- Deletions: `0`
- No unrelated files modified in this PR commit.
