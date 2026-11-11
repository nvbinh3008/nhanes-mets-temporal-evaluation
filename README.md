# Paper 3 reproducibility materials RC2

This release candidate accompanies *Survey-Aware Temporal Evaluation of a Frozen Model for Prevalent Metabolic Syndrome in NHANES: Prevalence-Referenced Performance and Measurement Sensitivity*. The article uses a frozen R5 model to examine the effect of prevalence benchmarks, outcome measurement and survey design on evaluation across NHANES cycles. The development period is 2007–2014 and the temporal evaluation is 2015–2018. The original primary analysis uses Day 1 dietary data. The model is a methodological case study.

The related study by Van-Binh Ngo and Thi-Tinh Nguyen developed and evaluated the diet-only HEI-2020 rule representation. The authors report that EAI FISAT 2026 accepted it and received the camera-ready manuscript. Publication remains unconfirmed. This repository documents the use of that representation within the separately developed 27-feature R5 model.

Development performance uses predictions from multiple fitted out-of-fold objects. Temporal predictions come from a single final model fitted on the development cohort. The survey uncertainty is conditional on these fixed predictions. Archived hashes identify artifacts and do not independently establish when the authors first accessed temporal outcomes. This study does not establish present-day external validity or clinical usefulness.

## Contents

| Path | Contents |
|---|---|
| `src/` | NHANES ETL, FPED/HEI, orchestration and modeling code |
| `config/` | Harmonization and model settings. The workbook and exact medication map are held for controlled review. The drug-map example contains headers only. |
| `specifications/` | Frozen feature order, monotonic directions, fitted preprocessing summaries, thresholds, model parameters and prior local verification summary |
| `results/aggregate_tables/` and `figures/` | Screened audited group-level results and selected audited figures |
| `scripts/` and `tests/` | Aggregate table formatting, hash checks, smoke tests and archived data-dependent research procedures |
| `requirements*.txt` and `pyproject.toml` | Pinned Windows environment and installed entry points |

Obtain public source inputs directly from [CDC/NCHS NHANES](https://wwwn.cdc.gov/nchs/nhanes/), [USDA FPED](https://www.ars.usda.gov/northeast-area/beltsville-md-bhnrc/beltsville-human-nutrition-research-center/food-surveys-research-group/docs/fped-doc-and-datasets/) and [USDA FNDDS](https://www.ars.usda.gov/northeast-area/beltsville-md-bhnrc/beltsville-human-nutrition-research-center/food-surveys-research-group/docs/fndds/). Linkage and ETL outputs are not distributed.

## Verified workflow

Tested on Windows x64 with CPython 3.12.14. Start in the repository root. Run the strict integrity check before creating the environment or writing any outputs. Use an installed 64-bit Python 3.12 interpreter.

```powershell
python scripts/verify_release.py --strict
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --require-hashes --only-binary=:all: -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/smoke_check.py
.\.venv\Scripts\python.exe scripts/regenerate_tables.py --output outputs/tables --check
```

The local check installed all pinned wheels from an existing wheel cache in a new environment. The first pip command uses your configured index. An offline installation can add `--no-index --find-links <trusted-wheel-directory>` if the exact Windows wheels are already available. The hashes apply to Windows x64 CPython 3.12 wheels. Linux and macOS are unverified. Hosted GitHub Actions has not run on this candidate. Its checkout, Python setup and remote index steps require verification on the actual repository.

The smoke check imports the installed code, parses the configuration, exercises `nhanes-etl`, `hei-pipeline`, `nhanes-full` and `nhanes-model` help, checks configuration and the expected absence of the controlled workbook and data with `nhanes-full preflight`, and runs weighted metric, PSU bootstrap and missing-input tests with synthetic data. The last command regenerates the displayed values and interval formatting of Tables 3 and 5 as two CSVs from the existing audited aggregates. It does not recalculate estimates or confidence intervals.

## Historical and restricted procedures

Read [execution scope](EXECUTION_SCOPE.md) for input requirements. The research scripts reference the original authorized archive, including linked matrices, development OOF predictions, temporal predictions and frozen model objects. Historical outputs and results cannot be reconstructed with the empty drug-map example. In RC2 the installed modeling CLI refuses temporal-evaluate, nested-validate, run-development and fit-final before reading their config. See [bootstrap provenance](BOOTSTRAP_PROVENANCE.md) for the corrected outputs. Data-dependent helper scripts remain for inspection and authorized research use. The commands above require no study records and perform no fitting, calibration, data download or model loading.

The previous private check recorded agreement of archived and regenerated frozen probabilities. Its estimator load warnings reported scikit-learn 1.9.0 objects under 1.9.1. The current environment is for verification of the release software, not a reconstructed historical fitting environment. Independent raw-data-to-results reproduction has not been established.

## Citation and reproducibility reports

Use the approved versioned software citation and cite the associated manuscript as appropriate. See `CITATION.cff`. To report an issue through the project’s GitHub issue tracker when available, include the release version, OS, Python version, exact command and a redacted error or synthetic example. Omit participant records, private paths, archives and credentials.
