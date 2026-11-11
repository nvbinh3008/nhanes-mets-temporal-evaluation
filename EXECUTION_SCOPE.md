# Execution scope

The verified public commands are hash verification, pinned installation, CLI help and preflight, synthetic metric tests, and formatting Tables 3 and 5. A successful preflight reports available YAML configuration and the expected absence of the controlled workbook and study inputs. The public environment and checks exercise code paths without loading study data.

| Procedure | Additional inputs and boundary |
|---|---|
| Original NHANES/FPED/HEI workflow | Exact source cycle files and releases, controlled original harmonization workbook, investigator medication mapping, staged files and checked linkage configuration. The public drug-map template contains headers only. Run data-processing commands in a separate authorized working copy if needed. |
| Phase B metrics, calibration and comparisons | Original linked data and fixed OOF and temporal prediction arrays. Comparisons also consume private intermediate predictions for the exploratory comparator and selection sensitivity. Set `NHANES_SOURCE_ROOT` to an authorized archive and `NHANES_OUTPUT_ROOT` to a disjoint writable location. |
| Baseline weighted SD | Frozen R2/R4/R5 representations and their registry hashes. The archived script is inspectable. No descriptive statistics are recomputed in the public smoke test. |
| Frozen-model verification | Trusted original binaries with matching hashes, exact input feature tables and an appropriate software environment. Joblib loading executes Python deserialization. Use only validated study artifacts in an authorized environment. |
| Comparator or training commands | The installed modeling CLI disables fit and legacy inference dispatch in RC2. Inspect the archived implementations and use an independently authorized archive for historical work. |

The original 500-replicate unrescaled bootstrap is historical. The screened output uses the corrected rescaled PSU method. Internal lock identifiers and the private reconstruction summary document lineage and do not date the first access to temporal outcomes independently.

See [bootstrap provenance](BOOTSTRAP_PROVENANCE.md) for the output map, seeds and limitations.
