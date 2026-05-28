# Anonymous ACL Submission Package

This package contains an anonymized code release accompanying the submission.
It includes the implementation of the main method, the adapted `self-REF`
implementation used in this project, baseline code, dataset preparation
scripts, and a small set of reference evaluation artifacts.

## Directory Overview

- `local_state_pref/`
  - Implementation of the main preference-based routing / defer method.
  - Organized by domain: `code/`, `math/`, and `mmlu/`.
- `self-REF/`
  - Anonymous copy of the `self-REF`-style implementation used in this work.
  - Includes the prompt construction, training, and evaluation code used for
    the adapted confidence-token setting in this project.
- `baseline/`
  - Implementations of baseline routing methods and supporting analysis
    utilities.
- `dataset/`
  - Dataset preparation scripts for:
    - `MMLU`
    - `math`
    - `code_benchmarks`
- `motivation_hidden/`
  - Hidden-state extraction utilities used to build probe feature manifests for
    the linear-routing baselines.
- `metadata/`
  - Reference routing metadata files retained for the main GenGate method.
- `eval/`
  - Expert-output CSV files used by the evaluation pipeline.

## Included Evaluation Artifacts

The package includes a limited set of evaluation artifacts intended to document
the core method outputs used in the paper.

Included:
- `6` routing metadata CSV files under `metadata/`
  - `6` files for the main method (`GenGate_*`)
- `3` expert-output CSV files under:
  - `eval/code/expert_output/`
  - `eval/math/expert_output/`
  - `eval/MMLU/expert_output/`

Not included:
- full baseline result tables
- intermediate local-output CSV files
- additional internal sweep artifacts not needed for understanding the core
  method implementation


## Environment

The file `requirements.txt` was exported from our environment
using `pip list --format=freeze`.

## Notes

- Some baseline directories are included for code completeness even when their
  corresponding evaluation artifacts are not distributed in this package.
- Some baseline plotting or reporting scripts may still refer to `PBDD`; this
  is an earlier name for the `GenGate` method used in the paper and in the
  metadata files included here.
- The `self-REF` directory reflects the adapted version used in this project,
  including the modified confidence-token setup required by the paper’s
  implementation.
