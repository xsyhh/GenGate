# self-REF only-decision scripts

This directory extracts the core only-decision Self-REF-style scripts from
the original `sft_upper_bound` workspace.

The copied files are organized first by model family, then by dataset:

- `qwen/code/`: from `sft_upper_bound/code_only_decision`
- `qwen/math/`: from `sft_upper_bound/math_only_decision`
- `qwen/mmlu/`: from `sft_upper_bound/mmlu_only_decision`
- `llama/code/`: copied from `qwen/code/`
- `llama/math/`: copied from `qwen/math/`
- `llama/mmlu/`: copied from `qwen/mmlu/`

Only source scripts, evaluation/plot helpers, and slurm launch scripts were
copied. Historical logs, checkpoints, generated jsonl/csv files, and cached
outputs were intentionally left in the original tree.

Most scripts still contain the original absolute paths from `sft_upper_bound`;
update those paths before launching new runs from this directory.

The `bash/` scripts are direct local-launch conversions of the original Slurm
scripts. They keep the original Python entry points and arguments where the
entry point exists. Two launch-only filename corrections are applied because
the original Slurm scripts referenced missing files:

- `*/code/bash/3_build.sh` uses `3_build_llama_dataset.py` instead of the
  missing `3_build_llama_factory_dataset.py`.
- `*/mmlu/bash/offline_eval.sh` uses `mmlu_generate_metadata.py` instead of
  the missing `offline_eval_gsm8k_generate.py`.
