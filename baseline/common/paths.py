from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BASELINE_ROOT.parent
MOTIVATION_OUTPUT_ROOT = REPO_ROOT / "Motivation" / "output"


MODEL_ALIASES = {
    "qwen-code": "qwen2.5-coder-3b-instruct",
    "qwen-coder": "qwen2.5-coder-3b-instruct",
    "qwen2.5-coder-3b-instruct": "qwen2.5-coder-3b-instruct",
    "qwen": "qwen2.5-3b-instruct",
    "qwen-math": "qwen2.5-3b-instruct",
    "qwen-mmlu": "qwen2.5-3b-instruct",
    "qwen2.5-3b-instruct": "qwen2.5-3b-instruct",
    "llama": "meta-llama-3-8b-instruct",
    "llama3": "meta-llama-3-8b-instruct",
    "llama3-8b": "meta-llama-3-8b-instruct",
    "meta-llama-3-8b-instruct": "meta-llama-3-8b-instruct",
}


DOMAIN_DATASET_SLUG = {
    "code": "humaneval_mbpp_leetcode",
    "math": "hendrycks_math",
    "mmlu": "mmlu",
}


TRAIN_SPLITS = {
    "code": "code_train",
    "math": "math_train",
    "mmlu": "mmlu_all_train",
}


EVAL_SPLITS = {
    "code": "code_val",
    "math": "math_test",
    "mmlu": "mmlu_all_test",
}


DEFAULT_DATA_PATHS = {
    "code": REPO_ROOT / "coder" / "data" / "code_benchmarks" / "mbppplus_leetcode_humanevalplus" / "RL" / "val_with_id.csv",
    "math": REPO_ROOT / "coder" / "data" / "MATH" / "hendrycks_math_test.jsonl",
    "mmlu": REPO_ROOT / "coder" / "data" / "mmlu" / "processed" / "mmlu_all_test.jsonl",
}


def slugify(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "run"


def canonical_model_slug(model: str, domain: str | None = None) -> str:
    raw = str(model or "").strip()
    key_source = Path(raw).name if any(sep in raw for sep in ("/", "\\")) else raw
    key = slugify(key_source)
    if key == "qwen":
        return "qwen2.5-coder-3b-instruct" if domain == "code" else "qwen2.5-3b-instruct"
    return MODEL_ALIASES.get(key, key)


def manifest_path(model: str, split: str) -> Path:
    model_slug = canonical_model_slug(model)
    return MOTIVATION_OUTPUT_ROOT / model_slug / split / f"{split}_probe_features_manifest.json"


def split_for(domain: str, role: str) -> str:
    domain = domain.lower()
    if role == "train":
        return TRAIN_SPLITS[domain]
    if role in {"eval", "val", "validation", "test"}:
        return EVAL_SPLITS[domain]
    raise ValueError(f"Unknown split role: {role}")


@dataclass(frozen=True)
class RunPaths:
    method: str
    domain: str
    model_slug: str
    dataset_slug: str
    run_name: str
    outputs: Path
    eval_output: Path
    ckpt: Path


def make_run_paths(
    *,
    method: str,
    domain: str,
    model: str,
    run_name: str = "default",
    dataset_slug: str | None = None,
) -> RunPaths:
    method_slug = method
    domain_slug = slugify(domain)
    model_slug = canonical_model_slug(model, domain_slug)
    dataset = dataset_slug or DOMAIN_DATASET_SLUG[domain_slug]
    run = slugify(run_name)
    base = BASELINE_ROOT / method_slug / domain_slug
    paths = RunPaths(
        method=method_slug,
        domain=domain_slug,
        model_slug=model_slug,
        dataset_slug=dataset,
        run_name=run,
        outputs=base / "outputs" / model_slug / dataset / run,
        eval_output=base / "eval_output" / model_slug / dataset / run,
        ckpt=base / "ckpt" / model_slug / dataset / run,
    )
    for path in (paths.outputs, paths.eval_output, paths.ckpt):
        path.mkdir(parents=True, exist_ok=True)
    return paths
