from __future__ import annotations

import argparse
import json
import os


def ensure_local_dir(path: str, label: str) -> str:
    expanded = os.path.expanduser(path)
    if not os.path.isdir(expanded):
        raise FileNotFoundError(f"{label} does not exist: {expanded}")
    return expanded


def read_adapter_base_model_path(lora_ckpt: str) -> str | None:
    adapter_config_path = os.path.join(lora_ckpt, "adapter_config.json")
    if not os.path.isfile(adapter_config_path):
        return None

    with open(adapter_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    base_model = str(config.get("base_model_name_or_path") or "").strip()
    return base_model or None


def resolve_base_model_path(lora_ckpt: str, base_model: str | None) -> str:
    if base_model:
        return ensure_local_dir(base_model, "Base model")

    inferred = read_adapter_base_model_path(lora_ckpt)
    if not inferred:
        raise ValueError(
            "Unable to infer base model from adapter_config.json; pass --base_model explicitly."
        )
    return ensure_local_dir(inferred, "Inferred base model")


def parse_torch_dtype(dtype: str):
    import torch

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return dtype_map[dtype]


def merge_lora(
    *,
    base_model: str,
    lora_ckpt: str,
    output_dir: str,
    dtype: str,
    trust_remote_code: bool,
) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = parse_torch_dtype(dtype)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[1/5] Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[2/5] Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )

    print(f"[3/5] Loading LoRA adapter: {lora_ckpt}")
    model = PeftModel.from_pretrained(model, lora_ckpt)

    print("[4/5] Merging LoRA weights into base model")
    merged_model = model.merge_and_unload()

    print(f"[5/5] Saving merged model: {output_dir}")
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print("Merged model is ready for vLLM.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge an MMLU LoRA checkpoint into its base model for vLLM generation.")
    parser.add_argument("--lora_ckpt", required=True, help="LoRA checkpoint or adapter directory.")
    parser.add_argument("--output_dir", required=True, help="Output directory for the merged full model.")
    parser.add_argument("--base_model", default=None, help="Base model path. Defaults to adapter_config.json base_model_name_or_path.")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    lora_ckpt = ensure_local_dir(args.lora_ckpt, "LoRA checkpoint")
    base_model = resolve_base_model_path(lora_ckpt, args.base_model)
    merge_lora(
        base_model=base_model,
        lora_ckpt=lora_ckpt,
        output_dir=args.output_dir,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
