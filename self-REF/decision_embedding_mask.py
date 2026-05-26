from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from peft import LoraConfig
from transformers import TrainerCallback


DECISION_ROW_STATE = "decision_embedding_rows.pt"
FULL_EMBED_OR_HEAD_TARGETS = {
    "embed_tokens",
    "lm_head",
    "wte",
    "word_embeddings",
}


@dataclass(frozen=True)
class EmbeddingTieInfo:
    config_tie_word_embeddings: bool
    same_module: bool
    same_weight_storage: bool
    input_weight_ptr: int | None
    output_weight_ptr: int | None


def inspect_embedding_tying(model, prefix: str = "[self-ref]") -> EmbeddingTieInfo:
    input_embedding = model.get_input_embeddings()
    output_embedding = model.get_output_embeddings()
    input_weight = getattr(input_embedding, "weight", None)
    output_weight = getattr(output_embedding, "weight", None)

    input_ptr = input_weight.data_ptr() if input_weight is not None else None
    output_ptr = output_weight.data_ptr() if output_weight is not None else None
    info = EmbeddingTieInfo(
        config_tie_word_embeddings=bool(getattr(model.config, "tie_word_embeddings", False)),
        same_module=input_embedding is not None and input_embedding is output_embedding,
        same_weight_storage=input_ptr is not None and input_ptr == output_ptr,
        input_weight_ptr=input_ptr,
        output_weight_ptr=output_ptr,
    )

    print(f"{prefix} model.config.tie_word_embeddings = {info.config_tie_word_embeddings}")
    print(f"{prefix} input/output embedding same module = {info.same_module}")
    print(f"{prefix} input/output embedding same weight storage = {info.same_weight_storage}")
    print(f"{prefix} embedding/head rows outside decision tokens will be gradient-masked.")
    return info


def _normalize_modules(modules: str | Iterable[str]) -> list[str]:
    if isinstance(modules, str):
        return [m.strip() for m in modules.split(",") if m.strip()]
    return [str(m).strip() for m in modules if str(m).strip()]


def filter_lora_target_modules(modules: str | Iterable[str], prefix: str = "[self-ref]") -> list[str]:
    requested = _normalize_modules(modules)
    effective: list[str] = []
    dropped: list[str] = []
    for module in requested:
        if module.split(".")[-1] in FULL_EMBED_OR_HEAD_TARGETS:
            dropped.append(module)
        else:
            effective.append(module)

    if dropped:
        print(
            f"{prefix} dropped embedding/head from LoRA target_modules: {', '.join(dropped)}. "
            "They are handled by direct decision-row training plus gradient masks."
        )
    print(f"{prefix} effective LoRA targets: {', '.join(effective) if effective else '(none)'}")
    return effective


def build_lora_config_with_embedding_mask(
    *,
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: str | Iterable[str],
    task_type: str = "CAUSAL_LM",
    bias: str = "none",
    prefix: str = "[self-ref]",
) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=filter_lora_target_modules(target_modules, prefix=prefix),
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
    )


def _register_row_mask_hook(param: torch.nn.Parameter, token_ids: Sequence[int], name: str) -> None:
    row_ids = sorted(set(int(token_id) for token_id in token_ids))
    if not row_ids:
        raise ValueError("decision token ids must not be empty")
    if min(row_ids) < 0 or max(row_ids) >= param.shape[0]:
        raise ValueError(f"decision token ids {row_ids} out of range for {name} with shape {tuple(param.shape)}")

    mask_shape = [param.shape[0]] + [1] * (param.dim() - 1)
    row_mask = torch.zeros(mask_shape, dtype=torch.float32, device=param.device)
    row_mask[row_ids] = 1.0

    def grad_hook(grad):
        return grad * row_mask.to(device=grad.device, dtype=grad.dtype)

    param.register_hook(grad_hook)
    print(f"[self-ref] registered decision-row gradient mask on {name}: {row_ids}")


def _base_model_for_embeddings(model):
    if (
        hasattr(model, "get_input_embeddings")
        and hasattr(model, "get_output_embeddings")
        and model.get_input_embeddings() is not None
        and model.get_output_embeddings() is not None
    ):
        return model
    base_model = getattr(model, "base_model", None)
    if (
        base_model is not None
        and hasattr(base_model, "get_input_embeddings")
        and hasattr(base_model, "get_output_embeddings")
        and base_model.get_input_embeddings() is not None
        and base_model.get_output_embeddings() is not None
    ):
        return base_model
    raise ValueError("model does not expose get_input_embeddings/get_output_embeddings")


def setup_decision_embedding_row_training(model, token_ids: Sequence[int]) -> None:
    token_ids = [int(token_id) for token_id in token_ids]
    model = _base_model_for_embeddings(model)
    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()
    if input_emb is None or not hasattr(input_emb, "weight"):
        raise ValueError("model input embedding weight not found")
    if output_emb is None or not hasattr(output_emb, "weight"):
        raise ValueError("model output embedding/lm_head weight not found")

    input_emb.weight.requires_grad = True
    output_emb.weight.requires_grad = True

    seen = set()
    for name, param in [
        ("input_embeddings.weight", input_emb.weight),
        ("output_embeddings.weight", output_emb.weight),
    ]:
        if id(param) in seen:
            continue
        seen.add(id(param))
        _register_row_mask_hook(param, token_ids, name)


@torch.no_grad()
def save_decision_embedding_rows(model, output_dir: str | os.PathLike, token_ids: Sequence[int]) -> Path:
    token_ids = [int(token_id) for token_id in token_ids]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model = _base_model_for_embeddings(model)
    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()
    if input_emb is None or output_emb is None:
        raise ValueError("model input/output embeddings not found")
    input_weight = input_emb.weight.detach()
    output_weight = output_emb.weight.detach()
    payload = {
        "token_ids": token_ids,
        "input_rows": input_weight[token_ids].detach().cpu().clone(),
        "output_rows": output_weight[token_ids].detach().cpu().clone(),
        "input_output_same_weight": bool(input_weight.data_ptr() == output_weight.data_ptr()),
    }
    state_path = output_path / DECISION_ROW_STATE
    torch.save(payload, state_path)
    print(f"[self-ref] saved decision embedding/head rows to {state_path}")
    return state_path


@torch.no_grad()
def load_decision_embedding_rows(
    model,
    row_state_dir: str | os.PathLike,
    *,
    strict: bool = False,
) -> bool:
    state_path = Path(row_state_dir) / DECISION_ROW_STATE
    if not state_path.is_file():
        message = f"[self-ref] decision row state not found: {state_path}"
        if strict:
            raise FileNotFoundError(message)
        print(message)
        return False

    payload = torch.load(state_path, map_location="cpu")
    token_ids = [int(token_id) for token_id in payload["token_ids"]]
    model = _base_model_for_embeddings(model)
    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()
    if input_emb is None or output_emb is None:
        raise ValueError("model input/output embeddings not found")

    input_rows = payload["input_rows"].to(device=input_emb.weight.device, dtype=input_emb.weight.dtype)
    output_rows = payload["output_rows"].to(device=output_emb.weight.device, dtype=output_emb.weight.dtype)
    input_index = torch.tensor(token_ids, device=input_emb.weight.device, dtype=torch.long)
    output_index = torch.tensor(token_ids, device=output_emb.weight.device, dtype=torch.long)
    input_emb.weight.data.index_copy_(0, input_index, input_rows)
    output_emb.weight.data.index_copy_(0, output_index, output_rows)
    print(f"[self-ref] loaded decision embedding/head rows from {state_path}")
    return True


class DecisionEmbeddingRowsCallback(TrainerCallback):
    def __init__(self, token_ids: Sequence[int]):
        self.token_ids = [int(token_id) for token_id in token_ids]

    def on_save(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        save_decision_embedding_rows(model, checkpoint_dir, self.token_ids)
        return control
