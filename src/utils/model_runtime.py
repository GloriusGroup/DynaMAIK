from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.metrics import token_accuracy


def set_modules_train(*modules: nn.Module | None) -> None:
    for module in modules:
        if module is not None:
            module.train()


def set_modules_eval(*modules: nn.Module | None) -> None:
    for module in modules:
        if module is not None:
            module.eval()


def move_optional_tensor(tensor: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.to(device, non_blocking=True)


def build_encoder_memory(
    *,
    encoder: nn.Module,
    decoder: nn.Module,
    spec_batch: torch.Tensor,
    spectrum_enc: bool = True,
    reactant_enc: nn.Module | None = None,
    reactant_ids: torch.Tensor | None = None,
    fusion_enc: nn.Module | None = None,
    formula_enc: nn.Module | None = None,
    formula_ids: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Build decoder memory from whichever input encoders are active."""
    spec_memory = None
    spec_mask = None
    if spectrum_enc:
        spec_memory = encoder(spec_batch)
        if spec_memory is not None:
            batch_size, seq_len, _ = spec_memory.shape
            spec_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=spec_memory.device)

    rxn_memory = None
    rxn_mask = None
    if reactant_enc is not None:
        rxn_memory = reactant_enc(reactant_ids)
        rxn_mask = reactant_ids == getattr(reactant_enc, "pad_id", 0)

    fml_memory = None
    if formula_enc is not None:
        fml_memory = formula_enc(formula_ids)

    if getattr(decoder, "multi_source", False):
        if spec_memory is None or rxn_memory is None:
            raise ValueError("multi_source_decoder=True requires both spectrum and reactant encoders.")
        return spec_memory, rxn_memory, spec_mask, rxn_mask

    memory_parts = []
    if fusion_enc is not None:
        if spec_memory is None or rxn_memory is None:
            raise ValueError("FusionEncoder requires spectrum_enc=True and reactant_enc!=None.")
        memory_parts.append(fusion_enc(spec_memory, rxn_memory, mask_rxn=rxn_mask))
        if fml_memory is not None:
            memory_parts.append(fml_memory)
    else:
        for memory in (spec_memory, rxn_memory, fml_memory):
            if memory is not None:
                memory_parts.append(memory)

    if not memory_parts:
        raise ValueError("No encoder memory available. Enable at least one of spectrum/reactant/formula.")
    return memory_parts[0] if len(memory_parts) == 1 else torch.cat(memory_parts, dim=1)


def repeat_memory(memory: Any, repeats: int) -> Any:
    """Repeat a single-source tensor or multi-source memory tuple along the batch dimension."""
    if repeats == 1:
        return memory
    if isinstance(memory, tuple):
        return tuple(None if item is None else item.repeat_interleave(repeats, dim=0) for item in memory)
    return memory.repeat_interleave(repeats, dim=0)


def teacher_forcing_metrics(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> dict[str, Any]:
    """Compute token and sequence metrics for teacher-forced decoder outputs."""
    valid_mask = targets != pad_id
    pred = logits.float().argmax(dim=-1)
    seq_correct = ((pred == targets) | (~valid_mask)).all(dim=1)
    has_tokens = valid_mask.any(dim=1)
    acc_seq = (seq_correct & has_tokens).sum().float() / has_tokens.sum().clamp_min(1)

    return {
        "valid_mask": valid_mask,
        "pred": pred,
        "acc_tok": token_accuracy(pred, targets, valid_mask),
        "seq_correct": seq_correct,
        "has_tokens": has_tokens,
        "acc_seq": acc_seq,
        "token_count": int(valid_mask.sum().item()),
    }


def pool_hidden_by_input_mask(hidden: torch.Tensor, input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    mask = (input_ids != pad_id).float()
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom
