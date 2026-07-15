from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from src.models.metrics import token_accuracy
from src.models.spec2prod import BinnedEncoder, PeakEncoder, create_2D_padding_mask, create_padding_mask


@dataclass(frozen=True)
class EncoderMemory:
    values: torch.Tensor
    padding_mask: torch.Tensor | None = None

    def size(self, *args):
        return self.values.size(*args)

    def __getitem__(self, item):
        padding_mask = None if self.padding_mask is None else self.padding_mask[item]
        return EncoderMemory(self.values[item], padding_mask)


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
) -> EncoderMemory | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Build decoder memory from whichever input encoders are active."""
    spec_memory = None
    spec_mask = None
    if spectrum_enc:
        spec_memory = encoder(spec_batch)
        if isinstance(encoder, BinnedEncoder):
            spec_mask = create_padding_mask(spec_batch)
        elif isinstance(encoder, PeakEncoder):
            spec_mask = create_2D_padding_mask(spec_batch)

    rxn_memory = None
    rxn_mask = None
    if reactant_enc is not None:
        rxn_memory = reactant_enc(reactant_ids)
        rxn_mask = reactant_ids == getattr(reactant_enc, "pad_id", 0)

    fml_memory = None
    fml_mask = None
    if formula_enc is not None:
        fml_memory = formula_enc(formula_ids)
        fml_mask = formula_ids == getattr(formula_enc, "pad_id", 0)

    if getattr(decoder, "multi_source", False):
        if spec_memory is None or rxn_memory is None:
            raise ValueError("multi_source_decoder=True requires both spectrum and reactant encoders.")
        return spec_memory, rxn_memory, spec_mask, rxn_mask

    memory_parts: list[tuple[torch.Tensor, torch.Tensor | None]] = []
    if fusion_enc is not None:
        if spec_memory is None or rxn_memory is None:
            raise ValueError("FusionEncoder requires spectrum_enc=True and reactant_enc!=None.")
        fused_mask = _combine_masks((spec_memory, spec_mask), (rxn_memory, rxn_mask))
        memory_parts.append((fusion_enc(spec_memory, rxn_memory, mask_spec=spec_mask, mask_rxn=rxn_mask), fused_mask))
        if fml_memory is not None:
            memory_parts.append((fml_memory, fml_mask))
    else:
        for memory, mask in ((spec_memory, spec_mask), (rxn_memory, rxn_mask), (fml_memory, fml_mask)):
            if memory is not None:
                memory_parts.append((memory, mask))

    if not memory_parts:
        raise ValueError("No encoder memory available. Enable at least one of spectrum/reactant/formula.")
    if len(memory_parts) == 1:
        values, padding_mask = memory_parts[0]
    else:
        values = torch.cat([memory for memory, _ in memory_parts], dim=1)
        padding_mask = _combine_masks(*memory_parts)
    return EncoderMemory(values, padding_mask)


def _combine_masks(*parts: tuple[torch.Tensor, torch.Tensor | None]) -> torch.Tensor | None:
    if not any(mask is not None for _, mask in parts):
        return None
    masks = [
        mask if mask is not None else torch.zeros(memory.shape[:2], dtype=torch.bool, device=memory.device)
        for memory, mask in parts
    ]
    return torch.cat(masks, dim=1)


def repeat_memory(memory: Any, repeats: int) -> Any:
    """Repeat a single-source tensor or multi-source memory tuple along the batch dimension."""
    if repeats == 1:
        return memory
    if isinstance(memory, EncoderMemory):
        return EncoderMemory(
            memory.values.repeat_interleave(repeats, dim=0),
            None if memory.padding_mask is None else memory.padding_mask.repeat_interleave(repeats, dim=0),
        )
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
