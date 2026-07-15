import json
import heapq

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ==== your modules (adjust paths if needed) ====
from src.data.tokenizers import SmilesTokenizerAdapter, FormulaTokenizerAdapter
from src.models.metrics import token_accuracy, sequence_accuracy, tanimoto_similarity
from src.utils.model_runtime import build_encoder_memory, pool_hidden_by_input_mask, repeat_memory, set_modules_eval, teacher_forcing_metrics
from src.utils.utils_train import build_prefix_batch, canonicalize_smiles, counts_from_formula_tokens

from torch.amp import autocast


@torch.no_grad()
def evaluate_formula_only(
        encoder: nn.Module,
        decoder_formula: nn.Module,
        loader: DataLoader,
        device: torch.device,
        formula_tokenizer: FormulaTokenizerAdapter,
        formula_eos_id: int,
        reactant_enc: nn.Module | None = None,
        fusion_enc: nn.Module | None = None,
        spectrum_enc: bool = True,
        temperature: float = 1.0,
        topk: bool = False,          # if True: sample 5 candidates for top-5 exact match
        count_head: nn.Module | None = None,
        elem_list: tuple[str] | None = None,
        log_count_mae: bool = False,
        ) -> dict:
    """
    Formula-only evaluation.

    Supported batch formats:
      4-tuple: (spec_batch, fml_in, fml_out, formula_strs)
      6-tuple: (spec_batch, fml_in, fml_out, formula_strs, smi_in, smi_out)
      7-tuple: (spec_batch, reactant_ids, fml_in, fml_out, formula_strs, smi_in, smi_out)

    SMILES tensors are ignored here.

    Returns metrics keyed with "..._formula" and sets val_loss = val_loss_formula.
    """

    set_modules_eval(encoder, decoder_formula, reactant_enc, fusion_enc, count_head)

    total_loss_fml = 0.0
    n_tok_fml = 0
    acc_tok_fml_sum = 0.0
    acc_seq_fml_sum = 0.0
    n_batches = 0

    exact_tf_fml_sum = 0
    exact_tf_fml_count = 0
    exact_gen_fml_sum = 0
    exact_gen_fml_count = 0
    if topk:
        exact_gen_fml_top5_sum = 0
        exact_gen_fml_top5_count = 0

    # --- element MAE accumulators ---
    mae_sum = None     # [n_elems]
    mae_count = 0      # number of molecules accumulated

    for batch in loader:
        if len(batch) == 4:
            # formula-only
            spec_batch, fml_in, fml_out, formula_strs = batch
            reactant_ids = None

        elif len(batch) == 6:
            # multitask without reactants
            spec_batch, fml_in, fml_out, formula_strs, _, _ = batch
            reactant_ids = None

        elif len(batch) == 7:
            # multitask with reactants
            spec_batch, reactant_ids, fml_in, fml_out, formula_strs, _, _ = batch

        else:
            raise ValueError(
                f"Unexpected batch length {len(batch)} in evaluate_formula_only. "
                "Expected 4, 6, or 7."
            )

        spec_batch = spec_batch.to(device, non_blocking=True)
        if reactant_ids is not None:
            reactant_ids = reactant_ids.to(device, non_blocking=True)
        fml_in = fml_in.to(device, non_blocking=True)
        fml_out = fml_out.to(device, non_blocking=True)

        # ---- build memory ----
        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            memory = build_encoder_memory(
                encoder=encoder,
                decoder=decoder_formula,
                spec_batch=spec_batch,
                spectrum_enc=spectrum_enc,
                reactant_enc=reactant_enc,
                reactant_ids=reactant_ids,
                fusion_enc=fusion_enc,
            )

            # ---- forward ----
            if log_count_mae and (count_head is not None) and (elem_list is not None):
                logits_fml, hs_fml = decoder_formula(fml_in, memory, return_hidden=True)
            else:
                logits_fml = decoder_formula(fml_in, memory)
                hs_fml = None

        # ---- TF loss/metrics in fp32 ----
        loss_fml = decoder_formula.compute_loss(logits_fml.float(), fml_out)

        tf_fml = teacher_forcing_metrics(logits_fml, fml_out, decoder_formula.pad_id)
        tok_fml = tf_fml["token_count"]
        total_loss_fml += float(loss_fml.item()) * max(tok_fml, 1)
        n_tok_fml += max(tok_fml, 1)
        acc_tok_fml_sum += float(tf_fml["acc_tok"])
        acc_seq_fml_sum += float(tf_fml["acc_seq"])

        exact_tf_fml_sum += (tf_fml["seq_correct"] & tf_fml["has_tokens"]).sum().item()
        exact_tf_fml_count += tf_fml["has_tokens"].sum().item()

        # ---- fixed-size count MAE (teacher-forced hidden states) ----
        if log_count_mae and (count_head is not None) and (elem_list is not None):
            pooled = pool_hidden_by_input_mask(hs_fml, fml_in, decoder_formula.pad_id)
            pred_counts = count_head(pooled).float()
            true_counts = counts_from_formula_tokens(fml_out=fml_out, formula_tokenizer=formula_tokenizer, elements=elem_list, pad_id=decoder_formula.pad_id, eos_id=formula_eos_id).to(pred_counts.device)  # [B,n_elems]
            batch_sum = (pred_counts - true_counts).abs().sum(dim=0)

            if mae_sum is None:
                mae_sum = batch_sum.detach().cpu()
            else:
                mae_sum += batch_sum.detach().cpu()

            mae_count += int(pred_counts.size(0))

        n_batches += 1

        # ---- autoregressive generation for exact match ----
        B, T_fml = fml_out.shape
        fml_prefix = torch.full((B, 1), formula_tokenizer.bos_id, dtype=torch.long, device=device)

        if topk:
            k = 5
            fml_prefix_rep = fml_prefix.repeat_interleave(k, dim=0)

            memory_rep = repeat_memory(memory, k)

            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                gen_fml_rep = decoder_formula.generate_from_prefix(
                    prefix_ids=fml_prefix_rep,
                    memory=memory_rep,
                    eos_id=formula_eos_id,
                    pad_id=decoder_formula.pad_id,
                    max_new_tokens=T_fml,
                    temperature=temperature,
                    topk=50,
                )

            gen_fml = gen_fml_rep.view(B, k, -1)

            for i in range(B):
                true_fml = formula_tokenizer.decode(fml_out[i].tolist(), skip_specials=True).replace(" ", "")
                hit = False
                for s in range(k):
                    pred_fml_s = formula_tokenizer.decode(gen_fml[i, s].tolist(), skip_specials=True).replace(" ", "")
                    if s == 0:
                        exact_gen_fml_sum += int(pred_fml_s == true_fml)
                        exact_gen_fml_count += 1
                    if pred_fml_s == true_fml:
                        hit = True
                exact_gen_fml_top5_sum += int(hit)
                exact_gen_fml_top5_count += 1

        else:
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                gen_fml = decoder_formula.generate_from_prefix(
                    prefix_ids=fml_prefix,
                    memory=memory,
                    eos_id=formula_eos_id,
                    pad_id=decoder_formula.pad_id,
                    max_new_tokens=T_fml,
                    temperature=temperature,
                )

            for i in range(B):
                true_fml = formula_tokenizer.decode(fml_out[i].tolist(), skip_specials=True).replace(" ", "")
                pred_fml_s = formula_tokenizer.decode(gen_fml[i].tolist(), skip_specials=True).replace(" ", "")
                exact_gen_fml_sum += int(pred_fml_s == true_fml)
                exact_gen_fml_count += 1

    mean_loss_fml = total_loss_fml / max(n_tok_fml, 1)

    metrics = {
        "val_loss": float(mean_loss_fml),  # <-- formula-only val_loss
        "val_loss_formula": float(mean_loss_fml),
        "val_token_acc_formula": float(acc_tok_fml_sum / max(n_batches, 1)),
        "val_sequence_acc_formula": float(acc_seq_fml_sum / max(n_batches, 1)),
        "val_exact_match_tf_formula": float(exact_tf_fml_sum / max(exact_tf_fml_count, 1)),
        "val_exact_match_gen_formula": float(exact_gen_fml_sum / max(exact_gen_fml_count, 1)),
    }

    if topk:
        metrics["val_exact_match_gen_top5_formula"] = float(
            exact_gen_fml_top5_sum / max(exact_gen_fml_top5_count, 1)
        )

    if log_count_mae and (count_head is not None) and (elem_list is not None) and (mae_sum is not None) and (mae_count > 0):
        mae = (mae_sum / mae_count)  # [n_elems] on CPU
        # per-element
        for j, e in enumerate(elem_list):
            metrics[f"val_mae_{e}_counts"] = float(mae[j].item())
        # aggregate
        metrics["val_mae_counts_mean"] = float(mae.mean().item())

    return metrics


@torch.no_grad()
def evaluate_product_only_multitask(
        encoder: nn.Module,
        decoder_smiles: nn.Module,
        decoder_formula: nn.Module,
        loader: DataLoader,
        device: torch.device,
        smiles_tokenizer: SmilesTokenizerAdapter,
        smiles_eos_id: int,
        formula_tokenizer: FormulaTokenizerAdapter,
        formula_eos_id: int,
        tanimoto_samples: int = 128,  # keep 0 unless you want to extend it
        reactant_enc: nn.Module | None = None,
        fusion_enc: nn.Module | None = None,
        spectrum_enc: bool = True,
        formula_enc: nn.Module | None = None,  # keep for compatibility if you still use formula as input encoder too
        temperature: float = 1.0,
        topk: bool = False,  # if True: sample 5 candidates (like your SMILES eval) for top-5 accuracy
        formula_loss_weight: float = 1.0,  # to report a combined loss comparable to training
        count_head: nn.Module | None = None,
        elem_list: tuple[str] | None = None,
        log_count_mae: bool = False,
        ) -> dict:
    """
    Expects loader batches from:
      collate_batch_product_only_with_reactants_and_formula_multitask:
        spec_batch, reactant_ids, fml_in, fml_out, formula_strs, smi_in, smi_out
    Computes TF loss/acc for both heads and GEN exact match for both heads (product mode only).
    """

    set_modules_eval(encoder, decoder_smiles, decoder_formula, reactant_enc, fusion_enc, formula_enc, count_head)

    # ---- accumulators: SMILES ----
    total_loss_smi = 0.0
    n_tok_smi = 0
    acc_tok_smi_sum = 0.0
    acc_seq_smi_sum = 0.0
    n_batches = 0

    exact_tf_smi_sum = 0
    exact_tf_smi_count = 0
    exact_gen_smi_sum = 0
    exact_gen_smi_raw_sum = 0
    exact_gen_smi_count = 0
    if topk:
        exact_gen_smi_top5_sum = 0
        exact_gen_smi_top5_count = 0

    # ---- accumulators: Formula ----
    total_loss_fml = 0.0
    n_tok_fml = 0
    acc_tok_fml_sum = 0.0
    acc_seq_fml_sum = 0.0

    exact_tf_fml_sum = 0
    exact_tf_fml_count = 0
    exact_gen_fml_sum = 0
    exact_gen_fml_count = 0
    if topk:
        exact_gen_fml_top5_sum = 0
        exact_gen_fml_top5_count = 0

    # --- element MAE accumulators ---
    mae_sum = None     # [n_elems]
    mae_count = 0      # number of molecules accumulated

    pred_smiles_all, true_smiles_all = [], []
    do_tanimoto = (
            (smiles_tokenizer is not None)
            and (smiles_eos_id is not None)
            and (tanimoto_samples is not None)
            and (tanimoto_samples > 0)
    )

    for batch in loader:
        # multitask batch
        if len(batch) == 7:
            # multitask with reactants
            spec_batch, reactant_ids, fml_in, fml_out, formula_strs, smi_in, smi_out = batch
        elif len(batch) == 6:
            # multitask without reactants
            spec_batch, fml_in, fml_out, formula_strs, smi_in, smi_out = batch
            reactant_ids = None
        else:
            raise ValueError(
                f"Unexpected batch length {len(batch)} in evaluate_product_only_multitask. "
                "Expected 6 or 7."
            )

        spec_batch = spec_batch.to(device, non_blocking=True)
        if reactant_ids is not None:
            reactant_ids = reactant_ids.to(device, non_blocking=True)

        smi_in = smi_in.to(device, non_blocking=True)
        smi_out = smi_out.to(device, non_blocking=True)

        fml_in = fml_in.to(device, non_blocking=True)
        fml_out = fml_out.to(device, non_blocking=True)

        # ---- build memory (same logic as your evaluate_product_only) ----
        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            memory = build_encoder_memory(
                encoder=encoder,
                decoder=decoder_smiles,
                spec_batch=spec_batch,
                spectrum_enc=spectrum_enc,
                reactant_enc=reactant_enc,
                reactant_ids=reactant_ids,
                fusion_enc=fusion_enc,
                formula_enc=formula_enc,
                formula_ids=fml_out,
            )

            # ---- forward passes ----
            logits_smi = decoder_smiles(smi_in, memory)
            if log_count_mae and (count_head is not None) and (elem_list is not None):
                logits_fml, hs_fml = decoder_formula(fml_in, memory, return_hidden=True)
            else:
                logits_fml = decoder_formula(fml_in, memory)
                hs_fml = None

        # ---- SMILES TF loss/metrics (fp32) ----
        loss_smi = decoder_smiles.compute_loss(logits_smi.float(), smi_out)
        tf_smi = teacher_forcing_metrics(logits_smi, smi_out, decoder_smiles.pad_id)
        tok_smi = tf_smi["token_count"]
        total_loss_smi += float(loss_smi.item()) * max(tok_smi, 1)
        n_tok_smi += max(tok_smi, 1)
        acc_tok_smi_sum += float(tf_smi["acc_tok"])
        acc_seq_smi_sum += float(tf_smi["acc_seq"])

        exact_tf_smi_sum += (tf_smi["seq_correct"] & tf_smi["has_tokens"]).sum().item()
        exact_tf_smi_count += tf_smi["has_tokens"].sum().item()

        # ---- Formula TF loss/metrics (fp32) ----
        loss_fml = decoder_formula.compute_loss(logits_fml.float(), fml_out)
        tf_fml = teacher_forcing_metrics(logits_fml, fml_out, decoder_formula.pad_id)
        tok_fml = tf_fml["token_count"]
        total_loss_fml += float(loss_fml.item()) * max(tok_fml, 1)
        n_tok_fml += max(tok_fml, 1)
        acc_tok_fml_sum += float(tf_fml["acc_tok"])
        acc_seq_fml_sum += float(tf_fml["acc_seq"])

        exact_tf_fml_sum += (tf_fml["seq_correct"] & tf_fml["has_tokens"]).sum().item()
        exact_tf_fml_count += tf_fml["has_tokens"].sum().item()

        # ---- fixed-size count MAE (teacher-forced hidden states) ----
        if log_count_mae and (count_head is not None) and (elem_list is not None):
            pooled = pool_hidden_by_input_mask(hs_fml, fml_in, decoder_formula.pad_id)
            pred_counts = count_head(pooled).float()
            true_counts = counts_from_formula_tokens(fml_out=fml_out, formula_tokenizer=formula_tokenizer, elements=elem_list, pad_id=decoder_formula.pad_id, eos_id=formula_eos_id).to(pred_counts.device)                                     # [B,n_elems]
            batch_sum = (pred_counts - true_counts).abs().sum(dim=0)

            if mae_sum is None:
                mae_sum = batch_sum.detach().cpu()
            else:
                mae_sum += batch_sum.detach().cpu()

            mae_count += int(pred_counts.size(0))

        n_batches += 1

        # ---- Autoregressive GEN exact match (SMILES + Formula) ----
        B, T_smi = smi_out.shape
        _, T_fml = fml_out.shape

        # prefixes are BOS only in product mode
        smi_prefix = torch.full((B, 1), smiles_tokenizer.bos_id, dtype=torch.long, device=device)
        fml_prefix = torch.full((B, 1), formula_tokenizer.bos_id, dtype=torch.long, device=device)

        if topk:
            k = 5
            smi_prefix_rep = smi_prefix.repeat_interleave(k, dim=0)
            fml_prefix_rep = fml_prefix.repeat_interleave(k, dim=0)

            memory_rep = repeat_memory(memory, k)

            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                gen_smi_rep = decoder_smiles.generate_from_prefix(
                    prefix_ids=smi_prefix_rep,
                    memory=memory_rep,
                    eos_id=smiles_eos_id,
                    pad_id=decoder_smiles.pad_id,
                    max_new_tokens=T_smi,
                    temperature=temperature,
                    topk=50,
                )
                gen_fml_rep = decoder_formula.generate_from_prefix(
                    prefix_ids=fml_prefix_rep,
                    memory=memory_rep,
                    eos_id=formula_eos_id,
                    pad_id=decoder_formula.pad_id,
                    max_new_tokens=T_fml,
                    temperature=temperature,
                    topk=50,
                )

            gen_smi = gen_smi_rep.view(B, k, -1)
            gen_fml = gen_fml_rep.view(B, k, -1)

            for i in range(B):
                # SMILES: canonicalize for "auto" exact match
                true_smi_raw = smiles_tokenizer.decode(smi_out[i].tolist(), skip_specials=True)
                true_smi_can = canonicalize_smiles(true_smi_raw)
                true_smi_auto = true_smi_can if true_smi_can is not None else true_smi_raw

                # Formula: whitespace-insensitive exact match
                true_fml = formula_tokenizer.decode(fml_out[i].tolist(), skip_specials=True).replace(" ", "")

                smi_top1_done = False
                smi_top5_hit = False
                fml_top1_done = False
                fml_top5_hit = False

                for s in range(k):
                    # SMILES candidate
                    pred_smi_raw = smiles_tokenizer.decode(gen_smi[i, s].tolist(), skip_specials=True)
                    pred_smi_can = canonicalize_smiles(pred_smi_raw)
                    pred_smi_auto = pred_smi_can if pred_smi_can is not None else pred_smi_raw

                    if not smi_top1_done:
                        exact_gen_smi_sum += int(pred_smi_auto == true_smi_auto)
                        exact_gen_smi_raw_sum += int(pred_smi_raw == true_smi_raw)
                        exact_gen_smi_count += 1
                        smi_top1_done = True

                        if do_tanimoto and len(pred_smiles_all) < tanimoto_samples:
                            pred_smiles_all.append(pred_smi_auto)
                            true_smiles_all.append(true_smi_auto)

                    if pred_smi_auto == true_smi_auto:
                        smi_top5_hit = True

                    # Formula candidate
                    pred_fml = formula_tokenizer.decode(gen_fml[i, s].tolist(), skip_specials=True).replace(" ", "")
                    if not fml_top1_done:
                        exact_gen_fml_sum += int(pred_fml == true_fml)
                        exact_gen_fml_count += 1
                        fml_top1_done = True
                    if pred_fml == true_fml:
                        fml_top5_hit = True

                exact_gen_smi_top5_sum += int(smi_top5_hit)
                exact_gen_smi_top5_count += 1
                exact_gen_fml_top5_sum += int(fml_top5_hit)
                exact_gen_fml_top5_count += 1

        else:
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                gen_smi = decoder_smiles.generate_from_prefix(
                    prefix_ids=smi_prefix,
                    memory=memory,
                    eos_id=smiles_eos_id,
                    pad_id=decoder_smiles.pad_id,
                    max_new_tokens=T_smi,
                    temperature=temperature,
                )
                gen_fml = decoder_formula.generate_from_prefix(
                    prefix_ids=fml_prefix,
                    memory=memory,
                    eos_id=formula_eos_id,
                    pad_id=decoder_formula.pad_id,
                    max_new_tokens=T_fml,
                    temperature=temperature,
                )

            for i in range(B):
                # SMILES
                true_smi_raw = smiles_tokenizer.decode(smi_out[i].tolist(), skip_specials=True)
                pred_smi_raw = smiles_tokenizer.decode(gen_smi[i].tolist(), skip_specials=True)

                true_smi_can = canonicalize_smiles(true_smi_raw)
                pred_smi_can = canonicalize_smiles(pred_smi_raw)
                true_smi_auto = true_smi_can if true_smi_can is not None else true_smi_raw
                pred_smi_auto = pred_smi_can if pred_smi_can is not None else pred_smi_raw

                exact_gen_smi_sum += int(pred_smi_auto == true_smi_auto)
                exact_gen_smi_raw_sum += int(pred_smi_raw == true_smi_raw)
                exact_gen_smi_count += 1

                if do_tanimoto and len(pred_smiles_all) < tanimoto_samples:
                    pred_smiles_all.append(pred_smi_auto)
                    true_smiles_all.append(true_smi_auto)

                # Formula
                true_fml = formula_tokenizer.decode(fml_out[i].tolist(), skip_specials=True).replace(" ", "")
                pred_fml = formula_tokenizer.decode(gen_fml[i].tolist(), skip_specials=True).replace(" ", "")
                exact_gen_fml_sum += int(pred_fml == true_fml)
                exact_gen_fml_count += 1

    # ---- aggregate ----
    mean_loss_smi = total_loss_smi / max(n_tok_smi, 1)
    mean_loss_fml = total_loss_fml / max(n_tok_fml, 1)
    mean_loss_combined = float(mean_loss_smi + formula_loss_weight * mean_loss_fml)

    metrics = {
        # combined (for "best checkpoint" comparisons, etc.)
        "val_loss": float(mean_loss_combined),

        # SMILES
        "val_loss_smiles": float(mean_loss_smi),
        "val_token_acc_smiles": float(acc_tok_smi_sum / max(n_batches, 1)),
        "val_sequence_acc_smiles": float(acc_seq_smi_sum / max(n_batches, 1)),
        "val_exact_match_tf_smiles": float(exact_tf_smi_sum / max(exact_tf_smi_count, 1)),
        "val_exact_match_gen_smiles": float(exact_gen_smi_sum / max(exact_gen_smi_count, 1)),
        "val_exact_match_gen_raw_smiles": float(exact_gen_smi_raw_sum / max(exact_gen_smi_count, 1)),

        # Formula
        "val_loss_formula": float(mean_loss_fml),
        "val_token_acc_formula": float(acc_tok_fml_sum / max(n_batches, 1)),
        "val_sequence_acc_formula": float(acc_seq_fml_sum / max(n_batches, 1)),
        "val_exact_match_tf_formula": float(exact_tf_fml_sum / max(exact_tf_fml_count, 1)),
        "val_exact_match_gen_formula": float(exact_gen_fml_sum / max(exact_gen_fml_count, 1)),
    }

    if topk:
        metrics["val_exact_match_gen_top5_smiles"] = float(exact_gen_smi_top5_sum / max(exact_gen_smi_top5_count, 1))
        metrics["val_exact_match_gen_top5_formula"] = float(exact_gen_fml_top5_sum / max(exact_gen_fml_top5_count, 1))

    if do_tanimoto and len(pred_smiles_all) > 0:
        tanimoto_mean, valid_frac = tanimoto_similarity(pred_smiles_all, true_smiles_all)
        metrics["val_tanimoto_smiles"] = float(tanimoto_mean) if tanimoto_mean == tanimoto_mean else float("nan")
        metrics["val_tanimoto_valid_frac_smiles"] = float(valid_frac)

    if log_count_mae and (count_head is not None) and (elem_list is not None) and (mae_sum is not None) and (mae_count > 0):
        mae = (mae_sum / mae_count)  # [n_elems] on CPU
        # per-element
        for j, e in enumerate(elem_list):
            metrics[f"val_mae_{e}_counts"] = float(mae[j].item())
        # aggregate
        metrics["val_mae_counts_mean"] = float(mae.mean().item())

    return metrics



@torch.no_grad()
def evaluate_product_only(
        encoder: nn.Module,
        decoder: nn.Module,
        loader: DataLoader,
        device: torch.device,
        tokenizer: SmilesTokenizerAdapter | None = None,
        eos_id: int | None = None,
        tanimoto_samples: int = 128,
        reactant_enc: nn.Module | None = None,
        fusion_enc: nn.Module | None = None,
        spectrum_enc: bool = True,
        formula_enc: nn.Module | None = None,
        temperature: float = 1.0,
        topk: bool = False,
) -> dict:

    set_modules_eval(encoder, decoder, reactant_enc, fusion_enc, formula_enc)

    total_loss = 0.0
    n_tokens = 0
    acc_tok_sum = 0.0
    acc_seq_sum = 0.0
    n_batches = 0

    # ---- NEW: match full evaluate() behaviour ----
    exact_tf_sum = 0
    exact_tf_count = 0
    exact_gen_sum = 0
    exact_gen_raw = 0
    exact_gen_count = 0
    if topk:
        exact_gen_top5_sum = 0
        exact_gen_top5_count = 0

    pred_smiles_all, true_smiles_all = [], []
    do_tanimoto = (tokenizer is not None) and (eos_id is not None) and tanimoto_samples > 0

    for batch in loader:
        formula_strs = None

        if reactant_enc is not None and formula_enc is not None:
            spec_batch, reactant_ids, formula_ids, formula_strs, tgt_in, tgt_out = batch
        elif reactant_enc is not None:
            spec_batch, reactant_ids, tgt_in, tgt_out = batch
            formula_ids = None
        elif formula_enc is not None:
            spec_batch, formula_ids, formula_strs, tgt_in, tgt_out = batch
            reactant_ids = None
        else:
            spec_batch, tgt_in, tgt_out = batch
            reactant_ids = None
            formula_ids = None

        spec_batch = spec_batch.to(device, non_blocking=True)
        tgt_in  = tgt_in.to(device, non_blocking=True)
        tgt_out = tgt_out.to(device, non_blocking=True)
        if reactant_ids is not None:
            reactant_ids = reactant_ids.to(device, non_blocking=True)
        if formula_ids is not None:
            formula_ids = formula_ids.to(device, non_blocking=True)
        # ---- forward pass (spectrum + optional reactant encoder) ----
        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
            memory = build_encoder_memory(
                encoder=encoder,
                decoder=decoder,
                spec_batch=spec_batch,
                spectrum_enc=spectrum_enc,
                reactant_enc=reactant_enc,
                reactant_ids=reactant_ids,
                fusion_enc=fusion_enc,
                formula_enc=formula_enc,
                formula_ids=formula_ids,
            )
            logits = decoder(tgt_in, memory)

        # ---- loss in float32 ----
        loss = decoder.compute_loss(logits.float(), tgt_out)

        # In product-only mode, tgt_out contains only product tokens (+PAD)
        tf_metrics = teacher_forcing_metrics(logits, tgt_out, decoder.pad_id)
        tokens_this_batch = tf_metrics["token_count"]
        total_loss += float(loss.item()) * max(tokens_this_batch, 1)
        n_tokens    += max(tokens_this_batch, 1)
        acc_tok_sum += tf_metrics["acc_tok"]
        acc_seq_sum += tf_metrics["acc_seq"]
        n_batches   += 1

        # ---- NEW: teacher-forced exact match on whole product ----
        exact_tf_sum   += (tf_metrics["seq_correct"] & tf_metrics["has_tokens"]).sum().item()
        exact_tf_count += tf_metrics["has_tokens"].sum().item()

        # ---- NEW: autoregressive generation in product-only mode ----
        if (tokenizer is not None) and (eos_id is not None):
            B, T = tgt_out.shape
            bos_id = tokenizer.bos_id
            prefixes = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

            if topk:
                num_samples = 5
                prefixes_rep = prefixes.repeat_interleave(num_samples, dim=0)

                memory_rep = repeat_memory(memory, num_samples)

                with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                    gen_ids_rep = decoder.generate_from_prefix(
                        prefix_ids=prefixes_rep,
                        memory=memory_rep,
                        eos_id=eos_id,
                        pad_id=decoder.pad_id,
                        max_new_tokens=T,
                        temperature=temperature,
                        topk=50,
                    )

                T_gen = gen_ids_rep.size(1)
                gen_ids = gen_ids_rep.view(B, num_samples, T_gen)

                for i in range(B):
                    true_prod_raw = tokenizer.decode(tgt_out[i].tolist(), skip_specials=True)
                    true_prod_can = canonicalize_smiles(true_prod_raw)
                    true_auto = true_prod_can if true_prod_can is not None else true_prod_raw

                    top1_done = False
                    top5_hit = False

                    for s in range(num_samples):
                        pred_prod_raw = tokenizer.decode(gen_ids[i, s].tolist(), skip_specials=True)
                        pred_prod_can = canonicalize_smiles(pred_prod_raw)
                        pred_auto = pred_prod_can if pred_prod_can is not None else pred_prod_raw

                        if not top1_done:
                            exact_gen_sum += int(pred_auto == true_auto)
                            exact_gen_raw += int(pred_prod_raw == true_prod_raw)
                            exact_gen_count += 1
                            top1_done = True

                            if do_tanimoto and len(pred_smiles_all) < tanimoto_samples:
                                pred_smiles_all.append(pred_auto)
                                true_smiles_all.append(true_auto)

                        if pred_auto == true_auto:
                            top5_hit = True

                    exact_gen_top5_sum += int(top5_hit)
                    exact_gen_top5_count += 1

            else:
                with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                    gen_ids = decoder.generate_from_prefix(
                        prefix_ids=prefixes,
                        memory=memory,
                        eos_id=eos_id,
                        pad_id=decoder.pad_id,
                        max_new_tokens=T,
                        temperature=temperature,
                    )

                for i in range(B):
                    true_prod_raw = tokenizer.decode(tgt_out[i].tolist(), skip_specials=True)
                    pred_prod_raw = tokenizer.decode(gen_ids[i].tolist(), skip_specials=True)

                    true_prod_can = canonicalize_smiles(true_prod_raw)
                    pred_prod_can = canonicalize_smiles(pred_prod_raw)

                    # fall back to raw if canonicalization fails
                    true_auto = true_prod_can if true_prod_can is not None else true_prod_raw
                    pred_auto = pred_prod_can if pred_prod_can is not None else pred_prod_raw

                    exact_gen_sum += int(pred_auto == true_auto)
                    exact_gen_raw += int(pred_prod_raw == true_prod_raw)
                    exact_gen_count += 1

                    # optional: Tanimoto collection (canonical-ish)
                    if do_tanimoto and len(pred_smiles_all) < tanimoto_samples:
                        pred_smiles_all.append(pred_auto)
                        true_smiles_all.append(true_auto)

    # ---- aggregate metrics ----
    mean_loss     = total_loss / max(n_tokens, 1)
    mean_acc_tok  = acc_tok_sum / max(n_batches, 1)
    mean_acc_seq  = acc_seq_sum / max(n_batches, 1)
    acc_tf        = float(exact_tf_sum / max(exact_tf_count, 1))
    acc_gen       = float(exact_gen_sum / max(exact_gen_count, 1))
    acc_gen_raw   = float(exact_gen_raw / max(exact_gen_count, 1))
    if topk:
        acc_gen_top5 = float(exact_gen_top5_sum / max(exact_gen_top5_count, 1))

    metrics = {
        "val_loss": float(mean_loss),
        "val_token_acc": float(mean_acc_tok),
        "val_sequence_acc": float(mean_acc_seq),
        "val_exact_match_tf": acc_tf,
        "val_exact_match_gen": acc_gen,
        "val_exact_match_gen_raw": acc_gen_raw,
    }

    if topk:
        metrics["val_exact_match_gen_top5"] = acc_gen_top5  # top-5 sampling accuracy

    if do_tanimoto and len(pred_smiles_all) > 0:
        tanimoto_mean, valid_frac = tanimoto_similarity(pred_smiles_all, true_smiles_all)
        metrics["val_tanimoto"] = float(tanimoto_mean) if tanimoto_mean == tanimoto_mean else float("nan")
        metrics["val_tanimoto_valid_frac"] = float(valid_frac)

    return metrics


@torch.no_grad()
def evaluate(
        encoder: nn.Module,
        decoder: nn.Module,
        loader: DataLoader,
        device: torch.device,
        tokenizer: SmilesTokenizerAdapter | None = None,
        eos_id: int | None = None,
        tanimoto_samples: int = 128,
        reactant_enc: nn.Module | None = None,
        spectrum_enc: bool = True,
        temperature: float = 1.0,) -> dict:

    encoder.eval()
    decoder.eval()
    if reactant_enc is not None:
        reactant_enc.eval()

    total_loss = 0.0
    n_tokens = 0
    acc_tok_sum = 0.0
    acc_seq_sum = 0.0
    n_batches = 0

    exact_tf_sum = 0
    exact_tf_count = 0
    exact_gen_sum = 0
    exact_gen_raw = 0
    exact_gen_count = 0

    pred_smiles_all, true_smiles_all = [], []
    do_tanimoto = (tokenizer is not None) and (eos_id is not None) and tanimoto_samples > 0

    rxn_sep_id = getattr(decoder, "rxn_sep_id")

    for batch in loader:
        if reactant_enc is not None:
            spec_batch, reactant_ids, tgt_in, tgt_out = batch
            reactant_ids = reactant_ids.to(device, non_blocking=True)
        else:
            spec_batch, tgt_in, tgt_out = batch
        spec_batch = spec_batch.to(device, non_blocking=True)
        tgt_in  = tgt_in.to(device, non_blocking=True)
        tgt_out = tgt_out.to(device, non_blocking=True)

        # ---- fast forward under AMP ----
        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
            spec_memory = None
            if spectrum_enc:
                spec_memory = encoder(spec_batch)

            if reactant_enc is not None:
                rxn_memory = reactant_enc(reactant_ids)
                if spectrum_enc:
                    memory = torch.cat([spec_memory, rxn_memory], dim=1)
                else:
                    memory = rxn_memory
            else:
                memory = spec_memory

            logits = decoder(tgt_in, memory)

        # ---- stable loss in float32 ----
        # (ensure your compute_loss reduces in fp32; casting logits is the simplest)
        loss = decoder.compute_loss(logits.float(), tgt_out)

        # product masks & metrics
        prod_mask  = decoder.build_product_mask(tgt_out)
        valid_mask = prod_mask & (tgt_out != decoder.pad_id)

        # argmax is fine in bf16/fp16, but casting is cheap and safe
        pred = logits.float().argmax(dim=-1)

        acc_tok = token_accuracy(pred, tgt_out, valid_mask)
        seq_correct = ((pred == tgt_out) | (~valid_mask)).all(dim=1)
        has_prod = valid_mask.any(dim=1)
        acc_seq = (seq_correct & has_prod).sum().float() / has_prod.sum().clamp_min(1)

        tokens_this_batch = int(valid_mask.sum().item())
        total_loss += float(loss.item()) * max(tokens_this_batch, 1)
        n_tokens    += max(tokens_this_batch, 1)
        acc_tok_sum += acc_tok
        acc_seq_sum += acc_seq
        n_batches   += 1

        exact_tf_sum   += (seq_correct & has_prod).sum().item()
        exact_tf_count += has_prod.sum().item()

        # ---- autoregressive generation ----
        if (tokenizer is not None) and (eos_id is not None):
            prefixes, prefix_lens, arrow_col = build_prefix_batch(tgt_in, rxn_sep_id, decoder.pad_id)

            # Keep model matmuls fast with AMP during generation too
            with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                gen_ids = decoder.generate_from_prefix(
                    prefix_ids=prefixes,
                    memory=memory,
                    eos_id=eos_id,
                    max_new_tokens=tgt_in.size(1),
                    pad_id=decoder.pad_id,
                    temperature=temperature,
                )

            # IMPORTANT: If generate_from_prefix internally does softmax/sampling,
            # make sure INSIDE that method it casts logits to float() before softmax.
            # (i.e., logits = logits.float(); probs = torch.softmax(logits, dim=-1))

            for i in range(gen_ids.size(0)):
                pred_rxn = tokenizer.decode(gen_ids[i].tolist(), skip_specials=True)
                true_rxn = tokenizer.decode(tgt_out[i].tolist(),  skip_specials=True)

                true_prod_raw = true_rxn.split(">>", 1)[1] if ">>" in true_rxn else true_rxn
                pred_prod_raw = pred_rxn.split(">>", 1)[1] if ">>" in pred_rxn else pred_rxn

                true_prod_can = canonicalize_smiles(true_prod_raw)
                pred_prod_can = canonicalize_smiles(pred_prod_raw)

                # Count only if both are valid; otherwise skip this sample for gen exact-match
                pred_auto = pred_prod_can if pred_prod_can is not None else pred_prod_raw
                true_auto = true_prod_can if true_prod_can is not None else true_prod_raw
                exact_gen_sum += int(pred_auto == true_auto)
                exact_gen_raw += int(pred_prod_raw == true_prod_raw)
                exact_gen_count += 1

                # For Tanimoto collection (it will parse anyway), you may store canonical forms:
                if do_tanimoto and len(pred_smiles_all) < tanimoto_samples:
                    pred_smiles_all.append(pred_auto)
                    true_smiles_all.append(true_auto)

    mean_loss     = total_loss / max(n_tokens, 1)
    mean_acc_tok  = acc_tok_sum / max(n_batches, 1)
    mean_acc_seq  = acc_seq_sum / max(n_batches, 1)
    acc_tf        = float(exact_tf_sum / max(exact_tf_count, 1))
    acc_gen       = float(exact_gen_sum / max(exact_gen_count, 1))
    acc_gen_raw   = float(exact_gen_raw / max(exact_gen_count, 1))

    metrics = {
        "val_loss": float(mean_loss),
        "val_token_acc": float(mean_acc_tok),
        "val_sequence_acc": float(mean_acc_seq),
        "val_exact_match_tf": acc_tf,
        "val_exact_match_gen": acc_gen,
        "val_exact_match_gen_raw": acc_gen_raw,
    }

    if do_tanimoto and len(pred_smiles_all) > 0:
        tanimoto_mean, valid_frac = tanimoto_similarity(pred_smiles_all, true_smiles_all)
        metrics["val_tanimoto"] = float(tanimoto_mean) if tanimoto_mean == tanimoto_mean else float("nan")
        metrics["val_tanimoto_valid_frac"] = float(valid_frac)

    return metrics


@torch.no_grad()
def evaluate_beam_molecule_accuracy(
    encoder, decoder, loader, device,
    tokenizer, eos_id,
    max_len: int = 256,
    beam_size: int = 5,
    limit_samples: int = 128,   # evaluate on a subset for speed
):
    """
    Returns dict with:
      - gen_mol_acc_top1: fraction where best beam exactly matches target
      - gen_mol_acc_top5: fraction where any of top-5 beams matches target
    """

    encoder.eval()
    decoder.eval()

    n_eval = 0
    correct_top1 = 0
    correct_top5 = 0

    rxn_sep_id = getattr(decoder, "rxn_sep_id")

    for spec_batch, _, tgt_out in loader:
        if n_eval >= limit_samples: break

        B = min(spec_batch.size(0), limit_samples - n_eval)
        spec_batch = spec_batch[:B].to(device)
        tgt_out    = tgt_out[:B].to(device)

        # Encode spectra once
        memory = encoder(spec_batch)

        # ---- simple per-sample beam search (batch loop for clarity) ----
        for i in range(B):
            # Build reactant prefix (includes BOS and '>>' if present)
            row = tgt_out[i]
            arrow = (row == rxn_sep_id).nonzero(as_tuple=False)
            if arrow.numel() == 0:
                prefix = row[:1]  # BOS only fallback
            else:
                j = int(arrow[0].item())
                prefix = row[:j+1]

            beams = [(0.0, prefix.clone())]   # start from prefix
            finished = []

            steps = max_len - prefix.size(0)
            for _ in range(max(steps, 1)):
                new_beams = []
                for neg_lp, seq in beams:
                    if seq[-1].item() == eos_id:
                        finished.append((neg_lp, seq))
                        continue
                    logits = decoder(seq.unsqueeze(0), memory[i:i+1])[:, -1, :]  # [1,V]
                    logp = torch.log_softmax(logits, dim=-1).squeeze(0)          # [V]
                    topk_logp, topk_idx = torch.topk(logp, k=beam_size, dim=-1) # [K]
                    for add_lp, tok in zip(topk_logp.tolist(), topk_idx.tolist()):
                        new_seq = torch.cat([seq, torch.tensor([tok], device=device)])
                        new_beams.append((neg_lp - add_lp, new_seq))
                # keep best K
                beams = heapq.nsmallest(beam_size, new_beams, key=lambda x: x[0])
                # early stop if we already have K finished sequences
                if len(finished) >= beam_size:
                    break

            candidates = finished if finished else beams
            candidates = sorted(candidates, key=lambda x: x[0])[:beam_size]

            tgt_str = tokenizer.decode(row.tolist(), skip_specials=True)
            cand_strs = [tokenizer.decode(seq.tolist(), skip_specials=True) for _, seq in candidates]

            if len(cand_strs) > 0 and cand_strs[0] == tgt_str:
                correct_top1 += 1
            if any(s == tgt_str for s in cand_strs[:5]):
                correct_top5 += 1

        n_eval += B

    return {
        "gen_mol_acc_top1": correct_top1 / max(n_eval, 1),
        "gen_mol_acc_top5": correct_top5 / max(n_eval, 1),
        "gen_mol_eval_N": n_eval,
    }
