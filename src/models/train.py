# src/train.py
from __future__ import annotations
import argparse
import math
import time
import os
import json
from typing import Callable, List, Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

import mlflow
import mlflow.pytorch

from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count
from urllib.parse import urlparse
from rdkit import RDLogger

# ==== your modules (adjust paths if needed) ====

from src.data.tokenizers import SmilesTokenizerAdapter, FormulaTokenizerAdapter
from src.models.spec2prod import Encoder, Decoder, ReactantEncoder, FormulaEncoder, FusionEncoder, ElementCountHead
from src.data.loaders import SpecSmilesDataset, load_table, parse_spectrum
from src.utils.utils_train import log_autoregressive, make_worker_init_fn, log_autoregressive_product_only, canonicalize_smiles, build_prefix_batch, load_smarts_dict, compute_substructure_confidence_when_true_weighted, compile_smarts
from src.models.configs import TrainConfig
from src.models.metrics import SubstructureStats
from src.utils.model_runtime import build_encoder_memory, repeat_memory, set_modules_eval, set_modules_train
from src.utils.train_runtime import (
    apply_and_log_lrs,
    build_model_components,
    count_head_loss,
    count_loss_tokens,
    epoch_lrs,
    evaluate_for_config,
    gradient_clip_parameters,
    print_epoch_summary,
    select_collate_fn,
    unpack_training_batch,
    validate_config,
)
from src.data.preprocess import get_sorted_reactants

RDLogger.DisableLog('rdApp.*')
# ---- enable fast TF32 operations (Ampere+ GPUs only) ----
torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"




# ------------------------------
# Training / Evaluation
# ------------------------------
def train_one_epoch(
        cfg: TrainConfig,
        encoder: Encoder,
        decoder: Decoder,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scaler: GradScaler,
        grad_clip: float = 1.0,
        amp_dtype: torch.dtype = torch.bfloat16,
        reactant_enc: ReactantEncoder | None = None,
        fusion_enc: FusionEncoder | None = None,
        spectrum_enc: bool = True,
        formula_enc: FormulaEncoder | None = None,
        formula_dec: Decoder | None = None,
        count_head: ElementCountHead | None = None,
        formula_tokenizer: FormulaTokenizerAdapter | None = None,
        ) -> float:
    set_modules_train(encoder, decoder, reactant_enc, fusion_enc, formula_enc, formula_dec, count_head)

    if not spectrum_enc and (reactant_enc is None) and (formula_enc is None):
        raise ValueError("No input encoder active: enable spectrum_encoder or reactant_encoder or formula_encoder.")

    if formula_enc is not None and formula_dec is not None:
        raise ValueError("Cannot use both formula_encoder and formula_decoder at the same time (data leakage).")

    total_loss = 0.0
    n_tokens = 0

    for batch in loader:
        batch_data = unpack_training_batch(batch, cfg, device, reactant_enc, formula_enc, formula_dec)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type='cuda', dtype=amp_dtype, enabled=device.type == 'cuda'):
            memory = build_encoder_memory(
                encoder=encoder,
                decoder=decoder,
                spec_batch=batch_data["spec_batch"],
                spectrum_enc=spectrum_enc,
                reactant_enc=reactant_enc,
                reactant_ids=batch_data["reactant_ids"],
                fusion_enc=fusion_enc,
                formula_enc=formula_enc,
                formula_ids=batch_data["formula_ids"],
            )

            if formula_dec is not None and cfg.train_formula_only and reactant_enc is None:
                logits_fml, hidden_fml = formula_dec(batch_data["fml_in"], memory, return_hidden=True)
                loss = formula_dec.compute_loss(logits_fml.float(), batch_data["fml_out"])

                if (count_head is not None) and (formula_tokenizer is not None) and cfg.use_count_head:
                    loss = loss + cfg.count_loss_weight * count_head_loss(
                        cfg, count_head, formula_tokenizer, formula_dec, hidden_fml, batch_data["fml_in"], batch_data["fml_out"]
                    )

            elif formula_dec is not None and not cfg.train_formula_only:
                logits_fml, hidden_fml = formula_dec(batch_data["fml_in"], memory, return_hidden=True)
                loss_fml = formula_dec.compute_loss(logits_fml.float(), batch_data["fml_out"])
                if (count_head is not None) and (formula_tokenizer is not None) and cfg.use_count_head:
                    loss_fml = loss_fml + cfg.count_loss_weight * count_head_loss(
                        cfg, count_head, formula_tokenizer, formula_dec, hidden_fml, batch_data["fml_in"], batch_data["fml_out"]
                    )

                logits_smi = decoder(batch_data["smi_in"], memory)
                loss_smi = decoder.compute_loss(logits_smi.float(), batch_data["smi_out"])
                loss = loss_smi + cfg.formula_loss_weight * loss_fml
            else:
                logits = decoder(batch_data["tgt_in"], memory)
                loss = decoder.compute_loss(logits.float(), batch_data["tgt_out"])

        scaler.scale(loss).backward()
        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            params = gradient_clip_parameters(cfg, encoder, decoder, reactant_enc, fusion_enc, formula_enc, formula_dec, count_head)
            torch.nn.utils.clip_grad_norm_(params, grad_clip)

        scaler.step(optimizer)
        scaler.update()

        pad_for_count = (formula_dec.pad_id if (formula_dec is not None and cfg.train_formula_only and reactant_enc is None) else decoder.pad_id)
        tokens_this_batch = count_loss_tokens(decoder, batch_data["tgt_out"], pad_for_count)
        total_loss += float(loss.item()) * max(tokens_this_batch, 1)
        n_tokens += max(tokens_this_batch, 1)

    return total_loss / max(n_tokens, 1)


def log_val_predictions_csv(
        epoch: int,
        encoder: Encoder,
        decoder: Decoder,
        loader_val: DataLoader,
        device: torch.device,
        tokenizer: SmilesTokenizerAdapter,
        eos_id: int,
        out_dir: Path,
        limit_samples: int = 200,
        reactant_enc: ReactantEncoder | None = None,
        fusion_enc: FusionEncoder | None = None,
        spectrum_enc: bool = True,
        formula_enc: FormulaEncoder | None = None,
        formula_dec: Decoder | None = None,
        formula_tokenizer: FormulaTokenizerAdapter | None = None,
        formula_eos_id: int | None = None,
        temperature: float = 1.0,
        formula_only: bool = False,
        substructure_dict: dict[str, str] | None = None,  # fg_name -> SMARTS
        k_samples: int = 10,  # number of sampled preds per example
        topk_sampling: int = 50) -> str:

    # full validation only every 10th epoch
    max_rows = None if (epoch % 10 == 0) else int(limit_samples)

    set_modules_eval(encoder, decoder, reactant_enc, fusion_enc, formula_enc, formula_dec)

    decoder_mode = getattr(decoder, "decoder_mode", "rxn")
    rxn_sep_id = getattr(decoder, "rxn_sep_id", None)

    compiled_smarts = None
    if (substructure_dict is not None) and (not formula_only) and (decoder_mode == "product"):
        compiled_smarts = compile_smarts(substructure_dict)
        if len(compiled_smarts) == 0:
            compiled_smarts = None  # avoid writing empty junk

    rows_gen = []
    seen = 0

    with torch.inference_mode(), autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
        for batch in loader_val:
            # stop early only if we are sampling
            if max_rows is not None and seen >= max_rows:
                break

            formula_strs = None
            formula_ids = None

            if reactant_enc is not None and formula_dec is not None:
                spec_batch, reactant_ids, fml_in, fml_out, formula_strs, smi_in, smi_out = batch

            elif formula_dec is not None and formula_only and reactant_enc is None:
                spec_batch, fml_in, fml_out, formula_strs = batch
                reactant_ids = None
                formula_ids = None
                tgt_in = tgt_out = None

            elif formula_dec is not None and reactant_enc is None:
                spec_batch, fml_in, fml_out, formula_strs, smi_in, smi_out = batch
                reactant_ids = None
                formula_ids = None

            elif reactant_enc is not None and formula_enc is not None:
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

            # how many rows to take from this batch
            if max_rows is None:
                B = int(spec_batch.size(0))  # full batch
            else:
                remaining = max_rows - seen
                if remaining <= 0:
                    break
                B = int(min(spec_batch.size(0), remaining))

            if B <= 0:
                break

            spec_batch = spec_batch[:B].to(device, non_blocking=True)
            if formula_dec is not None:
                # always move formula tensors
                fml_in = fml_in[:B].to(device, non_blocking=True)
                fml_out = fml_out[:B].to(device, non_blocking=True)

                # only move/assign smiles tensors if we actually have them
                if not formula_only:
                    smi_in = smi_in[:B].to(device, non_blocking=True)
                    smi_out = smi_out[:B].to(device, non_blocking=True)
                    tgt_in, tgt_out = smi_in, smi_out
                else:
                    tgt_in = tgt_out = None
            else:
                tgt_in = tgt_in[:B].to(device, non_blocking=True)
                tgt_out = tgt_out[:B].to(device, non_blocking=True)
            if reactant_ids is not None:
                reactant_ids = reactant_ids[:B].to(device, non_blocking=True)
            if formula_ids is not None:
                formula_ids = formula_ids[:B].to(device, non_blocking=True)
            if formula_strs is not None:
                formula_strs = formula_strs[:B]

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

            # ---- Autoregressive generation ----
            if decoder_mode == "product":
                if formula_dec is not None and formula_only:
                    if formula_tokenizer is None or formula_eos_id is None:
                        raise ValueError("formula_only=True requires formula_tokenizer and formula_eos_id")

                    log_dict_fml = log_autoregressive_product_only(
                        decoder=formula_dec,
                        memory=memory,
                        tgt_out=fml_out[:B].to(device, non_blocking=True),
                        tokenizer=formula_tokenizer,
                        eos_id=formula_eos_id,
                        pad_id=formula_dec.pad_id,
                        reactant_ids=reactant_ids,
                        formulas=formula_strs,
                        temperature=temperature,
                    )

                    # rename keys so CSV is clear and DOESN'T contain a SMILES pred column
                    for r in log_dict_fml:
                        r["pred_formula"] = r.pop("pred_product_gen", "")
                        r["true_formula"] = r.pop("true_product", "")
                        # if log_autoregressive_product_only gave exact_match_gen already, reuse it:
                        r["formula_exact"] = r.pop("exact_match_gen", 0)

                        # optional: drop smile-specific fields if they exist
                        r.pop("exact_match_gen_raw", None)
                        r.pop("full_true_rxn", None)
                        r.pop("full_pred_rxn_gen", None)

                    log_dict_gen = log_dict_fml  # write these rows

                else:
                    log_dict_gen = log_autoregressive_product_only(
                        decoder=decoder,
                        memory=memory,
                        tgt_out=tgt_out,
                        tokenizer=tokenizer,
                        eos_id=eos_id,
                        pad_id=decoder.pad_id,
                        reactant_ids=reactant_ids,
                        formulas=formula_strs,
                        reactant_tokenizer=tokenizer,
                        temperature=temperature
                    )
                    if formula_dec is not None:
                        log_dict_fml = log_autoregressive_product_only(
                            decoder=formula_dec,
                            memory=memory,
                            tgt_out=fml_out[:B].to(device, non_blocking=True),  # true formula tokens
                            tokenizer=formula_tokenizer,
                            eos_id=formula_eos_id,
                            pad_id=formula_dec.pad_id,
                            reactant_ids=reactant_ids,
                            formulas=formula_strs,  # these are true formula strings; good to keep in CSV
                            reactant_tokenizer=tokenizer,
                            temperature=temperature
                        )
                        # merge columns into the smiles rows
                        # assume both lists have length B and same order
                        # after you computed log_dict_fml
                        for i in range(len(log_dict_gen)):
                            # predicted formula string from formula decoder generation
                            log_dict_gen[i]["pred_formula"] = log_dict_fml[i].get("pred_product_gen", "")

                            # true formula string decoded from fml_out tokens
                            log_dict_gen[i]["true_formula"] = log_dict_fml[i].get("true_product", "")

                            # robust exact match (ignore spaces)
                            pf = (log_dict_gen[i]["pred_formula"] or "").replace(" ", "")
                            tf = (log_dict_gen[i]["true_formula"] or "").replace(" ", "")
                            log_dict_gen[i]["formula_exact"] = int(pf == tf)

            else:
                log_dict_gen = log_autoregressive(
                    decoder, rxn_sep_id, memory, tgt_out, tokenizer, eos_id, tgt_in, temperature=temperature
                )

            if (compiled_smarts is not None) and (decoder_mode == "product") and (not formula_only):
                B_eff = len(log_dict_gen)
                k = max(int(k_samples), 1)

                prefixes = torch.full((B_eff, 1), tokenizer.bos_id, dtype=torch.long, device=device)
                prefixes_rep = prefixes.repeat_interleave(k, dim=0)

                memory_rep = repeat_memory(memory, k)

                gen_ids_rep, logp_rep = decoder.generate_from_prefix(
                    prefix_ids=prefixes_rep,
                    memory=memory_rep,
                    eos_id=eos_id,
                    pad_id=decoder.pad_id,
                    max_new_tokens=tgt_out.size(1),
                    temperature=temperature,
                    topk=topk_sampling,
                    return_logprobs=True,
                )

                Tgen = gen_ids_rep.size(1)
                gen_ids = gen_ids_rep.view(B_eff, k, Tgen)
                logp = logp_rep.view(B_eff, k)

                for i in range(B_eff):
                    pred_samples = []
                    for s in range(k):
                        raw = tokenizer.decode(gen_ids[i, s].tolist(), skip_specials=True)
                        can = canonicalize_smiles(raw)
                        pred_samples.append(can if can is not None else raw)

                    w = torch.softmax(logp[i] - logp[i].max(), dim=0).detach().float().cpu().tolist()

                    pred_top1 = log_dict_gen[i].get("pred_product_gen", "")
                    conf_list = compute_substructure_confidence_when_true_weighted(
                        pred_top1=pred_top1,
                        pred_samples=pred_samples,
                        weights=w,
                        compiled_smarts=compiled_smarts,
                    )

                    # store JSON list-of-dicts in ONE csv cell
                    log_dict_gen[i]["substructure_confidence"] = json.dumps(conf_list)

            rows_gen.extend(log_dict_gen)
            seen += B

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_gen = out_dir / f"val_generated_epoch_{epoch:03d}.csv"
    pd.DataFrame(rows_gen).to_csv(csv_gen, index=False)
    return str(csv_gen)



@torch.no_grad()
def log_substructure_confidence_csv(
        epoch: int,
        encoder: Encoder,
        decoder: Decoder,
        loader_val: DataLoader,
        device: torch.device,
        tokenizer: SmilesTokenizerAdapter,
        eos_id: int,
        out_dir: Path,
        substructure_dict: dict[str, str],   # fg_name -> SMARTS
        num_samples_per_prefix: int = 10,
        limit_samples: int = 200,
        reactant_enc: ReactantEncoder | None = None,
        fusion_enc: FusionEncoder | None = None,
        formula_enc: FormulaEncoder | None = None,
        spectrum_enc: bool = True,
        temperature: float = 1.0,) -> str:

    set_modules_eval(encoder, decoder, reactant_enc, fusion_enc, formula_enc)

    decoder_mode = getattr(decoder, "decoder_mode", "rxn")
    rxn_sep_id = getattr(decoder, "rxn_sep_id", None)

    # dataset-level stats accumulator
    sub_stats = SubstructureStats(substructure_dict)

    seen = 0  # number of examples actually used (after filtering)

    with torch.inference_mode(), autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
        for batch in loader_val:
            if seen >= limit_samples:
                break

            formula_strs = None

            # multitask with reactants:
            # (spec, reactant_ids, fml_in, fml_out, formula_strs, smi_in, smi_out)
            if formula_enc is None and reactant_enc is not None and len(batch) == 7:
                spec_batch, reactant_ids, _, _, formula_strs, tgt_in, tgt_out = batch
                formula_ids = None

            # multitask without reactants:
            # (spec, fml_in, fml_out, formula_strs, smi_in, smi_out)
            elif formula_enc is None and reactant_enc is None and len(batch) == 6:
                spec_batch, _, _, formula_strs, tgt_in, tgt_out = batch
                reactant_ids = None
                formula_ids = None

            # product-only with reactants + formula encoder input
            elif reactant_enc is not None and formula_enc is not None:
                spec_batch, reactant_ids, formula_ids, formula_strs, tgt_in, tgt_out = batch

            # product-only with reactants
            elif reactant_enc is not None:
                spec_batch, reactant_ids, tgt_in, tgt_out = batch
                formula_ids = None

            # product-only with formula encoder input
            elif formula_enc is not None:
                spec_batch, formula_ids, formula_strs, tgt_in, tgt_out = batch
                reactant_ids = None

            # plain product-only
            else:
                spec_batch, tgt_in, tgt_out = batch
                reactant_ids = None
                formula_ids = None

            B = spec_batch.size(0)

            spec_batch = spec_batch.to(device, non_blocking=True)
            tgt_in     = tgt_in.to(device, non_blocking=True)
            tgt_out    = tgt_out.to(device, non_blocking=True)
            if reactant_ids is not None:
                reactant_ids = reactant_ids.to(device, non_blocking=True)

            if formula_ids is not None:
                formula_ids = formula_ids.to(device, non_blocking=True)

            # ---- optionally filter rows for rxn-mode (must contain '>>') ----
            if decoder_mode == "rxn":
                if rxn_sep_id is None:
                    raise ValueError("decoder_mode='rxn' but rxn_sep_id is None")

                has_arrow = (tgt_in == rxn_sep_id).any(dim=1)  # [B]
                if not has_arrow.any():
                    # no valid reaction rows in this batch
                    continue

                spec_batch = spec_batch[has_arrow]
                tgt_in     = tgt_in[has_arrow]
                tgt_out    = tgt_out[has_arrow]
                if reactant_ids is not None:
                    reactant_ids = reactant_ids[has_arrow]

                B = spec_batch.size(0)

            # enforce global example limit
            if seen + B > limit_samples:
                keep = limit_samples - seen
                spec_batch = spec_batch[:keep]
                tgt_in     = tgt_in[:keep]
                tgt_out    = tgt_out[:keep]
                if reactant_ids is not None:
                    reactant_ids = reactant_ids[:keep]
                if formula_ids is not None:
                    formula_ids = formula_ids[:keep]
                if formula_strs is not None:
                    formula_strs = formula_strs[:keep]

                B = keep

            if B <= 0:
                break

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


            num_samples = max(num_samples_per_prefix, 1)

            # ----- build prefixes & generate -----
            if decoder_mode == "rxn":
                # same pattern as log_autoregressive
                prefixes, _, _ = build_prefix_batch(
                    tgt_in_batch=tgt_in,
                    rxn_sep_id=rxn_sep_id,
                    pad_id=decoder.pad_id,
                )
            elif decoder_mode == "product":
                # product-only: start from first token (usually BOS)
                # tgt_in: [B, T] (BOS, y_0, y_1, ...)
                prefixes = tgt_in[:, :1]   # [B, 1]
            else:
                raise ValueError(f"Unsupported decoder_mode: {decoder_mode}")

            prefixes_rep = prefixes.repeat_interleave(num_samples, dim=0)
            memory_rep = repeat_memory(memory, num_samples)

            gen_ids_rep, logp_rep = decoder.generate_from_prefix(
                prefix_ids=prefixes_rep,
                memory=memory_rep,
                eos_id=eos_id,
                max_new_tokens=tgt_in.size(1),
                pad_id=decoder.pad_id,
                temperature=temperature,
                topk=50,
                return_logprobs=True,
            )

            # reshape to [B, num_samples, T_gen]
            T_gen = gen_ids_rep.size(1)
            gen_ids = gen_ids_rep.view(B, num_samples, T_gen)
            logp = logp_rep.view(B, num_samples)  # [B, k]

            # ----- per-example: decode & update stats -----
            for i in range(B):
                # true product
                if decoder_mode == "rxn":
                    true_rxn = tokenizer.decode(tgt_out[i].tolist(), skip_specials=True)
                    true_prod_raw = true_rxn.split(">>", 1)[1] if ">>" in true_rxn else true_rxn
                else:  # "product"
                    true_prod_raw = tokenizer.decode(tgt_out[i].tolist(), skip_specials=True)

                true_prod_can = canonicalize_smiles(true_prod_raw)
                true_smiles = true_prod_can if true_prod_can is not None else true_prod_raw

                # K predictions
                pred_smiles_list = []
                for s in range(num_samples):
                    pred_rxn_or_prod = tokenizer.decode(
                        gen_ids[i, s].tolist(), skip_specials=True
                    )
                    if decoder_mode == "rxn":
                        pred_prod_raw = pred_rxn_or_prod.split(">>", 1)[1] if ">>" in pred_rxn_or_prod else pred_rxn_or_prod
                    else:  # "product"
                        pred_prod_raw = pred_rxn_or_prod

                    pred_prod_can = canonicalize_smiles(pred_prod_raw)
                    pred_auto = pred_prod_can if pred_prod_can is not None else pred_prod_raw
                    pred_smiles_list.append(pred_auto)

                    # --- turn logp[i] into normalized probabilities (weights) ---
                    logp_i = logp[i]  # [k]
                    # subtract max for numerical stability
                    logp_i_stable = logp_i - logp_i.max()
                    weights = torch.softmax(logp_i_stable, dim=0)  # [k], sum to 1
                # update global stats for this example
                sub_stats.update(true_smiles=true_smiles, pred_smiles_list=pred_smiles_list, sample_weights=weights.tolist())

            seen += B
            if seen >= limit_samples:
                break

    # ----- aggregate and write CSV: ONE ROW PER FUNCTIONAL GROUP -----
    rows = sub_stats.summary_rows()

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"val_substructure_confidence_epoch_{epoch:03d}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return str(csv_path)


def train_loop(
        cfg: TrainConfig,
        encoder: Encoder,
        decoder: Decoder,
        loader_val: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        tokenizer: SmilesTokenizerAdapter,
        eos_id: int,
        ds_train: SpecSmilesDataset,
        collate: Callable[[List[Any]], Any],
        ckpt_dir: Path,
        preds_dir: Path,
        reactant_enc: ReactantEncoder | None = None,
        fusion_enc: FusionEncoder | None = None,
        spectrum_enc: bool = True,
        formula_enc: FormulaEncoder | None = None,
        formula_dec: Decoder | None = None,
        formula_tokenizer: FormulaTokenizerAdapter | None = None,
        count_head: ElementCountHead | None = None) -> None:
    #ckpt_dir = Path(cfg.ckpt_dir)
    best_val = math.inf
    global_step = 0
    scaler = GradScaler()

    def should_run_eval(epoch: int) -> bool:
        if cfg.eval_interval <= 0:
            return epoch == cfg.epochs  # only last epoch
        return (epoch % cfg.eval_interval == 0) or (epoch == cfg.epochs)

    for epoch in range(1, cfg.epochs + 1):
        lrs = epoch_lrs(cfg, epoch)
        apply_and_log_lrs(epoch, optimizer, lrs, reactant_enc, fusion_enc, formula_enc, formula_dec)
        ds_train.set_epoch(epoch)  # for shuffling if using a Sampler
        ds_train.set_rand_reactants(cfg.randomize_reactants)  # enable random reactant subset augmentation
        #loader_val.set_epoch(epoch)
        loader_train = DataLoader(ds_train, batch_size=cfg.batch_size, num_workers=cfg.num_workers, prefetch_factor=2, shuffle=True, pin_memory=True, worker_init_fn=make_worker_init_fn(epoch), collate_fn=collate)

        t0 = time.time()
        train_loss = train_one_epoch(cfg, encoder, decoder, loader_train, optimizer, device, scaler, cfg.grad_clip, amp_dtype=torch.bfloat16, reactant_enc=reactant_enc, fusion_enc=fusion_enc, spectrum_enc=spectrum_enc, formula_enc=formula_enc, formula_dec=formula_dec, count_head=count_head, formula_tokenizer=formula_tokenizer)

        do_eval = should_run_eval(epoch)

        if do_eval:
            print(f"Evaluation of Epoch {epoch} started...")

            val_metrics = evaluate_for_config(
                cfg,
                encoder,
                decoder,
                loader_val,
                device,
                tokenizer,
                eos_id,
                reactant_enc=reactant_enc,
                fusion_enc=fusion_enc,
                spectrum_enc=spectrum_enc,
                formula_enc=formula_enc,
                formula_dec=formula_dec,
                formula_tokenizer=formula_tokenizer,
                count_head=count_head,
            )

            print(f"Evaluation of Epoch {epoch} completed.")

            dt = time.time() - t0
            print_epoch_summary(epoch, cfg, train_loss, val_metrics, lrs, dt, formula_dec)

            csv_path = log_val_predictions_csv(
                epoch=epoch,
                encoder=encoder,
                decoder=decoder,
                loader_val=loader_val,
                device=device,
                tokenizer=tokenizer,
                eos_id=eos_id,
                out_dir=preds_dir,
                limit_samples=200,  # tweak as you like
                reactant_enc=reactant_enc,
                fusion_enc = fusion_enc,
                spectrum_enc=spectrum_enc,
                formula_enc=formula_enc,
                formula_dec=formula_dec,
                formula_tokenizer=formula_tokenizer if cfg.formula_decoder else None,
                formula_eos_id=formula_tokenizer.eos_id if cfg.formula_decoder else None,
                temperature=cfg.temperature,
                formula_only=(cfg.formula_decoder and cfg.train_formula_only),
                substructure_dict=load_smarts_dict(cfg.smarts_path) if (cfg.confidence and not cfg.train_formula_only) else None,
                k_samples=cfg.k_samples,
                topk_sampling=50,
            )

            if cfg.confidence and not cfg.train_formula_only:
                csv_sub = log_substructure_confidence_csv(
                    epoch=epoch,
                    encoder=encoder,
                    decoder=decoder,
                    loader_val=loader_val,
                    device=device,
                    tokenizer=tokenizer,
                    eos_id=eos_id,
                    out_dir=preds_dir,
                    substructure_dict=load_smarts_dict(cfg.smarts_path),
                    num_samples_per_prefix=cfg.k_samples,  # top-k samples per prefix
                    limit_samples=200,
                    reactant_enc=reactant_enc,
                    fusion_enc=fusion_enc,
                    spectrum_enc=spectrum_enc,
                    formula_enc=formula_enc,
                    temperature=cfg.temperature,
                )
                print(f"Logged uncertainties per smarts to: {csv_sub}")

            print(f"Logged validation predictions to: {csv_path}")

            # ---- Log metrics per epoch ----
            mlflow.log_metrics({
                "train_loss": float(train_loss),
                **val_metrics,
                #**gen_mol_metrics,
                "epoch_time_sec": float(dt),
            }, step=epoch)

            # checkpoints as before
            if val_metrics["val_loss"] < best_val:
                best_val = val_metrics["val_loss"]
                save_checkpoint(
                    ckpt_dir / "best.pt", encoder, decoder, optimizer, epoch, cfg,
                    reactant_enc=reactant_enc, fusion_enc=fusion_enc, formula_enc=formula_enc,
                    formula_dec=formula_dec, count_head=count_head,
                )
                mlflow.log_metric("best_val_loss", float(best_val), step=epoch)
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch:03d}.pt", encoder, decoder, optimizer, epoch, cfg,
                reactant_enc=reactant_enc, fusion_enc=fusion_enc, formula_enc=formula_enc,
                formula_dec=formula_dec, count_head=count_head,
            )

        else:
            dt = time.time() - t0
            mlflow.log_metrics({
                "train_loss": float(train_loss),
                "epoch_time_sec": float(dt),
            }, step=epoch)

        global_step += len(loader_train)


def save_checkpoint(
        path: Path,
        encoder: Encoder,
        decoder: Decoder,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        cfg: TrainConfig,
        reactant_enc: ReactantEncoder | None = None,
        fusion_enc: FusionEncoder | None = None,
        formula_enc: FormulaEncoder | None = None,
        formula_dec: Decoder | None = None,
        count_head: ElementCountHead | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg.__dict__,
    }

    if reactant_enc is not None:
        payload["reactant_encoder"] = reactant_enc.state_dict()
    if fusion_enc is not None:
        payload["fusion_encoder"] = fusion_enc.state_dict()
    if formula_enc is not None:
        payload["formula_encoder"] = formula_enc.state_dict()
    if formula_dec is not None:
        payload["formula_decoder"] = formula_dec.state_dict()
    if count_head is not None:
        payload["count_head"] = count_head.state_dict()
    torch.save(payload, path)
    #mlflow.log_artifact(str(path))


def _load_training_table(path: str, cfg: TrainConfig) -> pd.DataFrame:
    columns = [cfg.spectrum_column, cfg.reaction_smiles_column]
    if cfg.formula_encoder or cfg.formula_decoder:
        columns.append(cfg.formula_column)

    dataframe = load_table(path)
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {', '.join(missing)}")
    return dataframe.loc[:, columns].copy()


def _validate_input_paths(cfg: TrainConfig) -> None:
    paths = [("training_path", cfg.training_path)]
    if cfg.validation_path:
        paths.append(("validation_path", cfg.validation_path))
    if cfg.confidence and not cfg.train_formula_only:
        paths.append(("smarts_path", cfg.smarts_path))

    missing = [f"{name}={path}" for name, path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing configured input files: " + ", ".join(missing))


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Train DynaMAIK from a YAML configuration.")
    parser.add_argument("--config", required=True, help="Path to the training YAML file.")
    args = parser.parse_args(argv)

    tqdm.pandas()
    cfg = TrainConfig.from_yaml(args.config)
    validate_config(cfg)
    _validate_input_paths(cfg)
    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    # ---- MLflow setup ----
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", f"{cfg.experiment}"))
    dataset_name = Path(cfg.training_path).name
    run_name = f"cnnhidden{cfg.conv_hidden}_cnnblocks{cfg.conv_blocks}_kernelsize{cfg.k_size}_start_enc_{cfg.enc_lr_start}_start_dec_{cfg.dec_lr_start}_batchsize_{cfg.batch_size}_epochs{cfg.epochs}_d{cfg.d_model}_h{cfg.nhead}_enc{cfg.enc_layers}_dec{cfg.dec_layers}_{dataset_name}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_artifact(str(Path(args.config).resolve()), artifact_path="config")
        artifact_uri = mlflow.get_artifact_uri()  # e.g. file:/.../mlruns/exp/run/artifacts
        artifact_path = Path(urlparse(artifact_uri).path)
        print("Artifact root:", artifact_path)

        # ---- Prepare tokenizer ----
        tokenizer = SmilesTokenizerAdapter(max_len=cfg.max_smiles_len)
        pad_id, bos_id, eos_id = tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id

        print("Loading data...")
        df = _load_training_table(cfg.training_path, cfg)

        if cfg.experiment == 'test':
            df = df.head(100)

        if cfg.sort_reactants:
            print("Sorting reactants in dataset...")
            df[cfg.reaction_smiles_column] = df[cfg.reaction_smiles_column].progress_apply(get_sorted_reactants)

        print(f"Total samples: {len(df)}")

        formulas_train = None
        formulas_val = None
        # spectra_train/val: list of dict[int, float]
        if not cfg.validation_path:
            spectra = df[cfg.spectrum_column].tolist()

            with Pool(cpu_count()) as p:
                spectra = list(tqdm(p.imap(parse_spectrum, spectra), total=len(spectra), desc="Parsing spectra"))

            # smiles_train/val:  list of str
            smiles = df[cfg.reaction_smiles_column].tolist()
            need_formula = cfg.formula_encoder or cfg.formula_decoder
            if need_formula:
                # assume df has a column cfg.formula_column like "PRODUCT_FORMULA"
                formulas = df[cfg.formula_column].tolist()
                spectra_train, spectra_val, smiles_train, smiles_val, formulas_train, formulas_val = train_test_split(
                    spectra, smiles, formulas, test_size=0.2, random_state=42
                )
            else:
                spectra_train, spectra_val, smiles_train, smiles_val = train_test_split(spectra, smiles, test_size=0.2, random_state=42)

        else:
            # Load training data
            df_train = df
            spectra_train = df_train[cfg.spectrum_column].tolist()
            with Pool(cpu_count()) as p:
                spectra_train = list(tqdm(p.imap(parse_spectrum, spectra_train), total=len(spectra_train), desc="Parsing train spectra"))
            smiles_train = df_train[cfg.reaction_smiles_column].tolist()

            need_formula = cfg.formula_encoder or cfg.formula_decoder
            if need_formula:
                formulas_train = df_train[cfg.formula_column].tolist()

            # Load validation data
            df_val = _load_training_table(cfg.validation_path, cfg)
            if cfg.sort_reactants:
                print("Sorting reactants in validation dataset...")
                df_val[cfg.reaction_smiles_column] = df_val[cfg.reaction_smiles_column].progress_apply(get_sorted_reactants)

            spectra_val = df_val[cfg.spectrum_column].tolist()
            with Pool(cpu_count()) as p:
                spectra_val = list(tqdm(p.imap(parse_spectrum, spectra_val), total=len(spectra_val), desc="Parsing val spectra"))
            smiles_val = df_val[cfg.reaction_smiles_column].tolist()

            if need_formula:
                # assume df has a column cfg.formula_column like "PRODUCT_FORMULA"
                formulas_val = df_val[cfg.formula_column].tolist()

        print(f"Training samples: {len(smiles_train)}, Validation samples: {len(smiles_val)}")
        print("Fitting tokenizer...")
        for s in tqdm(smiles_train + smiles_val, desc="Tokenizer fit"):
            _ = tokenizer.encode(s, add_bos=True, add_eos=True)

        rxn_sep_id = tokenizer.token_to_id(">>")
        vocab_size = tokenizer.size()
        tokenizer.freeze()
        tokenizer.save(artifact_path / "smiles_tokenizer.json")

        print(f"Vocab size: {vocab_size}")

        # Log data sizes and tokenizer
        mlflow.log_params({
            "vocab_size": vocab_size,
            "num_train_samples": len(smiles_train),
            "num_val_samples": len(smiles_val),
        })

        print("Preparing datasets and dataloaders...")
        # ---- Datasets / Loaders ----
        use_reactants = cfg.reactant_encoder  # or just True if you always have them

        if cfg.spectrum_column == "SPECTRUM_PREDICTED":
            synthetic_spectra = True

        elif cfg.spectrum_column == "SPECTRUM":
            synthetic_spectra = False

        else:
            raise ValueError(f"Unsupported spectrum column: {cfg.spectrum_column}")


        formula_tokenizer = None
        if cfg.formula_encoder or cfg.formula_decoder:
            formula_tokenizer = FormulaTokenizerAdapter(max_len=cfg.formula_max_len)

            # IMPORTANT: fit on all available formulas (train+val if present)
            formulas_for_vocab = []
            if formulas_train is not None:
                formulas_for_vocab += formulas_train
            if formulas_val is not None:
                formulas_for_vocab += formulas_val

            # fit vocab
            for f in tqdm(formulas_for_vocab, desc="Formula tokenizer fit"):
                _ = formula_tokenizer.encode(f, add_bos=True, add_eos=True)

            formula_tokenizer.freeze()
            formula_vocab_size = formula_tokenizer.size()
            formula_tokenizer.save(artifact_path /"formula_tokenizer.json")
            mlflow.log_param("formula_vocab_size", formula_vocab_size)


        return_formula = cfg.formula_encoder or cfg.formula_decoder
        ds_train = SpecSmilesDataset(spectra_train, smiles_train, tokenizer, encoder_mode=cfg.encoder_mode, max_bin_length=cfg.max_bin_length, max_mz=cfg.max_mz, product_only=(cfg.decoder_mode == "product"), return_reactants=use_reactants, synthetic_spectra=synthetic_spectra, formulas=formulas_train, formula_tokenizer=formula_tokenizer, return_formula=return_formula, enum_smiles=cfg.enum_smiles, frags=cfg.frags)
        ds_val = SpecSmilesDataset(spectra_val, smiles_val, tokenizer, encoder_mode=cfg.encoder_mode, max_bin_length=cfg.max_bin_length, max_mz=cfg.max_mz, product_only=(cfg.decoder_mode == "product"), return_reactants=use_reactants, synthetic_spectra=synthetic_spectra, formulas=formulas_val, formula_tokenizer=formula_tokenizer, return_formula=return_formula)

        collate = select_collate_fn(cfg, pad_id)

        loader_val = DataLoader(
            ds_val, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate
        )

        encoder, decoder, reactant_enc, fusion_enc, formula_enc, decoder_formula, count_head = build_model_components(
            cfg=cfg,
            device=device,
            vocab_size=vocab_size,
            pad_id=pad_id,
            rxn_sep_id=rxn_sep_id,
            formula_tokenizer=formula_tokenizer,
        )

        # ---- Optimizer / Loss ----
        param_groups = [{
            "name": "spectrum_encoder",
            "params": encoder.parameters(),
            "lr": cfg.lr_enc,
            "weight_decay": cfg.weight_decay,
        }]

        if not (cfg.train_formula_only and cfg.formula_decoder):
            param_groups.append({
                "name": "smiles_decoder",
                "params": decoder.parameters(),
                "lr": cfg.lr_dec,
                "weight_decay": cfg.weight_decay,
            })

        if reactant_enc is not None:
            param_groups.append({
                "name": "reactant_encoder",
                "params": reactant_enc.parameters(),
                "lr": cfg.lr_enc_react,
                "weight_decay": cfg.weight_decay,
            })

        if formula_enc is not None:
            param_groups.append({
                "name": "formula_encoder",
                "params": formula_enc.parameters(),
                "lr": cfg.lr_enc_formula,
                "weight_decay": cfg.weight_decay,
            })

        if fusion_enc is not None:
            param_groups.append({
                "name": "fusion_encoder",
                "params": fusion_enc.parameters(),
                "lr": cfg.lr_enc,
                "weight_decay": cfg.weight_decay,
            })

        if cfg.formula_decoder:
            param_groups.append({
                "name": "formula_decoder",
                "params": decoder_formula.parameters(),
                "lr": cfg.lr_dec,
                "weight_decay": cfg.weight_decay,
            })

        if count_head is not None:
            param_groups.append({
                "name": "count_head",
                "params": count_head.parameters(),
                "lr": cfg.lr_dec,
                "weight_decay": cfg.weight_decay,
            })

        optimizer = torch.optim.AdamW(param_groups, fused=device.type == "cuda")

        mlflow.log_params({
            "optimizer": "AdamW", "warumup_epochs_enc": cfg.enc_warmup_epochs, "warumup_epochs_dec": cfg.dec_warmup_epochs, "use_count_head": cfg.use_count_head, "count_elements": ",".join(cfg.count_elements), "count_loss_weight": cfg.count_loss_weight,
            "enc_lr": cfg.lr_enc, "dec_lr": cfg.lr_dec, "weight_decay": cfg.weight_decay, "feedforward_dimensions": cfg.dim_ff, "enum_smiles": cfg.enum_smiles,
            "model_dim": cfg.d_model, "grad_clip": cfg.grad_clip, "pad_id": pad_id, "batch_size": cfg.batch_size, "Sampling": cfg.topk,
            "randomize_reactants": cfg.randomize_reactants, "lr_scheduler": cfg.lr_scheduler, "encoder_mode": cfg.encoder_mode, "decoder_mode": cfg.decoder_mode, "reactant_encoder": cfg.reactant_encoder,"formula_encoder": cfg.formula_encoder,  "formula_decoder": cfg.formula_decoder, "formula_loss_weight": cfg.formula_loss_weight, "fusion_encoder": cfg.fusion_encoder, 'multi_source_decoder': cfg.multi_source_decoder,
            "recursive_decoder": cfg.recursive_decoder, "recursive_reactant_encoder": cfg.recursive_reactant_encoder, "recursive_spectrum_encoder": cfg.recursive_spectrum_encoder, "spectrum_encoder": cfg.spectrum_encoder, "conv_hidden": cfg.conv_hidden, "conv_blocks": cfg.conv_blocks, "kernel_size": cfg.k_size, "n_heads": cfg.nhead, "dropout": cfg.dropout, "temperature": cfg.temperature, 'formula_decoder_only': cfg.train_formula_only,
        })

        ckpt_dir = artifact_path / "checkpoints"
        preds_dir = artifact_path / "predictions"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        preds_dir.mkdir(parents=True, exist_ok=True)

        spectrum_enc = getattr(cfg, "spectrum_encoder", True)
        # ---- Train loop ----
        print("Starting training...")
        train_loop(cfg, encoder, decoder, loader_val, optimizer, device, tokenizer, eos_id, ds_train, collate, ckpt_dir=ckpt_dir, preds_dir=preds_dir, reactant_enc=reactant_enc, fusion_enc=fusion_enc, spectrum_enc=spectrum_enc, formula_enc=formula_enc, formula_dec=decoder_formula if cfg.formula_decoder else None, formula_tokenizer=formula_tokenizer, count_head=count_head)

        # ---- Log final model weights as separate artifacts ----
        print("Logging final model artifacts...")
        enc_path = preds_dir / "encoder.pt"
        dec_path = preds_dir / "decoder.pt"
        enc_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(encoder.state_dict(), enc_path)
        torch.save(decoder.state_dict(), dec_path)
        #mlflow.log_artifact(str(enc_path))
        #mlflow.log_artifact(str(dec_path))
        if reactant_enc is not None:
            rxn_path = preds_dir / "reactant_encoder.pt"
            torch.save(reactant_enc.state_dict(), rxn_path)
            #mlflow.log_artifact(str(rxn_path))

        if formula_enc is not None:
            rxn_path = preds_dir / "formula_encoder.pt"
            torch.save(formula_enc.state_dict(), rxn_path)

        if fusion_enc is not None:
            fusion_path = preds_dir / "fusion_encoder.pt"
            torch.save(fusion_enc.state_dict(), fusion_path)

        if cfg.formula_decoder:
            formula_dec_path = preds_dir / "formula_decoder.pt"
            torch.save(decoder_formula.state_dict(), formula_dec_path)
        if count_head is not None:
            count_head_path = preds_dir / "count_head.pt"
            torch.save(count_head.state_dict(), count_head_path)
        print("Done.")


if __name__ == "__main__":
    main()
