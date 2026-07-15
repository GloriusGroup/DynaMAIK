from __future__ import annotations

import mlflow
import torch

from src.data.collate import (
    collate_batch,
    collate_batch_formula_only,
    collate_batch_product_only,
    collate_batch_product_only_with_formula,
    collate_batch_product_only_with_formula_multitask,
    collate_batch_product_only_with_reactants,
    collate_batch_product_only_with_reactants_and_formula,
    collate_batch_product_only_with_reactants_and_formula_multitask,
)
from src.models.configs import TrainConfig
from src.models.evaluation import evaluate, evaluate_formula_only, evaluate_product_only, evaluate_product_only_multitask
from src.models.spec2prod import (
    BinnedEncoder,
    Decoder,
    ElementCountHead,
    Encoder,
    FormulaEncoder,
    FusionEncoder,
    PeakEncoder,
    ReactantEncoder,
    RecursiveProductDecoder,
    RecursiveReactantEncoder,
    RecursiveSpectrumEncoder,
)
from src.utils.model_runtime import move_optional_tensor, pool_hidden_by_input_mask
from src.utils.utils_train import counts_from_formula_tokens, lr_for_epoch, set_group_lrs_by_name

try:
    from src.models.tiny_recursive.core_network import TRMAttentionNetwork
except ModuleNotFoundError:
    TRMAttentionNetwork = None


def require_recursive_network():
    if TRMAttentionNetwork is None:
        raise ImportError(
            "Recursive model options require src.models.tiny_recursive, "
            "but that package is not available in this project."
        )
    return TRMAttentionNetwork


def unpack_training_batch(batch, cfg, device, reactant_enc=None, formula_enc=None, formula_dec=None):
    """Normalize all supported training batch layouts into named tensors."""
    data = {
        "formula_strs": None,
        "reactant_ids": None,
        "formula_ids": None,
        "tgt_in": None,
        "tgt_out": None,
        "fml_in": None,
        "fml_out": None,
        "smi_in": None,
        "smi_out": None,
    }

    if reactant_enc is not None and formula_dec is not None:
        spec_batch, data["reactant_ids"], data["fml_in"], data["fml_out"], data["formula_strs"], data["smi_in"], data["smi_out"] = batch
        data["tgt_out"] = data["smi_out"]
    elif formula_dec is not None and cfg.train_formula_only and reactant_enc is None:
        spec_batch, data["fml_in"], data["fml_out"], data["formula_strs"] = batch
        data["tgt_out"] = data["fml_out"]
    elif formula_dec is not None and reactant_enc is None:
        spec_batch, data["fml_in"], data["fml_out"], data["formula_strs"], data["smi_in"], data["smi_out"] = batch
        data["tgt_out"] = data["smi_out"]
    elif reactant_enc is not None and formula_enc is not None:
        spec_batch, data["reactant_ids"], data["formula_ids"], data["formula_strs"], data["tgt_in"], data["tgt_out"] = batch
    elif reactant_enc is not None:
        spec_batch, data["reactant_ids"], data["tgt_in"], data["tgt_out"] = batch
    elif formula_enc is not None:
        spec_batch, data["formula_ids"], data["formula_strs"], data["tgt_in"], data["tgt_out"] = batch
    else:
        spec_batch, data["tgt_in"], data["tgt_out"] = batch

    data["spec_batch"] = spec_batch.to(device, non_blocking=True)
    for key in ("reactant_ids", "formula_ids", "tgt_in", "tgt_out", "fml_in", "fml_out", "smi_in", "smi_out"):
        data[key] = move_optional_tensor(data[key], device)
    return data


def count_loss_tokens(decoder, targets: torch.Tensor, pad_id: int) -> int:
    if decoder.decoder_mode == "product":
        valid_mask = targets != pad_id
    else:
        valid_mask = decoder.build_product_mask(targets) & (targets != pad_id)
    return int(valid_mask.sum().item())


def gradient_clip_parameters(cfg, encoder, decoder, reactant_enc=None, fusion_enc=None, formula_enc=None, formula_dec=None, count_head=None):
    modules = [encoder]
    for module in (reactant_enc, fusion_enc, formula_enc, formula_dec, count_head):
        if module is not None:
            modules.append(module)
    if not (cfg.train_formula_only and formula_dec is not None):
        modules.append(decoder)
    return [parameter for module in modules for parameter in module.parameters()]


def count_head_loss(cfg, count_head, formula_tokenizer, formula_dec, hidden_fml, fml_in, fml_out):
    pooled = pool_hidden_by_input_mask(hidden_fml, fml_in, formula_dec.pad_id)
    pred_counts = count_head(pooled)
    true_counts = counts_from_formula_tokens(
        fml_out=fml_out,
        formula_tokenizer=formula_tokenizer,
        elements=cfg.count_elements,
        pad_id=formula_dec.pad_id,
        eos_id=formula_tokenizer.eos_id,
    ).to(pred_counts.device)
    return torch.nn.functional.smooth_l1_loss(torch.relu(pred_counts), true_counts)


def epoch_lrs(cfg: TrainConfig, epoch: int) -> dict[str, float]:
    return {
        "spectrum_encoder": lr_for_epoch(e=epoch, T_warm=cfg.enc_warmup_epochs, T_total=cfg.epochs, lr_start=cfg.enc_lr_start, lr_base=cfg.lr_enc, lr_min=cfg.enc_lr_min),
        "reactant_encoder": lr_for_epoch(e=epoch, T_warm=cfg.enc_react_warmup_epochs, T_total=cfg.epochs, lr_start=cfg.enc_react_lr_start, lr_base=cfg.lr_enc_react, lr_min=cfg.enc_react_lr_min),
        "formula_encoder": lr_for_epoch(e=epoch, T_warm=cfg.enc_formula_warmup_epochs, T_total=cfg.epochs, lr_start=cfg.enc_formula_lr_start, lr_base=cfg.lr_enc_formula, lr_min=cfg.enc_formula_lr_min),
        "decoder": lr_for_epoch(e=epoch, T_warm=cfg.dec_warmup_epochs, T_total=cfg.epochs, lr_start=cfg.dec_lr_start, lr_base=cfg.lr_dec, lr_min=cfg.dec_lr_min),
    }


def apply_and_log_lrs(epoch, optimizer, lrs, reactant_enc=None, fusion_enc=None, formula_enc=None, formula_dec=None):
    lr_by_name = {
        "spectrum_encoder": lrs["spectrum_encoder"],
        "smiles_decoder": lrs["decoder"],
        "formula_decoder": lrs["decoder"],
        "count_head": lrs["decoder"],
    }
    if reactant_enc is not None:
        lr_by_name["reactant_encoder"] = lrs["reactant_encoder"]
    if formula_enc is not None:
        lr_by_name["formula_encoder"] = lrs["formula_encoder"]
    if fusion_enc is not None:
        lr_by_name["fusion_encoder"] = lrs["spectrum_encoder"]

    set_group_lrs_by_name(optimizer, lr_by_name)

    mlflow.log_metric("lr_encoder", float(lrs["spectrum_encoder"]), step=epoch)
    mlflow.log_metric("lr_decoder", float(lrs["decoder"]), step=epoch)
    if reactant_enc is not None:
        mlflow.log_metric("lr_react_encoder", float(lrs["reactant_encoder"]), step=epoch)
    if formula_enc is not None:
        mlflow.log_metric("lr_formula_encoder", float(lrs["formula_encoder"]), step=epoch)
    if formula_dec is not None:
        mlflow.log_metric("lr_formula_decoder", float(lrs["decoder"]), step=epoch)
    if fusion_enc is not None:
        mlflow.log_metric("lr_fusion_encoder", float(lrs["spectrum_encoder"]), step=epoch)


def evaluate_for_config(
    cfg,
    encoder,
    decoder,
    loader_val,
    device,
    tokenizer,
    eos_id,
    reactant_enc=None,
    fusion_enc=None,
    spectrum_enc=True,
    formula_enc=None,
    formula_dec=None,
    formula_tokenizer=None,
    count_head=None,
):
    if cfg.decoder_mode != "product":
        return evaluate(encoder, decoder, loader_val, device, tokenizer, eos_id, reactant_enc=reactant_enc, spectrum_enc=spectrum_enc, temperature=cfg.temperature)

    if formula_dec is not None and cfg.train_formula_only:
        return evaluate_formula_only(
            encoder=encoder,
            decoder_formula=formula_dec,
            loader=loader_val,
            device=device,
            formula_tokenizer=formula_tokenizer,
            formula_eos_id=formula_tokenizer.eos_id,
            reactant_enc=reactant_enc,
            fusion_enc=fusion_enc,
            spectrum_enc=spectrum_enc,
            temperature=cfg.temperature,
            topk=cfg.topk,
            count_head=count_head,
            elem_list=cfg.count_elements,
            log_count_mae=cfg.use_count_head,
        )

    if formula_dec is not None:
        return evaluate_product_only_multitask(
            encoder=encoder,
            decoder_smiles=decoder,
            decoder_formula=formula_dec,
            loader=loader_val,
            device=device,
            smiles_tokenizer=tokenizer,
            smiles_eos_id=eos_id,
            formula_tokenizer=formula_tokenizer,
            formula_eos_id=formula_tokenizer.eos_id,
            reactant_enc=reactant_enc,
            fusion_enc=fusion_enc,
            spectrum_enc=spectrum_enc,
            formula_enc=formula_enc,
            temperature=cfg.temperature,
            topk=cfg.topk,
            formula_loss_weight=cfg.formula_loss_weight,
            count_head=count_head,
            elem_list=cfg.count_elements,
            log_count_mae=cfg.use_count_head,
        )

    return evaluate_product_only(
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
        temperature=cfg.temperature,
        topk=cfg.topk,
    )


def print_epoch_summary(epoch, cfg, train_loss, val_metrics, lrs, dt, formula_dec=None):
    if formula_dec is not None and cfg.train_formula_only:
        print(f"... val_loss={val_metrics['val_loss']:.4f} tok_acc_fml={val_metrics['val_token_acc_formula']:.4f} seq_acc_fml={val_metrics['val_sequence_acc_formula']:.4f} ",
              f"lr_enc={lrs['spectrum_encoder']:.3e}  lr_dec={lrs['decoder']:.3e}  ({dt:.1f}s)")
    elif formula_dec is not None:
        print(f"... val_loss={val_metrics['val_loss']:.4f} tok_acc_smi={val_metrics['val_token_acc_smiles']:.4f} seq_acc_smi={val_metrics['val_sequence_acc_smiles']:.4f} tok_acc_fml={val_metrics['val_token_acc_formula']:.4f} seq_acc_fml={val_metrics['val_sequence_acc_formula']:.4f} ",
              f"lr_enc={lrs['spectrum_encoder']:.3e}  lr_dec={lrs['decoder']:.3e}  ({dt:.1f}s)")
    else:
        print(f"[Epoch {epoch:03d}] "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_metrics['val_loss']:.4f}  "
              f"tok_acc={val_metrics['val_token_acc']:.4f}  "
              f"seq_acc={val_metrics['val_sequence_acc']:.4f}  "
              f"tanimoto={val_metrics.get('val_tanimoto', float('nan')):.4f}  "
              f"lr_enc={lrs['spectrum_encoder']:.3e}  lr_dec={lrs['decoder']:.3e}  ({dt:.1f}s)")


def select_collate_fn(cfg: TrainConfig, pad_id: int):
    if cfg.decoder_mode != "product":
        return lambda batch: collate_batch(batch, pad_id)
    if cfg.train_formula_only and cfg.formula_decoder and not cfg.reactant_encoder:
        return lambda batch: collate_batch_formula_only(batch, pad_id)
    if cfg.reactant_encoder and cfg.formula_decoder:
        return lambda batch: collate_batch_product_only_with_reactants_and_formula_multitask(batch, pad_id)
    if (not cfg.reactant_encoder) and cfg.formula_decoder:
        return lambda batch: collate_batch_product_only_with_formula_multitask(batch, pad_id)
    if cfg.reactant_encoder and cfg.formula_encoder:
        return lambda batch: collate_batch_product_only_with_reactants_and_formula(batch, pad_id)
    if cfg.reactant_encoder:
        return lambda batch: collate_batch_product_only_with_reactants(batch, pad_id)
    if cfg.formula_encoder:
        return lambda batch: collate_batch_product_only_with_formula(batch, pad_id)
    return lambda batch: collate_batch_product_only(batch, pad_id)


def build_model_components(cfg, device, vocab_size, pad_id, rxn_sep_id, formula_tokenizer=None):
    if cfg.recursive_spectrum_encoder and cfg.encoder_mode == "raw":
        recursive_network_cls = require_recursive_network()
        trm_spec = recursive_network_cls(dimension=cfg.d_model, num_layers=2, num_heads=cfg.nhead, mlp_ratio=4.0, dropout=cfg.dropout)
        encoder = RecursiveSpectrumEncoder(
            seq_len=cfg.max_mz,
            d_model=cfg.d_model,
            network=trm_spec,
            conv_hidden=cfg.conv_hidden,
            conv_blocks=cfg.conv_blocks,
            k_size=cfg.k_size,
            dropout=cfg.dropout,
            num_refinement_blocks=cfg.trm_ref_blocks,
            num_latent_refinements=cfg.trm_latent_refines,
            cnn=cfg.cnn,
        ).to(device)
    elif cfg.encoder_mode == "raw":
        encoder = Encoder(seq_len=cfg.max_mz, d_model=cfg.d_model, nhead=cfg.nhead, num_layers=cfg.enc_layers,
                          dim_feedforward=cfg.dim_ff, dropout=cfg.dropout, conv_hidden=cfg.conv_hidden, conv_blocks=cfg.conv_blocks, k_size=cfg.k_size, cnn=cfg.cnn).to(device)
    elif cfg.encoder_mode == "binned":
        encoder = BinnedEncoder(seq_len=cfg.max_bin_length, d_model=cfg.d_model, nhead=cfg.nhead, num_layers=cfg.enc_layers,
                                dim_feedforward=cfg.dim_ff, dropout=cfg.dropout).to(device)
    elif cfg.encoder_mode == "binned_2D":
        encoder = PeakEncoder(seq_len=cfg.max_bin_length, d_model=cfg.d_model, nhead=cfg.nhead, num_layers=cfg.enc_layers,
                              dim_feedforward=cfg.dim_ff, dropout=cfg.dropout).to(device)
    else:
        raise ValueError(f"Unsupported encoder mode: {cfg.encoder_mode}")

    reactant_enc = None
    if cfg.reactant_encoder:
        if cfg.recursive_reactant_encoder:
            recursive_network_cls = require_recursive_network()
            trm_rxn = recursive_network_cls(dimension=cfg.d_model, num_layers=2, num_heads=cfg.nhead, mlp_ratio=4.0, dropout=cfg.dropout)
            reactant_enc = RecursiveReactantEncoder(
                vocab_size=vocab_size,
                pad_id=pad_id,
                max_len=cfg.max_smiles_len,
                d_model=cfg.d_model,
                network=trm_rxn,
                num_refinement_blocks=cfg.trm_ref_blocks,
                num_latent_refinements=cfg.trm_latent_refines,
                dropout=cfg.dropout,
            ).to(device)
        else:
            reactant_enc = ReactantEncoder(vocab_size=vocab_size, pad_id=pad_id, max_len=cfg.max_smiles_len, d_model=cfg.d_model,
                                           nhead=cfg.nhead, num_layers=cfg.enc_layers, dim_feedforward=cfg.dim_ff, dropout=cfg.dropout).to(device)

    formula_enc = None
    if cfg.formula_encoder:
        if formula_tokenizer is None:
            raise ValueError("cfg.formula_encoder=True but formula_tokenizer is None")
        formula_enc = FormulaEncoder(
            vocab_size=formula_tokenizer.size(),
            pad_id=formula_tokenizer.pad_id,
            max_len=cfg.formula_max_len,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.enc_layers,
            dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout,
        ).to(device)

    fusion_enc = None
    if cfg.fusion_encoder:
        if reactant_enc is None:
            raise ValueError("Fusion encoder requires a reactant encoder.")
        max_len_spec = cfg.max_mz if cfg.encoder_mode == "raw" else cfg.max_bin_length
        fusion_enc = FusionEncoder(d_model=cfg.d_model, nhead=cfg.fusion_nhead, num_layers=cfg.fusion_layers, dim_feedforward=cfg.fusion_dim_ff, dropout=cfg.dropout, max_len_spec=max_len_spec, max_len_rxn=cfg.max_smiles_len).to(device)

    if cfg.recursive_decoder:
        recursive_network_cls = require_recursive_network()
        trm_net = recursive_network_cls(dimension=cfg.d_model, num_layers=4, num_heads=cfg.trm_nhead, mlp_ratio=4.0, dropout=cfg.dropout)
        decoder = RecursiveProductDecoder(vocab_size=vocab_size, pad_id=pad_id, rxn_sep_id=rxn_sep_id,
                                          max_len=cfg.max_smiles_len, d_model=cfg.d_model, network=trm_net,
                                          decoder_mode=cfg.decoder_mode, halt_loss_weight=1.0).to(device)
    else:
        decoder = Decoder(vocab_size=vocab_size, pad_id=pad_id, rxn_sep_id=rxn_sep_id, max_len=cfg.max_smiles_len,
                          d_model=cfg.d_model, nhead=cfg.nhead, num_layers=cfg.dec_layers, dim_feedforward=cfg.dim_ff,
                          dropout=cfg.dropout, decoder_mode=cfg.decoder_mode, multi_source=cfg.multi_source_decoder).to(device)

    decoder_formula = None
    if cfg.formula_decoder:
        decoder_formula = Decoder(vocab_size=formula_tokenizer.size(),
                                  pad_id=formula_tokenizer.pad_id,
                                  rxn_sep_id=rxn_sep_id,
                                  max_len=cfg.formula_max_len,
                                  d_model=cfg.d_model,
                                  nhead=cfg.nhead,
                                  num_layers=cfg.dec_layers,
                                  dim_feedforward=cfg.dim_ff,
                                  dropout=cfg.dropout,
                                  decoder_mode=cfg.decoder_mode,
                                  multi_source=cfg.multi_source_decoder).to(device)

    count_head = None
    if cfg.use_count_head:
        count_head = ElementCountHead(cfg.d_model, n_elems=len(cfg.count_elements)).to(device)
        if not cfg.formula_decoder:
            raise ValueError("Count head is only supported when formula_decoder is True.")

    return encoder, decoder, reactant_enc, fusion_enc, formula_enc, decoder_formula, count_head
