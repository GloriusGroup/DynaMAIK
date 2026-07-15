# src/predict.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, List

import pandas as pd
import torch
import numpy as np
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from ast import literal_eval
from collections import Counter
from rdkit import Chem, RDLogger

# ==== your modules ====
from src.models.configs import TrainConfig
from src.data.tokenizers import SmilesTokenizerAdapter, FormulaTokenizerAdapter

from src.data.loaders import SpecSmilesDataset
from src.data.collate import (
    collate_batch,
    collate_batch_product_only,
    collate_batch_product_only_with_reactants,
    collate_batch_product_only_with_formula,
    collate_batch_product_only_with_reactants_and_formula,
    collate_batch_product_only_with_reactants_and_formula_multitask,
    collate_batch_formula_only,
)

from src.models.spec2prod import (
    Encoder, Decoder, BinnedEncoder, PeakEncoder,
    ReactantEncoder, FormulaEncoder, FusionEncoder,
    RecursiveProductDecoder, RecursiveReactantEncoder, RecursiveSpectrumEncoder,
)

try:
    from src.models.tiny_recursive.core_network import TRMAttentionNetwork
except ModuleNotFoundError:
    TRMAttentionNetwork = None


RDLogger.DisableLog("rdApp.*")


def _require_recursive_network():
    if TRMAttentionNetwork is None:
        raise ImportError(
            "Recursive model options require src.models.tiny_recursive, "
            "but that package is not available in this project."
        )
    return TRMAttentionNetwork

# -----------------------------
# small utilities
# -----------------------------
def _safe_load_state_dict(module: torch.nn.Module, state: Dict[str, Any], name: str) -> None:
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] {name}.load_state_dict(strict=False): missing={len(missing)} unexpected={len(unexpected)}")


def _maybe_load_pt(path: Path) -> Optional[Dict[str, Any]]:
    if path is None or (not path.exists()):
        return None
    return torch.load(path, map_location="cpu")


def _get_attr(cfg: Any, key: str, default: Any) -> Any:
    return getattr(cfg, key, default) if hasattr(cfg, key) else default


def _repeat_memory(memory, num_repeats: int):
    if num_repeats == 1:
        return memory
    if isinstance(memory, tuple):
        return tuple(
            None if x is None else x.repeat_interleave(num_repeats, dim=0)
            for x in memory
        )
    return memory.repeat_interleave(num_repeats, dim=0)


# -----------------------------
# Model construction (mirrors train.py)
# -----------------------------
def build_models_from_cfg(
    cfg: TrainConfig,
    device: torch.device,
    smiles_tokenizer: SmilesTokenizerAdapter,
    formula_tokenizer: Optional[FormulaTokenizerAdapter],
):
    pad_id = smiles_tokenizer.pad_id
    rxn_sep_id = smiles_tokenizer.token_to_id(">>") if ">>" in smiles_tokenizer.stoi else smiles_tokenizer.unk_id
    vocab_size = smiles_tokenizer.size()

    # Spectrum encoder
    if _get_attr(cfg, "recursive_spectrum_encoder", False) and cfg.encoder_mode == "raw":
        recursive_network_cls = _require_recursive_network()
        trm_spec = recursive_network_cls(
            dimension=cfg.d_model, num_layers=2, num_heads=cfg.nhead, mlp_ratio=4.0, dropout=cfg.dropout
        )
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
    else:
        if cfg.encoder_mode == "raw":
            encoder = Encoder(
                seq_len=cfg.max_mz,
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                num_layers=cfg.enc_layers,
                dim_feedforward=cfg.dim_ff,
                dropout=cfg.dropout,
                conv_hidden=cfg.conv_hidden,
                conv_blocks=cfg.conv_blocks,
                k_size=cfg.k_size,
                cnn=cfg.cnn,
            ).to(device)
        elif cfg.encoder_mode == "binned":
            encoder = BinnedEncoder(
                seq_len=cfg.max_bin_length,
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                num_layers=cfg.enc_layers,
                dim_feedforward=cfg.dim_ff,
                dropout=cfg.dropout,
            ).to(device)
        elif cfg.encoder_mode == "binned_2D":
            encoder = PeakEncoder(
                seq_len=cfg.max_bin_length,
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                num_layers=cfg.enc_layers,
                dim_feedforward=cfg.dim_ff,
                dropout=cfg.dropout,
            ).to(device)
        else:
            raise ValueError(f"Unsupported encoder_mode: {cfg.encoder_mode}")

    # Reactant encoder
    reactant_enc = None
    if _get_attr(cfg, "reactant_encoder", False):
        if _get_attr(cfg, "recursive_reactant_encoder", False):
            recursive_network_cls = _require_recursive_network()
            trm_rxn = recursive_network_cls(
                dimension=cfg.d_model, num_layers=2, num_heads=cfg.nhead, mlp_ratio=4.0, dropout=cfg.dropout
            )
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
            reactant_enc = ReactantEncoder(
                vocab_size=vocab_size,
                pad_id=pad_id,
                max_len=cfg.max_smiles_len,
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                num_layers=cfg.enc_layers,
                dim_feedforward=cfg.dim_ff,
                dropout=cfg.dropout,
            ).to(device)

    # Formula encoder
    formula_enc = None
    if _get_attr(cfg, "formula_encoder", False):
        if formula_tokenizer is None:
            raise ValueError("cfg.formula_encoder=True but formula_tokenizer.json not found")
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

    # Fusion encoder
    fusion_enc = None
    if _get_attr(cfg, "fusion_encoder", False):
        if reactant_enc is None:
            raise ValueError("Fusion encoder requires reactant encoder.")
        max_len_spec = cfg.max_mz if cfg.encoder_mode == "raw" else cfg.max_bin_length
        fusion_enc = FusionEncoder(
            d_model=cfg.d_model,
            nhead=cfg.fusion_nhead,
            num_layers=cfg.fusion_layers,
            dim_feedforward=cfg.fusion_dim_ff,
            dropout=cfg.dropout,
            max_len_spec=max_len_spec,
            max_len_rxn=cfg.max_smiles_len,
        ).to(device)

    # SMILES decoder
    if _get_attr(cfg, "recursive_decoder", False):
        recursive_network_cls = _require_recursive_network()
        trm_net = recursive_network_cls(
            dimension=cfg.d_model, num_layers=4, num_heads=cfg.trm_nhead, mlp_ratio=4.0, dropout=cfg.dropout
        )
        decoder = RecursiveProductDecoder(
            vocab_size=vocab_size,
            pad_id=pad_id,
            rxn_sep_id=rxn_sep_id,
            max_len=cfg.max_smiles_len,
            d_model=cfg.d_model,
            network=trm_net,
            decoder_mode=cfg.decoder_mode,
            halt_loss_weight=1.0,
        ).to(device)
    else:
        decoder = Decoder(
            vocab_size=vocab_size,
            pad_id=pad_id,
            rxn_sep_id=rxn_sep_id,
            max_len=cfg.max_smiles_len,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.dec_layers,
            dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout,
            decoder_mode=cfg.decoder_mode,
            multi_source=_get_attr(cfg, "multi_source_decoder", False),
        ).to(device)

    # Formula decoder (optional)
    decoder_formula = None
    if _get_attr(cfg, "formula_decoder", False):
        if formula_tokenizer is None:
            raise ValueError("cfg.formula_decoder=True but formula_tokenizer.json not found")
        decoder_formula = Decoder(
            vocab_size=formula_tokenizer.size(),
            pad_id=formula_tokenizer.pad_id,
            rxn_sep_id=rxn_sep_id,
            max_len=cfg.formula_max_len,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.dec_layers,
            dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout,
            decoder_mode=cfg.decoder_mode,
            multi_source=_get_attr(cfg, "multi_source_decoder", False),
        ).to(device)

    return encoder, decoder, reactant_enc, fusion_enc, formula_enc, decoder_formula


# -----------------------------
# Memory construction (mirrors train.py)
# -----------------------------
def build_memory(
    cfg: TrainConfig,
    encoder: torch.nn.Module,
    spec_batch: torch.Tensor,
    spectrum_enc: bool,
    reactant_enc: Optional[torch.nn.Module],
    reactant_ids: Optional[torch.Tensor],
    fusion_enc: Optional[torch.nn.Module],
    formula_enc: Optional[torch.nn.Module],
    formula_ids: Optional[torch.Tensor],
):
    spec_memory, spec_mask = None, None
    if spectrum_enc:
        spec_memory = encoder(spec_batch)
        if spec_memory is not None:
            B, S, _ = spec_memory.shape
            spec_mask = torch.zeros(B, S, dtype=torch.bool, device=spec_memory.device)

    rxn_memory, rxn_mask = None, None
    if reactant_enc is not None:
        rxn_memory = reactant_enc(reactant_ids)
        rxn_mask = (reactant_ids == reactant_enc.pad_id)

    fml_memory, fml_mask = None, None
    if formula_enc is not None:
        fml_memory = formula_enc(formula_ids)
        fml_mask = (formula_ids == formula_enc.pad_id)

    if _get_attr(cfg, "multi_source_decoder", False):
        if spec_memory is None or rxn_memory is None:
            raise ValueError("multi_source_decoder=True requires both spectrum and reactant encoders.")
        return (spec_memory, rxn_memory, spec_mask, rxn_mask)

    mem_list = []
    if fusion_enc is not None:
        if spec_memory is None or rxn_memory is None:
            raise ValueError("FusionEncoder requires spectrum_enc=True and reactant_enc!=None.")
        fused_sr = fusion_enc(spec_memory, rxn_memory, mask_rxn=rxn_mask)
        mem_list.append(fused_sr)
        if fml_memory is not None:
            mem_list.append(fml_memory)
    else:
        if spec_memory is not None:
            mem_list.append(spec_memory)
        if rxn_memory is not None:
            mem_list.append(rxn_memory)
        if fml_memory is not None:
            mem_list.append(fml_memory)

    if len(mem_list) == 0:
        raise ValueError("No encoder memory available. Enable at least one of spectrum/reactant/formula.")
    return mem_list[0] if len(mem_list) == 1 else torch.cat(mem_list, dim=1)


# -----------------------------
# Generation helpers
# -----------------------------
@torch.no_grad()
def generate_smiles_samples(
    decoder: Decoder,
    memory,
    tokenizer: SmilesTokenizerAdapter,
    max_new_tokens: int,
    temperature: float,
    topk: int,
    num_samples: int,
    device: torch.device,
) -> List[List[str]]:
    B = memory[0].size(0) if isinstance(memory, tuple) else memory.size(0)

    memory_rep = _repeat_memory(memory, num_samples)
    prefix = torch.full((B * num_samples, 1), tokenizer.bos_id, dtype=torch.long, device=device)

    gen_ids = decoder.generate_from_prefix(
        prefix_ids=prefix,
        memory=memory_rep,
        eos_id=tokenizer.eos_id,
        pad_id=decoder.pad_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        topk=topk,
        return_logprobs=False,
    )

    preds_flat = [tokenizer.decode(gen_ids[i].tolist(), skip_specials=True) for i in range(gen_ids.size(0))]
    preds_nested = [
        preds_flat[i * num_samples:(i + 1) * num_samples]
        for i in range(B)
    ]
    return preds_nested


@torch.no_grad()
def generate_formula_samples(
    decoder_formula: Decoder,
    memory,
    tokenizer: FormulaTokenizerAdapter,
    max_new_tokens: int,
    temperature: float,
    topk: int,
    num_samples: int,
    device: torch.device,
) -> List[List[str]]:
    B = memory[0].size(0) if isinstance(memory, tuple) else memory.size(0)

    memory_rep = _repeat_memory(memory, num_samples)
    prefix = torch.full((B * num_samples, 1), tokenizer.bos_id, dtype=torch.long, device=device)

    gen_ids = decoder_formula.generate_from_prefix(
        prefix_ids=prefix,
        memory=memory_rep,
        eos_id=tokenizer.eos_id,
        pad_id=decoder_formula.pad_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        topk=topk,
        return_logprobs=False,
    )

    preds_flat = [
        tokenizer.decode(gen_ids[i].tolist(), skip_specials=True).replace(" ", "")
        for i in range(gen_ids.size(0))
    ]
    preds_nested = [
        preds_flat[i * num_samples:(i + 1) * num_samples]
        for i in range(B)
    ]
    return preds_nested


def _extract_product_from_rxn(rxn: str) -> str:
    if ">>" in rxn:
        return rxn.split(">>", 1)[1]
    return rxn


def _canonicalize_for_rerank(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return s  # keep invalid/raw string instead of dropping it
    return Chem.MolToSmiles(mol, canonical=True)


def _frequency_rerank_smiles(samples: List[str], top_n: int) -> List[Dict[str, Any]]:
    first_seen = {}
    counts = Counter()
    raw_example = {}

    for idx, raw in enumerate(samples):
        can = _canonicalize_for_rerank(raw)
        counts[can] += 1

        if can not in first_seen:
            first_seen[can] = idx
            raw_example[can] = raw

    ranked = sorted(
        counts.items(),
        key=lambda x: (-x[1], first_seen[x[0]])
    )

    total = max(len(samples), 1)

    out = []
    for candidate, count in ranked[:top_n]:
        out.append({
            "candidate": candidate,          # canonical SMILES
            "raw_example": raw_example[candidate],
            "count": count,
            "frequency": count / total,
        })

    while len(out) < top_n:
        out.append({
            "candidate": "",
            "raw_example": "",
            "count": 0,
            "frequency": 0.0,
        })

    return out


def _frequency_rerank(samples: List[str], top_n: int) -> List[Dict[str, Any]]:
    """
    Rank sampled predictions by frequency.

    Tie-breaking:
      1. higher count first
      2. earlier first occurrence first
    """
    first_seen = {}
    counts = Counter()

    for idx, s in enumerate(samples):
        s = "" if s is None else str(s)
        counts[s] += 1
        if s not in first_seen:
            first_seen[s] = idx

    ranked = sorted(
        counts.items(),
        key=lambda x: (-x[1], first_seen[x[0]])
    )

    total = max(len(samples), 1)

    out = []
    for candidate, count in ranked[:top_n]:
        out.append({
            "candidate": candidate,
            "count": count,
            "frequency": count / total,
        })

    while len(out) < top_n:
        out.append({
            "candidate": "",
            "count": 0,
            "frequency": 0.0,
        })

    return out


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="CSV/Parquet with spectra and optional columns.")
    ap.add_argument("--out", type=str, required=True, help="Output CSV path.")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoints/best.pt (or epoch_XXX.pt).")
    ap.add_argument("--artifact_dir", type=str, default=None,
                    help="Artifact root containing smiles_tokenizer.json, formula_tokenizer.json, predictions/, checkpoints/. "
                         "If omitted, uses ckpt/.. (parent of checkpoints).")
    ap.add_argument("--spectrum_col", type=str, default="SPECTRUM_PREDICTED")
    ap.add_argument("--reaction_smiles_col", type=str, default="reaction_smiles")
    ap.add_argument("--formula_col", type=str, default="sum_formula")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--num_samples", type=int, default=1, help="Number of full sampled predictions per input row.")
    ap.add_argument("--rerank_by_frequency", action="store_true", help="Rank candidates by how often they appear among sampled predictions.")
    ap.add_argument("--num_candidates", type=int, default=None, help="Number of reranked candidates to write. Defaults to num_samples."
    )
    args = ap.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)
    if args.num_samples < 1:
        raise ValueError("--num_samples must be >= 1")

    ckpt_path = Path(args.ckpt)
    payload = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = payload.get("config", {})

    cfg = TrainConfig()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)

    device = torch.device(args.device if args.device else _get_attr(cfg, "device", "cuda"))
    print(f"[INFO] device={device}")

    if args.artifact_dir is not None:
        artifact_dir = Path(args.artifact_dir)
    else:
        artifact_dir = ckpt_path.parent.parent
    print(f"[INFO] artifact_dir={artifact_dir}")

    smiles_tok_path = artifact_dir / "smiles_tokenizer.json"
    if not smiles_tok_path.exists():
        raise FileNotFoundError(f"Missing smiles_tokenizer.json at: {smiles_tok_path}")
    smiles_tok = SmilesTokenizerAdapter.load(smiles_tok_path)
    smiles_tok.freeze()

    formula_tok = None
    formula_tok_path = artifact_dir / "formula_tokenizer.json"
    if formula_tok_path.exists():
        formula_tok = FormulaTokenizerAdapter.load(formula_tok_path)
        formula_tok.freeze()

    encoder, decoder, reactant_enc, fusion_enc, formula_enc, decoder_formula = build_models_from_cfg(
        cfg=cfg,
        device=device,
        smiles_tokenizer=smiles_tok,
        formula_tokenizer=formula_tok,
    )

    _safe_load_state_dict(encoder, payload["encoder"], "encoder")
    _safe_load_state_dict(decoder, payload["decoder"], "decoder")

    if reactant_enc is not None:
        if "reactant_encoder" in payload:
            _safe_load_state_dict(reactant_enc, payload["reactant_encoder"], "reactant_encoder")
        else:
            sd = _maybe_load_pt(artifact_dir / "predictions" / "reactant_encoder.pt")
            if sd is not None:
                _safe_load_state_dict(reactant_enc, sd, "reactant_encoder(from predictions/)")

    if fusion_enc is not None:
        if "fusion_encoder" in payload:
            _safe_load_state_dict(fusion_enc, payload["fusion_encoder"], "fusion_encoder")
        else:
            sd = _maybe_load_pt(artifact_dir / "predictions" / "fusion_encoder.pt")
            if sd is not None:
                _safe_load_state_dict(fusion_enc, sd, "fusion_encoder(from predictions/)")

    if formula_enc is not None:
        if "formula_encoder" in payload:
            _safe_load_state_dict(formula_enc, payload["formula_encoder"], "formula_encoder")
        else:
            sd = _maybe_load_pt(artifact_dir / "predictions" / "formula_encoder.pt")
            if sd is not None:
                _safe_load_state_dict(formula_enc, sd, "formula_encoder(from predictions/)")

    if decoder_formula is not None:
        sd = _maybe_load_pt(artifact_dir / "predictions" / "formula_decoder.pt")
        if sd is None:
            print("[WARN] cfg.formula_decoder=True but predictions/formula_decoder.pt not found. Formula predictions disabled.")
            decoder_formula = None
        else:
            _safe_load_state_dict(decoder_formula, sd, "formula_decoder")

    encoder.eval()
    decoder.eval()
    if reactant_enc is not None:
        reactant_enc.eval()
    if fusion_enc is not None:
        fusion_enc.eval()
    if formula_enc is not None:
        formula_enc.eval()
    if decoder_formula is not None:
        decoder_formula.eval()

    in_path = Path(args.input)
    if in_path.suffix.lower() == ".csv":
        df = pd.read_csv(in_path)
    elif in_path.suffix.lower() in [".parquet", ".pq"]:
        df = pd.read_parquet(in_path)
    else:
        raise ValueError(f"Unsupported input format: {in_path.suffix}")

    if args.spectrum_col not in df.columns:
        raise ValueError(f"Missing spectrum_col '{args.spectrum_col}' in input file.")

    need_reactants = bool(_get_attr(cfg, "reactant_encoder", False))
    need_formula_enc = bool(_get_attr(cfg, "formula_encoder", False))
    need_formula_dec = bool(_get_attr(cfg, "formula_decoder", False))
    need_formula_tokenizer = need_formula_enc or need_formula_dec
    spectrum_enc = bool(_get_attr(cfg, "spectrum_encoder", True))

    if need_formula_tokenizer and formula_tok is None:
        raise ValueError("Model needs formula tokenizer but formula_tokenizer.json not found in artifact_dir.")

    if need_reactants and args.reaction_smiles_col not in df.columns:
        raise ValueError(f"cfg.reactant_encoder=True but '{args.reaction_smiles_col}' missing in input file.")
    if need_formula_enc and args.formula_col not in df.columns:
        raise ValueError(f"cfg.formula_encoder=True but '{args.formula_col}' missing in input file.")

    spectra_raw = df[args.spectrum_col].tolist()
    spectra = []
    for s in tqdm(spectra_raw, desc="Parsing spectra"):
        spectra.append(literal_eval(s) if isinstance(s, str) else s)

    if args.reaction_smiles_col in df.columns:
        smiles_list = df[args.reaction_smiles_col].astype(str).tolist()
    else:
        smiles_list = ["C>>C" if cfg.decoder_mode == "rxn" else "C"] * len(df)

    formulas_list = None
    if need_formula_enc:
        if args.formula_col in df.columns:
            formulas_list = df[args.formula_col].astype(str).tolist()
        else:
            raise ValueError("cfg.formula_encoder=True requires real formulas column.")

        if formula_tok is None:
            raise ValueError("Missing formula tokenizer for formula_encoder.")

    synthetic_spectra = (args.spectrum_col == "SPECTRUM_PREDICTED")
    return_formula = need_formula_enc

    ds = SpecSmilesDataset(
        spectra=spectra,
        smiles=smiles_list,
        tokenizer=smiles_tok,
        encoder_mode=cfg.encoder_mode,
        max_bin_length=cfg.max_bin_length,
        max_mz=cfg.max_mz,
        product_only=(cfg.decoder_mode == "product"),
        return_reactants=need_reactants,
        synthetic_spectra=synthetic_spectra,
        formulas=formulas_list,
        formula_tokenizer=formula_tok,
        return_formula=return_formula,
    )

    pad_id = smiles_tok.pad_id

    if cfg.decoder_mode == "product":
        if need_reactants and need_formula_enc:
            collate_fn = lambda b: collate_batch_product_only_with_reactants_and_formula(b, pad_id)
        elif need_reactants:
            collate_fn = lambda b: collate_batch_product_only_with_reactants(b, pad_id)
        elif need_formula_enc:
            collate_fn = lambda b: collate_batch_product_only_with_formula(b, pad_id)
        else:
            collate_fn = lambda b: collate_batch_product_only(b, pad_id)
    else:
        collate_fn = lambda b: collate_batch(b, pad_id)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    temperature = float(args.temperature) if args.temperature is not None else float(_get_attr(cfg, "temperature", 1.0))
    topk = int(args.topk) if args.topk is not None else int(_get_attr(cfg, "topk", 50))
    num_samples = int(args.num_samples)
    num_candidates = int(args.num_candidates) if args.num_candidates is not None else num_samples
    if num_candidates < 1:
        raise ValueError("--num_candidates must be >= 1")

    max_smiles_new = int(_get_attr(cfg, "max_smiles_len", 256))
    max_fml_new = int(_get_attr(cfg, "formula_max_len", 64))

    do_smiles = not (_get_attr(cfg, "train_formula_only", False) and _get_attr(cfg, "formula_decoder", False))
    do_formula = (decoder_formula is not None)

    rows: List[Dict[str, Any]] = []

    with torch.inference_mode(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        for batch in tqdm(loader, desc="Predicting"):
            formula_strs = None
            formula_ids = None

            if need_reactants and _get_attr(cfg, "formula_decoder", False) and cfg.decoder_mode == "product" and len(batch) == 7:
                spec_batch, reactant_ids, fml_in, fml_out, formula_strs, smi_in, smi_out = batch
                formula_ids = None
            elif _get_attr(cfg, "formula_decoder", False) and _get_attr(cfg, "train_formula_only", False) and (not need_reactants) and len(batch) == 4:
                spec_batch, fml_in, fml_out, formula_strs = batch
                reactant_ids = None
                formula_ids = None
            elif need_reactants and need_formula_enc:
                spec_batch, reactant_ids, formula_ids, formula_strs, tgt_in, tgt_out = batch
            elif need_reactants:
                spec_batch, reactant_ids, tgt_in, tgt_out = batch
                formula_ids = None
            elif need_formula_enc:
                spec_batch, formula_ids, formula_strs, tgt_in, tgt_out = batch
                reactant_ids = None
            else:
                spec_batch, tgt_in, tgt_out = batch
                reactant_ids = None
                formula_ids = None

            spec_batch = spec_batch.to(device, non_blocking=True)
            if reactant_ids is not None:
                reactant_ids = reactant_ids.to(device, non_blocking=True)
            if formula_ids is not None:
                formula_ids = formula_ids.to(device, non_blocking=True)

            memory = build_memory(
                cfg=cfg,
                encoder=encoder,
                spec_batch=spec_batch,
                spectrum_enc=spectrum_enc,
                reactant_enc=reactant_enc,
                reactant_ids=reactant_ids,
                fusion_enc=fusion_enc,
                formula_enc=formula_enc,
                formula_ids=formula_ids,
            )

            B = spec_batch.size(0)

            pred_smiles_samples = None
            pred_rxn_samples = None
            if do_smiles:
                gen_samples = generate_smiles_samples(
                    decoder=decoder,
                    memory=memory,
                    tokenizer=smiles_tok,
                    max_new_tokens=max_smiles_new,
                    temperature=temperature,
                    topk=topk,
                    num_samples=num_samples,
                    device=device,
                )

                if cfg.decoder_mode == "rxn":
                    pred_rxn_samples = gen_samples
                    pred_smiles_samples = [
                        [_extract_product_from_rxn(x) for x in sample_list]
                        for sample_list in gen_samples
                    ]
                else:
                    pred_smiles_samples = gen_samples

            pred_formula_samples = None
            if do_formula:
                if formula_tok is None:
                    raise ValueError("decoder_formula exists but formula_tok is None")
                pred_formula_samples = generate_formula_samples(
                    decoder_formula=decoder_formula,
                    memory=memory,
                    tokenizer=formula_tok,
                    max_new_tokens=max_fml_new,
                    temperature=temperature,
                    topk=topk,
                    num_samples=num_samples,
                    device=device,
                )

            for i in range(B):
                r: Dict[str, Any] = {}

                if args.rerank_by_frequency:
                    # ---- SMILES consensus ranking ----
                    if do_smiles and pred_smiles_samples is not None:
                        ranked_smiles = _frequency_rerank_smiles(
                            pred_smiles_samples[i],
                            top_n=num_candidates,
                        )

                        for j, item in enumerate(ranked_smiles):
                            r[f"pred_smiles_{j + 1}"] = item["candidate"]  # canonical SMILES
                            r[f"pred_smiles_{j + 1}_raw_example"] = item["raw_example"]
                            r[f"pred_smiles_{j + 1}_count"] = item["count"]
                            r[f"pred_smiles_{j + 1}_frequency"] = item["frequency"]

                    # ---- reaction-string consensus ranking, optional ----
                    if pred_rxn_samples is not None:
                        ranked_rxn = _frequency_rerank(
                            pred_rxn_samples[i],
                            top_n=num_candidates,
                        )

                        for j, item in enumerate(ranked_rxn):
                            r[f"pred_rxn_{j + 1}"] = item["candidate"]
                            r[f"pred_rxn_{j + 1}_count"] = item["count"]
                            r[f"pred_rxn_{j + 1}_frequency"] = item["frequency"]

                    # ---- formula consensus ranking ----
                    if do_formula and pred_formula_samples is not None:
                        ranked_formula = _frequency_rerank(
                            pred_formula_samples[i],
                            top_n=num_candidates,
                        )

                        for j, item in enumerate(ranked_formula):
                            r[f"pred_formula_{j + 1}"] = item["candidate"]
                            r[f"pred_formula_{j + 1}_count"] = item["count"]
                            r[f"pred_formula_{j + 1}_frequency"] = item["frequency"]

                else:
                    # ---- original behavior: raw sampled outputs ----
                    if do_smiles and pred_smiles_samples is not None:
                        for j in range(num_samples):
                            r[f"pred_smiles_{j + 1}"] = pred_smiles_samples[i][j]

                        if pred_rxn_samples is not None:
                            for j in range(num_samples):
                                r[f"pred_rxn_{j + 1}"] = pred_rxn_samples[i][j]

                    if do_formula and pred_formula_samples is not None:
                        for j in range(num_samples):
                            r[f"pred_formula_{j + 1}"] = pred_formula_samples[i][j]

                rows.append(r)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()
