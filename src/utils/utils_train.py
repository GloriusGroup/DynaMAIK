import math
import torch
from torch.utils.data import get_worker_info
from rdkit import Chem
from pathlib import Path
from src.data.tokenizers import FormulaTokenizerAdapter


def build_prefix_batch(tgt_in_batch: torch.Tensor, rxn_sep_id: int, pad_id: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Extract <bos>...>> per row, and LEFT-PAD so that every row has the same length
    and ends with '>>'. Padding is placed BEFORE '>>'.

    Returns:
        prefixes: [B, T_prefix_max] (right-aligned prefixes; pads on the left)
        prefix_lens: [B] true (unpadded) prefix lengths
        arrow_col: int, column index where '>>' sits for all rows (== T_prefix_max-1)
    """
    prefixes, lens = [], []

    for row in tgt_in_batch:
        arrow = (row == rxn_sep_id).nonzero(as_tuple=False)
        if arrow.numel() == 0:
            raise ValueError(f"Missing '>>' (id={rxn_sep_id}) in row: {row.tolist()}")
        j = int(arrow[0].item())          # position of '>>' in the raw row
        p = row[:j+1]                     # <bos> ... '>>'
        prefixes.append(p)
        lens.append(p.size(0))            # L_i

    lens = torch.tensor(lens, device=tgt_in_batch.device, dtype=torch.long)
    T_max = int(lens.max().item())        # max prefix length across batch
    B = len(prefixes)

    out = torch.full((B, T_max), pad_id, dtype=torch.long, device=tgt_in_batch.device)
    for i, p in enumerate(prefixes):
        L = p.size(0)
        out[i, T_max - L : T_max] = p     # RIGHT-ALIGN: pads land on the left

    arrow_col = T_max - 1                 # '>>' is now at the last column for all rows
    return out, lens, arrow_col


# For readability, also decode *just* the product slices by IDS (no string split errors)
# product slice = (j+1 : until EOS or PAD)
def slice_product(ids_row: torch.Tensor, tokenizer, decoder, j: int) -> str:
    sl = ids_row[j + 1:]
    # stop at first EOS or PAD if present
    stops = ((sl == tokenizer.eos_id) | (sl == decoder.pad_id)).nonzero(as_tuple=False)
    if stops.numel() > 0:
        sl = sl[:int(stops[0].item())]
    return tokenizer.decode(sl.tolist(), skip_specials=True)


def log_teacher_forced_product_only(
    decoder,
    tgt_in,
    memory,
    tgt_out,
    tokenizer,
    reactant_ids=None,
) -> list[dict]:
    logits = decoder(tgt_in, memory)  # [B, T, V]
    pred_tf_ids = logits.argmax(dim=-1)  # [B, T]
    rows = []

    B = tgt_in.size(0)
    for i in range(B):
        true_prod_raw = tokenizer.decode(tgt_out[i].tolist(), skip_specials=True)
        pred_prod_raw = tokenizer.decode(pred_tf_ids[i].tolist(), skip_specials=True)

        if reactant_ids is not None:
            react_str = tokenizer.decode(reactant_ids[i].tolist(), skip_specials=True)
        else:
            react_str = ""

        true_prod_can = canonicalize_smiles(true_prod_raw)
        pred_prod_can = canonicalize_smiles(pred_prod_raw)

        pred = pred_prod_can if pred_prod_can is not None else pred_prod_raw
        true = true_prod_can if true_prod_can is not None else true_prod_raw

        true_rxn = react_str + ">>" + true if react_str else true
        pred_rxn = react_str + ">>" + pred if react_str else pred

        rows.append({
            "reactants": react_str,
            "true_product": true,
            "pred_product_gen": pred,
            "exact_match_gen": int(pred == true),
            "exact_match_gen_raw": int(pred_prod_raw == true_prod_raw),
            "full_true_rxn": true_rxn,
            "full_pred_rxn_gen": pred_rxn,
        })
    return rows


def log_autoregressive_product_only(
    decoder,
    memory,
    tgt_out,
    tokenizer,
    eos_id,
    pad_id,
    reactant_ids=None,
    formulas=None,
    reactant_tokenizer=None,
    temperature: float = 1.0,
) -> list[dict]:
    B, T = tgt_out.shape
    device = tgt_out.device

    # prefix = <bos> for each sample
    bos_id = tokenizer.bos_id
    prefixes = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

    gen_ids = decoder.generate_from_prefix(
        prefix_ids=prefixes,
        memory=memory,
        eos_id=eos_id,
        pad_id=pad_id,
        max_new_tokens=T,
        temperature=temperature,
    )

    rows = []
    for i in range(B):
        true_prod_raw = tokenizer.decode(tgt_out[i].tolist(), skip_specials=True)
        pred_prod_raw = tokenizer.decode(gen_ids[i].tolist(),   skip_specials=True)

        if reactant_ids is not None and reactant_tokenizer is not None:
            react_str = reactant_tokenizer.decode(reactant_ids[i].tolist(), skip_specials=True)
        else:
            react_str = ""

        true_prod_can = canonicalize_smiles(true_prod_raw)
        pred_prod_can = canonicalize_smiles(pred_prod_raw)

        pred = pred_prod_can if pred_prod_can is not None else pred_prod_raw
        true = true_prod_can if true_prod_can is not None else true_prod_raw

        true_rxn = react_str + ">>" + true if react_str else true
        pred_rxn = react_str + ">>" + pred if react_str else pred

        formula_str = formulas[i] if formulas is not None else ""
        rows.append({
            "reactants": react_str,
            "true_product": true,
            "inputed_sum_formula": formula_str,
            "pred_product_gen": pred,
            "exact_match_gen": int(pred == true),
            "exact_match_gen_raw": int(pred_prod_raw == true_prod_raw),
            "full_true_rxn": true_rxn,
            "full_pred_rxn_gen": pred_rxn,
        })
    return rows


def log_teacher_forced(decoder, tgt_in, memory, tgt_out, tokenizer, B) -> list[dict]:
    # Forward pass under teacher forcing
    logits = decoder(tgt_in, memory)  # [B, T, V]
    pred_tf_ids = logits.argmax(dim=-1)  # [B, T]

    rows = []
    for i in range(B):
        pred_rxn = tokenizer.decode(pred_tf_ids[i].tolist(), skip_specials=True)
        true_rxn = tokenizer.decode(tgt_out[i].tolist(),  skip_specials=True)

        true_prod_raw = true_rxn.split(">>", 1)[1] if ">>" in true_rxn else true_rxn
        pred_prod_raw = pred_rxn.split(">>", 1)[1] if ">>" in pred_rxn else pred_rxn

        # canonicalize products
        true_prod_can = canonicalize_smiles(true_prod_raw)
        pred_prod_can = canonicalize_smiles(pred_prod_raw)

        true_prod = true_prod_can if true_prod_can is not None else true_prod_raw
        pred_prod = pred_prod_can if pred_prod_can is not None else pred_prod_raw

        rows.append({
            "reactants": true_rxn.split(">>", 1)[0] if ">>" in true_rxn else "",
            "true_product": true_prod,
            "pred_product_tf": pred_prod,
            "exact_match_tf": int(pred_prod == true_prod),
            "exact_match_tf_raw": int(pred_prod_raw == true_prod_raw),
            "full_pred_rxn_tf": pred_rxn,
            "full_true_rxn": true_rxn,
        })

    return rows


def canonicalize_smiles(smiles) -> str | None:
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return None


def log_autoregressive(decoder, rxn_sep_id, memory, tgt_out, tokenizer, eos_id, tgt_in, temperature: float = 1.0,) -> list[dict]:
    rows = []
    print("DEBUG log_autoregressive: type(tgt_in) =", type(tgt_in), "shape=", getattr(tgt_in, "shape", None))
    with torch.no_grad():
        prefixes, _, _ = build_prefix_batch(tgt_in_batch=tgt_in, rxn_sep_id=rxn_sep_id, pad_id=decoder.pad_id)
        gen_ids = decoder.generate_from_prefix(
            prefix_ids=prefixes,
            memory=memory,
            eos_id=eos_id,
            max_new_tokens=tgt_in.size(1),
            pad_id=decoder.pad_id,
            temperature=temperature,
        )

    N = gen_ids.size(0)
    for i in range(N):
        pred_rxn = tokenizer.decode(gen_ids[i].tolist(),  skip_specials=True)
        true_rxn = tokenizer.decode(tgt_out[i].tolist(), skip_specials=True)

        true_prod_raw = true_rxn.split(">>", 1)[1] if ">>" in true_rxn else true_rxn
        pred_prod_raw = pred_rxn.split(">>", 1)[1] if ">>" in pred_rxn else pred_rxn

        true_prod_can = canonicalize_smiles(true_prod_raw)
        pred_prod_can = canonicalize_smiles(pred_prod_raw)

        pred = pred_prod_can if pred_prod_can is not None else pred_prod_raw
        true = true_prod_can if true_prod_can is not None else true_prod_raw

        rows.append({
            "reactants": true_rxn.split(">>", 1)[0] if ">>" in true_rxn else "",
            "true_product": true,
            "pred_product_gen": pred,
            "exact_match_gen": int(pred == true),
            "exact_match_gen_raw": int(pred_prod_raw == true_prod_raw),
            "full_pred_rxn_gen": pred_rxn,
            "full_true_rxn": true_rxn,
        })
    return rows


def log_autoregressive_old(decoder, rxn_sep_id, memory, tgt_out, tokenizer, eos_id, tgt_in) -> list[dict]:
    """
    Autoregressive logging with CSV-compatible fields (same keys as log_teacher_forced).
    Uses left-padded prefixes so '>>' is aligned at the last prefix column.
    """
    device = tgt_out.device
    pad_id = decoder.pad_id
    rxn_sep_id = rxn_sep_id if rxn_sep_id is not None else decoder.rxn_sep_id

    # ---- 1) Build LEFT-PADDED prefixes so every row ends with '>>' at the same column
    prefixes, prefix_lens, arrow_col = build_prefix_batch(
        tgt_in_batch=tgt_out, rxn_sep_id=rxn_sep_id, pad_id=pad_id
    )  # prefixes: [N, T_prefix_max], arrow_col == T_prefix_max - 1

    # ---- 2) Generate from the aligned prefixes
    gen_ids_full = decoder.generate_from_prefix(
        prefix_ids=prefixes,          # last column is '>>' for all rows
        memory=memory,
        eos_id=eos_id,
        pad_id=pad_id,                # ensure PAD is banned internally
        max_new_tokens=tgt_in.size(1) # generous cap; generation will also clamp to capacity
    )

    # ---- 3) REALIGN generated sequence to target coordinates
    # In prefixes, '>>' is at `arrow_col`; in tgt_out[i], arrow is at j_i.
    # We shift each generated row left by delta_i = arrow_col - j_i.
    N, T_target = tgt_out.shape
    T_gen = gen_ids_full.size(1)
    gen_aligned = torch.full((N, max(T_gen, T_target)), pad_id, dtype=gen_ids_full.dtype, device=device)

    for i in range(N):
        arrow = (tgt_out[i] == rxn_sep_id).nonzero(as_tuple=False)
        j = int(arrow[0].item())
        delta = arrow_col - j
        start = max(delta, 0)
        row = gen_ids_full[i, start:]             # drop the artificial left-padding frame
        gen_aligned[i, : row.size(0)] = row       # place at column 0 in target frame

    # Make gen_ids match tgt_out shape [N, T]
    if gen_aligned.size(1) < T_target:
        pad_cols = T_target - gen_aligned.size(1)
        gen_ids = torch.cat(
            [gen_aligned, torch.full((N, pad_cols), pad_id, device=device, dtype=gen_aligned.dtype)], dim=1
        )
    else:
        gen_ids = gen_aligned[:, :T_target]  # [N, T]

    # ---- 4) Rebuild masks just like TF
    prod_mask = decoder.build_product_mask(tgt_out)                 # [N, T], True on product (after '>>')
    valid_mask = prod_mask & (tgt_out != pad_id)                    # exclude PAD on target side

    # GT left + generated product
    merged_gen_ids = torch.where(valid_mask, gen_ids, tgt_out)      # [N, T]

    # ---- 5) Emit a single CSV-friendly row (same keys as teacher-forced)
    rows = []
    for i in range(N):
        arrow = (tgt_out[i] == rxn_sep_id).nonzero(as_tuple=False)
        j = int(arrow[0].item())

        # exact-match over product span (mirror TF style)
        eq_prod = (gen_ids[i] == tgt_out[i]) | (~valid_mask[i])
        seq_correct = bool(eq_prod.all().item())
        has_prod = bool(valid_mask[i].any().item())
        exact_match_tf = int(seq_correct and has_prod)   # keep TF key name for CSV compatibility

        # decode full reaction using GT left + generated product
        #p_full = tokenizer.decode(merged_gen_ids[i].tolist(), skip_specials=True)
        t_full = tokenizer.decode(tgt_out[i].tolist(), skip_specials=True)

        # product strings using your helper (consistent with TF logger)
        pred_prod_raw = slice_product(merged_gen_ids[i], tokenizer=tokenizer, decoder=decoder, j=j)
        true_prod_raw = slice_product(tgt_out[i], tokenizer=tokenizer, decoder=decoder, j=j)

        true_prod_can = canonicalize_smiles(true_prod_raw)
        pred_prod_can = canonicalize_smiles(pred_prod_raw)

        # string-level exact match under generation (canonical)
        pred = pred_prod_can if pred_prod_can is not None else pred_prod_raw
        true = true_prod_can if true_prod_can is not None else true_prod_raw

        exact_match_gen = int(pred == true)
        exact_match_gen_raw = int(pred_prod_raw == true_prod_raw)

        # optional: rebuild full reactions with canonical products for nicer CSVs
        reactants = t_full.split(">>", 1)[0] if ">>" in t_full else ""
        full_pred_rxn_gen = reactants + ">>" + pred
        full_true_rxn     = reactants + ">>" + true

        rows.append({
            "reactants": reactants,
            "true_product": true,
            "pred_product_gen": pred,
            "exact_match_gen": exact_match_gen,          # canonical string-level equality
            "exact_match_gen_raw": exact_match_gen_raw,  # raw string-level equality
            "full_pred_rxn_gen": full_pred_rxn_gen,      # GT left + canonical pred product
            "full_true_rxn": full_true_rxn,
            # keep the token-level signal too if you like:
            "exact_match_gen_tokenwise": exact_match_tf,
            "has_arrow": 1,
            "prod_len_target": int(valid_mask[i].sum().item()),
        })
    return rows


def make_worker_init_fn(epoch: int) -> callable:
    def _init(_wid: int):
        info = get_worker_info()
        if info is not None:
            info.dataset.set_epoch(epoch)  # ← worker’s own dataset instance
    return _init


def get_scheduler(optimizer, cfg, steps_per_epoch: int):
    # ---- Warmup + Cosine ----
    if cfg.lr_scheduler == "warmup_cosine":
        if cfg.scheduler_step_on == "batch":
            # Steps-based schedule
            total_steps  = int(cfg.epochs * steps_per_epoch)
            warmup_steps = int(getattr(cfg, "warmup_steps", cfg.warmup_epochs * steps_per_epoch))
            warmup_steps = max(1, warmup_steps)
            decay_steps  = max(1, total_steps - warmup_steps)

            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0 / warmup_steps,  # first step ~ lr/warmup_steps
                end_factor=1.0,
                total_iters=warmup_steps
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=decay_steps,
                eta_min=cfg.min_lr
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_steps]
            )
        else:
            # Epoch-based schedule
            warmup_epochs = max(1, int(cfg.warmup_epochs))
            decay_epochs  = max(1, int(cfg.epochs) - warmup_epochs)
            #lr = cfg.lr
            start_lr = float(cfg.warmup_start_lr)
            start_lr_factor = start_lr / cfg.lr

            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=start_lr_factor,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=decay_epochs,
                eta_min=cfg.min_lr
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs]
            )

    # ---- Plateau (epoch-level only) ----
    if cfg.lr_scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg.plateau_factor, patience=cfg.plateau_patience
        )

    # ---- OneCycle (must be stepped per batch) ----
    if cfg.lr_scheduler == "onecycle":
        total_steps = int(cfg.epochs * steps_per_epoch)
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="cos"
        )

    # ---- Constant ----
    return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)


def set_lr(optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = float(lr)


def set_group_lrs(optimizer, lr_enc: float, lr_dec: float, lr_enc_react: float | None, lr_enc_formula: float | None) -> None:
    """
    Assumes param_groups[0] = encoder, param_groups[1] = decoder.
    """
    optimizer.param_groups[0]["lr"] = float(lr_enc)
    optimizer.param_groups[1]["lr"] = float(lr_dec)
    if lr_enc_react is not None and lr_enc_formula is None and len(optimizer.param_groups) > 2:
        optimizer.param_groups[2]["lr"] = float(lr_enc_react)
    elif lr_enc_formula is not None and lr_enc_react is None and len(optimizer.param_groups) > 2:
        optimizer.param_groups[2]["lr"] = float(lr_enc_formula)
    elif lr_enc_formula is not None and lr_enc_react is not None and len(optimizer.param_groups) > 3:
        optimizer.param_groups[2]["lr"] = float(lr_enc_react)
        optimizer.param_groups[3]["lr"] = float(lr_enc_formula)


def set_group_lrs_by_name(optimizer, lr_by_name: dict[str, float]) -> None:
    for group in optimizer.param_groups:
        name = group.get("name")
        if name in lr_by_name:
            group["lr"] = float(lr_by_name[name])


def lr_for_epoch(e: int, *, T_warm: int, T_total: int, lr_start: float, lr_base: float, lr_min: float) -> float:
    """
    Learning rate to USE during epoch e (1-based).
    - Warmup: linear from lr_start (e=1) -> lr_base (e=T_warm)
    - Cosine: from lr_base (e=T_warm+1) -> lr_min (e=T_total)
    """
    # --- Warmup epochs: interpolate endpoints exactly ---
    if e <= T_warm:
        if T_warm <= 1:
            return lr_base  # degenerate warmup: jump to base
        alpha = (e - 1) / (T_warm - 1)     # e=1 -> 0.0, e=T_warm -> 1.0
        return lr_start + (lr_base - lr_start) * alpha

    # --- Cosine phase ---
    t = e - T_warm                        # t = 1..(T_total - T_warm)
    T = max(1, T_total - T_warm)
    cos_term = 0.5 * (1.0 + math.cos(math.pi * t / T))  # t=0->1 (not used), t=T->0
    return lr_min + (lr_base - lr_min) * cos_term


def get_lr(optimizer) -> tuple[float, float] | tuple[float, float, float]:
    if len(optimizer.param_groups) == 3:
        return optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"], optimizer.param_groups[2]["lr"]

    return optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"]


def load_smarts_dict(path: str | Path) -> dict[str, str]:
    smarts_dict = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split(":", maxsplit=2)
            if len(parts) < 3:
                continue

            name  = parts[0].strip()
            smarts = parts[2].strip()
            smarts_dict[name] = smarts

    return smarts_dict


def compile_smarts(substructure_dict: dict[str, str]):
    compiled = {}
    for name, smarts in substructure_dict.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None:
            compiled[name] = (smarts, patt)
    return compiled

def _safe_mol(smiles: str):
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None

def compute_substructure_confidence_when_true_weighted(
    pred_top1: str,
    pred_samples: list[str],      # length K
    weights: list[float],         # length K, sum ~1
    compiled_smarts: dict[str, tuple[str, "Chem.Mol"]],
) -> list[dict]:
    """
    Returns a LIST of dicts (JSON-serializable).
    Only includes substructures that are TRUE in the top-1 prediction.
    Confidence is weighted probability over sampled predictions.
    """
    mol_top1 = _safe_mol(pred_top1)
    if mol_top1 is None:
        return []

    # which substructures are present in top-1
    present_names = []
    for name, (smarts, patt) in compiled_smarts.items():
        try:
            if mol_top1.HasSubstructMatch(patt):
                present_names.append(name)
        except Exception:
            pass

    if not present_names:
        return []

    # pre-build mols for samples
    mols = [_safe_mol(s) for s in pred_samples]

    out = []
    for name in present_names:
        smarts, patt = compiled_smarts[name]
        conf = 0.0
        for m, w in zip(mols, weights):
            if m is None:
                continue
            try:
                if m.HasSubstructMatch(patt):
                    conf += float(w)
            except Exception:
                continue
        out.append({
            "name": name,
            "smarts": smarts,
            "confidence_when_true_weighted": round(conf, 2),
        })
    return out


def counts_from_formula_tokens(
    fml_out: torch.Tensor,               # [B,T]
    formula_tokenizer: FormulaTokenizerAdapter,
    elements: list[str],
    pad_id: int,
    eos_id: int,
) -> torch.Tensor:
    device = fml_out.device
    B, T = fml_out.shape
    elem_to_idx = {e:i for i,e in enumerate(elements)}
    counts = torch.zeros((B, len(elements)), device=device, dtype=torch.float32)

    # map ids -> strings on CPU (fast enough; formulas are short)
    ids_cpu = fml_out.detach().cpu().tolist()
    itos = formula_tokenizer.itos  # id->token

    for b in range(B):
        toks = []
        for tid in ids_cpu[b]:
            if tid in (pad_id, eos_id):
                break
            tok = itos.get(int(tid), "<unk>")
            if tok in ("<bos>", "<pad>", "<eos>", "<unk>"):
                continue
            toks.append(tok)

        i = 0
        while i < len(toks):
            tok = toks[i]
            if tok in elem_to_idx:
                # default count=1 unless next token is number
                c = 1
                if i + 1 < len(toks) and toks[i+1].isdigit():
                    c = int(toks[i+1])
                    i += 1
                counts[b, elem_to_idx[tok]] += float(c)
            i += 1

    return counts