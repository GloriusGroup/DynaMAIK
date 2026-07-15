import torch
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import json
from collections import defaultdict


def token_accuracy(pred, tgt_out, valid_mask) -> float:
    """
    Token-level accuracy ignoring PAD.
    logits: [N, T, V], tgt_out: [N, T]
    """
    correct = (pred == tgt_out) & valid_mask
    acc_tok = correct.sum().float() / valid_mask.sum().clamp_min(1)
    return acc_tok


def sequence_accuracy(logits: torch.Tensor, tgt_out: torch.Tensor, pad_id: int) -> float:
    """
    Exact sequence match ignoring PAD positions.
    """
    with torch.no_grad():
        pred = logits.argmax(dim=-1)                      # [B, T]
        mask = (tgt_out != pad_id)
        # count a sequence correct if all non-PAD positions match
        per_seq_correct = (pred.eq(tgt_out) | ~mask).all(dim=1)  # [B]
        return per_seq_correct.float().mean().item()


def _smiles_to_fp(smiles: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto_similarity(pred_smiles: list[str], true_smiles: list[str]) -> tuple[float, float]:
    """
    Returns (mean_tanimoto, valid_fraction).
    Invalid SMILES pairs are skipped in the mean; valid_fraction = (#valid_pairs / N).
    """

    sims = []
    valid_pairs = 0
    for p, t in zip(pred_smiles, true_smiles):
        fp_p = _smiles_to_fp(p)
        fp_t = _smiles_to_fp(t)
        if fp_p is None or fp_t is None:
            continue
        valid_pairs += 1
        sims.append(DataStructs.TanimotoSimilarity(fp_p, fp_t))
    mean_sim = float(np.mean(sims)) if sims else float("nan")
    valid_frac = float(valid_pairs / max(len(pred_smiles), 1))
    return mean_sim, valid_frac


class SubstructureStats:
    """
    Dataset-level stats per functional group (fg_name).

    Per FG we track:

      - unscaled presence prob: (#samples with FG) / k
      - weighted presence prob: sum_j w_j * 1{FG in sample j}
        where w_j is the model probability of sample j (normalized per prefix)

    And we aggregate both over:
      - all examples
      - examples where FG is present in the true product
    """
    def __init__(self, substructure_dict: dict[str, str]):
        self.substructure_dict = substructure_dict

        # compile SMARTS
        self.smarts_mols: dict[str, Chem.Mol] = {}
        for name, smarts in substructure_dict.items():
            mol = Chem.MolFromSmarts(smarts)
            if mol is None:
                raise ValueError(f"Invalid SMARTS for {name}: {smarts}")
            self.smarts_mols[name] = mol

        # sums over ALL examples
        self.sum_p_all_unscaled   = defaultdict(float)  # plain (#pos / k)
        self.sum_p_all_weighted   = defaultdict(float)  # probability-weighted

        # sums over examples where FG is in the true product
        self.sum_p_true_unscaled  = defaultdict(float)
        self.sum_p_true_weighted  = defaultdict(float)

        # counts
        self.n_all                = defaultdict(int)
        self.n_true               = defaultdict(int)
        self.n_true_and_pred      = defaultdict(int)

    def update(
        self,
        true_smiles: str,
        pred_smiles_list: list[str],
        sample_weights: list[float] | None = None,
    ):
        """
        true_smiles: canonical (or raw) true product
        pred_smiles_list: list of K predicted product SMILES
        sample_weights: length-K weights per sample (should sum to ~1).
                        If None, all samples treated equally (1/k).
        """
        true_mol = Chem.MolFromSmiles(true_smiles) if true_smiles is not None else None
        pred_mols = [
            Chem.MolFromSmiles(s) if s is not None else None
            for s in pred_smiles_list
        ]
        k = max(len(pred_mols), 1)

        # weights for scaled version
        if sample_weights is None:
            w = [1.0 / k] * k
        else:
            w = [float(x) for x in sample_weights]
            s = sum(w)
            if s > 0:
                w = [x / s for x in w]
            else:
                w = [1.0 / k] * k

        for fg_name, patt in self.smarts_mols.items():
            # --- in_true: does GT product contain this FG? ---
            in_true = False
            if true_mol is not None:
                in_true = true_mol.HasSubstructMatch(patt)

            # --- per-sample FG presence ---
            count_pred = 0
            weighted_presence = 0.0
            any_in_pred = False

            for pm, weight in zip(pred_mols, w):
                if pm is None:
                    continue
                has_fg = pm.HasSubstructMatch(patt)
                if has_fg:
                    count_pred += 1
                    weighted_presence += weight
                    any_in_pred = True

            # unscaled: plain frequency (#samples with FG) / k
            p_unscaled = count_pred / k

            # weighted: sum_j w_j * 1{FG in sample j}
            p_weighted = weighted_presence

            # --- accumulate over ALL examples ---
            self.sum_p_all_unscaled[fg_name] += float(p_unscaled)
            self.sum_p_all_weighted[fg_name] += float(p_weighted)
            self.n_all[fg_name]              += 1

            # --- accumulate over examples where FG is present in GT ---
            if in_true:
                self.sum_p_true_unscaled[fg_name] += float(p_unscaled)
                self.sum_p_true_weighted[fg_name] += float(p_weighted)
                self.n_true[fg_name]              += 1
                if any_in_pred:
                    self.n_true_and_pred[fg_name] += 1

    def summary_rows(self) -> list[dict]:
        rows = []
        for fg_name, smarts in self.substructure_dict.items():
            n_all  = self.n_all[fg_name]
            n_true = self.n_true[fg_name]
            n_tap  = self.n_true_and_pred[fg_name]

            # means over ALL examples
            mean_p_all_unscaled = (
                self.sum_p_all_unscaled[fg_name] / n_all if n_all > 0 else 0.0
            )
            mean_p_all_weighted = (
                self.sum_p_all_weighted[fg_name] / n_all if n_all > 0 else 0.0
            )

            # means when FG is present in GT
            mean_p_true_unscaled = (
                self.sum_p_true_unscaled[fg_name] / n_true if n_true > 0 else 0.0
            )
            mean_p_true_weighted = (
                self.sum_p_true_weighted[fg_name] / n_true if n_true > 0 else 0.0
            )

            frac_true = n_true / n_all if n_all > 0 else 0.0
            coverage_given_true = n_tap / n_true if n_true > 0 else 0.0

            rows.append({
                "fg_name": fg_name,
                "smarts": smarts,
                "validation_set_size": int(n_all),
                "smarts_in_true": int(n_true),  # How often does this FG really appear in ground-truth?
                "smarts_in_true_and_at_least_one_pred_of_k": int(n_tap),  # Out of the examples where the FG was truly present, how many times did at least one of the model’s k samples predict that FG?
                "mean_confidence_all_unscaled": float(mean_p_all_unscaled),  # On average, across the entire dataset, how likely is the model to include this FG in a random sample?
                "mean_confidence_all_weighted": float(mean_p_all_weighted),  # On average, across the entire dataset, how likely is the model to include this FG in a random sample, weighted by sample probabilities.
                "mean_confidence_when_true_unscaled": float(mean_p_true_unscaled),  # On average, across the entire dataset, how likely is the model to include this FG in a random sample?
                "mean_confidence_when_true_weighted": float(mean_p_true_weighted),  # On average, across the entire dataset, how likely is the model to include this FG in a random sample, weighted by sample probabilities.
                "fraction_with_fg_true": float(frac_true),  # When the FG should be there (in ground truth), how often does the sampled model output include it?
                "coverage_given_true": float(coverage_given_true),  # How common is this FG in the validation dataset?
            })
        return rows
