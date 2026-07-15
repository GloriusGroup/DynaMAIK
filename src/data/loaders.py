import random
from ast import literal_eval
from math import isfinite
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.preprocess import (
    dict_to_topk_peaks,
    dict_to_vector_spectra,
    enumerate_reaction_smiles,
    generate_one_combination,
    get_ordered_bins,
)
from src.data.tokenizers import SmilesTokenizerAdapter, FormulaTokenizerAdapter


def parse_spectrum(value) -> Dict[int, float]:
    parsed = literal_eval(value) if isinstance(value, str) else value
    if not isinstance(parsed, dict):
        raise ValueError(f"Spectrum must be a dictionary, got {type(parsed).__name__}")

    spectrum: Dict[int, float] = {}
    for raw_mz, raw_intensity in parsed.items():
        mz_value = float(raw_mz)
        intensity = float(raw_intensity)
        if not mz_value.is_integer() or not isfinite(intensity):
            raise ValueError(f"Invalid spectrum peak: m/z={raw_mz!r}, intensity={raw_intensity!r}")
        spectrum[int(mz_value)] = intensity
    return spectrum


def load_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    match p.suffix.lower():
        case ".csv":
            return pd.read_csv(p)
        case ".parquet" | ".pq":
            return pd.read_parquet(p)
        case suffix:
            raise ValueError(f"Unsupported file format: {suffix}")


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    match p.suffix.lower():
        case ".csv":
            df.to_csv(p, index=False)
        case ".parquet" | ".pq":
            df.to_parquet(p, index=False)
        case suffix:
            raise ValueError(f"Unsupported file format: {suffix}")


class SpecSmilesDataset(Dataset):
    """
    Dataset for paired spectra and reaction/product SMILES.

    Depending on the constructor flags, each item can include reactant ids,
    formula ids/string, or product-only decoder ids.
    """
    def __init__(
        self,
        spectra: List[Dict[int, float]],
        smiles: List[str],
        tokenizer: SmilesTokenizerAdapter,
        encoder_mode: str,
        max_bin_length: int = 300,
        max_mz: int = 650,
        norm: str = "basepeak",
        transform: str = "log1p",
        product_only: bool = False,
        return_reactants: bool = False,
        synthetic_spectra: bool = False,
        return_formula: bool = False,
        formulas: List[str] | None = None,
        formula_tokenizer: FormulaTokenizerAdapter | None = None,
        enum_smiles: bool = False,
        frags: bool = False,
    ):
        if len(spectra) != len(smiles):
            raise ValueError("spectra and smiles must be same length")
        self.spectra = spectra
        self.smiles = smiles
        self.tk = tokenizer
        self.encoder_mode = encoder_mode
        self.max_bin_length = max_bin_length
        self.max_mz = max_mz
        self.norm = norm
        self.transform = transform
        self.epoch = 0
        self.rand_reactants = False
        self.synthetic_spectra = synthetic_spectra

        self.product_only = product_only
        self.return_reactants = return_reactants

        self.return_formula = return_formula
        self.formulas = formulas
        self.formula_tk = formula_tokenizer
        self.enum_smiles = enum_smiles
        self.frags = frags

        if self.return_formula:
            if not self.product_only:
                raise ValueError("return_formula=True is currently supported only with product_only=True.")
            if self.formulas is None:
                raise ValueError("return_formula=True requires formulas to be provided (list[str]).")
            if len(self.formulas) != len(self.spectra):
                raise ValueError("formulas and spectra must be same length")
            if self.formula_tk is None:
                raise ValueError("return_formula=True requires formula_tokenizer to be provided.")

    def set_rand_reactants(self, flag: bool) -> None:
        self.rand_reactants = flag

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(self, i: int):
        smi = self._prepare_smiles(self.smiles[i], i)
        spec_vec = self._encode_spectrum(self.spectra[i])

        if not self.product_only:
            ids = self.tk.encode(smi, add_bos=True, add_eos=True)
            return spec_vec, ids

        return self._product_only_item(spec_vec, smi, i)

    def _prepare_smiles(self, smiles: str, index: int) -> str:
        """Apply per-epoch/data-augmentation transforms to the stored SMILES value."""
        if self.frags:
            reactants, separator, product = smiles.partition(">>")
            selected = random.choice(reactants.split("."))
            smiles = selected + separator + product
        if self.enum_smiles:
            smiles = enumerate_reaction_smiles(smiles)
        if self.rand_reactants:
            smiles = generate_one_combination(smiles, seed=(self.epoch << 32) ^ index)
        return smiles

    def _encode_spectrum(self, spec_dict: Dict[int, float]) -> torch.Tensor:
        vector = dict_to_vector_spectra(spec_dict, max_mz=self.max_mz, synthetic=self.synthetic_spectra)
        match self.encoder_mode:
            case "raw":
                return torch.from_numpy(vector)
            case "binned":
                binned_vector = get_ordered_bins(vector, max_mz=self.max_mz, max_bin_length=self.max_bin_length)
                return torch.tensor(binned_vector, dtype=torch.float32)
            case "binned_2D":
                return dict_to_topk_peaks(spec_dict, max_mz=self.max_mz, max_bin_length=self.max_bin_length)
            case mode:
                raise ValueError(f"Unknown encoder_mode: {mode}")

    def _product_only_item(self, spec_vec: torch.Tensor, smiles: str, index: int):
        react_str, prod_str = self._split_reaction_smiles(smiles)
        prod_ids = self.tk.encode(prod_str, add_bos=True, add_eos=True)

        formula_ids = None
        formula_str = None
        if self.return_formula:
            formula_str = self._formula_at(index)
            formula_ids = self.formula_tk.encode(formula_str, add_bos=True, add_eos=True)

        if self.return_reactants and self.return_formula:
            react_ids = self.tk.encode(react_str, add_bos=True, add_eos=True)
            return spec_vec, react_ids, formula_ids, formula_str, prod_ids

        if self.return_reactants and not self.return_formula:
            react_ids = self.tk.encode(react_str, add_bos=True, add_eos=True)
            return spec_vec, react_ids, prod_ids

        if self.return_formula and not self.return_reactants:
            return spec_vec, formula_ids, formula_str, prod_ids

        return spec_vec, prod_ids

    def _formula_at(self, index: int) -> str:
        if self.formulas is None:
            raise ValueError("return_formula=True but formulas is None")
        return self.formulas[index]

    @staticmethod
    def _split_reaction_smiles(smiles: str) -> tuple[str, str]:
        if ">>" not in smiles:
            return "", smiles
        return smiles.split(">>", 1)
