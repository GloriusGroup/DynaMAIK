import random

import numpy as np
import torch
from rdkit import Chem


def dict_to_vector_spectra(spec: dict[int, float], max_mz: int = 650, synthetic: bool = False) -> np.ndarray:
    """Convert an m/z intensity dictionary to a base-peak-normalized vector."""
    vector = np.zeros(max_mz, dtype=np.float32)
    for mz, intensity in spec.items():
        if 1 <= mz <= max_mz:
            vector[mz - 1] = intensity

    max_intensity = vector.max()
    if max_intensity > 0:
        vector /= max_intensity

    if synthetic:
        threshold = 0.01
        vector[vector < threshold] = 0.0

    return vector


def get_ordered_bins(intensity_vector: np.ndarray, max_mz: int, max_bin_length: int) -> list[int]:
    """Return non-zero m/z bins ordered by intensity and padded/truncated to a fixed length."""
    peak_list = [(i + 1, intensity_vector[i]) for i in range(max_mz - 1)]
    peak_list.sort(key=lambda peak: peak[1], reverse=True)

    bin_list = [mz for mz, intensity in peak_list if intensity > 0]
    if len(bin_list) < max_bin_length:
        return bin_list + [0] * (max_bin_length - len(bin_list))
    return bin_list[:max_bin_length]


def dict_to_topk_peaks(spec: dict[int, float], max_mz: int, max_bin_length: int = 300) -> torch.Tensor:
    """Extract top-K positive peaks and return a fixed tensor of shape [K, 2]."""
    items = [(float(mz), float(intensity)) for mz, intensity in spec.items() if 0 <= mz <= max_mz and intensity > 0]
    if not items:
        return torch.zeros(max_bin_length, 2, dtype=torch.float32)

    peaks = torch.tensor(items, dtype=torch.float32)
    topk = min(max_bin_length, peaks.size(0))

    _, indices = torch.topk(peaks[:, 1], k=topk, largest=True, sorted=True)
    peaks = peaks[indices]
    peaks = peaks[peaks[:, 0].argsort()]
    peaks[:, 1] /= peaks[:, 1].max()

    out = torch.zeros(max_bin_length, 2, dtype=torch.float32)
    out[:topk] = peaks
    return out


def generate_one_combination(reaction_smiles: str, seed: int) -> str:
    """Select one deterministic random non-empty subset of reactants."""
    reactants, _, products = reaction_smiles.partition(">>")
    reactants = reactants.split(".")
    if len(reactants) == 1:
        return reaction_smiles

    rng = random.Random(seed)
    while True:
        mask = rng.getrandbits(len(reactants))
        if mask:
            subset = [reactants[i] for i in range(len(reactants)) if (mask >> i) & 1]
            return ".".join(subset) + ">>" + products


def get_sorted_reactants(reaction_smiles: str) -> str:
    """Sort reactants alphabetically while keeping the product unchanged."""
    reactants, _, products = reaction_smiles.partition(">>")
    reactants = reactants.split(".")
    reactants.sort()
    return ".".join(reactants) + ">>" + products


def enumerate_smiles(smiles: str) -> str:
    """Return one randomized SMILES string, or the original value if RDKit cannot parse it."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, doRandom=True, canonical=False)


def enumerate_reaction_smiles(reaction_smiles: str) -> str:
    """Randomize each molecule in a reaction SMILES string independently."""
    if ">>" not in reaction_smiles:
        return enumerate_smiles(reaction_smiles)

    reactants, _, product = reaction_smiles.partition(">>")
    enum_reactants = [enumerate_smiles(reactant) for reactant in reactants.split(".")]
    return ".".join(enum_reactants) + ">>" + enumerate_smiles(product)
