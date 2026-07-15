from typing import List, Tuple

import torch

TokenIds = List[int]


def pad_sequences(seqs: List[TokenIds], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    lens = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    max_len = int(lens.max().item())
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(seqs):
        out[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out, lens


def make_decoder_inputs(token_ids: List[TokenIds], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create teacher-forcing input/output tensors from BOS/EOS-wrapped ids."""
    dec_in = [ids[:-1] for ids in token_ids]
    dec_out = [ids[1:] for ids in token_ids]
    dec_in_pad, _ = pad_sequences(dec_in, pad_id)
    dec_out_pad, _ = pad_sequences(dec_out, pad_id)
    return dec_in_pad, dec_out_pad


def _stack_spectra(specs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.stack(specs, dim=0).to(torch.float32)


def collate_batch(
    batch: List[Tuple[torch.Tensor, List[int]]],
    pad_id: int,
) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor]:
    """
    Returns:
      spec_batch: [N, 650] float
      tgt_in:     [N, T]   long (starts with BOS, teacher forcing inputs)
      tgt_out:    [N, T]   long (next-token targets, ends with EOS/PAD)
    """
    specs, ids_list = zip(*batch)
    spec_batch = _stack_spectra(specs)
    tgt_in, tgt_out = make_decoder_inputs(list(ids_list), pad_id)
    return spec_batch, tgt_in, tgt_out


def collate_batch_product_only_with_reactants(
    batch: List[Tuple[torch.Tensor, List[int], List[int]]],
    pad_id: int,
) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """
    Each element of batch: (spec_vec, react_ids, prod_ids)

    Returns:
      spec_batch:     [N, 650] float
      react_batch:    [N, T_rxn_max] long (padded reactant ids)
      tgt_in:         [N, T_dec] long (product-only, teacher forcing inputs)
      tgt_out:        [N, T_dec] long (product-only, targets)
    """
    specs, react_ids_list, prod_ids_list = zip(*batch)

    spec_batch = _stack_spectra(specs)
    react_batch, _ = pad_sequences(list(react_ids_list), pad_id)
    tgt_in, tgt_out = make_decoder_inputs(list(prod_ids_list), pad_id)

    return spec_batch, react_batch, tgt_in, tgt_out


def collate_batch_product_only(
    batch: List[Tuple[torch.Tensor, List[int]]],
    pad_id: int,
):
    specs, prod_ids_list = zip(*batch)
    spec_batch = _stack_spectra(specs)
    tgt_in, tgt_out = make_decoder_inputs(list(prod_ids_list), pad_id)
    return spec_batch, tgt_in, tgt_out


def collate_batch_product_only_with_formula(batch, pad_id):
    specs, formula_ids_list, formula_strs, prod_ids_list = zip(*batch)
    spec_batch = _stack_spectra(specs)
    formula_batch, _ = pad_sequences(list(formula_ids_list), pad_id)
    tgt_in, tgt_out = make_decoder_inputs(list(prod_ids_list), pad_id)
    return spec_batch, formula_batch, list(formula_strs), tgt_in, tgt_out


def collate_batch_product_only_with_reactants_and_formula(batch, pad_id):
    specs, react_ids_list, formula_ids_list, formula_strs, prod_ids_list = zip(*batch)
    spec_batch = _stack_spectra(specs)
    react_batch, _ = pad_sequences(list(react_ids_list), pad_id)
    formula_batch, _ = pad_sequences(list(formula_ids_list), pad_id)
    tgt_in, tgt_out = make_decoder_inputs(list(prod_ids_list), pad_id)
    return spec_batch, react_batch, formula_batch, list(formula_strs), tgt_in, tgt_out


def collate_batch_product_only_with_reactants_and_formula_multitask(batch, pad_id):
    """
    Each element: (spec_vec, react_ids, formula_ids, formula_str, prod_ids)

    Returns:
      spec_batch:     [B, ...]
      react_batch:    [B, T_rxn]
      fml_in:         [B, T_fml-1]
      fml_out:        [B, T_fml-1]
      fml_strs:       list[str]
      smi_in:         [B, T_smi-1]
      smi_out:        [B, T_smi-1]
    """
    specs, react_ids_list, formula_ids_list, formula_strs, prod_ids_list = zip(*batch)

    spec_batch = _stack_spectra(specs)
    react_batch, _ = pad_sequences(list(react_ids_list), pad_id)

    # formula teacher-forcing
    fml_in, fml_out = make_decoder_inputs(list(formula_ids_list), pad_id)

    # smiles teacher-forcing (product-only)
    smi_in, smi_out = make_decoder_inputs(list(prod_ids_list), pad_id)

    return spec_batch, react_batch, fml_in, fml_out, list(formula_strs), smi_in, smi_out


def collate_batch_formula_only(
    batch,  # each item must contain formula token ids
    pad_id_fml: int,
):
    # expected item: (spec, formula_ids, formula_str, prod_ids?) OR (spec, formula_ids, formula_str)
    # adapt unpacking to your dataset item layout:
    specs, formula_ids_list, formula_strs, *rest = zip(*batch)

    spec_batch = _stack_spectra(specs)

    fml_batch, _ = pad_sequences(list(formula_ids_list), pad_id_fml)  # [B, T]
    fml_in = fml_batch[:, :-1].contiguous()
    fml_out = fml_batch[:, 1:].contiguous()

    return spec_batch, fml_in, fml_out, list(formula_strs)


def collate_batch_product_only_with_formula_multitask(batch, pad_id: int):
    """
    Each dataset sample is expected to be:
        spec, formula_ids, formula_str, prod_ids

    Returns:
        spec_batch, fml_in, fml_out, formula_strs, smi_in, smi_out
    """
    specs, formula_ids_list, formula_strs, prod_ids_list = zip(*batch)

    spec_batch = _stack_spectra(specs)

    # formula teacher-forcing
    fml_in, fml_out = make_decoder_inputs(list(formula_ids_list), pad_id)

    # smiles teacher-forcing (product-only)
    smi_in, smi_out = make_decoder_inputs(list(prod_ids_list), pad_id)

    return spec_batch, fml_in, fml_out, list(formula_strs), smi_in, smi_out
