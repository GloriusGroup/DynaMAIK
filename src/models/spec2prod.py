import torch
import torch.nn as nn
from typing import Any, Optional, Tuple

try:
    from src.models.tiny_recursive.tiny_recursive import RecursiveDecoder
    from src.models.tiny_recursive.core_network import TRMAttentionNetwork
except ModuleNotFoundError:
    RecursiveDecoder = None
    TRMAttentionNetwork = Any


def _require_recursive_decoder():
    if RecursiveDecoder is None:
        raise ImportError(
            "RecursiveProductDecoder requires src.models.tiny_recursive, "
            "but that package is not available in this project."
        )
    return RecursiveDecoder


class ConvStem1D(nn.Module):
    """
    Multi-scale conv front-end for 1D spectra.
    - Depthwise-separable convs + dilations to capture peak shapes and local context.
    - Preserves sequence length (padding computed to 'same').
    """
    def __init__(self, in_ch=1, hidden=64, n_blocks=3, dropout=0.05, k_size=7):
        super().__init__()
        layers = []
        ch = in_ch
        for i in range(n_blocks):
            # depthwise
            dilation = 2**i  # 1,2,4...
            #k = 7            # reasonably wide to cover peak widths
            pad = dilation * (k_size - 1) // 2
            layers += [
                nn.Conv1d(ch, ch, kernel_size=k_size, padding=pad, dilation=dilation, groups=ch, bias=False),
                nn.GELU(),
                nn.Conv1d(ch, hidden, kernel_size=1, bias=False),  # pointwise
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            ch = hidden
        self.net = nn.Sequential(*layers)

    def forward(self, x):            # x: [N, T, 1]
        x = x.transpose(1, 2)           # [N, 1, T]
        x = self.net(x)                 # [N, C, T]
        x = x.transpose(1, 2)           # [N, T, C]
        return x


class Encoder(nn.Module):
    """
    GC-MS Spectrum Encoder
    Input:  [N, 650]
    Output: memory [N, 650, d_model]
    """
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        conv_hidden: int = 64,
        conv_blocks: int = 3,
        k_size: int = 7,
        norm_first: bool = True,
        add_final_layernorm: bool = True,
        cnn: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.cnn = cnn
        # --- Conv front-end ---
        self.conv_stem = ConvStem1D(in_ch=1, hidden=conv_hidden, n_blocks=conv_blocks, dropout=dropout, k_size=k_size)

        # Project conv features to d_model (residual with raw intensity optional)
        self.to_model = nn.Linear(conv_hidden, d_model)         # [N, T, C] -> [N, T, d_model]
        nn.init.xavier_uniform_(self.to_model.weight)
        nn.init.zeros_(self.to_model.bias)

        # (Optional) add the raw intensity as a skip if you want:
        self.raw_proj = nn.Linear(1, d_model)
        nn.init.xavier_uniform_(self.raw_proj.weight); nn.init.zeros_(self.raw_proj.bias)

        self.pos_encoding = PositionalEncoding(max_len=seq_len, d_model=d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            norm_first=norm_first,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.final_ln = nn.LayerNorm(d_model) if add_final_layernorm else nn.Identity()

    def forward(self, spec_vec: torch.Tensor) -> Tuple[torch.Tensor]:
        assert spec_vec.dim() == 2 and spec_vec.size(1) == self.seq_len, f"spec_vec must be [B, {self.seq_len}]"
        # Prepare inputs
        raw = spec_vec.unsqueeze(-1)             # [N, T, 1]
        if self.cnn:
            conv_feats = self.conv_stem(raw)         # [N, T, C]
            x = self.to_model(conv_feats) + self.raw_proj(raw)  # [N, T, d_model] (skip projection)
        else:
            x = self.raw_proj(raw)                   # [N, T, d_model]

        x = self.pos_encoding(x)
        memory = self.encoder(x)                 # [N, T, d_model]
        memory = self.final_ln(memory)
        return memory


class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, T, d_model]
        N, T, _ = x.shape
        if T > self.pos_embedding.num_embeddings:
            raise ValueError(
                f"T={T} exceeds positional max_len={self.pos_embedding.num_embeddings}. "
                f"Increase Decoder(max_len=...) or clamp tokenizer length."
            )
        idx = torch.arange(T, device=x.device).unsqueeze(0).expand(N, T)
        return x + self.pos_embedding(idx)


class ReactantEncoder(nn.Module):
    """
    Reactant SMILES Encoder

    Input:
        reactant_ids: [B, T_rxn]  (token ids for reactant SMILES, padded with pad_id)
    Output:
        memory_rxn:   [B, T_rxn, d_model]
    """
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        max_len: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        norm_first: bool = True,
        add_final_layernorm: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[pad_id].zero_()

        self.pos_encoding = PositionalEncoding(max_len=max_len, d_model=d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            norm_first=norm_first,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.final_ln = nn.LayerNorm(d_model) if add_final_layernorm else nn.Identity()

    def forward(self, reactant_ids: torch.Tensor) -> torch.Tensor:
        """
        reactant_ids: [B, T_rxn]
        """
        assert reactant_ids.dim() == 2, "reactant_ids must be [B, T_rxn]"
        #B, T = reactant_ids.shape

        # [B, T, d_model]
        x = self.token_embedding(reactant_ids)
        x = self.pos_encoding(x)

        # padding mask for TransformerEncoder
        src_key_padding_mask = (reactant_ids == self.pad_id)  # [B, T]

        memory_rxn = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # [B, T, d_model]

        memory_rxn = self.final_ln(memory_rxn)
        return memory_rxn


class FormulaEncoder(nn.Module):
    """
    Sum Formula Encoder (RDKit-style formulas like: C6H12O6)

    Input:
        formula_ids: [B, T_f]  (token ids for sum formula, padded with pad_id)
    Output:
        memory_f:    [B, T_f, d_model]
    """
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        max_len: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        norm_first: bool = True,
        add_final_layernorm: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[pad_id].zero_()

        self.pos_encoding = PositionalEncoding(max_len=max_len, d_model=d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            norm_first=norm_first,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.final_ln = nn.LayerNorm(d_model) if add_final_layernorm else nn.Identity()

    def forward(self, formula_ids: torch.Tensor) -> torch.Tensor:
        """
        formula_ids: [B, T_f]
        """
        assert formula_ids.dim() == 2, "formula_ids must be [B, T_f]"

        # [B, T, d_model]
        x = self.token_embedding(formula_ids)
        x = self.pos_encoding(x)

        # padding mask for TransformerEncoder
        src_key_padding_mask = (formula_ids == self.pad_id)  # [B, T]

        memory_f = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # [B, T, d_model]
        memory_f = self.final_ln(memory_f)
        return memory_f


def create_padding_mask(spec_vec: torch.Tensor, pad_value: float = 0.0) -> torch.Tensor:
    """
    Creates a boolean padding mask for TransformerEncoder.
    spec_vec: [N, T] input spectrum (binned)
    pad_value: value indicating a padded bin (default = 0.0)

    Returns: mask [N, T] where True = PAD / ignore
    """
    if spec_vec.dim() != 2:
        raise ValueError("spec_vec must be [N, T] tensor.")

    # True where it's padding
    mask = (spec_vec == pad_value)
    fully_padded = mask.all(dim=1)
    if fully_padded.any():
        mask[fully_padded, 0] = False

    return mask  # bool tensor [N, T]


def create_2D_padding_mask(peaks: torch.Tensor) -> torch.Tensor:
    """
    peaks: [N, T, 2]
    Returns: [N, T] bool mask, True = PAD row
    """
    if peaks.dim() != 3 or peaks.size(-1) != 2:
        raise ValueError("peaks must be [N, T, 2].")

    # real rows: any non-zero feature
    real = (peaks.abs().sum(dim=2) > 0)   # [B,K]
    mask = ~real                          # True = PAD
    fully_padded = mask.all(dim=1)
    if fully_padded.any():
        mask[fully_padded, 0] = False
    return mask


class BinnedEncoder(nn.Module):
    """
    GC-MS Spectrum Encoder
    Input:  [N, max_bin_length]
    Output: memory [N, max_bin_length, d_model]
    """
    def __init__(self,
                 seq_len: int,
                 d_model: int,
                 nhead: int,
                 num_layers: int,
                 dim_feedforward: int,
                 dropout: float,
                 norm_first: bool = True,
                 add_final_layernorm: bool = True,
                 ):
        super().__init__()
        self.seq_len = seq_len
        self.to_model = nn.Linear(1, d_model)         # [N, T, 1] -> [N, T, d_model]
        nn.init.xavier_uniform_(self.to_model.weight)
        nn.init.zeros_(self.to_model.bias)
        self.pos_encoding = PositionalEncoding(max_len=seq_len, d_model=d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            norm_first=norm_first,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.final_ln = nn.LayerNorm(d_model) if add_final_layernorm else nn.Identity()

    def forward(self, spec_vec: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor]:
        assert spec_vec.dim() == 2 and spec_vec.size(1) == self.seq_len, f"spec_vec must be [N, {self.seq_len}]"
        if padding_mask is None:
            padding_mask = create_padding_mask(spec_vec)
        x = spec_vec.unsqueeze(-1)             # [N, T, 1]
        x = self.to_model(x)                   # [N, T, d_model]
        x = self.pos_encoding(x)
        memory = self.encoder(x, src_key_padding_mask=padding_mask)   # [N, T, d_model]
        memory = self.final_ln(memory)
        return memory


class PeakEncoder(nn.Module):
    def __init__(self,
                 seq_len: int,
                 d_model: int,
                 nhead: int,
                 num_layers: int,
                 dim_feedforward: int,
                 dropout: float,
                 norm_first: bool = True,
                 add_final_layernorm: bool = True,
                 ):
        super().__init__()
        self.to_model = nn.Linear(2, d_model)    # [N, T, 2] -> [N, T, d_model]
        nn.init.xavier_uniform_(self.to_model.weight)
        nn.init.zeros_(self.to_model.bias)
        self.pos_encoding = PositionalEncoding(max_len=seq_len, d_model=d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            norm_first=norm_first,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.final_ln = nn.LayerNorm(d_model) if add_final_layernorm else nn.Identity()

    def forward(self, spec_vec: torch.Tensor) -> torch.Tensor:
        pad_mask = create_2D_padding_mask(spec_vec)  # [N,T], True=PAD

        x = self.to_model(spec_vec)                  # [N, T, d_model]
        x = self.pos_encoding(x)                     # [N,T,d_model]

        memory = self.encoder(x, src_key_padding_mask=pad_mask)
        memory = self.final_ln(memory)               # [N,T,d]
        return memory



class FusionEncoder(nn.Module):
    """
    Attention-based fusion over spectrum + reactant memories.

    Inputs:
      memory_spec: [B, S_spec, d_model]
      memory_rxn:  [B, S_rxn,  d_model]
      mask_spec:   [B, S_spec] (True = PAD) or None
      mask_rxn:    [B, S_rxn]  (True = PAD) or None

    Output:
      memory_fused: [B, S_spec + S_rxn, d_model]
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_len_spec: int,
        max_len_rxn: int,
        norm_first: bool = True,
    ):
        super().__init__()
        self.max_len = max_len_spec + max_len_rxn

        # positional encoding over the *combined* sequence
        self.pos_encoding = PositionalEncoding(max_len=self.max_len, d_model=d_model)

        # segment embeddings: 0 = spectrum, 1 = reactants
        self.seg_embed = nn.Embedding(2, d_model)
        nn.init.normal_(self.seg_embed.weight, mean=0.0, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            norm_first=norm_first,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.final_ln = nn.LayerNorm(d_model)

    def forward(
        self,
        memory_spec: torch.Tensor,         # [B, S_spec, d]
        memory_rxn: torch.Tensor,          # [B, S_rxn,  d]
        mask_spec: torch.Tensor | None = None,  # [B, S_spec], True=PAD
        mask_rxn: torch.Tensor | None = None,   # [B, S_rxn],  True=PAD
    ) -> torch.Tensor:
        B, S_spec, d = memory_spec.shape
        B2, S_rxn, d2 = memory_rxn.shape
        assert B == B2 and d == d2, "Shapes mismatch between spec and reactant memories"
        S_total = S_spec + S_rxn
        assert S_total <= self.max_len, "Increase FusionEncoder.max_len"

        # 1) concat along sequence dim
        x = torch.cat([memory_spec, memory_rxn], dim=1)  # [B, S_spec+S_rxn, d]

        # 2) add segment embeddings
        seg_ids_spec = torch.zeros(B, S_spec, dtype=torch.long, device=x.device)  # 0
        seg_ids_rxn  = torch.ones(B, S_rxn,  dtype=torch.long, device=x.device)   # 1
        seg_ids = torch.cat([seg_ids_spec, seg_ids_rxn], dim=1)                   # [B, S_total]
        x = x + self.seg_embed(seg_ids)                                          # [B, S_total, d]

        # 3) add positional encoding
        x = self.pos_encoding(x)                                                 # [B, S_total, d]

        # 4) build padding mask for encoder
        if mask_spec is None:
            mask_spec = torch.zeros(B, S_spec, dtype=torch.bool, device=x.device)
        if mask_rxn is None:
            mask_rxn = torch.zeros(B, S_rxn, dtype=torch.bool, device=x.device)
        src_key_padding_mask = torch.cat([mask_spec, mask_rxn], dim=1)           # [B, S_total]

        # 5) transformer encoder does the fusion via self-attention
        fused = self.encoder(x, src_key_padding_mask=src_key_padding_mask)       # [B, S_total, d]
        fused = self.final_ln(fused)
        return fused


class RecursiveReactantEncoder(nn.Module):
    """
    Reactant encoder using TRM recursive refinement.
    Keeps the same external interface as ReactantEncoder:
      forward(reactant_ids) -> [B,T,D]
      has .pad_id
    """
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        max_len: int,
        d_model: int,
        network: TRMAttentionNetwork,
        num_refinement_blocks: int = 3,
        num_latent_refinements: int = 2,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[pad_id].zero_()

        self.pos = PositionalEncoding(max_len=max_len, d_model=d_model)

        # learnable init states
        self.output_init = nn.Parameter(torch.randn(d_model) * 1e-2)
        self.latent_init = nn.Parameter(torch.randn(d_model) * 1e-2)

        self.network = network
        self.num_refinement_blocks = num_refinement_blocks
        self.num_latent_refinements = num_latent_refinements

    def _get_initial(self, B: int, T: int):
        outputs = self.output_init.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        latents = self.latent_init.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        return outputs, latents

    def forward(self, reactant_ids: torch.Tensor) -> torch.Tensor:
        """
        reactant_ids: [B,T]
        returns:      [B,T,D]
        """
        B, T = reactant_ids.shape
        valid_mask = (reactant_ids != self.pad_id)  # True = valid (include)

        x = self.token_embedding(reactant_ids)      # [B,T,D]
        x = self.pos(x)

        outputs, latents = self._get_initial(B, T)

        # recursive refinement
        for _ in range(self.num_refinement_blocks):
            for _ in range(self.num_latent_refinements):
                latents = self.network(outputs + latents + x, mask=valid_mask)
            outputs = self.network(outputs + latents, mask=valid_mask)

        return outputs


class RecursiveSpectrumEncoder(nn.Module):
    """
    Spectrum encoder:
      spec_vec [B, S] -> CNN -> [B,S,D] -> TRM recursive refinement -> memory [B,S,D]
    """
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        network: TRMAttentionNetwork,
        conv_hidden: int = 64,
        conv_blocks: int = 3,
        k_size: int = 7,
        dropout: float = 0.05,
        num_refinement_blocks: int = 3,
        num_latent_refinements: int = 2,
        cnn: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.cnn = cnn

        self.conv_stem = ConvStem1D(in_ch=1, hidden=conv_hidden, n_blocks=conv_blocks, dropout=dropout, k_size=k_size)
        self.to_model = nn.Linear(conv_hidden, d_model)
        nn.init.xavier_uniform_(self.to_model.weight)
        nn.init.zeros_(self.to_model.bias)

        self.raw_proj = nn.Linear(1, d_model)
        nn.init.xavier_uniform_(self.raw_proj.weight)
        nn.init.zeros_(self.raw_proj.bias)

        self.pos = PositionalEncoding(max_len=seq_len, d_model=d_model)

        # learnable init states
        self.output_init = nn.Parameter(torch.randn(d_model) * 1e-2)
        self.latent_init = nn.Parameter(torch.randn(d_model) * 1e-2)

        self.network = network
        self.num_refinement_blocks = num_refinement_blocks
        self.num_latent_refinements = num_latent_refinements

    def _get_initial(self, B: int, T: int):
        outputs = self.output_init.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        latents = self.latent_init.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        return outputs, latents

    def forward(self, spec_vec: torch.Tensor) -> torch.Tensor:
        """
        spec_vec: [B,S] float
        return:   [B,S,D]
        """
        assert spec_vec.dim() == 2 and spec_vec.size(1) == self.seq_len, f"spec_vec must be [B,{self.seq_len}]"
        B, S = spec_vec.shape

        # spectrum has no padding in your pipeline -> all valid
        valid_mask = torch.ones(B, S, dtype=torch.bool, device=spec_vec.device)

        raw = spec_vec.unsqueeze(-1)  # [B,S,1]
        if self.cnn:
            conv_feats = self.conv_stem(raw)  # [B,S,C]
            x = self.to_model(conv_feats) + self.raw_proj(raw)  # [B,S,D]
        else:
            x = self.raw_proj(raw)

        x = self.pos(x)

        outputs, latents = self._get_initial(B, S)

        for _ in range(self.num_refinement_blocks):
            for _ in range(self.num_latent_refinements):
                latents = self.network(outputs + latents + x, mask=valid_mask)
            outputs = self.network(outputs + latents, mask=valid_mask)

        return outputs


class ElementCountHead(nn.Module):
    def __init__(self, d_model: int, n_elems: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_elems),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        # pooled: [B,d]
        return self.net(pooled)  # [B, n_elems]


class Decoder(nn.Module):
    """
    Reaction-SMILES Decoder (prefix LM)
      - Inputs (training):
            y_in_ids  [N, T] (teacher-forced inputs; starts with <bos>, then reactants, '>>', product...)
            memory    [N, 650, d_model] from Encoder
        Output:
            logits    [N, T, V]

    Loss behavior (controlled by self.decoder_mode):
        - 'rxn':     only product positions (tokens AFTER the first '>>') are used.
        - 'product': all non-PAD positions are used.
    """
    def __init__(
            self,
            vocab_size: int,
            pad_id: int,
            rxn_sep_id: int,                 # token id for '>>'
            max_len: int,
            d_model: int,
            nhead: int,
            num_layers: int,
            dim_feedforward: int,
            dropout: float,
            decoder_mode: str,
            norm_first: bool = True,
            multi_source: bool = False,) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.rxn_sep_id = rxn_sep_id
        self.decoder_mode = decoder_mode
        self.multi_source = multi_source

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[pad_id].zero_()
        self.positional = PositionalEncoding(max_len=max_len, d_model=d_model)

        if not multi_source:
            # original single-source decoder
            dec_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=norm_first,
            )
            self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
            self.layers = None
        else:
            # multi-source: stack of MultiSourceDecoderLayer
            self.decoder = None
            self.layers = nn.ModuleList(
                [
                    MultiSourceDecoderLayer(
                        d_model=d_model,
                        nhead=nhead,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                        norm_first=norm_first,
                    )
                    for _ in range(num_layers)
                ]
            )

        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(
            self,
            y_in_ids: torch.Tensor,     # [N, T]    teacher-forced inputs
            memory: torch.Tensor | Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]],  # single-source: [B,S,d], multi-source: tuple
            return_hidden: bool = False,
            ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        N, T = y_in_ids.shape
        device = y_in_ids.device

        tgt_key_padding_mask = (y_in_ids == self.pad_id)        # [N, T]
        tgt = self.token_embedding(y_in_ids)                    # [N, T, d]
        tgt = self.positional(tgt)

        tgt_mask = self._causal_mask(T, device)

        if not self.multi_source:
            # original single-source path
            memory_padding_mask = None
            if hasattr(memory, "values") and hasattr(memory, "padding_mask"):
                memory_padding_mask = memory.padding_mask
                memory = memory.values
            hs = self.decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_padding_mask,
            )                                                   # [B, T, d]
        else:
            spec_memory, rxn_memory, spec_mask, rxn_mask = memory
            x = tgt
            for layer in self.layers:
                x = layer(
                    tgt=x,
                    memory_spec=spec_memory,
                    memory_rxn=rxn_memory,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_spec_key_padding_mask=spec_mask,
                    memory_rxn_key_padding_mask=rxn_mask,
                )
            hs = x                                                     # [N, T, d]

        logits = self.ln(hs)
        logits = self.lm_head(logits)                           # [N, T, V]

        if return_hidden:
            return logits, hs  # or return logits, x if you prefer post-LN hidden
        return logits


    # ---------- Product-only masking helpers ----------

    def build_product_mask(self, y_out_ids: torch.Tensor) -> torch.Tensor:
        """
        Return a boolean mask [N, T] that is True where the target token belongs to the PRODUCT,
        i.e., strictly to the right of the first '>>' token in each sequence.
        If a sequence has no '>>', the mask is all False for that row.
        """
        N, T = y_out_ids.shape
        device = y_out_ids.device
        mask = torch.zeros((N, T), dtype=torch.bool, device=device)

        # Find first '>>' position per sample; loop is fine here and clear.
        for i in range(N):
            arrow_pos = (y_out_ids[i] == self.rxn_sep_id).nonzero(as_tuple=False)
            if arrow_pos.numel() == 0:
                # No arrow found -> no product tokens to supervise
                print(
                    f"No reaction separator token '>>' (id={self.rxn_sep_id}) found in target sequence index {i}.")
                continue
            j = int(arrow_pos[0].item())  # first occurrence
            if j + 1 < T:
                mask[i, j + 1:] = True  # tokens AFTER '>>' are product
        return mask

    def compute_loss(
            self,
            logits: torch.Tensor,  # [N, T, V] from forward(y_in_ids, memory)
            y_out_ids: torch.Tensor,  # [N, T]    gold targets (next-token labels)
            label_smoothing: float = 0.0
    ) -> torch.Tensor:
        """
        CrossEntropy over ONLY product positions; ignores PAD everywhere.
        """
        N, T, V = logits.shape

        if self.decoder_mode == 'rxn':
            product_mask = self.build_product_mask(y_out_ids)  # [N, T]
            pad_mask = (y_out_ids == self.pad_id)
            valid_mask = product_mask & (~pad_mask)  # [N, T]

        elif self.decoder_mode == 'product':
            valid_mask = (y_out_ids != self.pad_id)  # [N, T]

        else:
            raise ValueError(f"Unknown mode '{self.decoder_mode}', expected 'rxn' or 'product'.")

        # Flatten
        logits_flat = logits.reshape(N * T, V)
        targets_flat = y_out_ids.reshape(N * T)
        valid_flat = valid_mask.view(N * T)

        if label_smoothing > 0:
            # Use PyTorch's built-in with ignore_index to keep code clean
            loss_f = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing, reduction='sum')
        else:
            loss_f = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')

        # Set ignored targets to -100 (PyTorch ignore_index)
        masked_targets = targets_flat.clone()
        masked_targets[~valid_flat] = -100

        loss = loss_f(logits_flat, masked_targets)
        denom = valid_flat.sum().clamp_min(1)  # avoid div by zero if no product tokens
        return loss / denom

    # ---------- Inference with reactant prefix ----------

    @torch.no_grad()
    def generate_from_prefix(
            self,
            prefix_ids: torch.Tensor,  # [B, T_prefix_max] from build_prefix_batch_leftpad
            memory: torch.Tensor | Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]],  # memory: single-source [B,S,d] or multi-source tuple
            eos_id: int,
            pad_id: int,
            max_new_tokens: int = 256,
            temperature: float = 1.0,
            topk: int | None = None,
            return_logprobs: bool = False,
    ):

        ys = prefix_ids.clone()  # [B, T_cur]; last column is '>>'
        B = ys.size(0)

        max_len = self.positional.pos_embedding.num_embeddings
        if ys.size(1) >= max_len:
            if return_logprobs:
                # prefix only, zero log-prob increment
                return ys[:, :max_len], torch.zeros(B, device=ys.device)
            return ys[:, :max_len]

        remain = max_len - ys.size(1)
        max_new_tokens = min(max_new_tokens, remain)
        if max_new_tokens <= 0:
            if return_logprobs:
                return ys, torch.zeros(B, device=ys.device)
            return ys

        temp = max(float(temperature), 1e-6)
        alive = torch.ones(B, dtype=torch.bool, device=ys.device)
        if return_logprobs:
            seq_logp = torch.zeros(B, device=ys.device)

        for _ in range(max_new_tokens):
            # forward pass (possibly under autocast)
            logits = self.forward(ys, memory)  # [B, T_cur, V]
            next_logit = logits[:, -1, :] / temp  # [B, V]
            next_logit = next_logit.float()  # ← critical cast for stable softmax/sampling

            # mask finished rows
            if (~alive).any():
                next_logit[~alive, :] = float("-inf")
                next_logit[~alive, eos_id] = 0.0

            # mask PAD token
            next_logit[:, pad_id] = float("-inf")

            # full log-softmax for token log-probs
            if return_logprobs:
                log_probs = torch.log_softmax(next_logit, dim=-1)  # [B, V]

            # sample or greedy
            if topk is not None and topk > 0:
                k = min(topk, next_logit.size(-1))
                topk_vals, topk_idx = torch.topk(next_logit, k=k, dim=-1)
                probs = torch.softmax(topk_vals, dim=-1)  # safe: now float32
                choice = torch.multinomial(probs, num_samples=1)  # [B,1]
                next_id = topk_idx.gather(1, choice)  # [B,1]
            else:
                next_id = next_logit.argmax(dim=-1, keepdim=True)  # [B,1]

            if return_logprobs:
                step_logp = log_probs.gather(1, next_id).squeeze(1)  # [B]
                seq_logp = seq_logp + step_logp * alive.float()

            ys = torch.cat([ys, next_id], dim=1)
            alive = alive & (next_id.squeeze(1) != eos_id)

            if not alive.any() or ys.size(1) >= max_len:
                break

        if return_logprobs:
            return ys, seq_logp

        return ys


class MultiSourceDecoderLayer(nn.Module):
    """
    One decoder layer with:
      - self-attention on tgt
      - cross-attn to spectrum memory
      - cross-attn to reactant memory
      - *gated* fusion of the two cross-attn outputs
      - feed-forward block

    Shapes:
      tgt:         [B, T, d]
      memory_spec: [B, S_spec, d]
      memory_rxn:  [B, S_rxn,  d]
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.05,
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        self.norm_first = norm_first

        # self-attention on target
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # cross-attn to each source
        self.cross_spec = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_rxn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # --- gated fusion of spec/rxn outputs ---
        # gate: sigmoid( W_g [spec_out; rxn_out] )
        # fused = gate * spec_out + (1 - gate) * rxn_out
        self.gate_linear = nn.Linear(2 * d_model, d_model)

        # feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,                      # [B, T, d]
        memory_spec: torch.Tensor,             # [B, S_spec, d]
        memory_rxn: torch.Tensor,              # [B, S_rxn, d]
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_spec_key_padding_mask: Optional[torch.Tensor] = None,
        memory_rxn_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = tgt  # [B, T, d]

        # ---- 1) self-attention ----
        if self.norm_first:
            x_norm = self.norm1(x)
            attn_out, _ = self.self_attn(
                query=x_norm,
                key=x_norm,
                value=x_norm,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
            )
            x = x + self.dropout(attn_out)
        else:
            attn_out, _ = self.self_attn(
                query=x,
                key=x,
                value=x,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
            )
            x = self.norm1(x + self.dropout(attn_out))

        # ---- 2) cross-attention to spec & reactants ----
        if self.norm_first:
            q = self.norm2(x)
        else:
            q = x

        # spec
        spec_out, _ = self.cross_spec(
            query=q,
            key=memory_spec,
            value=memory_spec,
            key_padding_mask=memory_spec_key_padding_mask,
        )
        # reactants
        rxn_out, _ = self.cross_rxn(
            query=q,
            key=memory_rxn,
            value=memory_rxn,
            key_padding_mask=memory_rxn_key_padding_mask,
        )

        # ---- gated fusion ----
        # cat: [B, T, 2d]
        cat = torch.cat([spec_out, rxn_out], dim=-1)
        gate = torch.sigmoid(self.gate_linear(cat))   # [B, T, d]

        # convex combination between spec and reactant contexts
        fused = gate * spec_out + (1.0 - gate) * rxn_out  # [B, T, d]

        if self.norm_first:
            x = x + self.dropout(fused)
        else:
            x = self.norm2(x + self.dropout(fused))

        # ---- 3) feed-forward ----
        if self.norm_first:
            x_norm = self.norm3(x)
            ff_out = self.ffn(x_norm)
            x = x + self.dropout(ff_out)
        else:
            ff_out = self.ffn(x)
            x = self.norm3(x + self.dropout(ff_out))

        return x


class RecursiveProductDecoder(nn.Module):
    """
    Drop-in replacement (for training) that wraps your RecursiveDecoder
    but keeps the spec2prod.Decoder interface as much as needed by train.py.
    Single-source only: memory is [B, S, d].
    """
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        rxn_sep_id: int,
        max_len: int,
        d_model: int,
        network: TRMAttentionNetwork,
        decoder_mode: str = "product",   # or "rxn"
        halt_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        recursive_decoder_cls = _require_recursive_decoder()
        self.vocab_size   = vocab_size
        self.pad_id       = pad_id
        self.rxn_sep_id   = rxn_sep_id
        self.decoder_mode = decoder_mode
        self.multi_source = False        # important: so train.py treats memory as single-source

        self.positional = PositionalEncoding(max_len=max_len, d_model=d_model)

        # your recursive decoder core
        self.core = recursive_decoder_cls(
            dim=d_model,
            num_tokens=vocab_size,
            network=network,
            pad_token_id=pad_id,
            halt_loss_weight=halt_loss_weight,
        )

        # we’ll reuse ln + lm_head from your original Decoder
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, y_in_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        y_in_ids: [B, T]
        memory:   [B, S, d]
        returns:  logits [B, T, V]
        """
        B, T = y_in_ids.shape
        device = y_in_ids.device

        padding_mask = None
        if hasattr(memory, "values") and hasattr(memory, "padding_mask"):
            padding_mask = memory.padding_mask
            memory = memory.values

        Bm, S, _ = memory.shape
        assert B == Bm, "Batch size mismatch between y_in_ids and memory"
        memory_mask = (
            torch.ones(B, S, dtype=torch.bool, device=device)
            if padding_mask is None
            else ~padding_mask
        )

        # Let core handle embeddings & halting:
        # core.forward(tgt, memory, memory_mask) expects tgt ids
        logits_core, _halt_probs = self.core(y_in_ids, memory, memory_mask)  # [B,T,V]

        # optional extra LN+LM head:
        # if you want to use the logits from core directly, you can skip this.
        logits = logits_core

        return logits   # shape [B,T,V]

    # ---------- product-only helpers copied from original Decoder ----------

    def build_product_mask(self, y_out_ids: torch.Tensor) -> torch.Tensor:
        """
        Same as in spec2prod.Decoder.
        True where the token belongs to the product (right of '>>').
        """
        N, T = y_out_ids.shape
        device = y_out_ids.device
        mask = torch.zeros((N, T), dtype=torch.bool, device=device)

        for i in range(N):
            arrow_pos = (y_out_ids[i] == self.rxn_sep_id).nonzero(as_tuple=False)
            if arrow_pos.numel() == 0:
                continue
            j = int(arrow_pos[0].item())
            if j + 1 < T:
                mask[i, j + 1:] = True
        return mask

    def compute_loss(
        self,
        logits: torch.Tensor,    # [N, T, V]
        y_out_ids: torch.Tensor, # [N, T]
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        """
        Reuse your original compute_loss logic.
        """
        N, T, V = logits.shape

        if self.decoder_mode == "rxn":
            product_mask = self.build_product_mask(y_out_ids)
            pad_mask = (y_out_ids == self.pad_id)
            valid_mask = product_mask & (~pad_mask)
        elif self.decoder_mode == "product":
            valid_mask = (y_out_ids != self.pad_id)
        else:
            raise ValueError(f"Unknown mode '{self.decoder_mode}'")

        logits_flat  = logits.reshape(N * T, V)
        targets_flat = y_out_ids.reshape(N * T)
        valid_flat   = valid_mask.view(N * T)

        if label_smoothing > 0:
            loss_f = nn.CrossEntropyLoss(ignore_index=-100,
                                         label_smoothing=label_smoothing,
                                         reduction="sum")
        else:
            loss_f = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

        masked_targets = targets_flat.clone()
        masked_targets[~valid_flat] = -100

        loss = loss_f(logits_flat, masked_targets)
        denom = valid_flat.sum().clamp_min(1)
        return loss / denom

    # you can also copy generate_from_prefix(...) from your original Decoder
    # and reuse self.forward(...) inside it.
    @torch.no_grad()
    def generate_from_prefix(
        self,
        prefix_ids: torch.Tensor,  # [B, T_prefix_max]
        memory: torch.Tensor,      # single-source: [B, S, d]
        eos_id: int,
        pad_id: int,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        topk: int | None = None,
        return_logprobs: bool = False,
    ):
        ys = prefix_ids.clone()  # [B, T_cur]; last column is '>>' in rxn-mode
        B = ys.size(0)

        # maximum total length from positional encoding
        max_len = self.positional.pos_embedding.num_embeddings
        if ys.size(1) >= max_len:
            if return_logprobs:
                # prefix only, zero log-prob increment
                return ys[:, :max_len], torch.zeros(B, device=ys.device)
            return ys[:, :max_len]

        remain = max_len - ys.size(1)
        max_new_tokens = min(max_new_tokens, remain)
        if max_new_tokens <= 0:
            if return_logprobs:
                return ys, torch.zeros(B, device=ys.device)
            return ys

        temp = max(float(temperature), 1e-6)
        alive = torch.ones(B, dtype=torch.bool, device=ys.device)
        if return_logprobs:
            seq_logp = torch.zeros(B, device=ys.device)

        for _ in range(max_new_tokens):
            # forward pass through recursive decoder
            logits = self.forward(ys, memory)        # [B, T_cur, V]
            next_logit = logits[:, -1, :] / temp     # [B, V]
            next_logit = next_logit.float()          # stable softmax/sampling

            # mask finished rows
            if (~alive).any():
                next_logit[~alive, :] = float("-inf")
                next_logit[~alive, eos_id] = 0.0

            # mask PAD token
            next_logit[:, pad_id] = float("-inf")

            # log-probs if needed
            if return_logprobs:
                log_probs = torch.log_softmax(next_logit, dim=-1)  # [B, V]

            # sample or greedy
            if topk is not None and topk > 0:
                k = min(topk, next_logit.size(-1))
                topk_vals, topk_idx = torch.topk(next_logit, k=k, dim=-1)
                probs = torch.softmax(topk_vals, dim=-1)
                choice = torch.multinomial(probs, num_samples=1)  # [B,1]
                next_id = topk_idx.gather(1, choice)              # [B,1]
            else:
                next_id = next_logit.argmax(dim=-1, keepdim=True) # [B,1]

            if return_logprobs:
                step_logp = log_probs.gather(1, next_id).squeeze(1)  # [B]
                seq_logp = seq_logp + step_logp * alive.float()

            ys = torch.cat([ys, next_id], dim=1)
            alive = alive & (next_id.squeeze(1) != eos_id)

            if not alive.any() or ys.size(1) >= max_len:
                break

        if return_logprobs:
            return ys, seq_logp

        return ys
