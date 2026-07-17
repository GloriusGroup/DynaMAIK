from dataclasses import dataclass, fields
from pathlib import Path

import torch
import yaml


DEFAULT_SMARTS_PATH = str(Path(__file__).resolve().parents[1] / "data" / "smarts_filtered.txt")


@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1024
    n_encoder_layers: int = 3
    n_decoder_layers: int = 3
    max_seq_len: int = 512
    dropout: float = 0.1


@dataclass
class TrainConfig:
    # data
    training_path: str = ""
    validation_path: str | None = None  # None uses an automatic train/validation split
    max_mz: int = 650
    max_bin_length: int = 300
    max_smiles_len: int = 512
    formula_max_len: int = 32
    encoder_mode: str = "raw"  # "raw" or "binned" or "binned_2D"
    decoder_mode: str = "product"  # "rxn" -> Decoder sees reactants or "product" -> Decoder sees product only
    sort_reactants: bool = False  # whether to sort reactants alphabetically in reaction SMILES
    enum_smiles: bool = True
    frags: bool = True  # whether to use fragments in the dataset

    # Encoder
    reactant_encoder: bool = True  # if decoder_mode=="product" and reactant_encoder=False -> Spec2Mol
    spectrum_encoder: bool = True  # whether to encode spectrum information
    formula_encoder: bool = False  # whether to encode molecular formula information -> ONLY IF FORMULA_DECODER=FALSE
    fusion_encoder: bool = True  # whether to fuse reactant and spectrum encodings -> will only work if reactant_encoder=True and spectrum_encoder=True
    recursive_spectrum_encoder: bool = False  # whether to use a recursive spectrum encoder
    recursive_reactant_encoder: bool = False

    # Decoder
    formula_decoder: bool = True  # whether to use multi-task SMILES + SumFormula prediction -> ONLY IF FORMULA_ENCODER=FALSE
    train_formula_only: bool = False # if True: only SumFormula prediction with the SumFormula Tokenizer
    multi_source_decoder: bool = False  # whether to use multi-source decoder attention -> will only work if reactant_encoder=True and spectrum_encoder=True
    recursive_decoder: bool = False
    confidence: bool = True  # get confidence scores for funtional groups
    topk: bool = False  # if True: uses top-k sampling during inference else: greedy
    k_samples: int = 10
    formula_loss_weight: float = 1.5  # weight for formula loss in multi-task training (if formula dominates, reduce)
    use_count_head: bool = False
    count_loss_weight: float = 1.0
    count_elements: tuple[str, ...] = ("C", "H", "B", "Br", "Cl", "F", "I", "N", "O", "P", "S", "Si")

    # Paths
    smarts_path: str = DEFAULT_SMARTS_PATH
    mlflow_tracking_uri: str = "mlruns"

    # training
    batch_size: int = 100
    epochs: int = 50
    #lr: float = 1e-3
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    num_workers: int = 8
    eval_interval: int = 1

    # model
    d_model: int = 256
    nhead: int = 16
    enc_layers: int = 5
    dec_layers: int = 5
    dim_ff: int = 4096

    # FusionEncoder
    fusion_nhead: int = 4
    fusion_layers:int = 1
    fusion_dim_ff: int = 1024

    # RecursiveEncoder:
    trm_ref_blocks: int = 3
    trm_latent_refines: int = 3
    trm_nhead: int = 8

    dropout: float = 0.05
    temperature: float = 1.0

    #CNN
    cnn: bool = True
    conv_hidden: int = 256
    conv_blocks: int = 3
    k_size: int = 7

    # misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    #device: str = 'cpu'
    #ckpt_dir: str = "checkpoints"
    print_every: int = 100
    experiment: str = 'fragments'
    randomize_reactants: bool = False
    spectrum_column: str = "SPECTRUM_PREDICTED"
    reaction_smiles_column: str = "reaction_smiles"
    formula_column: str = "sum_formula"

    # learning rate scheduler
    lr_scheduler: str = "warmup_cosine"  # ["warmup_cosine", "plateau", "onecycle", "constant"]
    warmup_epochs: int = 15   # overrides warmup_steps when set
    #min_lr: float = 5e-6  # floor LR for warmup_cosine
    #warmup_start_lr: float = 1e-8  # initial LR for warmup_cosine
    cosine_cycles: float = 0.5  # 0.5 = decay once to min_lr over training
    scheduler_step_on: str = "epoch"  # ["batch", "epoch"] for warmup_cosine/constant
    plateau_patience: int = 5  # for ReduceLROnPlateau

    # learning rates
    lr_enc: float = 5e-4
    lr_enc_react: float = 5e-4
    lr_enc_formula: float = 5e-4
    lr_dec: float = 5e-4

    enc_warmup_epochs: int = 5
    enc_react_warmup_epochs: int = 5
    enc_formula_warmup_epochs: int = 5
    dec_warmup_epochs: int = 5

    enc_lr_start: float = 1e-8  # start LR for encoder warmup
    enc_react_lr_start: float = 1e-8
    enc_formula_lr_start: float = 1e-8
    dec_lr_start: float = 1e-8

    enc_lr_min: float = 1e-6
    enc_react_lr_min: float = 1e-6
    enc_formula_lr_min: float = 1e-6
    dec_lr_min: float = 1e-6

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as stream:
            values = yaml.safe_load(stream) or {}
        if not isinstance(values, dict):
            raise ValueError("Training configuration must be a YAML mapping")

        known_fields = {field.name for field in fields(cls)}
        unknown_fields = sorted(set(values) - known_fields)
        if unknown_fields:
            raise ValueError(f"Unknown training configuration fields: {', '.join(unknown_fields)}")

        if "count_elements" in values:
            values["count_elements"] = tuple(values["count_elements"])

        base_dir = config_path.parent
        for field_name in ("training_path", "validation_path", "smarts_path"):
            value = values.get(field_name)
            if value:
                candidate = Path(value).expanduser()
                if not candidate.is_absolute():
                    candidate = base_dir / candidate
                values[field_name] = str(candidate.resolve())

        tracking_uri = values.get("mlflow_tracking_uri")
        if tracking_uri and "://" not in tracking_uri and not tracking_uri.startswith("file:"):
            candidate = Path(tracking_uri).expanduser()
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            values["mlflow_tracking_uri"] = str(candidate.resolve())

        config = cls(**values)
        if not config.training_path:
            raise ValueError("training_path must be set in the YAML configuration")
        if config.device == "auto":
            config.device = "cuda" if torch.cuda.is_available() else "cpu"
        return config
