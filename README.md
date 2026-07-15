# DynaMAIK

DynaMAIK is a PyTorch project for predicting product SMILES and molecular formulas from GC-MS spectra. Models can optionally use reaction reactants or a known molecular formula as additional inputs and can jointly predict product SMILES and molecular formulas.

<p align="center">
  <img src="img/DynaMAIK_toc.png" width="500">
</p>

## Features

- Raw, ordered-bin, and top-k peak spectrum encoders.
- Product-only and full reaction-sequence decoding.
- Optional reactant and molecular-formula encoders.
- Optional joint SMILES and molecular-formula decoding.
- Attention-based fusion or multi-source decoder attention.
- Optional element-count auxiliary loss.
- Greedy or top-k autoregressive generation.
- Validation-time molecular and SMARTS-based confidence metrics.
- CSV and Parquet input support.
- MLflow experiment tracking and checkpoint artifacts.

## Requirements

- Python 3.10 or newer.
- The dependencies pinned in `requirements.txt`.
- CUDA is optional. Training and prediction can run on CPU, although model training will be substantially slower.

Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

RDKit and PyTorch installation requirements can vary by platform. If installation through `pip` is not suitable for your system, use an equivalent Conda environment with the versions listed in `requirements.txt`.

## Project Structure

```text
src/
  data/
    collate.py          Batch collation and teacher-forcing tensors
    loaders.py          Dataset, table I/O, and spectrum validation
    preprocess.py       Spectrum and SMILES preprocessing
    tokenizers.py       SMILES and molecular-formula tokenizers
    smarts_filtered.txt SMARTS definitions used for confidence metrics
  models/
    configs.py          Training and model configuration
    evaluation.py       Validation and generation metrics
    metrics.py          Molecular and token-level metrics
    predict.py          Checkpoint-based prediction CLI
    spec2prod.py        Encoder, decoder, and fusion modules
    train.py            Training entry point and checkpoint handling
  utils/
    model_runtime.py    Shared encoder-memory and metric helpers
    train_runtime.py    Configuration, model, and training dispatch
    utils_train.py      Scheduling, chemistry, logging, and prefix helpers
tests/
  test_core.py          Regression and supported-mode smoke tests
```

## Input Data

Training and prediction accept tabular molecular data. Default column names are defined by `TrainConfig` in `src/models/configs.py`.

| Column | Default name | Description |
| --- | --- | --- |
| Spectrum | `SPECTRUM_PREDICTED` | Dictionary mapping integer m/z values to numeric intensities |
| Reaction SMILES | `reaction_smiles` | Usually `REACTANTS>>PRODUCT` |
| Molecular formula | `sum_formula` | Product formula such as `C6H12O6` |

CSV spectrum values are Python-literal dictionary strings:

```text
"{43: 0.5, 91: 1.0}"
```

Parquet files may store the same value as a native dictionary. Spectrum m/z keys must be integer-valued, and intensities must be finite numbers.

The formula column is required during training when either `formula_encoder` or `formula_decoder` is enabled. During prediction it is required only when `formula_encoder` is enabled.

## Configuration

Training is configured through `TrainConfig` in `src/models/configs.py`. Update at least these path fields before starting a run:

- `home_path`: base directory used for data and MLflow artifacts.
- `dataset`: training CSV or Parquet filename.
- `validation_path`: optional validation CSV; use `None` for an automatic train/validation split.

Important model options include:

| Option | Description |
| --- | --- |
| `encoder_mode` | `raw`, `binned`, or `binned_2D` |
| `decoder_mode` | `product` or `rxn` |
| `spectrum_encoder` | Use spectra as model input |
| `reactant_encoder` | Encode reactants separately in product mode |
| `formula_encoder` | Use a known formula as model input |
| `formula_decoder` | Jointly predict the molecular formula |
| `train_formula_only` | Train only the formula decoder |
| `fusion_encoder` | Fuse spectrum and reactant memories before decoding |
| `multi_source_decoder` | Attend to spectrum and reactant memories separately |
| `use_count_head` | Add an element-count auxiliary objective |
| `frags` | Select one reactant fragment while preserving the product |
| `enum_smiles` | Apply randomized RDKit SMILES enumeration |

Invalid combinations are rejected before data loading. In particular:

- `formula_encoder` and `formula_decoder` are mutually exclusive.
- Formula-only training requires `formula_decoder=True` and currently does not support a reactant encoder.
- Fusion and multi-source decoding both require spectrum and reactant encoders and cannot be enabled together.
- Multi-source decoding does not currently support formula-encoder memory.
- `decoder_mode="rxn"` currently supports spectrum input only.
- Recursive model options require `src.models.tiny_recursive`, which is not included in this repository.

## Training

After configuring `TrainConfig`, start training from the repository root:

```bash
python -m src.models.train
```

Training performs the following steps:

1. Loads and validates the configured data.
2. Fits and freezes the SMILES and optional formula tokenizers.
3. Builds the configured encoders and decoders.
4. Tracks losses, metrics, and learning rates through MLflow.
5. Writes best and per-epoch checkpoints.
6. Writes validation predictions and final component weights.

The default configuration contains machine-specific dataset and MLflow paths. It will not be portable until `home_path`, `dataset`, and `validation_path` are updated for the local environment.

## Artifacts

Artifacts are written below the active MLflow artifact directory:

```text
artifacts/
  smiles_tokenizer.json
  formula_tokenizer.json          # when formula modeling is enabled
  checkpoints/
    best.pt
    epoch_XXX.pt
  predictions/
    encoder.pt
    decoder.pt
    reactant_encoder.pt           # when enabled
    formula_encoder.pt            # when enabled
    formula_decoder.pt            # when enabled
    fusion_encoder.pt             # when enabled
    count_head.pt                 # when enabled
    val_generated_epoch_XXX.csv
```

Checkpoints contain all enabled model components, optimizer state, epoch, and training configuration. Prediction prefers component states stored in the checkpoint and supports standalone component files as a fallback for older artifacts.

## Prediction

Run prediction with a training checkpoint and an input CSV or Parquet file:

```bash
python -m src.models.predict \
  --input /path/to/input.csv \
  --out /path/to/predictions.csv \
  --ckpt /path/to/artifacts/checkpoints/best.pt
```

If `--artifact_dir` is omitted, the predictor assumes the checkpoint is located in `<artifact_dir>/checkpoints/` and loads tokenizers from the inferred artifact directory.

Useful options:

```bash
python -m src.models.predict \
  --input data.parquet \
  --out predictions.csv \
  --ckpt mlruns/.../artifacts/checkpoints/best.pt \
  --artifact_dir mlruns/.../artifacts \
  --spectrum_col SPECTRUM_PREDICTED \
  --reaction_smiles_col reaction_smiles \
  --formula_col sum_formula \
  --batch_size 64 \
  --device cpu \
  --temperature 1.0 \
  --topk 50 \
  --num_samples 10 \
  --rerank_by_frequency
```

Prediction uses CPU automatically when the checkpoint requests CUDA but CUDA is unavailable. An explicitly requested unavailable CUDA device produces an error instead of silently changing devices.

For reaction-sequence models, provide reaction prefixes such as `CC.O>>` in the reaction SMILES column. Generation starts from the supplied reactants and reaction separator.

## Verification

Run the regression and supported-configuration smoke tests:

```bash
python -m unittest discover -s tests -v
```

Run syntax and import checks:

```bash
python -m compileall src tests
python -c "import src.models.train, src.models.predict, src.models.evaluation"
```

The tests cover tokenizer round trips and truncation, spectrum boundaries, fragment augmentation, configuration validation, padding masks, checkpoint contents, and optimization steps for representative supported model configurations.

## Current Limitations

- Training paths are configured in Python rather than through a dedicated CLI or configuration file.
- External validation data is currently loaded from CSV.
- Recursive encoder and decoder options depend on modules not included in this repository.
- Full model quality and throughput depend on the training dataset and hardware and are not covered by the unit tests.
