import pickle
import random
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.loaders import SpecSmilesDataset, parse_spectrum
from src.data.preprocess import dict_to_topk_peaks, dict_to_vector_spectra, get_ordered_bins
from src.data.tokenizers import FormulaTokenizerAdapter, SmilesTokenizerAdapter
from src.models.configs import TrainConfig
from src.models.spec2prod import BinnedEncoder, Decoder, ReactantEncoder
from src.models.train import _validate_input_paths, save_checkpoint, train_one_epoch
from src.utils.model_runtime import EncoderMemory, build_encoder_memory
from src.utils.train_runtime import build_model_components, select_collate_fn, validate_config


class TokenizerTests(unittest.TestCase):
    def test_stereochemical_smiles_round_trip(self):
        tokenizer = SmilesTokenizerAdapter()
        for smiles in ("F/C=C/F", "N[C@@H](C)C(=O)O", "[C@H](O)Cl"):
            ids = tokenizer.encode(smiles)
            self.assertEqual(tokenizer.decode(ids), smiles)

    def test_truncation_preserves_eos(self):
        tokenizer = SmilesTokenizerAdapter(max_len=5)
        ids = tokenizer.encode("CCCCCC")
        self.assertEqual(len(ids), 5)
        self.assertEqual(ids[-1], tokenizer.eos_id)

        formula_tokenizer = FormulaTokenizerAdapter(max_len=3)
        formula_ids = formula_tokenizer.encode("C6H12O6")
        self.assertEqual(formula_ids[-1], formula_tokenizer.eos_id)

    def test_reaction_truncation_requires_product_capacity(self):
        tokenizer = SmilesTokenizerAdapter(max_len=5)
        with self.assertRaisesRegex(ValueError, "product token"):
            tokenizer.encode("CCC>>C")


class DataTests(unittest.TestCase):
    def test_fragment_augmentation_keeps_product(self):
        tokenizer = SmilesTokenizerAdapter()
        tokenizer.encode("C.N.O>>CO")
        tokenizer.freeze()
        dataset = SpecSmilesDataset(
            spectra=[{1: 1.0}],
            smiles=["C.N.O>>CO"],
            tokenizer=tokenizer,
            encoder_mode="raw",
            product_only=True,
            return_reactants=True,
            frags=True,
        )

        random.seed(4)
        for _ in range(20):
            _, reactant_ids, product_ids = dataset[0]
            self.assertIn(tokenizer.decode(reactant_ids), {"C", "N", "O"})
            self.assertEqual(tokenizer.decode(product_ids), "CO")

    def test_mz_boundaries_are_consistent(self):
        vector = dict_to_vector_spectra({0: 1.0, 1: 0.5, 650: 1.0}, max_mz=650)
        bins = get_ordered_bins(vector, max_mz=650, max_bin_length=5)
        self.assertIn(650, bins)
        self.assertIn(1, bins)

        peaks = dict_to_topk_peaks({0: 1.0, 1: 0.5, 650: 1.0}, max_mz=650, max_bin_length=3)
        self.assertNotIn(0.0, peaks[:2, 0].tolist())
        self.assertEqual(set(peaks[:2, 0].tolist()), {1.0, 650.0})

    def test_parse_spectrum_supports_strings_and_dicts(self):
        expected = {43: 0.5, 91: 1.0}
        self.assertEqual(parse_spectrum("{43: 0.5, 91: 1}"), expected)
        self.assertEqual(parse_spectrum(expected), expected)
        with self.assertRaisesRegex(ValueError, "dictionary"):
            parse_spectrum([1, 2])

    def test_formula_length_is_validated(self):
        with self.assertRaisesRegex(ValueError, "formulas and spectra"):
            SpecSmilesDataset(
                spectra=[{1: 1.0}],
                smiles=["C"],
                tokenizer=SmilesTokenizerAdapter(),
                encoder_mode="raw",
                product_only=True,
                return_formula=True,
                formulas=[],
                formula_tokenizer=FormulaTokenizerAdapter(),
            )


class RuntimeTests(unittest.TestCase):
    def test_yaml_template_covers_every_training_parameter(self):
        config_path = Path(__file__).resolve().parents[1] / "configs" / "train.yaml"
        with config_path.open("r", encoding="utf-8") as stream:
            values = yaml.safe_load(stream)
        self.assertEqual(set(values), {field.name for field in fields(TrainConfig)})

        cfg = TrainConfig.from_yaml(config_path)
        self.assertTrue(Path(cfg.training_path).is_absolute())
        self.assertTrue(Path(cfg.validation_path).is_absolute())
        self.assertIsInstance(cfg.count_elements, tuple)
        self.assertIn(cfg.device, {"cpu", "cuda"})
        validate_config(cfg)

    def test_yaml_loader_rejects_missing_paths_and_unknown_fields(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "train.yaml"
            config_path.write_text("unknown_option: true\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown training configuration fields"):
                TrainConfig.from_yaml(config_path)

            config_path.write_text("epochs: 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training_path"):
                TrainConfig.from_yaml(config_path)

            config_path.write_text(
                "training_path: data/train.csv\n"
                "validation_path: null\n"
                "mlflow_tracking_uri: runs\n",
                encoding="utf-8",
            )
            cfg = TrainConfig.from_yaml(config_path)
            self.assertEqual(cfg.training_path, str(Path(tmp_dir, "data", "train.csv")))
            self.assertIsNone(cfg.validation_path)
            self.assertEqual(cfg.mlflow_tracking_uri, str(Path(tmp_dir, "runs")))

    def test_configured_input_paths_are_checked_before_training(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_path = Path(tmp_dir) / "train.csv"
            training_path.touch()
            cfg = TrainConfig(
                training_path=str(training_path),
                validation_path=str(Path(tmp_dir) / "missing.csv"),
                confidence=False,
            )
            with self.assertRaisesRegex(FileNotFoundError, "validation_path"):
                _validate_input_paths(cfg)

            cfg.validation_path = None
            _validate_input_paths(cfg)

    def test_default_config_and_collator_are_valid(self):
        cfg = TrainConfig()
        validate_config(cfg)
        self.assertIsNotNone(pickle.dumps(select_collate_fn(cfg, pad_id=0)))

    def test_invalid_reaction_configuration_fails_early(self):
        cfg = TrainConfig(decoder_mode="rxn")
        with self.assertRaisesRegex(ValueError, "spectrum input only"):
            validate_config(cfg)

    def test_binned_padding_mask_reaches_decoder_memory(self):
        encoder = BinnedEncoder(
            seq_len=4,
            d_model=4,
            nhead=1,
            num_layers=1,
            dim_feedforward=8,
            dropout=0.0,
        )
        decoder = Decoder(
            vocab_size=8,
            pad_id=0,
            rxn_sep_id=4,
            max_len=8,
            d_model=4,
            nhead=1,
            num_layers=1,
            dim_feedforward=8,
            dropout=0.0,
            decoder_mode="product",
        )
        memory = build_encoder_memory(
            encoder=encoder,
            decoder=decoder,
            spec_batch=torch.tensor([[10.0, 2.0, 0.0, 0.0]]),
        )
        self.assertIsInstance(memory, EncoderMemory)
        self.assertTrue(torch.equal(memory.padding_mask, torch.tensor([[False, False, True, True]])))
        logits = decoder(torch.tensor([[1, 5]]), memory)
        self.assertEqual(tuple(logits.shape), (1, 2, 8))

        empty_memory = build_encoder_memory(
            encoder=encoder,
            decoder=decoder,
            spec_batch=torch.zeros(1, 4),
        )
        self.assertTrue(torch.equal(empty_memory.padding_mask, torch.tensor([[False, True, True, True]])))
        empty_logits = decoder(torch.tensor([[1, 5]]), empty_memory)
        self.assertTrue(torch.isfinite(empty_logits).all())

    def test_padding_embedding_stays_zero_after_initialization(self):
        encoder = ReactantEncoder(
            vocab_size=8,
            pad_id=0,
            max_len=8,
            d_model=4,
            nhead=1,
            num_layers=1,
            dim_feedforward=8,
            dropout=0.0,
        )
        self.assertTrue(torch.equal(encoder.token_embedding.weight[0], torch.zeros(4)))

    def test_checkpoint_contains_all_optional_components(self):
        modules = [torch.nn.Linear(2, 2) for _ in range(7)]
        encoder, decoder, reactant, fusion, formula_encoder, formula_decoder, count_head = modules
        optimizer = torch.optim.AdamW(
            parameter for module in modules for parameter in module.parameters()
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "checkpoint.pt"
            save_checkpoint(
                path,
                encoder,
                decoder,
                optimizer,
                epoch=3,
                cfg=TrainConfig(),
                reactant_enc=reactant,
                fusion_enc=fusion,
                formula_enc=formula_encoder,
                formula_dec=formula_decoder,
                count_head=count_head,
            )
            payload = torch.load(path, map_location="cpu")

        self.assertTrue(
            {
                "reactant_encoder",
                "fusion_encoder",
                "formula_encoder",
                "formula_decoder",
                "count_head",
            }.issubset(payload)
        )

    def test_supported_training_modes_complete_a_step(self):
        modes = (
            {
                "reactant_encoder": True,
                "formula_encoder": False,
                "formula_decoder": True,
                "fusion_encoder": True,
                "decoder_mode": "product",
            },
            {
                "reactant_encoder": False,
                "formula_encoder": True,
                "formula_decoder": False,
                "fusion_encoder": False,
                "decoder_mode": "product",
            },
            {
                "reactant_encoder": False,
                "formula_encoder": False,
                "formula_decoder": False,
                "fusion_encoder": False,
                "decoder_mode": "rxn",
            },
        )

        for mode in modes:
            with self.subTest(mode=mode):
                cfg = TrainConfig(
                    **mode,
                    spectrum_encoder=True,
                    max_mz=8,
                    max_bin_length=4,
                    max_smiles_len=16,
                    formula_max_len=8,
                    d_model=4,
                    nhead=1,
                    enc_layers=1,
                    dec_layers=1,
                    dim_ff=8,
                    fusion_nhead=1,
                    fusion_layers=1,
                    fusion_dim_ff=8,
                    cnn=False,
                    dropout=0.0,
                    batch_size=2,
                    num_workers=0,
                    confidence=False,
                )
                validate_config(cfg)

                smiles_tokenizer = SmilesTokenizerAdapter(max_len=cfg.max_smiles_len)
                smiles = ["C.N>>CN", "CO>>CO"]
                for value in smiles:
                    smiles_tokenizer.encode(value)
                smiles_tokenizer.freeze()

                formula_tokenizer = None
                formulas = None
                if cfg.formula_encoder or cfg.formula_decoder:
                    formulas = ["CH", "CH2O"]
                    formula_tokenizer = FormulaTokenizerAdapter(max_len=cfg.formula_max_len)
                    for value in formulas:
                        formula_tokenizer.encode(value)
                    formula_tokenizer.freeze()

                dataset = SpecSmilesDataset(
                    spectra=[{1: 1.0, 8: 0.5}, {2: 1.0}],
                    smiles=smiles,
                    tokenizer=smiles_tokenizer,
                    encoder_mode=cfg.encoder_mode,
                    max_mz=cfg.max_mz,
                    max_bin_length=cfg.max_bin_length,
                    product_only=cfg.decoder_mode == "product",
                    return_reactants=cfg.reactant_encoder,
                    return_formula=cfg.formula_encoder or cfg.formula_decoder,
                    formulas=formulas,
                    formula_tokenizer=formula_tokenizer,
                )
                loader = DataLoader(
                    dataset,
                    batch_size=2,
                    collate_fn=select_collate_fn(cfg, smiles_tokenizer.pad_id),
                )
                components = build_model_components(
                    cfg,
                    torch.device("cpu"),
                    smiles_tokenizer.size(),
                    smiles_tokenizer.pad_id,
                    smiles_tokenizer.token_to_id(">>"),
                    formula_tokenizer,
                )
                encoder, decoder, reactant, fusion, formula_encoder, formula_decoder, count_head = components
                parameters = [
                    parameter
                    for module in components
                    if module is not None
                    for parameter in module.parameters()
                ]
                optimizer = torch.optim.AdamW(parameters, lr=1e-3)
                scaler = torch.amp.GradScaler("cuda", enabled=False)
                loss = train_one_epoch(
                    cfg,
                    encoder,
                    decoder,
                    loader,
                    optimizer,
                    torch.device("cpu"),
                    scaler,
                    reactant_enc=reactant,
                    fusion_enc=fusion,
                    spectrum_enc=True,
                    formula_enc=formula_encoder,
                    formula_dec=formula_decoder,
                    count_head=count_head,
                    formula_tokenizer=formula_tokenizer,
                )
                self.assertTrue(torch.isfinite(torch.tensor(loss)))


if __name__ == "__main__":
    unittest.main()
