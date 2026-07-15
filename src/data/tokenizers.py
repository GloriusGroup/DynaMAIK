import json
import re
from pathlib import Path
from typing import Dict, List


SPECIALS = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}


def _initial_vocab() -> tuple[Dict[str, int], Dict[int, str]]:
    stoi = dict(SPECIALS)
    itos = {i: token for token, i in stoi.items()}
    return stoi, itos


def _strip_whitespace(value: str) -> str:
    return re.sub(r"\s+", "", value)


def _restore_vocab_state(tokenizer, obj: dict) -> None:
    tokenizer.stoi = {str(k): int(v) for k, v in obj["stoi"].items()}
    tokenizer.itos = {int(v): str(k) for k, v in tokenizer.stoi.items()}
    tokenizer._next_id = max(tokenizer.stoi.values()) + 1
    tokenizer.pad_id = tokenizer.stoi["<pad>"]
    tokenizer.bos_id = tokenizer.stoi["<bos>"]
    tokenizer.eos_id = tokenizer.stoi["<eos>"]
    tokenizer.unk_id = tokenizer.stoi["<unk>"]
    tokenizer.frozen = bool(obj.get("frozen", True))


def _save_tokenizer_payload(tokenizer, path: str | Path, tokenizer_type: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": tokenizer_type,
        "max_len": tokenizer.max_len,
        "stoi": tokenizer.stoi,
        "specials": SPECIALS,
        "frozen": tokenizer.frozen,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def _truncate_ids(ids: List[int], max_len: int, eos_id: int, add_eos: bool) -> List[int]:
    if max_len < 1:
        raise ValueError("max_len must be at least 1")
    if len(ids) <= max_len:
        return ids

    truncated = ids[:max_len]
    if add_eos:
        truncated[-1] = eos_id
    return truncated

SMI_REGEX = re.compile(r"""
    (
      \[[^\]]+\]                       # bracket atoms, including stereochemistry
    | Br | Cl | Si                       # two-letter halogens (order matters)
    | C | N | O | S | P | F | I | B | H   # single-letter atoms
    | b | c | n | o | s | p          # aromatic atoms
    | \(| \)                         # branches
    | \. | = | \# | - | / | \\      # bonds/symbols, including stereo bonds
    | \+                              # charge +
    | : | \* | \$                    # (optional non-stereo symbols)
    | >>                              # reaction separator ONLY
    | % [0-9]{2}                     # ring numbers like %10
    | [0-9]                          # ring digits 0-9
    )
""", re.VERBOSE)


class SmilesTokenizerAdapter:
    def __init__(self, max_len: int = 256):
        self.max_len = max_len
        self.stoi, self.itos = _initial_vocab()
        self._next_id = max(self.stoi.values()) + 1
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.unk_id = self.stoi["<unk>"]
        self.frozen = False

    def freeze(self):
        self.frozen = True

    def size(self) -> int:
        return len(self.stoi)

    def _add_token(self, tok: str) -> int:
        if tok not in self.stoi:
            if self.frozen:
                return self.unk_id
            self.stoi[tok] = self._next_id
            self.itos[self._next_id] = tok
            self._next_id += 1
        return self.stoi[tok]

    def tokenize(self, smiles: str) -> List[str]:
        clean = _strip_whitespace(smiles)
        tokens = SMI_REGEX.findall(clean)
        if "".join(tokens) != clean:
            raise ValueError(f"Unrecognized pattern in SMILES: {smiles} ...")
        return tokens

    def token_to_id(self, token: str):
        return int(self.stoi[token])

    def encode(self, smiles: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        toks = self.tokenize(smiles)
        ids = [self._add_token(t) for t in toks]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]

        rxn_id = self.stoi.get(">>")
        if len(ids) > self.max_len and rxn_id is not None and rxn_id in ids:
            arrow_pos = ids.index(rxn_id)
            last_content_pos = self.max_len - int(add_eos) - 1
            if arrow_pos >= last_content_pos:
                raise ValueError(
                    f"Reaction prefix and at least one product token do not fit "
                    f"(len={len(ids)}, max_len={self.max_len}) for: {smiles[:120]} ...")

        return _truncate_ids(ids, self.max_len, self.eos_id, add_eos)

    def decode(self, ids: List[int], skip_specials: bool = True) -> str:
        toks: List[str] = []
        for i in ids:
            tok = self.itos.get(int(i), "<unk>")
            if skip_specials and tok in SPECIALS:
                continue
            toks.append(tok)
        return "".join(toks)

    def save(self, path: str | Path) -> None:
        _save_tokenizer_payload(self, path, "smiles")

    @classmethod
    def load(cls, path: str | Path) -> "SmilesTokenizerAdapter":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if obj.get("type") not in (None, "smiles"):
            raise ValueError(f"Tokenizer file type mismatch: {obj.get('type')}")

        tok = cls(max_len=int(obj["max_len"]))
        _restore_vocab_state(tok, obj)
        return tok


# RDKit-style sum formula tokens:
#   - Element symbols: C, H, He, Cl, Fe, ...
#   - Integer counts: 1, 2, 12, ...
FORMULA_REGEX = re.compile(r"""
    (
        [A-Z][a-z]?    # element symbol
      | \d+            # atom count
    )
""", re.VERBOSE)

# Allowed characters after whitespace removal
_ALLOWED_CHARS = re.compile(r"^[A-Za-z0-9]+$")


class FormulaTokenizerAdapter:
    """
    Tokenizer for RDKit-generated sum formulas, e.g.:
        C6H12O6
        C2H6O
        CH4
        Cl2
        C12H22O11
    """

    def __init__(self, max_len: int = 64):
        self.max_len = max_len
        self.stoi, self.itos = _initial_vocab()
        self._next_id = max(self.stoi.values()) + 1
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.unk_id = self.stoi["<unk>"]
        self.frozen = False

    def freeze(self):
        self.frozen = True

    def size(self) -> int:
        return len(self.stoi)

    def _add_token(self, tok: str) -> int:
        if tok not in self.stoi:
            if self.frozen:
                return self.unk_id
            self.stoi[tok] = self._next_id
            self.itos[self._next_id] = tok
            self._next_id += 1
        return self.stoi[tok]

    def tokenize(self, formula: str) -> List[str]:
        clean = _strip_whitespace(formula)

        if not _ALLOWED_CHARS.match(clean):
            raise ValueError(f"Invalid character in sum formula: {formula!r}")

        tokens = FORMULA_REGEX.findall(clean)
        if "".join(tokens) != clean:
            raise ValueError(f"Unrecognized pattern in sum formula: {formula!r}")

        return tokens

    def token_to_id(self, token: str) -> int:
        return int(self.stoi[token])

    def encode(self, formula: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        toks = self.tokenize(formula)
        ids = [self._add_token(t) for t in toks]

        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]

        return _truncate_ids(ids, self.max_len, self.eos_id, add_eos)

    def decode(self, ids: List[int], skip_specials: bool = True) -> str:
        toks: List[str] = []
        for i in ids:
            tok = self.itos.get(int(i), "<unk>")
            if skip_specials and tok in SPECIALS:
                continue
            toks.append(tok)
        return "".join(toks)

    def save(self, path: str | Path) -> None:
        _save_tokenizer_payload(self, path, "formula")

    @classmethod
    def load(cls, path: str | Path) -> "FormulaTokenizerAdapter":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if obj.get("type") not in (None, "formula"):
            raise ValueError(f"Tokenizer file type mismatch: {obj.get('type')}")

        tok = cls(max_len=int(obj["max_len"]))
        _restore_vocab_state(tok, obj)
        return tok
