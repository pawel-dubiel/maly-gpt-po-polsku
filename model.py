"""Dekoder GPT: jawne Q/K/V, wiele głów i bloków, RMSNorm i feed-forward."""

import json
import math
import os
from pathlib import Path
import tempfile

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError as error:
    raise SystemExit("Brak PyTorch. Zainstaluj requirements-full.txt w .venv.") from error

from bpe_tokenizer import BPETokenizer


MODEL_FORMAT = "tiny-gpt-bpe-v2"
RMS_EPSILON = 1e-5
INITIAL_WEIGHT_STD = 0.02


def validate_architecture(architecture):
    fields = {"context_size", "d_model", "d_ff", "n_heads", "n_layers",
              "dropout", "rms_epsilon", "activation", "tie_embeddings"}
    if not isinstance(architecture, dict) or set(architecture) != fields:
        raise ValueError("Architektura wymaga wszystkich pól: " + ", ".join(sorted(fields)))
    for name in ("context_size", "d_model", "d_ff", "n_heads", "n_layers"):
        if type(architecture[name]) is not int or architecture[name] < 1:
            raise ValueError(f"{name} musi być dodatnią liczbą całkowitą.")
    if architecture["d_model"] % architecture["n_heads"]:
        raise ValueError("d_model musi być podzielne przez n_heads.")
    probability = architecture["dropout"]
    if type(probability) not in (int, float) or not math.isfinite(probability) or not 0 <= probability < 1:
        raise ValueError("dropout musi być skończoną liczbą od 0 włącznie do 1 wyłącznie.")
    if architecture["rms_epsilon"] != RMS_EPSILON:
        raise ValueError(f"rms_epsilon musi wynosić {RMS_EPSILON}.")
    if architecture["activation"] != "gelu" or architecture["tie_embeddings"] is not True:
        raise ValueError("Ten przykład wymaga activation='gelu' i tie_embeddings=true.")
    return dict(architecture)


def matrix_shapes(vocabulary_size, architecture):
    a = validate_architecture(architecture)
    d, ff = a["d_model"], a["d_ff"]
    shapes = {"token_embedding": (vocabulary_size, d),
              "position_embedding": (a["context_size"], d), "norm_final": (1, d)}
    for layer in range(a["n_layers"]):
        prefix = f"block_{layer}_"
        shapes.update({prefix + name: shape for name, shape in {
            "Wq": (d, d), "Wk": (d, d), "Wv": (d, d), "Wo": (d, d),
            "W1": (d, ff), "W2": (ff, d),
            "norm_attention": (1, d), "norm_ff": (1, d),
        }.items()})
    return shapes


class GPT(nn.Module):
    def __init__(self, tokenizer, architecture, weights):
        super().__init__()
        self.architecture = validate_architecture(architecture)
        if not isinstance(tokenizer, BPETokenizer):
            raise ValueError("Model wymaga tokenizera BPE.")
        self.tokenizer = BPETokenizer.from_dict(tokenizer.to_dict())
        self.context_size = self.architecture["context_size"]
        self.d_model, self.d_ff = self.architecture["d_model"], self.architecture["d_ff"]
        self.n_heads, self.n_layers = self.architecture["n_heads"], self.architecture["n_layers"]
        self.dropout = nn.Dropout(self.architecture["dropout"])
        shapes = matrix_shapes(len(self.tokenizer.vocabulary), self.architecture)
        if not isinstance(weights, dict) or set(weights) != set(shapes):
            raise ValueError("Brak wymaganych macierzy wag lub nieobsługiwane macierze.")
        checked = {}
        for name, (rows, columns) in shapes.items():
            matrix = weights[name]
            if not isinstance(matrix, list) or len(matrix) != rows:
                raise ValueError(f"Macierz {name} wymaga {rows} wierszy.")
            if any(not isinstance(row, list) or len(row) != columns for row in matrix):
                raise ValueError(f"Macierz {name} wymaga {columns} kolumn.")
            if any(type(value) not in (int, float) or not math.isfinite(value) for row in matrix for value in row):
                raise ValueError(f"Macierz {name} musi zawierać skończone liczby.")
            tensor = torch.tensor(matrix, dtype=torch.float32, device="cpu")
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"Wagi {name} nie mieszczą się w float32.")
            checked[name] = nn.Parameter(tensor)
        self.weights = nn.ParameterDict(checked)

    def norm(self, x, gain):
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + RMS_EPSILON) * gain

    def forward_with_attention(self, ids):
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] < 1 or not 1 <= ids.shape[1] <= self.context_size:
            raise ValueError(f"Wejście musi mieć kształt [niepusty batch, 1..{self.context_size} tokenów].")
        if ids.device.type != "cpu" or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("ID tokenów muszą być liczbami całkowitymi na CPU.")
        if (ids < 0).any().item() or (ids >= len(self.tokenizer.vocabulary)).any().item():
            raise ValueError("Nieprawidłowe ID tokenu.")
        w = self.weights
        batch, length = ids.shape
        head_size = self.d_model // self.n_heads
        x = self.dropout(w["token_embedding"][ids] + w["position_embedding"][:length])
        future = torch.ones((length, length), dtype=torch.bool, device="cpu").triu(1)
        maps = []
        for layer in range(self.n_layers):
            prefix = f"block_{layer}_"
            normalized = self.norm(x, w[prefix + "norm_attention"])
            # Q = X Wq, K = X Wk, V = X Wv; każda głowa otrzymuje d_model/n_heads liczb.
            q, k, v = [(normalized @ w[prefix + name]).view(
                batch, length, self.n_heads, head_size).transpose(1, 2)
                for name in ("Wq", "Wk", "Wv")]
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_size)
            attention = torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)
            joined = (attention @ v).transpose(1, 2).contiguous().view(batch, length, self.d_model)
            x = x + self.dropout(joined @ w[prefix + "Wo"])
            hidden = F.gelu(self.norm(x, w[prefix + "norm_ff"]) @ w[prefix + "W1"])
            x = x + self.dropout(hidden @ w[prefix + "W2"])
            maps.append(attention)
        # Wspólna macierz embeddingów dla wejścia i klasyfikatora wyjścia.
        logits = self.norm(x, w["norm_final"]) @ w["token_embedding"].T
        return logits, maps

    def forward(self, ids):
        return self.forward_with_attention(ids)[0]


def new_model(tokenizer, architecture, seed):
    a = validate_architecture(architecture)
    if type(seed) is not int:
        raise ValueError("Seed musi być liczbą całkowitą.")
    if not isinstance(tokenizer, BPETokenizer):
        raise ValueError("Model wymaga tokenizera BPE.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weights = {}
    for name, shape in matrix_shapes(len(tokenizer.vocabulary), a).items():
        if "norm_" in name:
            tensor = torch.ones(shape, dtype=torch.float32)
        else:
            scale = INITIAL_WEIGHT_STD
            if name.endswith(("_Wo", "_W2")):
                scale /= math.sqrt(2 * a["n_layers"])
            tensor = torch.randn(shape, generator=generator, dtype=torch.float32) * scale
        weights[name] = tensor.tolist()
    return GPT(tokenizer, a, weights)


def load_model(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = {"format", "architecture", "tokenizer", "vocabulary", "weights"}
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("Niepełne lub nieobsługiwane pola checkpointu.")
    if payload["format"] != MODEL_FORMAT:
        raise ValueError(f"Wymagany format {MODEL_FORMAT}; starszy model wymaga nowego treningu.")
    tokenizer = BPETokenizer.from_dict(payload["tokenizer"])
    if payload["vocabulary"] != list(tokenizer.vocabulary):
        raise ValueError("Słownik modelu musi dokładnie odpowiadać tokenizerowi.")
    return GPT(tokenizer, payload["architecture"], payload["weights"])


def save_model(path, model):
    path = Path(path)
    if not path.parent.is_dir() or path.is_dir():
        raise ValueError(f"Checkpoint wymaga pliku w istniejącym katalogu: {path}")
    # Walidacja przed zapisem chroni ostatni poprawny checkpoint.
    weights = {name: matrix.detach().tolist() for name, matrix in model.weights.items()}
    GPT(model.tokenizer, model.architecture, weights)
    payload = {
        "format": MODEL_FORMAT,
        "architecture": dict(model.architecture),
        "tokenizer": model.tokenizer.to_dict(), "vocabulary": list(model.tokenizer.vocabulary),
        "weights": weights,
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
