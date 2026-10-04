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
    if architecture["activation"] not in ("gelu", "swiglu") or architecture["tie_embeddings"] is not True:
        raise ValueError("Ten przykład wymaga activation w ('gelu', 'swiglu') i tie_embeddings=true.")
    return dict(architecture)


def matrix_shapes(vocabulary_size, architecture):
    a = validate_architecture(architecture)
    d, ff = a["d_model"], a["d_ff"]
    head_size = d // a["n_heads"]
    n_kv_heads = max(1, a["n_heads"] // 2) if a["activation"] == "swiglu" else a["n_heads"]
    kv_d = n_kv_heads * head_size
    shapes = {"token_embedding": (vocabulary_size, d), "norm_final": (1, d)}
    for layer in range(a["n_layers"]):
        prefix = f"block_{layer}_"
        block_shapes = {
            "Wq": (d, d), "Wk": (d, kv_d), "Wv": (d, kv_d), "Wo": (d, d),
            "W1": (d, ff), "W2": (ff, d),
            "norm_attention": (1, d), "norm_ff": (1, d),
        }
        if a["activation"] == "swiglu":
            block_shapes["W_gate"] = (d, ff)
        shapes.update({prefix + name: shape for name, shape in block_shapes.items()})
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
        self.activation = self.architecture["activation"]
        self.n_kv_heads = max(1, self.n_heads // 2) if self.activation == "swiglu" else self.n_heads
        self.dropout = nn.Dropout(self.architecture["dropout"])
        shapes = matrix_shapes(len(self.tokenizer.vocabulary), self.architecture)
        if not isinstance(weights, dict) or set(weights) != set(shapes):
            raise ValueError("Brak wymaganych macierzy wag lub nieobsługiwane macierze.")
        checked = {}
        for name, (rows, columns) in shapes.items():
            matrix = weights[name]
            if isinstance(matrix, torch.Tensor):
                if matrix.ndim != 2 or tuple(matrix.shape) != (rows, columns) or matrix.dtype == torch.bool:
                    raise ValueError(f"Macierz {name} wymaga kształtu {(rows, columns)}.")
                tensor = matrix.detach().to(dtype=torch.float32, device="cpu").clone()
            else:
                if not isinstance(matrix, list) or len(matrix) != rows:
                    raise ValueError(f"Macierz {name} wymaga {rows} wierszy.")
                if any(not isinstance(row, list) or len(row) != columns for row in matrix):
                    raise ValueError(f"Macierz {name} wymaga {columns} kolumn.")
                try:
                    tensor = torch.tensor(matrix, dtype=torch.float32, device="cpu")
                except (TypeError, ValueError, RuntimeError) as error:
                    raise ValueError(f"Macierz {name} musi zawierać skończone liczby.") from error
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"Macierz {name} musi zawierać skończone liczby w float32.")
            checked[name] = nn.Parameter(tensor)
        self.weights = nn.ParameterDict(checked)
        head_size = self.d_model // self.n_heads
        half = head_size // 2
        if half > 0:
            positions = torch.arange(self.context_size, dtype=torch.float32).unsqueeze(1)
            frequencies = 10000.0 ** (-torch.arange(half, dtype=torch.float32).unsqueeze(0) / half)
            angles = positions * frequencies
            self.register_buffer("rope_cos", torch.cat((angles.cos(), angles.cos()), dim=-1), persistent=False)
            self.register_buffer("rope_sin", torch.cat((angles.sin(), angles.sin()), dim=-1), persistent=False)
        else:
            self.rope_cos = self.rope_sin = None

    def norm(self, x, gain):
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + RMS_EPSILON) * gain

    def apply_rope(self, x, offset=0):
        if self.rope_cos is None:
            return x
        length = x.shape[-2]
        half = x.shape[-1] // 2
        cos = self.rope_cos[offset:offset + length].to(dtype=x.dtype, device=x.device)
        sin = self.rope_sin[offset:offset + length].to(dtype=x.dtype, device=x.device)
        rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return x * cos + rotated * sin

    def _repeat_kv(self, x):
        if self.n_kv_heads == self.n_heads:
            return x
        return x.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

    def _ffn(self, x, prefix):
        w = self.weights
        normed = self.norm(x, w[prefix + "norm_ff"])
        if self.activation == "swiglu":
            hidden = F.silu(normed @ w[prefix + "W_gate"]) * (normed @ w[prefix + "W1"])
        else:
            hidden = F.gelu(normed @ w[prefix + "W1"])
        return x + self.dropout(hidden @ w[prefix + "W2"])

    def _validate_ids(self, ids):
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] < 1 or not 1 <= ids.shape[1] <= self.context_size:
            raise ValueError(f"Wejście musi mieć kształt [niepusty batch, 1..{self.context_size} tokenów].")
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("ID tokenów muszą być liczbami całkowitymi.")
        model_device = self.weights["token_embedding"].device.type
        if ids.device.type != model_device:
            raise ValueError(f"ID tokenów muszą być na urządzeniu modelu ({model_device}).")
        if (ids < 0).any().item() or (ids >= len(self.tokenizer.vocabulary)).any().item():
            raise ValueError("Nieprawidłowe ID tokenu.")

    def forward_with_attention(self, ids):
        self._validate_ids(ids)
        w = self.weights
        batch, length = ids.shape
        head_size = self.d_model // self.n_heads
        x = self.dropout(w["token_embedding"][ids])
        future = torch.ones((length, length), dtype=torch.bool, device=ids.device).triu(1)
        maps = []
        for layer in range(self.n_layers):
            prefix = f"block_{layer}_"
            normalized = self.norm(x, w[prefix + "norm_attention"])
            q = (normalized @ w[prefix + "Wq"]).view(batch, length, self.n_heads, head_size).transpose(1, 2)
            k = (normalized @ w[prefix + "Wk"]).view(batch, length, self.n_kv_heads, head_size).transpose(1, 2)
            v = (normalized @ w[prefix + "Wv"]).view(batch, length, self.n_kv_heads, head_size).transpose(1, 2)
            q, k = self.apply_rope(q), self.apply_rope(k)
            k, v = self._repeat_kv(k), self._repeat_kv(v)
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_size)
            attention = torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)
            joined = (self.dropout(attention) @ v).transpose(1, 2).contiguous().view(batch, length, self.d_model)
            x = x + self.dropout(joined @ w[prefix + "Wo"])
            x = self._ffn(x, prefix)
            maps.append(attention)
        logits = self.norm(x, w["norm_final"]) @ w["token_embedding"].T
        return logits, maps

    def forward(self, ids):
        self._validate_ids(ids)
        w = self.weights
        batch, length = ids.shape
        head_size = self.d_model // self.n_heads
        x = self.dropout(w["token_embedding"][ids])
        dropout_p = self.dropout.p if self.training else 0.0
        for layer in range(self.n_layers):
            prefix = f"block_{layer}_"
            normalized = self.norm(x, w[prefix + "norm_attention"])
            q = (normalized @ w[prefix + "Wq"]).view(batch, length, self.n_heads, head_size).transpose(1, 2)
            k = (normalized @ w[prefix + "Wk"]).view(batch, length, self.n_kv_heads, head_size).transpose(1, 2)
            v = (normalized @ w[prefix + "Wv"]).view(batch, length, self.n_kv_heads, head_size).transpose(1, 2)
            q, k = self.apply_rope(q), self.apply_rope(k)
            k, v = self._repeat_kv(k), self._repeat_kv(v)
            attended = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)
            joined = attended.transpose(1, 2).contiguous().view(batch, length, self.d_model)
            x = x + self.dropout(joined @ w[prefix + "Wo"])
            x = self._ffn(x, prefix)
        return self.norm(x, w["norm_final"]) @ w["token_embedding"].T

    @torch.no_grad()
    def forward_cached(self, ids, kv_cache=None):
        """Szybki krok generowania z KV-Cache: przelicza tylko nowe tokeny, gdy mieścimy się w oknie."""
        self._validate_ids(ids)
        w = self.weights
        batch, length = ids.shape
        head_size = self.d_model // self.n_heads
        cached_len = 0 if kv_cache is None else kv_cache[0][0].shape[2]
        if kv_cache is not None and cached_len + length > self.context_size:
            raise ValueError("KV-Cache przekracza rozmiar okna kontekstu; odśwież bufor.")
        x = w["token_embedding"][ids]
        new_cache = []
        for layer in range(self.n_layers):
            prefix = f"block_{layer}_"
            normalized = self.norm(x, w[prefix + "norm_attention"])
            q = (normalized @ w[prefix + "Wq"]).view(batch, length, self.n_heads, head_size).transpose(1, 2)
            k = (normalized @ w[prefix + "Wk"]).view(batch, length, self.n_kv_heads, head_size).transpose(1, 2)
            v = (normalized @ w[prefix + "Wv"]).view(batch, length, self.n_kv_heads, head_size).transpose(1, 2)
            q = self.apply_rope(q, offset=cached_len)
            k = self.apply_rope(k, offset=cached_len)
            if kv_cache is not None:
                k = torch.cat((kv_cache[layer][0], k), dim=2)
                v = torch.cat((kv_cache[layer][1], v), dim=2)
            new_cache.append((k, v))
            k_rep, v_rep = self._repeat_kv(k), self._repeat_kv(v)
            attended = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=(length > 1), dropout_p=0.0)
            joined = attended.transpose(1, 2).contiguous().view(batch, length, self.d_model)
            x = x + (joined @ w[prefix + "Wo"])
            x = self._ffn(x, prefix)
        logits = self.norm(x, w["norm_final"]) @ w["token_embedding"].T
        return logits, new_cache


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
        weights[name] = tensor
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
    # Szybka walidacja tensorów przed zapisem chroni ostatni poprawny checkpoint.
    GPT(model.tokenizer, model.architecture, dict(model.weights))
    weights = {name: matrix.detach().cpu().tolist() for name, matrix in model.weights.items()}
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
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
