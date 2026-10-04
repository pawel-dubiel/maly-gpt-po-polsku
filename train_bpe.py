"""Trening transformera BPE z jawną konfiguracją i wyborem najlepszej walidacji."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from model import load_model, new_model, save_model, torch, validate_architecture
from torch.nn import functional as F
from bpe_tokenizer import BPETokenizer


def make_windows(ids, context_size):
    if type(context_size) is not int or context_size < 1:
        raise ValueError("Rozmiar kontekstu musi być dodatnią liczbą całkowitą.")
    if any(type(token) is not int or token < 0 for token in ids) or any(token == 0 for token in ids[:-1]):
        raise ValueError("Strumień treningowy musi zawierać ID tokenów bez znacznika <end> wewnątrz tekstu.")
    if len(ids) <= context_size:
        raise ValueError(f"Tekst wymaga co najmniej {context_size + 1} tokenów BPE.")
    stream = torch.tensor(ids, dtype=torch.long, device="cpu")
    windows = stream.unfold(0, context_size + 1, 1)
    return windows[:, :-1], windows[:, 1:]


def split_windows(ids, context_size, validation_fraction):
    if type(context_size) is not int or context_size < 1:
        raise ValueError("Rozmiar kontekstu musi być dodatnią liczbą całkowitą.")
    if (type(validation_fraction) not in (int, float)
            or not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1):
        raise ValueError("Udział walidacji musi być skończoną liczbą większą od 0 i mniejszą od 1.")
    boundary = int(len(ids) * (1 - validation_fraction))
    if min(boundary, len(ids) - boundary) <= context_size:
        raise ValueError(f"Podział wymaga co najmniej {context_size + 1} tokenów w każdej części; "
                         f"trening ma {boundary}, walidacja {len(ids) - boundary}.")
    # Podział przed tworzeniem okien zapobiega przeciekowi tokenów przez granicę zbiorów.
    train_inputs, train_targets = make_windows(ids[:boundary], context_size)
    validation_inputs, validation_targets = make_windows(ids[boundary:], context_size)
    return train_inputs, train_targets, validation_inputs, validation_targets


def batch_loss(model, inputs, targets):
    logits = model(inputs)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


@torch.no_grad()
def mean_loss(model, inputs, targets, batch_size):
    if len(inputs) == 0 or len(inputs) != len(targets):
        raise ValueError("Ocena wymaga niepustych wejść i odpowiadających im celów.")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Batch size musi być dodatnią liczbą całkowitą.")
    was_training = model.training
    model.eval()
    device = model.weights["token_embedding"].device
    try:
        total = 0.0
        for start in range(0, len(inputs), batch_size):
            x = inputs[start:start + batch_size].to(device)
            y = targets[start:start + batch_size].to(device)
            loss = batch_loss(model, x, y)
            if not torch.isfinite(loss).item():
                raise ValueError("Loss jest nieskończony lub NaN; zmniejsz learning rate.")
            total += loss.item() * len(x)
        return total / len(inputs)
    finally:
        model.train(was_training)



def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    fields = {"data", "tokenizer", "model", "report", "architecture", "training"}
    if not isinstance(config, dict) or set(config) != fields:
        raise ValueError("Konfiguracja wymaga dokładnie pól: " + ", ".join(sorted(fields)))
    validate_architecture(config["architecture"])
    t = config["training"]
    fields = {"seed", "steps", "batch_size", "learning_rate", "min_learning_rate",
              "warmup_steps", "weight_decay", "beta1", "beta2", "adam_epsilon",
              "gradient_clip", "evaluate_every", "train_eval_windows",
              "validation_fraction", "threads"}
    if not isinstance(t, dict) or set(t) != fields:
        raise ValueError("training wymaga dokładnie pól: " + ", ".join(sorted(fields)))
    for name in ("steps", "batch_size", "warmup_steps", "evaluate_every", "train_eval_windows", "threads"):
        if type(t[name]) is not int or t[name] < 1:
            raise ValueError(f"{name} musi być dodatnią liczbą całkowitą.")
    if type(t["seed"]) is not int or not 0 <= t["seed"] < 2**63:
        raise ValueError("seed musi być liczbą całkowitą od 0 do 2**63-1.")
    for name in ("learning_rate", "min_learning_rate", "adam_epsilon", "gradient_clip"):
        if type(t[name]) not in (int, float) or not math.isfinite(t[name]) or t[name] <= 0:
            raise ValueError(f"{name} musi być dodatnią, skończoną liczbą.")
    for name in ("beta1", "beta2", "validation_fraction"):
        if type(t[name]) not in (int, float) or not math.isfinite(t[name]) or not 0 < t[name] < 1:
            raise ValueError(f"{name} musi być skończoną liczbą między 0 a 1.")
    if (type(t["weight_decay"]) not in (int, float) or not math.isfinite(t["weight_decay"])
            or t["weight_decay"] < 0):
        raise ValueError("weight_decay musi być nieujemną, skończoną liczbą.")
    if t["min_learning_rate"] > t["learning_rate"] or t["warmup_steps"] >= t["steps"]:
        raise ValueError("Wymagane min_learning_rate <= learning_rate i warmup_steps < steps.")
    if t["evaluate_every"] > t["steps"]:
        raise ValueError("evaluate_every nie może przekraczać steps.")
    paths = {}
    for name in ("data", "tokenizer", "model", "report"):
        if not isinstance(config[name], str) or not config[name].strip():
            raise ValueError(f"{name} wymaga niepustej ścieżki.")
        paths[name] = (path.parent / config[name]).resolve()
    if len(set(paths.values()) | {path}) != 5:
        raise ValueError("Konfiguracja, dane, tokenizer, model i raport muszą być różnymi plikami.")
    for name in ("data", "tokenizer"):
        if not paths[name].is_file():
            raise ValueError(f"Brak pliku {name}: {paths[name]}")
    for name in ("model", "report"):
        if not paths[name].parent.is_dir() or paths[name].is_dir():
            raise ValueError(f"{name} wymaga pliku w istniejącym katalogu: {paths[name]}")
    return config, paths


def learning_rate(step, settings):
    if step <= settings["warmup_steps"]:
        return settings["learning_rate"] * step / settings["warmup_steps"]
    progress = (step - settings["warmup_steps"]) / (settings["steps"] - settings["warmup_steps"])
    return settings["min_learning_rate"] + (settings["learning_rate"] - settings["min_learning_rate"]) * (
        1 + math.cos(math.pi * progress)) / 2


def select_training_device(model):
    if model.context_size >= 128 or model.d_model >= 192:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def train(model, inputs, targets, validation_inputs, validation_targets, settings, model_path):
    device = select_training_device(model)
    model.to(device)
    # Normy i embeddingi nie podlegają weight decay; grupujemy po nazwie, nie po ndim.
    no_decay_names = ("norm_", "embedding")
    decay = [p for name, p in model.weights.items() if not any(k in name for k in no_decay_names)]
    no_decay = [p for name, p in model.weights.items() if any(k in name for k in no_decay_names)]
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": settings["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=settings["learning_rate"], betas=(settings["beta1"], settings["beta2"]),
        eps=settings["adam_epsilon"])
    torch.manual_seed(settings["seed"])  # Dropout; oddzielny generator losuje okna.
    rng = torch.Generator(device="cpu").manual_seed(settings["seed"])
    count = min(len(inputs), settings["train_eval_windows"])
    chosen = torch.linspace(0, len(inputs) - 1, steps=count).long()
    eval_x, eval_y = inputs[chosen], targets[chosen]
    large_vocab = len(model.tokenizer.vocabulary) >= 2048
    max_val_windows = 512 if large_vocab else 2048
    max_train_windows = 1024 if large_vocab else 4096
    val_stride = max(1, model.context_size // 2, len(validation_inputs) // max_val_windows)
    train_stride = max(1, model.context_size // 2, len(inputs) // max_train_windows)
    val_x, val_y = validation_inputs[::val_stride], validation_targets[::val_stride]
    full_train_x, full_train_y = inputs[::train_stride], targets[::train_stride]
    batch = settings["batch_size"]
    eval_batch = max(batch, 32 if large_vocab else (128 if model.context_size >= 128 else 256))
    history = []
    started = time.perf_counter()
    initial = {"train_loss_sample": mean_loss(model, eval_x, eval_y, eval_batch),
               "validation_loss": mean_loss(model, val_x, val_y, eval_batch)}
    print(f"Initial ({device.type}): {json.dumps(initial)}", flush=True)
    best_loss, best_step = math.inf, None
    for step in range(1, settings["steps"] + 1):
        model.train()
        rate = learning_rate(step, settings)
        for group in optimizer.param_groups:
            group["lr"] = rate
        indices = torch.randint(len(inputs), (batch,), generator=rng)
        optimizer.zero_grad(set_to_none=True)
        loss = batch_loss(model, inputs[indices].to(device), targets[indices].to(device))
        if not torch.isfinite(loss).item():
            raise ValueError("Loss jest nieskończony lub NaN; zmniejsz learning rate.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % settings["evaluate_every"] == 0 or step == settings["steps"]:
            record = {"step": step, "learning_rate": rate,
                      "train_loss_sample": mean_loss(model, eval_x, eval_y, eval_batch),
                      "validation_loss": mean_loss(model, val_x, val_y, eval_batch)}
            if record["validation_loss"] < best_loss:
                save_model(model_path, model)
                best_loss, best_step = record["validation_loss"], step
                record["saved"] = True
            else:
                record["saved"] = False
            record["seconds"] = time.perf_counter() - started
            history.append(record)
            print(json.dumps(record), flush=True)
    # Końcowa ocena dotyczy faktycznie zapisanego modelu, a nie ostatnich wag w pamięci.
    best = load_model(model_path).to(device)
    full_train = mean_loss(best, full_train_x, full_train_y, eval_batch)
    full_validation = mean_loss(best, val_x, val_y, eval_batch)
    model.to("cpu")
    print(f"Best checkpoint: step {best_step}; training loss = {full_train:.6f}; "
          f"validation loss = {full_validation:.6f}; {model_path}", flush=True)
    return {"initial": initial, "history": history, "best_step": best_step,
            "full_training_loss": full_train, "full_validation_loss": full_validation,
            "elapsed_seconds": time.perf_counter() - started}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        config, paths = load_config(args.config)
        settings = config["training"]
        torch.set_num_threads(settings["threads"])
        tokenizer = BPETokenizer.load(paths["tokenizer"])
        text = paths["data"].read_text(encoding="utf-8")
        ids = tokenizer.encode(text) + [0]
        windows = split_windows(ids, config["architecture"]["context_size"], settings["validation_fraction"])
        model = new_model(tokenizer, config["architecture"], settings["seed"])
        parameters = sum(p.numel() for p in model.parameters())
        print(f"Transformer: {json.dumps(model.architecture)}; parameters={parameters}", flush=True)
        print(f"Tokens: {len(ids)}; training windows: {len(windows[0])}; validation windows: {len(windows[2])}", flush=True)
        report = {"config": config, "torch_version": str(torch.__version__),
                  "data_sha256": sha256(paths["data"]), "tokenizer_sha256": sha256(paths["tokenizer"]),
                  "parameters": parameters, "tokens": len(ids),
                  "training_tokens": len(windows[0]) + model.context_size,
                  "validation_tokens": len(windows[2]) + model.context_size}
        report.update(train(model, *windows, settings, paths["model"]))
        report["model_sha256"] = sha256(paths["model"])
        paths["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Report saved: {paths['report']}", flush=True)
    except (OSError, ValueError, OverflowError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
