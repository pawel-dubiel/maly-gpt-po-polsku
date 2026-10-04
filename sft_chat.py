"""Supervised Fine-Tuning (SFT) z maskowaniem straty na pytaniu i trybem rozmowy Q&A."""

import argparse
import json
from pathlib import Path
import time

from model import load_model, save_model, torch
from predict_gpt import generate
from text_data import END, positive_float, positive_integer
from torch.nn import functional as F


IGNORE_INDEX = -100

# Zbiór par instrukcyjnych (Q&A oraz polecenia poetyckie) dla etapu SFT.
SFT_PAIRS = [
    ("Kto napisał Pana Tadeusza?", "Autorem epopei narodowej «Pan Tadeusz» jest Adam Mickiewicz."),
    ("Kim jest Jacek Soplica?", "Jacek Soplica to ojciec Tadeusza, który jako Ksiądz Robak przygotowuje powstanie na Litwie."),
    ("Kim jest Tadeusz Soplica?", "Tadeusz to młody dziedzic z rodu Sopliców, bratanek Sędziego, zakochany w Zosi."),
    ("Gdzie toczy się akcja Pana Tadeusza?", "Akcja toczy się na Litwie, w dworze w Soplicowie i zamku Horeszków w 1811 i 1812 roku."),
    ("Jak zaczyna się inwokacja?", "Litwo! Ojczyzno moja! ty jesteś jak zdrowie;\nIle cię trzeba cenić, ten tylko się dowie,\nKto cię stracił. Dziś piękność twą w całej ozdobie\nWidzę i opisuję, bo tęsknię po tobie."),
    ("Kim jest Zosia?", "Zosia to młoda córka Ewy Horeszkówny, wychowanka Telimeny i narzeczona Tadeusza."),
    ("O co spierają się Asesor i Rejent?", "Toczą spór o to, który z ich chartów — Kusy czy Sokół — jest szybszy na polowaniu."),
    ("Kim jest Gerwazy?", "Gerwazy Rębajło to wierny klucznik Horeszków, noszący wielki rapier zwany Scyzorykiem."),
    ("Kim jest Wojski?", "Wojski Hreczecha to przyjaciel Sędziego i znawca myślistwa, który pięknie gra na rogu."),
    ("Napisz dwuwiersz o Soplicowie.", "Słońce weszło nad borem i złociło łany,\nA w Soplicowie budził się dwór pobielany."),
    ("Co robi Ksiądz Robak?", "Ksiądz Robak to emisariusz, który potajemnie przygotowuje szlachtę litewską do powstania."),
    ("Jak kończy się Pan Tadeusz?", "Poemat kończy się zgodą rodów, zaręczynami Tadeusza z Zosią oraz uroczystym polonezem."),
]


def format_prompt(question):
    return f"Pytanie: {question.strip()}\nOdpowiedź: "


def build_sft_batch(tokenizer, pairs, context_size, device="cpu"):
    inputs_list, targets_list = [], []
    end_id = tokenizer.ids[END]
    for question, answer in pairs:
        prompt_ids = tokenizer.encode(format_prompt(question), strict=False)
        answer_ids = tokenizer.encode(answer, strict=False) + [end_id]
        if len(prompt_ids) + len(answer_ids) - 1 > context_size:
            keep_prompt = max(1, context_size + 1 - len(answer_ids))
            prompt_ids = prompt_ids[-keep_prompt:]
        full = (prompt_ids + answer_ids)[:context_size + 1]
        x = full[:-1]
        prompt_mask_len = max(0, min(len(x), len(prompt_ids) - 1))
        y = [IGNORE_INDEX] * prompt_mask_len + full[prompt_mask_len + 1:]
        pad_len = context_size - len(x)
        if pad_len > 0:
            pad_token = tokenizer.ids.get(" ", 1)
            x = x + [pad_token] * pad_len
            y = y + [IGNORE_INDEX] * pad_len
        inputs_list.append(x)
        targets_list.append(y)
    return (
        torch.tensor(inputs_list, dtype=torch.long, device=device),
        torch.tensor(targets_list, dtype=torch.long, device=device),
    )


def train_sft(model_path, output_path, steps=150, lr=5e-4, seed=1, threads=8):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    model = load_model(model_path)
    max_pair_tokens = max(
        len(model.tokenizer.encode(format_prompt(q), strict=False)) + len(model.tokenizer.encode(a, strict=False))
        for q, a in SFT_PAIRS
    )
    sft_seq_len = min(model.context_size, max_pair_tokens)
    inputs, targets = build_sft_batch(model.tokenizer, SFT_PAIRS, sft_seq_len)
    no_decay_names = ("norm_", "embedding")
    decay = [p for name, p in model.weights.items() if not any(k in name for k in no_decay_names)]
    no_decay = [p for name, p in model.weights.items() if any(k in name for k in no_decay_names)]
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": 0.05},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr)
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        init_logits = model(inputs)
        init_loss = F.cross_entropy(
            init_logits.reshape(-1, init_logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
        ).item()
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final_logits = model(inputs)
        final_loss = F.cross_entropy(
            final_logits.reshape(-1, final_logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
        ).item()
    save_model(output_path, model)
    return {
        "initial_sft_loss": init_loss,
        "final_sft_loss": final_loss,
        "steps": steps,
        "elapsed_seconds": time.perf_counter() - started,
        "output": str(output_path),
    }


def ask(model, question, seed=1, max_new_tokens=96, temperature=0.3, top_k=20, top_p=0.9, repetition_penalty=1.15):
    prompt = format_prompt(question)
    full_text, ended = generate(
        model, prompt, seed, max_new_tokens,
        temperature, top_k, top_p=top_p, repetition_penalty=repetition_penalty,
    )
    answer = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
    return answer, ended


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train_cmd = commands.add_parser("train", help="Doucz model bazowy na parach Pytanie -> Odpowiedź (SFT).")
    train_cmd.add_argument("--model", required=True)
    train_cmd.add_argument("--output", required=True)
    train_cmd.add_argument("--steps", type=positive_integer, default=150)
    train_cmd.add_argument("--lr", type=positive_float, default=5e-4)
    train_cmd.add_argument("--seed", type=int, default=1)
    train_cmd.add_argument("--threads", type=positive_integer, default=8)

    ask_cmd = commands.add_parser("ask", help="Zadaj pytanie modelowi po SFT.")
    ask_cmd.add_argument("--model", required=True)
    ask_cmd.add_argument("--question", required=True)
    ask_cmd.add_argument("--seed", type=int, default=1)
    ask_cmd.add_argument("--max-new-tokens", type=positive_integer, default=96)
    ask_cmd.add_argument("--temperature", type=positive_float, default=0.3)
    ask_cmd.add_argument("--top-k", type=positive_integer, default=20)
    ask_cmd.add_argument("--top-p", type=positive_float, default=0.9)
    ask_cmd.add_argument("--repetition-penalty", type=positive_float, default=1.15)
    args = parser.parse_args()
    if args.command == "train":
        summary = train_sft(args.model, Path(args.output), args.steps, args.lr, args.seed, args.threads)
        print(json.dumps(summary, ensure_ascii=False))
    else:
        model = load_model(args.model).eval()
        answer, ended = ask(
            model, args.question, args.seed, args.max_new_tokens,
            args.temperature, args.top_k, args.top_p, args.repetition_penalty,
        )
        print(f"Pytanie: {args.question}")
        print(f"Odpowiedź: {answer}")
        print("Stopped: <end>" if ended else "Stopped: max-new-tokens limit reached")


if __name__ == "__main__":
    main()
