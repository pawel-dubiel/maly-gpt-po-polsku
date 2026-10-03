"""Wczytanie checkpointu BPE i przewidywanie kolejnych tokenów na CPU."""

import argparse
import math
import random

from model import load_model, torch
from text_data import END, positive_float, positive_integer


@torch.no_grad()
def generate(model, prompt, seed, max_new_tokens, temperature, top_k):
    if type(seed) is not int or type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("Wymagany całkowity seed i dodatni limit nowych tokenów.")
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature musi być dodatnią, skończoną liczbą.")
    if type(top_k) is not int or not 1 <= top_k <= len(model.tokenizer.vocabulary):
        raise ValueError("Top-k musi być liczbą całkowitą od 1 do rozmiaru słownika.")
    ids = model.tokenizer.encode(prompt)
    if len(ids) > model.context_size:
        raise ValueError(f"Prompt przekracza kontekst {model.context_size} tokenów.")
    rng = random.Random(seed)
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            context = torch.tensor([ids[-model.context_size:]], dtype=torch.long, device="cpu")
            values, indices = torch.topk(model(context)[0, -1], top_k)
            probabilities = torch.softmax((values - values.max()) / temperature, dim=-1).tolist()
            next_id = rng.choices(indices.tolist(), weights=probabilities, k=1)[0]
            if next_id == model.tokenizer.ids[END]:
                return model.tokenizer.decode(ids), True
            ids.append(next_id)
        return model.tokenizer.decode(ids), False
    finally:
        model.train(was_training)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=positive_integer, required=True)
    parser.add_argument("--threads", type=positive_integer, required=True)
    parser.add_argument("--temperature", type=positive_float, required=True)
    parser.add_argument("--top-k", type=positive_integer, required=True)
    args = parser.parse_args()
    try:
        torch.set_num_threads(args.threads)
        model = load_model(args.model).eval()
        ids = model.tokenizer.encode(args.prompt)
        with torch.no_grad():
            logits, attention = model.forward_with_attention(torch.tensor([ids], dtype=torch.long, device="cpu"))
            probabilities = torch.softmax(logits[0, -1], dim=-1).tolist()
        words = model.tokenizer.vocabulary
        print(f"Loaded transformer: {args.model}; d_model={model.d_model}, d_ff={model.d_ff}; "
              f"blocks={model.n_layers}, heads={model.n_heads}, context={model.context_size}")
        print(f"Context tokens: {[words[token] for token in ids]}")
        print(f"Token IDs: {ids}")
        print("\nAttention from the last prompt token to each prompt token:")
        for layer, maps in enumerate(attention, start=1):
            for head, row in enumerate(maps[0, :, -1].tolist(), start=1):
                print(f"  Block {layer}, head {head}: " + ", ".join(
                    f"{words[token]!r}: {probability:.6f}" for token, probability in zip(ids, row)))
        print("\nNext-token probabilities:")
        for token, probability in zip(words, probabilities):
            print(f"  {token!r:>10}: {probability:.6f}")
        best = max(range(len(words)), key=lambda token: probabilities[token])
        print(f"\nMost likely next token: {words[best]!r}")
        text, ended = generate(model, args.prompt, args.seed, args.max_new_tokens, args.temperature, args.top_k)
        print(f"Generated: {text!r}")
        print("Stopped: <end>" if ended else "Stopped: max-new-tokens limit reached")
    except (OSError, ValueError, OverflowError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
