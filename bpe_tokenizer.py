"""Inspectable Unicode BPE: learn frequent adjacent pairs and preserve exact text."""

from collections import Counter
import argparse
import json
from pathlib import Path
import re

from text_data import END, positive_integer


TOKENIZER_TYPE = "unicode-bpe-v1"
SEGMENTS = re.compile(r" ?\w+| ?[^\w\s]+|\s+", re.UNICODE)


def merge_pair(ids, left, right, result):
    merged = []
    position = 0
    while position < len(ids):
        if position + 1 < len(ids) and ids[position] == left and ids[position + 1] == right:
            merged.append(result)
            position += 2
        else:
            merged.append(ids[position])
            position += 1
    return merged


class BPETokenizer:
    def __init__(self, alphabet, vocabulary, merges):
        self.alphabet = tuple(alphabet)
        self.vocabulary = tuple(vocabulary)
        self.merges = tuple(tuple(merge) for merge in merges)
        self.ids = {piece: i for i, piece in enumerate(self.vocabulary)}
        self.ranks = {(left, right): (rank, result) for rank, (left, right, result) in enumerate(self.merges)}

    @classmethod
    def train(cls, text, vocabulary_size):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Tokenizer training text is empty.")
        if END in text:
            raise ValueError(f"Text contains reserved token {END!r}.")
        if type(vocabulary_size) is not int or vocabulary_size < 1:
            raise ValueError("Vocabulary size must be a positive integer.")
        alphabet = sorted(set(text))
        vocabulary = [END] + alphabet
        if vocabulary_size < len(vocabulary):
            raise ValueError(f"Vocabulary size must be at least {len(vocabulary)} to include all characters and <end>.")
        piece_ids = {piece: i for i, piece in enumerate(vocabulary)}
        frequencies = Counter(SEGMENTS.findall(text))
        splits = [([piece_ids[char] for char in segment], frequency)
                  for segment, frequency in frequencies.items()]
        merges = []
        while len(vocabulary) < vocabulary_size:
            pairs = Counter()
            for ids, frequency in splits:
                for pair in zip(ids, ids[1:]):
                    pairs[pair] += frequency
            if not pairs:
                raise ValueError(f"Cannot reach vocabulary size {vocabulary_size}; no adjacent pairs remain.")
            # Ties choose the lowest IDs, so the same text gives the same tokenizer.
            left, right = max(pairs, key=lambda pair: (pairs[pair], -pair[0], -pair[1]))
            piece = vocabulary[left] + vocabulary[right]
            if piece not in piece_ids:
                piece_ids[piece] = len(vocabulary)
                vocabulary.append(piece)
            result = piece_ids[piece]
            merges.append([left, right, result])
            splits = [(merge_pair(ids, left, right, result), frequency) for ids, frequency in splits]
            if len(merges) % 50 == 0:
                print(f"BPE merge {len(merges)}: {vocabulary[left]!r} + {vocabulary[right]!r} -> {piece!r}; vocabulary={len(vocabulary)}", flush=True)
        return cls.from_dict({"type": TOKENIZER_TYPE, "alphabet": alphabet,
                              "vocabulary": vocabulary, "merges": merges})

    def _fallback_ids(self, char):
        # Byte-level fallback (<0xXX>) if present in vocabulary, otherwise Unicode/ASCII normalization.
        raw = char.encode("utf-8")
        if all(f"<0x{b:02X}>" in self.ids for b in raw):
            return [self.ids[f"<0x{b:02X}>"] for b in raw]
        replacements = {'"': "«", "'": ",", "–": "—", "—": "-", "\t": " "}
        if char in replacements and replacements[char] in self.ids:
            return [self.ids[replacements[char]]]
        if char.lower() in self.ids:
            return [self.ids[char.lower()]]
        if " " in self.ids:
            return [self.ids[" "]]
        return [1]

    def encode(self, text, strict=True):
        if not isinstance(text, str) or not text:
            raise ValueError("Text or prompt must contain at least one character.")
        if END in text:
            raise ValueError(f"{END!r} is reserved; do not put it inside text.")
        output, cache = [], {}
        for segment in SEGMENTS.findall(text):
            if segment not in cache:
                ids = []
                for char in segment:
                    if char in self.ids:
                        ids.append(self.ids[char])
                    elif not strict:
                        ids.extend(self._fallback_ids(char))
                    else:
                        raise ValueError(f"Unknown character: {char!r}. The tokenizer was not trained on it.")
                # Apply learned rules in their original order, including overlapping-pair handling.
                while len(ids) >= 2:
                    best = min(zip(ids, ids[1:]), key=lambda pair: self.ranks.get(pair, (len(self.merges), -1)))
                    if best not in self.ranks:
                        break
                    ids = merge_pair(ids, best[0], best[1], self.ranks[best][1])
                cache[segment] = ids
            output.extend(cache[segment])
        return output

    def decode(self, ids):
        pieces = []
        byte_buffer = bytearray()
        for token in ids:
            if type(token) is not int or not 0 <= token < len(self.vocabulary):
                raise ValueError("Invalid token ID in BPE decoding.")
            piece = self.vocabulary[token]
            if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
                byte_buffer.append(int(piece[3:5], 16))
            else:
                if byte_buffer:
                    pieces.append(byte_buffer.decode("utf-8", errors="replace"))
                    byte_buffer.clear()
                pieces.append(piece)
        if byte_buffer:
            pieces.append(byte_buffer.decode("utf-8", errors="replace"))
        return "".join(pieces)

    def to_dict(self):
        return {"type": TOKENIZER_TYPE, "alphabet": list(self.alphabet),
                "vocabulary": list(self.vocabulary), "merges": [list(merge) for merge in self.merges]}

    @classmethod
    def from_dict(cls, payload):
        required = {"type", "alphabet", "vocabulary", "merges"}
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("Tokenizer fields are missing or unsupported; type, alphabet, vocabulary, and merges are required.")
        if payload["type"] != TOKENIZER_TYPE:
            raise ValueError(f"Tokenizer type must be {TOKENIZER_TYPE!r}.")
        alphabet, vocabulary, merges = (payload[name] for name in ("alphabet", "vocabulary", "merges"))
        if not isinstance(alphabet, list) or not alphabet or any(not isinstance(char, str) or len(char) != 1 for char in alphabet):
            raise ValueError("Tokenizer alphabet must contain single characters.")
        if alphabet != sorted(set(alphabet)):
            raise ValueError("Tokenizer alphabet must contain unique characters in sorted order.")
        if not isinstance(vocabulary, list) or any(not isinstance(piece, str) or not piece for piece in vocabulary):
            raise ValueError("Tokenizer vocabulary must contain nonempty strings.")
        if not isinstance(merges, list):
            raise ValueError("Tokenizer merges must be a list.")
        built = [END] + alphabet
        known = {piece: i for i, piece in enumerate(built)}
        for merge in merges:
            if not isinstance(merge, list) or len(merge) != 3 or any(type(token) is not int for token in merge):
                raise ValueError("Each BPE merge must contain three integer token IDs.")
            left, right, result = merge
            if not 1 <= left < len(built) or not 1 <= right < len(built):
                raise ValueError("BPE merge references a missing token or the reserved end token.")
            piece = built[left] + built[right]
            if piece not in known:
                known[piece] = len(built)
                built.append(piece)
            if result != known[piece]:
                raise ValueError("BPE merge result does not match its concatenated pieces.")
        if vocabulary != built:
            raise ValueError("Tokenizer vocabulary does not match its alphabet and merge history.")
        return cls(alphabet, vocabulary, merges)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Learn and save BPE rules from UTF-8 text.")
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--vocab-size", type=positive_integer, required=True)
    inspect = commands.add_parser("inspect", help="Show the actual pieces and IDs for text.")
    inspect.add_argument("--tokenizer", required=True)
    inspect.add_argument("--text", required=True)
    args = parser.parse_args()
    try:
        if args.command == "train":
            path = Path(args.output)
            if not path.parent.is_dir() or path.is_dir():
                raise ValueError(f"Tokenizer output must be a file in an existing directory: {path}")
            tokenizer = BPETokenizer.train(Path(args.data).read_text(encoding="utf-8"), args.vocab_size)
            tokenizer.save(path)
            print(f"Saved tokenizer: {path}; vocabulary={len(tokenizer.vocabulary)}; merges={len(tokenizer.merges)}")
        else:
            tokenizer = BPETokenizer.load(args.tokenizer)
            ids = tokenizer.encode(args.text)
            print(f"Pieces: {[tokenizer.vocabulary[token] for token in ids]}")
            print(f"IDs: {ids}")
            print(f"Decoded: {tokenizer.decode(ids)!r}")
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
