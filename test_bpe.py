"""Independent checks of BPE merges, exact text reconstruction, and strict metadata."""

import copy
import unittest


class BPETests(unittest.TestCase):
    def test_most_frequent_pair_is_merged_and_ids_are_reproducible(self):
        from bpe_tokenizer import BPETokenizer

        # Alphabet: space, a, b, c, d, plus END. 'ab' occurs three times.
        first = BPETokenizer.train("ab ab ab cd", 7)
        second = BPETokenizer.train("ab ab ab cd", 7)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(len(first.vocabulary), 7)
        self.assertEqual(first.vocabulary[-1], "ab")
        self.assertEqual(first.encode("ab cd"), [6, 1, 4, 5])

    def test_round_trip_preserves_spaces_lines_punctuation_and_unseen_words(self):
        from bpe_tokenizer import BPETokenizer

        text = "Litwo!  kota kotem\n\nżółć; kotem kota\n"
        tokenizer = BPETokenizer.train(text, len(set(text)) + 5)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        # This word is absent from training but its letters are known.
        unseen = "kotek"
        self.assertNotIn(unseen, text)
        self.assertEqual(tokenizer.decode(tokenizer.encode(unseen)), unseen)
        with self.assertRaisesRegex(ValueError, "Unknown character"):
            tokenizer.encode("🙂")
        with self.assertRaises(ValueError):
            tokenizer.decode([-1])
        with self.assertRaises(ValueError):
            tokenizer.encode("<end>")

    def test_decode_preserves_tokens_from_single_pass_iterators(self):
        from bpe_tokenizer import BPETokenizer

        tokenizer = BPETokenizer.train("ab ab ab cd", 7)
        text = "ab cd"
        ids = tokenizer.encode(text)
        self.assertEqual(tokenizer.decode(iter(ids)), text)
        self.assertEqual(tokenizer.decode(token for token in ids), text)
        for invalid in (-1, len(tokenizer.vocabulary), 1.0, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                tokenizer.decode(iter(ids + [invalid]))

    def test_metadata_is_complete_and_invalid_merges_fail_explicitly(self):
        from bpe_tokenizer import BPETokenizer

        tokenizer = BPETokenizer.train("ab ab ab cd", 7)
        payload = tokenizer.to_dict()
        loaded = BPETokenizer.from_dict(payload)
        self.assertEqual(loaded.encode("abcd"), tokenizer.encode("abcd"))
        for key in payload:
            bad = copy.deepcopy(payload)
            del bad[key]
            with self.subTest(missing=key):
                with self.assertRaises(ValueError):
                    BPETokenizer.from_dict(bad)
        bad = copy.deepcopy(payload)
        bad["merges"][0][0] = 999
        with self.assertRaisesRegex(ValueError, "merge"):
            BPETokenizer.from_dict(bad)
        bad = copy.deepcopy(payload)
        bad["vocabulary"][-1] = "wrong"
        with self.assertRaises(ValueError):
            BPETokenizer.from_dict(bad)

    def test_unattainable_or_missing_settings_do_not_reduce_vocabulary_silently(self):
        from bpe_tokenizer import BPETokenizer

        with self.assertRaisesRegex(ValueError, "at least"):
            BPETokenizer.train("abc", 3)
        with self.assertRaisesRegex(ValueError, "Cannot reach"):
            BPETokenizer.train("a a", 100)
        with self.assertRaises(ValueError):
            BPETokenizer.train("", 10)


if __name__ == "__main__":
    unittest.main()
