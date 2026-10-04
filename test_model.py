"""Sprawdzenie jednego przepływu: BPE, attention, trening, zapis i predykcja."""

import ast
import copy
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from torch.nn import functional as F

from bpe_tokenizer import BPETokenizer
from model import GPT, load_model, new_model, save_model
from predict_gpt import generate
import train_bpe
from train_bpe import make_windows


ROOT = Path(__file__).parent
ARCHITECTURE = {"context_size": 3, "d_model": 64, "d_ff": 128,
                "n_heads": 1, "n_layers": 1, "dropout": 0.0,
                "rms_epsilon": 1e-5, "activation": "gelu", "tie_embeddings": True}
SETTINGS = {"seed": 1, "steps": 20, "batch_size": 4, "learning_rate": 0.01,
            "min_learning_rate": 0.001, "warmup_steps": 2, "weight_decay": 0.1,
            "beta1": 0.9, "beta2": 0.99, "adam_epsilon": 1e-8, "gradient_clip": 1.0,
            "evaluate_every": 5, "train_eval_windows": 16, "validation_fraction": 0.1, "threads": 1}


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tokenizer = BPETokenizer.train("ab ba ab ba", 4)
        self.model = new_model(self.tokenizer, ARCHITECTURE, 1)

    def test_initial_predictions_are_not_confident_without_training(self):
        tokenizer = BPETokenizer.load(ROOT / "tokenizer_bpe.json")
        model = new_model(tokenizer, dict(ARCHITECTURE, context_size=16), 1)
        inputs = torch.arange(1, 129).reshape(8, 16)
        targets = inputs + 1
        with torch.no_grad():
            logits, attention = model.forward_with_attention(inputs)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten()).item()
            last = attention[0][:, :, -1]
            entropy = -(last * last.clamp_min(1e-30).log()).sum(-1).mean().item()
        self.assertAlmostEqual(loss, math.log(len(tokenizer.vocabulary)), delta=0.1)
        self.assertGreater(entropy, math.log(16) - 0.1)

    def test_multiple_heads_and_blocks_preserve_causality(self):
        architecture = {"context_size": 8, "d_model": 16, "d_ff": 64,
                        "n_heads": 4, "n_layers": 2, "dropout": 0.1,
                        "rms_epsilon": 1e-5, "activation": "gelu", "tie_embeddings": True}
        model = new_model(self.tokenizer, architecture, 1).eval()
        first, maps = model.forward_with_attention(torch.tensor([[2, 3, 2, 3]]))
        changed = model(torch.tensor([[2, 3, 2, 2]]))
        self.assertEqual(len(maps), 2)
        for attention in maps:
            self.assertEqual(tuple(attention.shape), (1, 4, 4, 4))
            self.assertEqual(attention.triu(1).count_nonzero().item(), 0)
            torch.testing.assert_close(attention.sum(-1), torch.ones((1, 4, 4)))
        torch.testing.assert_close(first[:, :3], changed[:, :3], rtol=1e-5, atol=1e-6)
        with self.assertRaises(ValueError):
            new_model(self.tokenizer, dict(architecture, d_model=17), 1)

    def test_attention_scale_and_future_mask_match_calculation(self):
        with torch.no_grad():
            for matrix in self.model.weights.values():
                matrix.zero_()
            self.model.weights["token_embedding"][2, 0] = 1
            self.model.weights["token_embedding"][3, 1] = 1
            self.model.weights["block_0_norm_attention"].fill_(1)
            for name in ("Wq", "Wk", "Wv"):
                self.model.weights["block_0_" + name].copy_(torch.eye(64))
        _, attention = self.model.forward_with_attention(torch.tensor([[2, 3]]))
        score = 1 / (1 / 64 + 1e-5) / math.sqrt(64)
        probability = math.exp(score) / (1 + math.exp(score))
        self.assertEqual(attention[0][0, 0, 0].tolist(), [1, 0])
        self.assertAlmostEqual(attention[0][0, 0, 1, 1].item(), probability, places=6)

    def test_future_tokens_do_not_change_prefix_but_prefix_affects_prediction(self):
        first = self.model(torch.tensor([[2, 3, 2]]))
        future = self.model(torch.tensor([[2, 3, 3]]))
        prefix = self.model(torch.tensor([[3, 3, 2]]))
        self.assertTrue(torch.equal(first[:, :2], future[:, :2]))
        self.assertFalse(torch.equal(first[:, 1], prefix[:, 1]))

    def test_qkv_gradients_match_finite_differences(self):
        self.model.double()
        inputs, targets = torch.tensor([[2, 3]]), torch.tensor([[3, 2]])

        def loss():
            logits = self.model(inputs)
            return F.cross_entropy(logits.reshape(-1, 4), targets.reshape(-1))

        loss().backward()
        for name in ("Wq", "Wk", "Wv"):
            parameter = self.model.weights["block_0_" + name]
            flat = parameter.grad.abs().argmax().item()
            i, j = divmod(flat, 64)
            gradient = parameter.grad[i, j].item()
            self.assertGreater(abs(gradient), 1e-9)
            initial = parameter[i, j].item()
            with torch.no_grad():
                parameter[i, j] = initial + 1e-6
                plus = loss().item()
                parameter[i, j] = initial - 1e-6
                minus = loss().item()
                parameter[i, j] = initial
            self.assertAlmostEqual(gradient, (plus - minus) / 2e-6, delta=1e-6)

    def test_checkpoint_round_trip_and_required_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            save_model(path, self.model)
            saved = path.read_bytes()
            loaded = load_model(path)
            self.assertEqual((loaded.d_model, loaded.d_ff, loaded.context_size), (64, 128, 3))
            self.assertTrue(torch.equal(self.model(torch.tensor([[2, 3]])), loaded(torch.tensor([[2, 3]]))))
            self.assertEqual(loaded.tokenizer.to_dict(), self.tokenizer.to_dict())
            payload = json.loads(saved)
            for field in payload:
                bad = copy.deepcopy(payload)
                del bad[field]
                path.write_text(json.dumps(bad), encoding="utf-8")
                with self.subTest(missing=field), self.assertRaises(ValueError):
                    load_model(path)
            for field in payload["architecture"]:
                bad = copy.deepcopy(payload)
                del bad["architecture"][field]
                path.write_text(json.dumps(bad), encoding="utf-8")
                with self.subTest(missing=field), self.assertRaises(ValueError):
                    load_model(path)
            for invalid in (0, -1, 64.0, True):
                bad = copy.deepcopy(payload)
                bad["architecture"]["d_model"] = invalid
                path.write_text(json.dumps(bad), encoding="utf-8")
                with self.subTest(d_model=invalid), self.assertRaises(ValueError):
                    load_model(path)
            for change in ("shape", "nonfinite", "vocabulary"):
                bad = copy.deepcopy(payload)
                if change == "shape":
                    bad["weights"]["block_0_Wq"].pop()
                elif change == "nonfinite":
                    bad["weights"]["block_0_Wq"][0][0] = float("nan")
                else:
                    bad["vocabulary"][1], bad["vocabulary"][2] = bad["vocabulary"][2], bad["vocabulary"][1]
                path.write_text(json.dumps(bad), encoding="utf-8")
                with self.subTest(invalid=change), self.assertRaises(ValueError):
                    load_model(path)
            path.write_bytes(saved)
            with torch.no_grad():
                self.model.weights["block_0_Wq"][0, 0] = float("inf")
            with self.assertRaises(ValueError):
                save_model(path, self.model)
            self.assertEqual(path.read_bytes(), saved)

    def test_invalid_inputs_and_dimensions_fail_explicitly(self):
        for ids in (torch.tensor([[-1]]), torch.tensor([[4]]), torch.tensor([[1.0]]),
                    torch.tensor([[True]]), torch.empty((0, 1), dtype=torch.long),
                    torch.tensor([[2, 3, 2, 3]])):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.model(ids)
        for dimension in (0, -1, 64.0, True):
            with self.subTest(dimension=dimension), self.assertRaises(ValueError):
                new_model(self.tokenizer, dict(ARCHITECTURE, d_model=dimension), 1)

    def test_windows_have_actual_next_tokens_without_synthetic_end_markers(self):
        inputs, targets = make_windows([1, 2, 3, 1, 2], 2)
        self.assertEqual(inputs.tolist(), [[1, 2], [2, 3], [3, 1]])
        self.assertEqual(targets.tolist(), [[2, 3], [3, 1], [1, 2]])
        for ids, context in (([1, 2], 2), ([1, 2, 3], 0), ([1, 2, 3], True),
                             ([1, 0, 2], 1), ([1, 2.0, 3], 1)):
            with self.subTest(ids=ids, context=context), self.assertRaises(ValueError):
                make_windows(ids, context)

    def test_train_validation_split_has_no_shared_source_positions(self):
        train_x, train_y, val_x, val_y = train_bpe.split_windows(list(range(1, 21)), 3, 0.25)
        self.assertEqual(train_x[0].tolist(), [1, 2, 3])
        self.assertEqual(train_y[-1].tolist(), [13, 14, 15])
        self.assertEqual(val_x.tolist(), [[16, 17, 18], [17, 18, 19]])
        self.assertEqual(val_y.tolist(), [[17, 18, 19], [18, 19, 20]])
        train_tokens = set(train_x.flatten().tolist() + train_y.flatten().tolist())
        val_tokens = set(val_x.flatten().tolist() + val_y.flatten().tolist())
        self.assertFalse(train_tokens & val_tokens)
        for fraction in (0, 1, -0.1, 1.1, float("nan"), float("inf"), True, "0.1"):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                train_bpe.split_windows(list(range(1, 21)), 3, fraction)
        for fraction in (0.05, 0.95):
            with self.subTest(short_split=fraction), self.assertRaisesRegex(ValueError, "token"):
                train_bpe.split_windows(list(range(1, 21)), 3, fraction)

    def test_evaluation_weights_partial_batches_and_preserves_model(self):
        inputs, targets = make_windows([2, 3, 2, 2, 3, 3, 2, 3], 3)
        before = {name: p.detach().clone() for name, p in self.model.weights.items()}
        expected = F.cross_entropy(self.model(inputs).flatten(0, 1), targets.flatten()).item()
        observed = []
        hook = self.model.register_forward_pre_hook(
            lambda model, args: observed.append((model.training, torch.is_grad_enabled())))
        try:
            actual = train_bpe.mean_loss(self.model, inputs, targets, 3)
        finally:
            hook.remove()
        self.assertAlmostEqual(actual, expected, places=6)
        self.assertEqual(observed, [(False, False), (False, False)])
        self.assertTrue(self.model.training)
        for name, parameter in self.model.weights.items():
            self.assertTrue(torch.equal(parameter, before[name]))
            self.assertIsNone(parameter.grad)

    def test_validation_selects_earlier_checkpoint_without_training_on_it(self):
        train_x, train_y = make_windows([2] * 24, 3)
        val_x, _ = make_windows([2] * 12, 3)
        val_y = torch.full_like(val_x, 3)
        other_x, other_y = make_windows([2] * 12, 3)
        with tempfile.TemporaryDirectory() as directory:
            model = new_model(self.tokenizer, dict(ARCHITECTURE, d_model=8, d_ff=16, dropout=0.1), 1)
            other = new_model(self.tokenizer, dict(ARCHITECTURE, d_model=8, d_ff=16, dropout=0.1), 1)
            path = Path(directory) / "best.json"
            log = io.StringIO()
            with redirect_stdout(log):
                train_bpe.train(model, train_x, train_y, val_x, val_y, SETTINGS, path)
                train_bpe.train(other, train_x, train_y, other_x, other_y, SETTINGS, Path(directory) / "other.json")
            saved = load_model(path)
            saved_loss = train_bpe.mean_loss(saved, val_x, val_y, 4)
            final_loss = train_bpe.mean_loss(model, val_x, val_y, 4)
            self.assertLess(saved_loss, final_loss)
            self.assertIn("Best checkpoint: step 1;", log.getvalue())
            self.assertTrue(any(not torch.equal(saved.weights[name], p) for name, p in model.weights.items()))
            for name, parameter in model.weights.items():
                self.assertTrue(torch.equal(parameter, other.weights[name]))

    def test_multiblock_forward_and_all_gradients_match_pytorch_reference(self):
        model = new_model(self.tokenizer, dict(ARCHITECTURE, d_model=16, d_ff=32,
                                              n_heads=4, n_layers=2), 3).double()
        weights = {name: p.detach().clone().requires_grad_() for name, p in model.weights.items()}
        ids = torch.tensor([[2, 3, 2], [3, 2, 2]])
        x = F.embedding(ids, weights["token_embedding"])
        for layer in range(2):
            prefix = f"block_{layer}_"
            z = F.rms_norm(x, (16,), weights[prefix + "norm_attention"].squeeze(0), 1e-5)
            q, k, v = [F.linear(z, weights[prefix + name].T).reshape(2, 3, 4, 4).transpose(1, 2)
                       for name in ("Wq", "Wk", "Wv")]
            q, k = model.apply_rope(q), model.apply_rope(k)
            attended = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
            x = x + F.linear(attended.transpose(1, 2).reshape(2, 3, 16), weights[prefix + "Wo"].T)
            z = F.rms_norm(x, (16,), weights[prefix + "norm_ff"].squeeze(0), 1e-5)
            x = x + F.linear(F.gelu(F.linear(z, weights[prefix + "W1"].T)), weights[prefix + "W2"].T)
        expected = F.linear(F.rms_norm(x, (16,), weights["norm_final"].squeeze(0), 1e-5),
                            weights["token_embedding"])
        actual = model(ids)
        with_maps, _ = model.forward_with_attention(ids)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(with_maps, expected, rtol=1e-10, atol=1e-12)
        targets = torch.tensor([3, 2, 3, 2, 2, 3])
        F.cross_entropy(actual.flatten(0, 1), targets).backward()
        F.cross_entropy(expected.flatten(0, 1), targets).backward()
        for name, p in model.weights.items():
            with self.subTest(parameter=name):
                torch.testing.assert_close(p.grad, weights[name].grad, rtol=1e-9, atol=1e-11)

    def test_generation_is_deterministic_with_dropout_and_validates_sampling(self):
        model = new_model(self.tokenizer, dict(ARCHITECTURE, dropout=0.5), 1)
        first = generate(model, "ab", 1, 20, 0.8, 4)
        second = generate(model, "ab", 1, 20, 0.8, 4)
        self.assertEqual(first, second)
        self.assertTrue(model.training)
        for temperature, top_k in ((0, 4), (float("nan"), 4), (1, 0), (1, 5), (True, 4)):
            with self.subTest(temperature=temperature, top_k=top_k), self.assertRaises(ValueError):
                generate(model, "ab", 1, 3, temperature, top_k)

    def test_config_requires_every_setting_and_protects_input_paths(self):
        config = json.loads((ROOT / "train_config.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for name in ("data", "tokenizer"):
                config[name] = str(ROOT / config[name])
            for name in ("model", "report"):
                config[name] = str(Path(directory) / config[name])
            path.write_text(json.dumps(config))
            train_bpe.load_config(path)
            for section in ("architecture", "training"):
                for field in config[section]:
                    bad = copy.deepcopy(config)
                    del bad[section][field]
                    path.write_text(json.dumps(bad))
                    with self.subTest(section=section, missing=field), self.assertRaises(ValueError):
                        train_bpe.load_config(path)
            for field, value in (("steps", True), ("warmup_steps", 16000), ("gradient_clip", 0),
                                 ("weight_decay", -1), ("learning_rate", float("nan"))):
                bad = copy.deepcopy(config)
                bad["training"][field] = value
                path.write_text(json.dumps(bad))
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    train_bpe.load_config(path)
            config["model"] = config["data"]
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "różnymi"):
                train_bpe.load_config(path)

    def test_learning_rate_warms_up_then_decays_to_explicit_minimum(self):
        self.assertAlmostEqual(train_bpe.learning_rate(1, SETTINGS), 0.005)
        self.assertAlmostEqual(train_bpe.learning_rate(2, SETTINGS), 0.01)
        self.assertAlmostEqual(train_bpe.learning_rate(20, SETTINGS), 0.001)
        self.assertGreater(train_bpe.learning_rate(5, SETTINGS), train_bpe.learning_rate(15, SETTINGS))

    def test_end_token_stops_without_changing_prompt_spacing(self):
        from unittest.mock import patch

        logits = torch.tensor([[[1000.0, -1000.0, -1000.0, -1000.0]]])
        with patch.object(self.model, "forward_cached", return_value=(logits, None)):
            text, ended = generate(self.model, "a ", 1, 3, 1.0, 4)
        self.assertTrue(ended)
        self.assertEqual(text, "a ")

    def test_swiglu_gqa_qknorm_and_kv_cache_match_full_forward(self):
        arch = dict(ARCHITECTURE, context_size=8, d_model=16, d_ff=32,
                    n_heads=4, n_layers=2, activation="swiglu")
        model = new_model(self.tokenizer, arch, 7).eval()
        self.assertEqual(model.n_kv_heads, 2)
        self.assertIn("block_0_W_gate", model.weights)
        seq = torch.tensor([[2, 3, 1, 2, 3]])
        full_logits = model(seq)
        prefill_logits, cache = model.forward_cached(seq[:, :3], kv_cache=None)
        torch.testing.assert_close(prefill_logits, full_logits[:, :3], rtol=1e-5, atol=1e-6)
        step4_logits, cache = model.forward_cached(seq[:, 3:4], kv_cache=cache)
        step5_logits, cache = model.forward_cached(seq[:, 4:5], kv_cache=cache)
        torch.testing.assert_close(step4_logits[:, 0], full_logits[:, 3], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(step5_logits[:, 0], full_logits[:, 4], rtol=1e-5, atol=1e-6)

    def test_top_p_repetition_penalty_and_utf8_fallback(self):
        text, ended = generate(self.model, "a🙂", 1, 2, 0.8, 4, top_p=0.9, repetition_penalty=1.2)
        self.assertIsInstance(text, str)
        self.assertIsInstance(ended, bool)

    def test_sft_masked_loss_teaches_end_token(self):
        import sft_chat

        x, y = sft_chat.build_sft_batch(self.tokenizer, [("a", "b")], context_size=8)
        self.assertEqual(x.shape, (1, 8))
        self.assertEqual(y.shape, (1, 8))
        self.assertIn(sft_chat.IGNORE_INDEX, y[0].tolist())
        self.assertIn(0, y[0].tolist())

    def test_train_then_predict_using_only_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            data, tokenizer_path, path = (Path(directory) / name for name in ("book.txt", "tokenizer.json", "model.json"))
            data.write_text("Litwo! Ojczyzno moja!\nty jesteś jak zdrowie;\n" * 6, encoding="utf-8")
            tokenizer = BPETokenizer.train(data.read_text(encoding="utf-8"), 32)
            tokenizer.save(tokenizer_path)
            config_path = Path(directory) / "config.json"
            report_path = Path(directory) / "report.json"
            config = {"data": str(data), "tokenizer": str(tokenizer_path), "model": str(path),
                      "report": str(report_path),
                      "architecture": dict(ARCHITECTURE, context_size=16, n_heads=4, n_layers=2, dropout=0.1),
                      "training": dict(SETTINGS, steps=300, warmup_steps=10, learning_rate=0.001,
                                       min_learning_rate=0.0001, batch_size=8, evaluate_every=100)}
            config_path.write_text(json.dumps(config), encoding="utf-8")
            command = [sys.executable, str(ROOT / "train_bpe.py"), "--config", str(config_path)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(report_path.read_text())
            self.assertLess(report["full_training_loss"], report["initial"]["train_loss_sample"])
            self.assertLess(report["full_training_loss"], 1.0,
                            "Model powinien nauczyć się krótkiego, powtarzanego fragmentu.")
            self.assertEqual(report["model_sha256"], train_bpe.sha256(path))
            self.assertAlmostEqual(report["full_validation_loss"],
                                   min(row["validation_loss"] for row in report["history"]), places=6)
            self.assertIn("Best checkpoint: step", result.stdout)
            saved = path.read_bytes()
            data.unlink()
            tokenizer_path.unlink()
            prompt = "Litwo!  moja!\n"
            result = subprocess.run([sys.executable, str(ROOT / "predict_gpt.py"), "--model", str(path),
                                     "--prompt", prompt, "--seed", "1", "--max-new-tokens", "3", "--threads", "1", "--temperature", "0.8", "--top-k", "4"],
                                    cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            generated = ast.literal_eval(result.stdout.split("Generated: ")[1].splitlines()[0])
            self.assertTrue(generated.startswith(prompt))
            self.assertEqual(path.read_bytes(), saved)
            missing = subprocess.run(command[:-2], capture_output=True, text=True)
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("--config", missing.stderr)


if __name__ == "__main__":
    unittest.main()
