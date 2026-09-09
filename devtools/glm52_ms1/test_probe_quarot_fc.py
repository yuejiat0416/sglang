"""CPU oracles for the fixed-input FC/normalization diagnostic.

Small synthetic matrices test algebra and BF16 storage rounding. Their numerical
assertions are not precision thresholds for the real model or an NPU kernel.
"""

import builtins
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).with_name("probe_quarot_fc.py")
SPEC = importlib.util.spec_from_file_location("probe_quarot_fc", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
with patch.object(sys, "path", [str(SCRIPT.parent), *sys.path]):
    SPEC.loader.exec_module(probe)
    from test_probe_quarot_vocab import write_safetensors


def fc_fixture(scales=(1, 0.01, 0.0001)):
    q = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
    fc = np.array(
        [
            [1, 2, 3, 4, 5, 6],
            [-2, 3, 1, 2, -1, 4],
            [3, -1, 2, -3, 4, 1],
            [2, 1, -4, 3, 2, -2],
        ],
        dtype=np.float32,
    )
    inputs = (
        np.array(
            [
                [[1, 2, -1], [3, -2, 1]],
                [[-2, 1, 4], [1, 3, -1]],
                [[2, -3, 1], [-1, 2, 4]],
            ],
            dtype=np.float32,
        )
        * np.array(scales, dtype=np.float32)[:, None, None]
    )
    norm = np.array([2, 0.5, 1.5, 3], dtype=np.float32)
    return fc, q, norm, inputs


def block_oracle(fc, q, inputs):
    """Independent tiny dense block matrix; production need not build this."""
    count, width = inputs.shape[1:]
    transform = np.kron(np.eye(count), q.astype(np.float64))
    flat = inputs.reshape(len(inputs), count * width).astype(np.float64)
    weights = fc.astype(np.float64)
    reference = flat @ weights.T
    rotated_inputs = flat @ transform
    unchanged = rotated_inputs @ weights.T
    folded = weights @ transform
    equivalent = rotated_inputs @ folded.T
    return reference, unchanged, equivalent, rotated_inputs, folded


def norm_oracle(values, weight, eps):
    return np.array(
        [
            [
                float(v)
                * float(w)
                / math.sqrt(sum(float(x) ** 2 for x in row) / len(row) + eps)
                for v, w in zip(row, weight)
            ]
            for row in values
        ]
    )


def normal_bf16_oracle(values):
    """Arithmetic ties-to-even oracle for these finite, normal test weights."""
    values = values.astype(np.float64)
    _, exponent = np.frexp(np.abs(values))
    step = np.exp2(exponent - 8)
    return np.rint(values / step) * step


def write_fc_checkpoints(root, norm_shape=(3,)):
    target, draft = root / "target", root / "draft"
    fc, q, norm, _ = fc_fixture()
    write_safetensors(
        target / "optional" / "quarot.safetensors",
        {
            "global_rotation": ("F32", q * np.float32(0.99)),
        },
    )
    (target / "config.json").write_text(json.dumps({"hidden_size": 3}))
    (target / "quant_model_description.json").write_text(
        json.dumps(
            {
                "optional": {
                    "quarot": {
                        "rotation_map": {
                            "global_rotation": "optional/quarot.safetensors",
                        }
                    }
                },
            }
        )
    )
    write_safetensors(
        draft / "model.safetensors",
        {
            "unused.weight": ("F32", np.full((17, 3), 99)),
            "fc.weight": ("BF16", fc[:3]),
            "hidden_norm.weight": ("BF16", norm[:3].reshape(norm_shape)),
            "norm.weight": ("BF16", np.full(3, 9)),
        },
    )
    (draft / "config.json").write_text(
        json.dumps(
            {
                "aux_hidden_state_layer_ids": [0, 1],
                "transformer_layer_config": {"hidden_size": 3, "rms_norm_eps": 1e-5},
            }
        )
    )
    return target, draft


def assert_comparison(test, report, reference, candidate):
    test.assertEqual(len(report["rows"]), len(reference))
    for row, ref, cand in zip(report["rows"], reference, candidate):
        rn, cn = np.linalg.norm(ref), np.linalg.norm(cand)
        expected = {
            "reference_norm": rn,
            "candidate_norm": cn,
            "norm_ratio": cn / rn,
            "relative_l2": np.linalg.norm(cand - ref) / rn,
            "cosine": np.dot(ref, cand) / (rn * cn),
            "max_abs_error": np.max(np.abs(cand - ref)),
        }
        for key, value in expected.items():
            with test.subTest(input_id=row["id"], metric=key):
                np.testing.assert_allclose(row[key], value, rtol=1e-5, atol=2e-5)


class Bf16AndNormTests(unittest.TestCase):
    def test_bf16_ties_even_signed_zero_subnormal_and_overflow(self):
        source = np.array(
            [
                0x3F808000,
                0x3F818000,
                0xBF808000,
                0xBF818000,
                0x00008000,
                0x00018000,
                0x80008000,
                0x7F7FFFFF,
                0xFF7FFFFF,
                0x7F800000,
                0xFF800000,
            ],
            dtype=np.uint32,
        )
        expected = np.array(
            [
                0x3F800000,
                0x3F820000,
                0xBF800000,
                0xBF820000,
                0x00000000,
                0x00020000,
                0x80000000,
                0x7F800000,
                0xFF800000,
                0x7F800000,
                0xFF800000,
            ],
            dtype=np.uint32,
        )
        original = source.copy()
        result = probe.bf16_round(source.view(np.float32))
        self.assertEqual(result.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(result.view(np.uint32), expected)
        np.testing.assert_array_equal(source, original)

    def test_nan_does_not_round_to_infinity(self):
        words = np.array([0x7F800001, 0x7FC01234, 0xFF800001], dtype=np.uint32)
        self.assertTrue(np.isnan(probe.bf16_round(words.view(np.float32))).all())

    def test_rms_norm_uses_nonuniform_weight_and_epsilon_near_zero(self):
        values = np.array([[3, -4], [1e-4, -1e-4], [0, 0]], dtype=np.float32)
        weight = np.array([2, 0.5], dtype=np.float32)
        result = probe.rms_norm(values, weight, 1e-5)
        np.testing.assert_allclose(
            result, norm_oracle(values, weight, 1e-5), rtol=1e-6, atol=1e-8
        )
        self.assertEqual(result.dtype, np.dtype(np.float32))
        self.assertTrue(np.isfinite(result).all())
        without_effective_epsilon = probe.rms_norm(values[:2], weight, 1e-12)
        self.assertLess(
            np.linalg.norm(result[1]), np.linalg.norm(without_effective_epsilon[1])
        )


class FcCoordinateTests(unittest.TestCase):
    def test_multiblock_nonsymmetric_rotation_matches_independent_block_oracle(self):
        fc, q, norm, inputs = fc_fixture()
        report = probe.evaluate_fc(fc, q, norm, 1e-5, inputs, [3, 1])
        ref, unchanged, equivalent, _, _ = block_oracle(fc, q, inputs)
        expected = {
            "unchanged_fc_pre_norm": (ref, unchanged),
            "q_fold_equivalent_pre_norm": (ref, equivalent),
            "unchanged_fc_post_norm": (
                norm_oracle(ref, norm, 1e-5),
                norm_oracle(unchanged, norm, 1e-5),
            ),
            "q_fold_equivalent_post_norm": (
                norm_oracle(ref, norm, 1e-5),
                norm_oracle(equivalent, norm, 1e-5),
            ),
        }
        for name, pair in expected.items():
            with self.subTest(comparison=name):
                assert_comparison(self, report["comparisons"][name], *pair)
        self.assertEqual(len(report["q_roundtrip"]["rows"]), 6)
        self.assertTrue(
            all(
                row["relative_l2"] > 0.1
                for row in report["comparisons"]["unchanged_fc_pre_norm"]["rows"]
            )
        )

    def test_scaled_q_norm_cancellation_depends_on_input_scale_and_epsilon(self):
        fc, q, norm, inputs = fc_fixture()
        q = q * np.float32(0.99)
        report = probe.evaluate_fc(fc, q, norm, 1e-5, inputs, [0, 2])
        ref, _, equivalent, _, _ = block_oracle(fc, q, inputs)
        assert_comparison(
            self, report["comparisons"]["q_fold_equivalent_pre_norm"], ref, equivalent
        )
        assert_comparison(
            self,
            report["comparisons"]["q_fold_equivalent_post_norm"],
            norm_oracle(ref, norm, 1e-5),
            norm_oracle(equivalent, norm, 1e-5),
        )
        post = report["comparisons"]["q_fold_equivalent_post_norm"]["rows"]
        self.assertGreater(post[2]["relative_l2"], post[0]["relative_l2"])
        for row in report["q_roundtrip"]["rows"]:
            self.assertAlmostEqual(row["norm_ratio"], 0.99**2, places=6)

    def test_nonscalar_nonorthogonality_is_not_hidden_by_normalization(self):
        fc, q, norm, inputs = fc_fixture()
        q[0, 0] = 0.15
        report = probe.evaluate_fc(fc, q, norm, 1e-5, inputs, [1, 3])
        ref, _, equivalent, _, _ = block_oracle(fc, q, inputs)
        assert_comparison(
            self,
            report["comparisons"]["q_fold_equivalent_post_norm"],
            norm_oracle(ref, norm, 1e-5),
            norm_oracle(equivalent, norm, 1e-5),
        )
        self.assertTrue(
            any(
                row["cosine"] < 0.999
                for row in report["comparisons"]["q_fold_equivalent_post_norm"]["rows"]
            )
        )

    def test_sampled_folded_weights_include_bf16_storage_rounding(self):
        fc, _, norm, inputs = fc_fixture()
        fc = fc * np.float32(0.37)
        angle = 0.37
        q = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0],
                [math.sin(angle), math.cos(angle), 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        selected = [3, 1]
        report = probe.evaluate_fc(fc, q, norm, 1e-5, inputs, selected)
        ref, _, equivalent, rotated_inputs, folded = block_oracle(fc, q, inputs)
        bf16_weights = normal_bf16_oracle(folded[selected].astype(np.float32))
        bf16_output = rotated_inputs @ bf16_weights.T
        reports = report["sampled_weights"]
        assert_comparison(
            self,
            reports["folded_fp32_vs_equivalent"],
            equivalent[:, selected],
            rotated_inputs @ folded[selected].T,
        )
        assert_comparison(
            self,
            reports["folded_bf16_vs_fp32"],
            rotated_inputs @ folded[selected].T,
            bf16_output,
        )
        assert_comparison(
            self, reports["folded_bf16_vs_reference"], ref[:, selected], bf16_output
        )
        self.assertTrue(
            any(
                row["relative_l2"] > 0 for row in reports["folded_bf16_vs_fp32"]["rows"]
            )
        )

    def test_shapes_and_selected_rows_cannot_silently_change_the_comparison(self):
        fc, q, norm, inputs = fc_fixture()
        cases = (
            (fc[:, :-1], q, norm, inputs, [0]),
            (fc, q[:2], norm, inputs, [0]),
            (fc, q, norm[:-1], inputs, [0]),
            (fc, q, norm, inputs.reshape(3, 6), [0]),
            (fc, q, norm, inputs, [-1]),
            (fc, q, norm, inputs, [4]),
        )
        for index, args in enumerate(cases):
            with self.subTest(case=index), self.assertRaises((ValueError, IndexError)):
                probe.evaluate_fc(*args[:3], 1e-5, *args[3:])

    def test_zero_inputs_keep_diagnostics_json_serializable(self):
        fc, q, norm, inputs = fc_fixture()
        inputs[1] = 0
        report = probe.evaluate_fc(fc, q, norm, 1e-5, inputs, [0, 3])
        json.dumps(report, allow_nan=False)
        self.assertTrue(
            report["comparisons"]["q_fold_equivalent_post_norm"]["needs_review"]
        )


class NormReaderAndCliTests(unittest.TestCase):
    def test_one_dimensional_norm_is_exact_and_not_the_final_draft_norm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text("{}")
            for dtype in ("BF16", "F16", "F32"):
                with self.subTest(dtype=dtype):
                    weight = np.array([2, 0.5, -1.5], dtype=np.float32)
                    write_safetensors(
                        root / "model.safetensors",
                        {
                            "norm.weight": ("F32", np.full(3, 99)),
                            "hidden_norm.weight": (dtype, weight),
                        },
                    )
                    checkpoint = probe.vocab.Checkpoint(root)
                    actual = probe.read_hidden_norm(checkpoint, 3)
                    np.testing.assert_array_equal(actual, weight)
                    self.assertEqual(actual.shape, (3,))
                    self.assertEqual(actual.dtype, np.dtype(np.float32))

    def test_norm_alias_conflict_and_missing_hidden_norm_do_not_guess(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text("{}")
            cases = (
                {"norm.weight": ("BF16", np.ones(3))},
                {
                    "hidden_norm.weight": ("BF16", np.ones(3)),
                    "model.hidden_norm.weight": ("BF16", np.ones(3)),
                },
                {"hidden_norm.weight": ("BF16", np.ones((1, 3)))},
            )
            for index, tensors in enumerate(cases):
                with self.subTest(case=index), self.assertRaises(ValueError):
                    write_safetensors(root / "model.safetensors", tensors)
                    probe.read_hidden_norm(probe.vocab.Checkpoint(root), 3)

    def test_cli_reads_only_fc_hidden_norm_and_q_and_preserves_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target, draft = write_fc_checkpoints(root)
            sources = {
                path: hashlib.sha256(path.read_bytes()).hexdigest()
                for directory in (target, draft)
                for path in directory.rglob("*")
                if path.is_file()
            }
            original_import = builtins.__import__

            def guarded_import(name, *args, **kwargs):
                if name.split(".")[0] in {"torch", "torch_npu", "sglang"}:
                    raise AssertionError(f"Diagnostic must not import {name}")
                return original_import(name, *args, **kwargs)

            with (
                patch("builtins.__import__", side_effect=guarded_import),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                code = probe.main(
                    [
                        "--target",
                        str(target),
                        "--draft",
                        str(draft),
                        "--out",
                        str(root / "evidence"),
                        "--weight-rows",
                        "2",
                    ]
                )
            self.assertEqual(code, 0)
            reports = list((root / "evidence").glob("*/report.json"))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text())
            self.assertEqual(report["status"], "FIXED_INPUT_DIAGNOSTIC_COLLECTED")
            self.assertEqual(report["fixed_inputs"]["shape"], [3, 2, 3])
            self.assertEqual(report["results"]["selected_fc_output_rows"], [0, 2])
            self.assertEqual(report["rms_norm_eps"], 1e-5)
            expected_inputs = np.random.default_rng(0).standard_normal((3, 2, 3))
            expected_inputs = expected_inputs.astype(np.float32)
            expected_inputs *= np.array([1, 0.01, 0.0001], dtype=np.float32)[
                :, None, None
            ]
            self.assertEqual(
                report["fixed_inputs"]["sha256"],
                hashlib.sha256(expected_inputs.tobytes()).hexdigest(),
            )
            self.assertEqual(len(report["runner_sha256"]), 64)
            self.assertEqual(len(report["reader_sha256"]), 64)
            read_keys = set()
            for source in report["files"]:
                for read in source["reads"]:
                    read_keys.add(read["key"])
                    for span in read["ranges"]:
                        with Path(source["path"]).open("rb") as stream:
                            stream.seek(span["offset"])
                            raw = stream.read(span["length"])
                        self.assertEqual(
                            span["sha256"], hashlib.sha256(raw).hexdigest()
                        )
            self.assertEqual(
                read_keys, {"fc.weight", "hidden_norm.weight", "global_rotation"}
            )
            self.assertFalse(list((root / "evidence").rglob("*.safetensors")))
            for path, before in sources.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
            json.dumps(report, allow_nan=False)

    def test_bad_norm_shape_preserves_earlier_reads_in_failed_cli_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target, draft = write_fc_checkpoints(root, norm_shape=(1, 3))
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                code = probe.main(
                    [
                        "--target",
                        str(target),
                        "--draft",
                        str(draft),
                        "--out",
                        str(root / "evidence"),
                    ]
                )
            self.assertEqual(code, 1)
            report_path = next((root / "evidence").glob("*/report.json"))
            report = json.loads(report_path.read_text())
            self.assertEqual(report["status"], "FAILED")
            self.assertIn("hidden_norm", report["error"])
            self.assertTrue(
                any(
                    read["key"] == "fc.weight" and read["ranges"]
                    for source in report["files"]
                    for read in source["reads"]
                )
            )
            json.dumps(report, allow_nan=False)

    def test_invalid_epsilon_fails_before_payload_reads_with_valid_json_evidence(self):
        for epsilon in ("nan", "inf", float("nan"), float("inf"), 0):
            with (
                self.subTest(epsilon=repr(epsilon)),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                target, draft = write_fc_checkpoints(root)
                config_path = draft / "config.json"
                config = json.loads(config_path.read_text())
                config["transformer_layer_config"]["rms_norm_eps"] = epsilon
                # Deliberately include nonstandard numeric literals in two
                # cases; the diagnostic must reject them without corrupting
                # its own failure report.
                config_path.write_text(json.dumps(config))
                with (
                    patch.object(
                        probe.vocab.TensorFile,
                        "read_tensor",
                        side_effect=AssertionError("Unexpected payload read"),
                    ) as payload_reader,
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    code = probe.main(
                        [
                            "--target",
                            str(target),
                            "--draft",
                            str(draft),
                            "--out",
                            str(root / "evidence"),
                        ]
                    )
                self.assertEqual(code, 1)
                payload_reader.assert_not_called()
                reports = list((root / "evidence").glob("*/report.json"))
                self.assertEqual(len(reports), 1)
                report = json.loads(reports[0].read_text())
                self.assertEqual(report["status"], "FAILED")
                self.assertEqual(
                    report["inputs"], {"target": str(target), "draft": str(draft)}
                )
                self.assertEqual(
                    report["runner_sha256"],
                    hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
                )
                self.assertTrue(report["error"])
                self.assertTrue(report["traceback"])
                self.assertFalse(
                    any(
                        read["ranges"]
                        for source in report["files"]
                        for read in source["reads"]
                    )
                )
                json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
