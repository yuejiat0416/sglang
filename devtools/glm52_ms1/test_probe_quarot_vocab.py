"""CPU checks for bounded checkpoint reads and coordinate comparison evidence.

Synthetic weights establish file/matrix contracts; they do not validate the
real GLM checkpoint, NPU execution, or a model-quality threshold.
"""

import builtins
import contextlib
import hashlib
import importlib.util
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).with_name("probe_quarot_vocab.py")
SPEC = importlib.util.spec_from_file_location("probe_quarot_vocab", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def write_safetensors(path, tensors):
    """Write genuine header/data offsets, independently of the reader under test."""
    header = {"__metadata__": {"fixture": "independent-cpu-oracle"}}
    payloads = []
    offset = 0
    for key, (dtype, values) in tensors.items():
        values = np.asarray(values)
        if dtype == "BF16":
            words = values.astype("<f4").view("<u4")
            raw = (words >> 16).astype("<u2").tobytes()
        else:
            storage_dtype = {"F16": "<f2", "F32": "<f4", "I8": "i1"}[dtype]
            raw = values.astype(storage_dtype).tobytes()
        header[key] = {
            "dtype": dtype,
            "shape": list(values.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        payloads.append(raw)
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(payloads))
    return header, 8 + len(encoded)


def coordinate_fixture():
    # Q is orthogonal but NOT symmetric: using Q instead of Q.T is observable.
    q = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
    gamma = np.diag(np.array([2, 0.5, 3], dtype=np.float32))
    r = q.T @ gamma @ q
    embedding = np.array(
        [[1, 2, 3], [-2, 1, 4], [4, -3, 2], [2, 4, -1]], dtype=np.float32
    )
    head = np.array([[2, -1, 3], [1, 3, -2], [-3, 2, 1], [4, 1, -2]], dtype=np.float32)
    return embedding @ q, embedding, head @ gamma @ q, head, q, r


def write_checkpoints(root):
    target, draft = root / "target", root / "draft"
    te, de, th, dh, q, r = coordinate_fixture()
    write_safetensors(
        target / "part-00001.safetensors",
        {"model.embed_tokens.weight": ("BF16", te)},
    )
    write_safetensors(
        target / "part-00002.safetensors",
        {"lm_head.weight": ("BF16", th)},
    )
    (target / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "part-00001.safetensors",
                    "lm_head.weight": "part-00002.safetensors",
                }
            }
        )
    )
    write_safetensors(
        draft / "model.safetensors",
        {"embed_tokens.weight": ("BF16", de), "lm_head.weight": ("BF16", dh)},
    )
    write_safetensors(
        target / "optional" / "quarot.safetensors",
        {"global_rotation": ("F32", q)},
    )
    write_safetensors(target / "rot.safetensors", {"rot.weight": ("BF16", r)})
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
    for directory in (target, draft):
        (directory / "config.json").write_text(
            json.dumps({"hidden_size": 3, "vocab_size": 4, "mask_token_id": 3})
        )
    return target, draft


class TensorReadTests(unittest.TestCase):
    def test_real_offsets_bf16_decoding_and_noncontiguous_row_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.safetensors"
            values = np.array(
                [[1, -2, 0.5], [3, -4, 1.5], [-8, 0, 2]], dtype=np.float32
            )
            for dtype in ("BF16", "F16", "F32"):
                with self.subTest(dtype=dtype):
                    header, data_start = write_safetensors(
                        path,
                        {
                            "leading": ("I8", np.arange(13)),
                            "wanted": (dtype, values),
                            "trailing": ("F32", np.arange(17)),
                        },
                    )
                    reader = probe.TensorFile(path)
                    self.assertEqual(reader.data_start, data_start)
                    self.assertEqual(reader.header["wanted"], header["wanted"])
                    result = reader.read_tensor("wanted", rows=[2, 0])
                    self.assertEqual(result.dtype, np.dtype("float32"))
                    np.testing.assert_array_equal(result, values[[2, 0]])
                    np.testing.assert_array_equal(reader.read_tensor("wanted"), values)

    def test_row_reads_do_not_read_the_whole_tensor_or_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "large-enough.safetensors"
            values = np.arange(3000, dtype=np.float32).reshape(1000, 3)
            header, data_start = write_safetensors(
                path,
                {
                    "leading": ("F32", np.zeros(123)),
                    "wanted": ("F32", values),
                    "unrelated": ("F32", np.zeros(4000)),
                },
            )
            reader = probe.TensorFile(path)
            with patch.object(probe, "read_bytes", wraps=probe.read_bytes) as spy:
                actual = reader.read_tensor("wanted", rows=[999, 0, 503])
            np.testing.assert_array_equal(actual, values[[999, 0, 503]])
            calls = [(call.args[1], call.args[2]) for call in spy.call_args_list]
            start = data_start + header["wanted"]["data_offsets"][0]
            self.assertCountEqual(calls, [(start + i * 12, 12) for i in (999, 0, 503)])
            self.assertEqual(sum(length for _, length in calls), 36)

    def test_negative_out_of_range_and_truncated_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.safetensors"
            write_safetensors(path, {"weight": ("BF16", np.ones((3, 4)))})
            reader = probe.TensorFile(path)
            for rows in ([-1], [3]):
                with (
                    self.subTest(rows=rows),
                    self.assertRaises((ValueError, IndexError)),
                ):
                    reader.read_tensor("weight", rows=rows)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises((ValueError, EOFError)):
                probe.TensorFile(path).read_tensor("weight", rows=[2])

    def test_integer_weights_are_not_silently_treated_as_float(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.safetensors"
            write_safetensors(path, {"weight": ("I8", np.ones((2, 3)))})
            with self.assertRaises((ValueError, TypeError)):
                probe.TensorFile(path).read_tensor("weight", rows=[0])


class CheckpointLookupTests(unittest.TestCase):
    def test_index_resolves_exact_target_key_without_mtp_substitution(self):
        with tempfile.TemporaryDirectory() as tmp:
            target, _ = write_checkpoints(Path(tmp))
            # Unindexed MTP data must not win over the exact index entry.
            write_safetensors(
                target / "mtp.safetensors",
                {
                    "model.layers.78.embed_tokens.weight": ("F32", np.zeros((4, 3))),
                },
            )
            reader, key = probe.Checkpoint(target).find_tensor(
                ("model.embed_tokens.weight", "embed_tokens.weight")
            )
            self.assertEqual(key, "model.embed_tokens.weight")
            np.testing.assert_array_equal(
                reader.read_tensor(key, rows=[0]), coordinate_fixture()[0][[0]]
            )

    def test_multiple_aliases_are_ambiguous_in_header_only_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text("{}")
            write_safetensors(
                root / "weights.safetensors",
                {
                    "embed_tokens.weight": ("F32", np.ones((2, 3))),
                    "model.embed_tokens.weight": ("F32", np.zeros((2, 3))),
                },
            )
            with self.assertRaises(ValueError):
                probe.Checkpoint(root).find_tensor(
                    ("model.embed_tokens.weight", "embed_tokens.weight")
                )

    def test_conflicting_indexes_are_not_silently_prioritized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text("{}")
            for name, value in (("first", 1), ("second", 2)):
                write_safetensors(
                    root / f"{name}.safetensors",
                    {
                        "lm_head.weight": ("F32", np.full((2, 3), value)),
                    },
                )
            for index, shard in (
                ("model.safetensors.index.json", "first.safetensors"),
                ("quant_model_weights.safetensors.index.json", "second.safetensors"),
            ):
                (root / index).write_text(
                    json.dumps({"weight_map": {"lm_head.weight": shard}})
                )
            with self.assertRaises(ValueError):
                probe.Checkpoint(root).find_tensor(("lm_head.weight",))


class CoordinateTests(unittest.TestCase):
    def test_nonsymmetric_rotation_and_nonuniform_norm_oracle(self):
        values = coordinate_fixture()
        reports = probe.evaluate_candidates(*values, ids=[13, 7, 92, 0])
        for group, candidate in (("embedding", "rotated"), ("head", "rotated_norm")):
            with self.subTest(group=group):
                for row in reports[group][candidate]["rows"]:
                    self.assertEqual(row["max_abs_error"], 0)
                    self.assertEqual(row["relative_l2"], 0)
        for group, candidate in (
            ("embedding", "raw"),
            ("head", "raw"),
            ("head", "rotated"),
        ):
            with self.subTest(group=group, candidate=candidate):
                self.assertTrue(
                    all(
                        row["max_abs_error"] > 0
                        for row in reports[group][candidate]["rows"]
                    )
                )

    def test_same_cosine_does_not_hide_a_factor_of_two(self):
        reference = np.array([[1, -2, 3], [4, 2, -1]], dtype=np.float32)
        report = probe.compare_rows(reference, reference * 2, [9, 2])
        self.assertEqual([row["id"] for row in report["rows"]], [9, 2])
        for row in report["rows"]:
            self.assertAlmostEqual(row["cosine"], 1)
            self.assertAlmostEqual(row["norm_ratio"], 2)
            self.assertAlmostEqual(row["relative_l2"], 1)

    def test_zero_and_nonfinite_rows_produce_strict_json_with_review(self):
        reference = np.array([[0, 0], [1, 2], [1, 2]], dtype=np.float32)
        candidate = np.array([[0, 0], [np.nan, 2], [1, np.inf]], dtype=np.float32)
        report = probe.compare_rows(reference, candidate, [3, 5, 7])
        json.dumps(report, allow_nan=False)
        self.assertTrue(report["needs_review"])
        self.assertIsNone(report["rows"][0]["cosine"])
        self.assertIsNone(report["rows"][0]["norm_ratio"])
        self.assertIsNone(report["rows"][0]["relative_l2"])
        self.assertFalse(report["rows"][1]["finite"])
        self.assertFalse(report["rows"][2]["finite"])

    def test_explicit_ids_preserve_order_and_default_includes_mask(self):
        configs = [{"mask_token_id": 97, "eos_token_id": [91, 92]}]
        explicit = probe.select_token_ids(100, configs, explicit=[9, 2, 9, 4])
        self.assertEqual(explicit["ids"], [9, 2, 4])
        default = probe.select_token_ids(100, configs, count=8)
        self.assertTrue({0, 99, 97, 91, 92}.issubset(default["ids"]))
        self.assertEqual(len(default["ids"]), len(set(default["ids"])))
        self.assertTrue(default["sources"])


class CliEvidenceTests(unittest.TestCase):
    def test_mid_read_failure_preserves_completed_region_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target, draft = write_checkpoints(root)
            head_path = target / "part-00002.safetensors"
            with head_path.open("rb") as stream:
                payload_start = 8 + struct.unpack("<Q", stream.read(8))[0]
            read_bytes = probe.read_bytes

            def fail_head_payload(stream, offset, length):
                if (
                    Path(stream.name).resolve() == head_path.resolve()
                    and offset >= payload_start
                ):
                    raise OSError("synthetic read failure")
                return read_bytes(stream, offset, length)

            with (
                patch.object(probe, "read_bytes", side_effect=fail_head_payload),
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
                        "--token-ids",
                        "3,0,2",
                    ]
                )
            self.assertEqual(code, 1)
            report_path = next((root / "evidence").glob("*/report.json"))
            report = json.loads(report_path.read_text())
            self.assertEqual(report["status"], "FAILED")
            self.assertIn("synthetic read failure", report["error"])
            self.assertIn("OSError", report["traceback"])
            completed = [
                read
                for source in report["files"]
                for read in source["reads"]
                if read["key"] == "model.embed_tokens.weight"
            ]
            self.assertEqual(len(completed), 1)
            self.assertEqual(len(completed[0]["ranges"]), 3)
            json.dumps(report, allow_nan=False)

    def test_nonfinite_matrix_returns_review_with_json_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target, draft = write_checkpoints(root)
            q = coordinate_fixture()[4].copy()
            q[0, 0] = np.nan
            write_safetensors(
                target / "optional" / "quarot.safetensors",
                {
                    "global_rotation": ("F32", q),
                },
            )
            with contextlib.redirect_stdout(io.StringIO()):
                code = probe.main(
                    [
                        "--target",
                        str(target),
                        "--draft",
                        str(draft),
                        "--out",
                        str(root / "evidence"),
                        "--token-ids",
                        "3,0,2",
                    ]
                )
            self.assertEqual(code, 2)
            report_path = next((root / "evidence").glob("*/report.json"))
            report = json.loads(report_path.read_text())
            self.assertEqual(report["status"], "NUMERICAL_REVIEW_REQUIRED")
            self.assertFalse(report["matrix_finite"]["Q"])
            json.dumps(report, allow_nan=False)

    def test_finite_mismatch_is_collected_without_inventing_a_quality_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target, draft = write_checkpoints(root)
            write_safetensors(
                target / "part-00002.safetensors",
                {
                    "lm_head.weight": ("BF16", coordinate_fixture()[2] * 8),
                },
            )
            with contextlib.redirect_stdout(io.StringIO()):
                code = probe.main(
                    [
                        "--target",
                        str(target),
                        "--draft",
                        str(draft),
                        "--out",
                        str(root / "evidence"),
                        "--token-ids",
                        "3,0,2",
                    ]
                )
            self.assertEqual(code, 0)
            report_path = next((root / "evidence").glob("*/report.json"))
            report = json.loads(report_path.read_text())
            self.assertEqual(report["status"], "COORDINATE_COMPARISON_COLLECTED")
            for candidate in report["comparisons"]["head"].values():
                self.assertTrue(
                    all(row["max_abs_error"] > 0 for row in candidate["rows"])
                )

    def test_cpu_cli_writes_read_fingerprints_without_model_runtime_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target, draft = write_checkpoints(root)
            source_hashes = {
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
                        "--token-ids",
                        "3,0,2",
                    ]
                )
            self.assertEqual(code, 0)
            reports = list((root / "evidence").glob("*/report.json"))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text())
            self.assertEqual(report["status"], "COORDINATE_COMPARISON_COLLECTED")
            self.assertEqual(report["samples"]["ids"], [3, 0, 2])
            self.assertTrue(report["versions"]["numpy"])
            self.assertIn("git_head", report)
            self.assertIn("q_roundtrip", report)
            self.assertGreaterEqual(len(report["files"]), 5)
            for evidence in report["files"]:
                source = Path(evidence["path"])
                self.assertEqual(evidence["file_size"], source.stat().st_size)
                self.assertIsInstance(evidence["mtime_ns"], int)
                self.assertEqual(len(evidence["header_sha256"]), 64)
                for read in evidence["reads"]:
                    for span in read["ranges"]:
                        with source.open("rb") as stream:
                            stream.seek(span["offset"])
                            raw = stream.read(span["length"])
                        self.assertEqual(
                            span["sha256"], hashlib.sha256(raw).hexdigest()
                        )
            json.dumps(report, allow_nan=False)
            for path, previous_hash in source_hashes.items():
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), previous_hash
                )


if __name__ == "__main__":
    unittest.main()
