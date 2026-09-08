# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the experiment harness; these do not validate NPU execution."""

from collections import namedtuple
import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch


SCRIPT = Path(__file__).with_name("probe_native192.py")
SPEC = importlib.util.spec_from_file_location("probe_native192", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class ImportTests(unittest.TestCase):
    def test_import_needs_no_device_packages(self):
        # -S hides site-packages; the blocker also catches attempted lazy imports.
        code = """
import importlib.abc
import runpy
import sys
class RejectDevicePackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torch_npu', 'triton'}:
            raise AssertionError('unexpected device package import: ' + fullname)
sys.meta_path.insert(0, RejectDevicePackages())
runpy.run_path(sys.argv[1], run_name='probe_import_check')
"""
        result = subprocess.run(
            [sys.executable, "-S", "-c", code, str(SCRIPT)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_help_needs_no_site_packages(self):
        result = subprocess.run(
            [sys.executable, "-S", str(SCRIPT), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--graph", result.stdout)
        self.assertIn("--benchmark", result.stdout)


class ReferenceTests(unittest.TestCase):
    def test_reference_uses_whole_head_rms_and_neox_partner(self):
        data = probe.make_case(torch, 3, 2, position_shift=11)
        dim, heads = 192, 2
        width = dim * heads
        # Unequal halves expose accidental per-half normalization. Different
        # token/head/Q/K values and weights expose broadcasting and swap errors.
        channels = torch.arange(dim, dtype=torch.float32)
        for token in range(3):
            for segment in range(3):
                for head in range(heads):
                    values = ((channels % 17) - 7) * (segment + head + 1)
                    values[:96] *= 3 + token
                    values[96:] += token + head + 0.25
                    start = segment * width + head * dim
                    data["qkv"][token, start : start + dim] = values.to(torch.bfloat16)
        data["q_weight"] = (0.25 + channels / 97).to(torch.bfloat16)
        data["k_weight"] = (-0.75 + channels / 131).to(torch.bfloat16)

        actual = probe.reference(torch, data)
        for segment, name in enumerate(("q_weight", "k_weight")):
            expected = torch.empty((3, width), dtype=torch.float32)
            weights = data[name].float().tolist()
            for token in range(3):
                sin = data["sin"][token].flatten().tolist()
                cos = data["cos"][token].flatten().tolist()
                for head in range(heads):
                    start = segment * width + head * dim
                    values = data["qkv"][token, start : start + dim].float().tolist()
                    full_mean_square = sum(value * value for value in values) / 192
                    left_mean_square = sum(value * value for value in values[:96]) / 96
                    self.assertGreater(abs(full_mean_square - left_mean_square), 1)
                    inverse_rms = 1 / math.sqrt(full_mean_square + 1e-5)
                    for channel in range(dim):
                        partner = channel + 96 if channel < 96 else channel - 96
                        sign = -1 if channel < 96 else 1
                        own = values[channel] * inverse_rms * weights[channel]
                        paired = values[partner] * inverse_rms * weights[partner]
                        expected[token, head * dim + channel] = (
                            own * cos[channel] + sign * paired * sin[channel]
                        )
            torch.testing.assert_close(actual[segment], expected, rtol=1e-6, atol=2e-6)
        torch.testing.assert_close(actual[2], data["qkv"][:, 2 * width :])

    def test_v_copy_keeps_signed_zero_bits(self):
        data = probe.make_case(torch, 1, 1)
        bit_pattern = torch.tensor(
            [0, -32768, 1, -32767, 16256, -16512], dtype=torch.int16
        )
        data["qkv"][:, 384:] = bit_pattern.repeat(32).view(torch.bfloat16)
        expected = probe.reference(torch, data)
        outputs = tuple(value.to(torch.bfloat16).clone() for value in expected)
        self.assertTrue(probe.compare(torch, outputs, expected)["passed"])
        self.assertTrue(
            torch.equal(expected[2].view(torch.int16).flatten(), bit_pattern.repeat(32))
        )

        # Numeric equality considers +0 and -0 equal; the harness must not.
        outputs[2].view(torch.int16)[0, 1] = 0
        self.assertTrue(torch.equal(outputs[2], expected[2]))
        result = probe.compare(torch, outputs, expected)
        self.assertFalse(result["v_bitwise_equal"])
        self.assertFalse(result["passed"])

    def test_incorrect_q_or_k_fails(self):
        expected = probe.reference(
            torch, probe.make_case(torch, 2, 2, position_shift=17)
        )
        for index in (0, 1):
            with self.subTest(output=("q", "k")[index]):
                outputs = [value.to(torch.bfloat16).clone() for value in expected]
                outputs[index][0, 0] += 1
                self.assertFalse(probe.compare(torch, outputs, expected)["passed"])

    def test_nan_in_actual_or_reference_fails(self):
        source = probe.reference(torch, probe.make_case(torch, 2, 1, position_shift=17))
        for reference_nan in (False, True):
            for index in (0, 1):
                with self.subTest(reference_nan=reference_nan, output=index):
                    expected = [value.clone() for value in source]
                    outputs = [value.to(torch.bfloat16).clone() for value in source]
                    (expected if reference_nan else outputs)[index][0, 0] = float("nan")
                    result = probe.compare(torch, outputs, expected)
                    self.assertFalse(result["passed"])
                    self.assertFalse(result[("q", "k")[index]]["finite"])
                    self.assertIsNone(result[("q", "k")[index]]["max_abs_vs_fp32"])


class LaunchContractTests(unittest.TestCase):
    def test_real_192_launch_preserves_pointer_order_and_default_mode(self):
        class FakeKernel:
            def __getitem__(self, grid):
                self.grid = grid

                def call(*arguments, **options):
                    self.arguments, self.options = arguments, options
                    return self

                return call

        for heads in (1, 4, 8):
            with self.subTest(heads=heads):
                data = probe.make_case(torch, 9, heads)
                outputs = probe.allocate_outputs(torch, data)
                fake = FakeKernel()
                launch, grid = probe.make_launcher(fake, data, outputs, vector_cores=64)
                self.assertIs(launch(), fake)
                self.assertEqual(grid, (math.ceil(64 / heads), heads, 1))
                self.assertEqual(fake.grid, grid)
                pointers = (
                    data["qkv"],
                    data["sin"],
                    data["cos"],
                    *outputs,
                    data["q_weight"],
                    None,
                    data["k_weight"],
                    None,
                )
                for actual, expected in zip(fake.arguments[:10], pointers):
                    self.assertIs(actual, expected)
                width = heads * 192
                self.assertEqual(
                    fake.arguments[10:],
                    (
                        9,
                        width,
                        width,
                        3 * width,
                        1e-5,
                        192,
                        192,
                        False,
                        True,
                        192,
                        192,
                        96,
                        0,
                    ),
                )
                # No compile_mode, force_simt_only, or other compiler override.
                self.assertEqual(fake.options, {"DO_PARTIAL": False, "DO_HALF": True})

    def test_missing_compiler_metadata_is_unknown(self):
        metadata = namedtuple("Metadata", "compile_mode")("unstructured_in_simt")
        compiled = type("Compiled", (), {"metadata": metadata})()
        result = probe.compiled_metadata(compiled)
        self.assertEqual(result["compile_mode"], "unstructured_in_simt")
        self.assertIsNone(result["force_simt_only"])
        self.assertTrue(
            all(value is None for value in probe.compiled_metadata(object()).values())
        )


class EvidenceTests(unittest.TestCase):
    def test_optional_phase_failure_is_distinct_from_not_run(self):
        for key in ("graph_status", "benchmark_status"):
            with self.subTest(phase=key), tempfile.TemporaryDirectory() as temporary:
                with contextlib.redirect_stdout(io.StringIO()):
                    evidence = probe.Evidence(Path(temporary) / "evidence")
                    evidence.report[key] = "RUNNING"
                    try:
                        evidence.step(key, lambda: {"passed": False})
                    except RuntimeError:
                        evidence.fail()
                report = json.loads((evidence.directory / "report.json").read_text())
                self.assertEqual(report[key], "FAILED")
                untouched = (
                    "benchmark_status" if key == "graph_status" else "graph_status"
                )
                self.assertEqual(report[untouched], "NOT_RUN")

    def test_stage_exception_is_never_logged_as_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "evidence"
            with contextlib.redirect_stdout(io.StringIO()):
                evidence = probe.Evidence(directory)
                try:
                    evidence.step(
                        "compile192",
                        lambda: (_ for _ in ()).throw(RuntimeError("compiler failure")),
                    )
                except RuntimeError:
                    evidence.fail()
            report = json.loads((directory / "report.json").read_text())
            self.assertEqual(report["status"], "FAILED")
            self.assertEqual(report["active_stage"], "compile192")
            self.assertFalse(any(stage["passed"] for stage in report["stages"]))
            self.assertNotIn("PASS", (directory / "run.log").read_text())
            self.assertIn("compiler failure", (directory / "traceback.txt").read_text())

    def test_failed_comparison_aborts_main_and_preserves_metrics(self):
        def failed_run(args, evidence):
            evidence.step("numeric192", lambda: {"passed": False, "max_abs": 1.0})

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "evidence"
            with (
                mock.patch.object(probe, "run", failed_run),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                code = probe.main(["--out", str(directory)])
            self.assertEqual(code, 1)
            report = json.loads((directory / "report.json").read_text())
            self.assertEqual(report["status"], "FAILED")
            self.assertEqual(report["stages"][0]["result"]["max_abs"], 1.0)
            self.assertFalse(report["stages"][0]["passed"])
            self.assertNotIn("PASS", (directory / "run.log").read_text())


if __name__ == "__main__":
    unittest.main()
