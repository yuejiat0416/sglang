# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the temporary recipe, without NPU imports or model loading."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("single_dspark_static.sh").resolve()
REPO = SCRIPT.parents[2]


class SingleDSparkStaticTests(unittest.TestCase):
    def run_script(self, *args, overrides=None):
        environment = os.environ.copy()
        for key in (
            "MODE",
            "GRAPH",
            "ENABLE_METRICS",
            "MS1_HOST",
            "MS1_PORT",
            "TARGET_MODEL",
            "DRAFT_MODEL",
            "KERNEL_REPO",
            "MS1_STATE",
            "BASH_ENV",
            "SERVED_MODEL_NAME",
        ):
            environment.pop(key, None)
        environment["PYTHONPATH"] = "/existing/python/path"
        environment.update(
            MS1_HOST="192.0.2.10",
            MS1_STATE="/test/run-state",
            TARGET_MODEL="/test/checkpoints/target",
            DRAFT_MODEL="/test/checkpoints/draft",
        )
        environment.update(overrides or {})
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            cwd="/",
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def printed_command(self, **overrides):
        result = self.run_script("--print-command", overrides=overrides)
        self.assertEqual(result.returncode, 0, result.stderr)
        command = shlex.split(result.stdout)
        self.assertEqual(command[0], "env")
        deleted = []
        index = 1
        while command[index] == "-u":
            deleted.append(command[index + 1])
            index += 2
        environment = {}
        while "=" in command[index]:
            key, value = command[index].split("=", 1)
            environment[key] = value
            index += 1
        return environment, deleted, command[index:]

    def server_args(self, command):
        boundary = command.index("--")
        self.assertEqual(
            command[boundary + 1 : boundary + 4],
            ["python3", "-m", "sglang.launch_server"],
        )
        return command[boundary + 4 :]

    def assert_option(self, args, name, value):
        self.assertEqual(args.count(name), 1, name)
        self.assertEqual(args[args.index(name) + 1], str(value), name)

    def test_default_preserves_working_recipe_and_adds_dspark_eager(self):
        environment, deleted, command = self.printed_command()
        self.assertEqual(
            command[:2], ["python3", str(SCRIPT.with_name("with_kernel_checkout.py"))]
        )
        self.assert_option(command, "--kernel-repo", f"{REPO}/../sgl-kernel-npu")
        self.assert_option(command, "--state-dir", "/test/run-state")
        args = self.server_args(command)
        expected = {
            "--model-path": "/test/checkpoints/target",
            "--attention-backend": "ascend",
            "--device": "npu",
            "--tp-size": 16,
            "--nnodes": 1,
            "--dp-size": 1,
            "--chunked-prefill-size": -1,
            "--max-prefill-tokens": 69632,
            "--mem-fraction-static": 0.85,
            "--max-running-requests": 8,
            "--served-model-name": "model",
            "--quantization": "modelslim",
            "--moe-a2a-backend": "deepep",
            "--deepep-mode": "auto",
            "--load-balance-method": "round_robin",
            "--host": "192.0.2.10",
            "--port": 8810,
            "--speculative-algorithm": "DSPARK",
            "--speculative-draft-model-path": "/test/checkpoints/draft",
            "--speculative-draft-model-quantization": "unquant",
            "--speculative-draft-attention-backend": "ascend",
            "--speculative-dspark-block-size": 8,
            "--speculative-num-draft-tokens": 9,
        }
        for name, value in expected.items():
            self.assert_option(args, name, value)
        for flag in (
            "--enable-dp-attention",
            "--trust-remote-code",
            "--disable-cuda-graph",
        ):
            self.assertEqual(args.count(flag), 1)
        self.assertNotIn("--cuda-graph-bs", args)
        self.assertNotIn("--speculative-num-steps", args)
        self.assertNotIn("--speculative-eagle-topk", args)
        self.assertNotIn("NEXTN", args)
        for key, value in {
            "SGLANG_RAGGED_VERIFY_MODE": "static",
            "HCCL_SOCKET_IFNAME": "lo",
            "GLOO_SOCKET_IFNAME": "lo",
            "HCCL_BUFFSIZE": "1000",
            "HCCL_OP_EXPANSION_MODE": "AIV",
            "DEEPEP_NORMAL_LONG_SEQ_ROUND": "72",
            "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "1024",
            "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
            "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
            "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
            "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "100",
        }.items():
            self.assertEqual(environment[key], value, key)
        self.assertEqual(
            environment["PYTHONPATH"], f"{REPO}/python:/existing/python/path"
        )
        self.assertEqual(
            set(deleted),
            {
                "http_proxy",
                "https_proxy",
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ASCEND_LAUNCH_BLOCKING",
            },
        )

    def test_graph_switch_restores_original_capture_bucket(self):
        _, _, command = self.printed_command(GRAPH="1")
        args = self.server_args(command)
        self.assert_option(args, "--cuda-graph-bs", 16)
        self.assertNotIn("--disable-cuda-graph", args)

    def test_target_control_changes_only_speculative_arguments(self):
        for graph in ("0", "1"):
            with self.subTest(graph=graph):
                draft_env, _, draft_command = self.printed_command(GRAPH=graph)
                target_env, _, target_command = self.printed_command(
                    MODE="target-only", GRAPH=graph
                )
                expected = []
                draft_args = iter(self.server_args(draft_command))
                for item in draft_args:
                    if item.startswith("--speculative-"):
                        next(draft_args)
                    else:
                        expected.append(item)
                self.assertEqual(self.server_args(target_command), expected)
                self.assertEqual(draft_env, target_env)

    def test_nextn_preserves_working_recipe_without_dspark_arguments(self):
        for graph in ("0", "1"):
            with self.subTest(graph=graph):
                target_env, target_deleted, target_command = self.printed_command(
                    MODE="target-only", GRAPH=graph
                )
                nextn_env, nextn_deleted, nextn_command = self.printed_command(
                    MODE="nextn",
                    GRAPH=graph,
                    DRAFT_MODEL="/unused/dspark/checkpoint",
                )
                args = self.server_args(nextn_command)
                expected_spec = {
                    "--speculative-algorithm": "NEXTN",
                    "--speculative-num-steps": 4,
                    "--speculative-eagle-topk": 1,
                    "--speculative-num-draft-tokens": 5,
                    "--speculative-draft-model-quantization": "unquant",
                }
                for name, value in expected_spec.items():
                    self.assert_option(args, name, value)
                self.assertEqual(
                    {arg for arg in args if arg.startswith("--speculative-")},
                    set(expected_spec),
                )
                self.assertNotIn("/unused/dspark/checkpoint", args)
                actual_base = []
                iterator = iter(args)
                for arg in iterator:
                    if arg in expected_spec:
                        next(iterator)
                    else:
                        actual_base.append(arg)
                self.assertEqual(actual_base, self.server_args(target_command))
                self.assertEqual(nextn_env, target_env)
                self.assertEqual(nextn_deleted, target_deleted)
                self.assertEqual(
                    nextn_command[: nextn_command.index("--")],
                    target_command[: target_command.index("--")],
                )

    def test_metrics_switch_only_adds_runtime_metrics_flag(self):
        for mode in ("dspark", "target-only", "nextn"):
            for graph in ("0", "1"):
                with self.subTest(mode=mode, graph=graph):
                    default = self.printed_command(MODE=mode, GRAPH=graph)
                    disabled = self.printed_command(
                        MODE=mode, GRAPH=graph, ENABLE_METRICS="0"
                    )
                    enabled = self.printed_command(
                        MODE=mode, GRAPH=graph, ENABLE_METRICS="1"
                    )
                    self.assertEqual(default, disabled)
                    self.assertEqual(enabled[:2], default[:2])
                    self.assertEqual(enabled[2], [*default[2], "--enable-metrics"])
                    self.assertNotIn("--enable-metrics", default[2])

    def test_print_is_side_effect_free_and_preserves_quoted_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "unexpected"
            hooks = root / "hooks.sh"
            hooks.write_text(
                'source() { touch "$MS1_TEST_MARKER"; return 99; }\n'
                'python3() { touch "$MS1_TEST_MARKER"; return 99; }\n'
            )
            state = root / "not-created"
            kernel = f"{root}/kernel checkout"
            draft = f"{root}/draft $(touch {marker})"
            environment, _, command = self.printed_command(
                BASH_ENV=str(hooks),
                MS1_TEST_MARKER=str(marker),
                MS1_STATE=str(state),
                KERNEL_REPO=kernel,
                DRAFT_MODEL=draft,
                TARGET_MODEL=f"{root}/target model",
                MS1_HOST="127.0.0.1",
                MS1_PORT="18810",
                SERVED_MODEL_NAME="test served model",
            )
            self.assert_option(command, "--kernel-repo", kernel)
            self.assert_option(command, "--state-dir", state)
            args = self.server_args(command)
            self.assert_option(args, "--speculative-draft-model-path", draft)
            self.assert_option(args, "--model-path", f"{root}/target model")
            self.assert_option(args, "--host", "127.0.0.1")
            self.assert_option(args, "--port", 18810)
            self.assert_option(args, "--served-model-name", "test served model")
            self.assertFalse(marker.exists())
            self.assertFalse(state.exists())

    def test_launch_allows_vendor_recovery_and_clears_proxies(self):
        # Substitute shell source and the python3 executable; no vendor scripts,
        # helper, device packages or model processes run in this CPU test.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_log = root / "source.log"
            launch_log = root / "launch.json"
            hooks = root / "hooks.sh"
            hooks.write_text(
                'source() { printf "%s\\n" "$1" >> "$MS1_SOURCE_LOG"; '
                # A recoverable command failure must not terminate source.
                "false; "
                'export PYTHONPATH="${PYTHONPATH}:vendor"; }\n'
            )
            fake_python = root / "python3"
            fake_python.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "with open(os.environ['MS1_LAUNCH_LOG'], 'w') as stream:\n"
                "    json.dump({'argv': sys.argv[1:], 'pythonpath': os.environ['PYTHONPATH'], "
                "'proxies': [k for k in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', "
                "'ASCEND_LAUNCH_BLOCKING') if k in os.environ]}, stream)\n"
            )
            fake_python.chmod(0o755)
            overrides = {
                "BASH_ENV": str(hooks),
                "MS1_SOURCE_LOG": str(source_log),
                "MS1_LAUNCH_LOG": str(launch_log),
                "PATH": f"{root}:{os.environ['PATH']}",
            }
            for key in (
                "http_proxy",
                "https_proxy",
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ASCEND_LAUNCH_BLOCKING",
            ):
                overrides[key] = "test-value"
            result = self.run_script(overrides=overrides)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                source_log.read_text().splitlines(),
                [
                    "/usr/local/Ascend/ascend-toolkit/set_env.sh",
                    "/usr/local/Ascend/nnal/atb/set_env.sh",
                ],
            )
            launch = json.loads(launch_log.read_text())
            self.assertEqual(launch["proxies"], [])
            self.assertEqual(
                launch["pythonpath"],
                f"{REPO}/python:/existing/python/path:vendor:vendor",
            )
            self.assertEqual(
                launch["argv"][0], str(SCRIPT.with_name("with_kernel_checkout.py"))
            )
            self.assert_option(launch["argv"], "--speculative-algorithm", "DSPARK")

    def test_final_vendor_failure_stops_before_helper(self):
        sources = [
            "/usr/local/Ascend/ascend-toolkit/set_env.sh",
            "/usr/local/Ascend/nnal/atb/set_env.sh",
        ]
        for index, failed_source in enumerate(sources):
            with (
                self.subTest(source=failed_source),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                source_log = root / "source.log"
                marker = root / "unexpected-helper"
                hooks = root / "hooks.sh"
                hooks.write_text(
                    'source() { printf "%s\\n" "$1" >> "$MS1_SOURCE_LOG"; '
                    'if [ "$1" = "$MS1_FAIL_SOURCE" ]; then return 1; fi; '
                    "return 0; }\n"
                )
                fake_python = root / "python3"
                fake_python.write_text('#!/bin/sh\n: > "$MS1_HELPER_MARKER"\n')
                fake_python.chmod(0o755)
                result = self.run_script(
                    overrides={
                        "BASH_ENV": str(hooks),
                        "MS1_SOURCE_LOG": str(source_log),
                        "MS1_FAIL_SOURCE": failed_source,
                        "MS1_HELPER_MARKER": str(marker),
                        "PATH": f"{root}:{os.environ['PATH']}",
                    }
                )
                self.assertEqual(result.returncode, 1)
                self.assertIn(f"Failed to source {failed_source}", result.stderr)
                self.assertEqual(
                    source_log.read_text().splitlines(), sources[: index + 1]
                )
                self.assertFalse(marker.exists())

    def test_invalid_switches_fail_before_launch(self):
        for overrides in (
            {"GRAPH": "2"},
            {"MODE": "unsupported"},
            {"ENABLE_METRICS": "2"},
            {"ENABLE_METRICS": "true"},
        ):
            with self.subTest(overrides=overrides):
                result = self.run_script("--print-command", overrides=overrides)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")
        self.assertEqual(self.run_script("--unknown").returncode, 2)
        self.assertEqual(self.run_script("--print-command", "extra").returncode, 2)

    def test_empty_required_local_settings_stop_before_launch(self):
        for key in ("MS1_HOST", "MS1_STATE", "TARGET_MODEL", "DRAFT_MODEL"):
            with self.subTest(key=key):
                result = self.run_script("--print-command", overrides={key: ""})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(key, result.stderr)
                self.assertEqual(result.stdout, "")
        result = self.run_script(
            "--print-command", overrides={"MODE": "target-only", "DRAFT_MODEL": ""}
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_bash_syntax_and_help(self):
        result = subprocess.run(
            ["bash", "-n", str(SCRIPT)], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.run_script("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("MODE=target-only", result.stdout)
        self.assertIn("MODE=nextn", result.stdout)
        self.assertIn("ENABLE_METRICS=1", result.stdout)


if __name__ == "__main__":
    unittest.main()
