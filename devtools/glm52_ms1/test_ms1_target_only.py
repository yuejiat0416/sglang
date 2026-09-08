"""Tests for the deployment helper, not substitutes for SGLang/NPU tests."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import ms1_target_only as helper


class TestTargetOnlyHelper(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # Self-contained so the tests can accompany the helper in a Git clone.
        self.cfg = dict(
            source_dir=str(self.root / "source"),
            model_path=str(self.root / "model"),
            image_reference="local-a3-cann91:test",
            cann_version="9.1",
            nnodes=2,
            tp_size=32,
            dp_size=8,
            dist_init_addr="192.0.2.1:29500",
            host="0.0.0.0",
            port=30000,
            quantization="modelslim",
            dtype="bfloat16",
            kv_cache_dtype="auto",
            context_length=4096,
            chunked_prefill_size=8192,
            max_prefill_tokens=8192,
            max_running_requests=8,
            mem_fraction_static=0.7,
            env={"HCCL_SOCKET_IFNAME": "eth0", "GLOO_SOCKET_IFNAME": "eth0"},
        )

    def git(self, source, *args):
        return subprocess.run(
            ["git", "-C", str(source), *args],
            check=True,
            capture_output=True,
            text=True,
            env={k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
        ).stdout.strip()

    def git_source(self):
        source = self.root / "source"
        (source / "python/sglang").mkdir(parents=True)
        (source / "python/sglang/__init__.py").write_text("VERSION = 1\n")
        (source / ".gitignore").write_text(
            "python/sglang/ignored.py\npython/sglang/__pycache__/\nshadow.py\n"
        )
        (source / "README.md").write_text("Fixture repository\n")
        self.git(source, "init", "-q")
        self.git(source, "config", "user.name", "MS1 test")
        self.git(source, "config", "user.email", "ms1-test@example.invalid")
        self.git(source, "config", "commit.gpgsign", "false")
        self.git(source, "add", ".")
        self.git(source, "commit", "-qm", "Fixture source")
        return source, self.git(source, "rev-parse", "HEAD")

    def server_info(self):
        info = {
            key: self.cfg[key]
            for key in (
                "model_path",
                "quantization",
                "tp_size",
                "dp_size",
                "nnodes",
                "context_length",
            )
        }
        info.update(
            device="npu",
            served_model_name="glm52-target-only",
            speculative_algorithm=None,
            speculative_draft_model_path=None,
            enable_dp_attention=True,
            cuda_graph_config={
                phase: {"backend": "disabled"} for phase in ("decode", "prefill")
            },
        )
        return info

    def test_candidate_is_target_only_eager_and_two_node(self):
        commands = [helper.build_command(self.cfg, rank) for rank in (0, 1)]
        for rank, argv in enumerate(commands):
            self.assertEqual(argv[argv.index("--node-rank") + 1], str(rank))
            self.assertFalse(
                any("speculative" in value or "draft" in value for value in argv)
            )
            for phase in ("prefill", "decode"):
                self.assertEqual(
                    argv[argv.index(f"--cuda-graph-backend-{phase}") + 1], "disabled"
                )
            self.assertIn("--enable-dp-attention", argv)
            self.assertEqual(self.cfg["max_running_requests"] // self.cfg["dp_size"], 1)

    def test_invalid_preflight_stops_before_runtime_probes(self):
        for override in (
            {"max_running_requests": 1},
            {"tp_size": 31},
            {"chunked_prefill_size": 8},
            {"model_path": "/SET_PATH"},
            {"env": {"HCCL_SOCKET_IFNAME": "eth0"}},
        ):
            with (
                self.subTest(override=override),
                patch.object(helper, "run_probe") as probe,
            ):
                cfg = dict(self.cfg, **override)
                with self.assertRaises(ValueError):
                    helper.preflight(cfg, self.root)
                probe.assert_not_called()

    def test_config_rejects_hidden_overrides(self):
        path = self.root / "config.json"
        for override in (
            {"extra_args": ["--speculative-algorithm", "DSPARK"]},
            {"env": {"PYTHONPATH": "/another/checkout"}},
        ):
            helper.write_json(path, dict(self.cfg, **override))
            with self.assertRaises(ValueError):
                helper.config_from(path)

    def test_runtime_blocks_external_model_and_rendezvous_overrides(self):
        for name in (
            "SGLANG_EXTERNAL_MODEL_PACKAGE",
            "SGLANG_DISABLED_MODEL_ARCHS",
            "SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE",
        ):
            with (
                self.subTest(name=name),
                patch.dict(os.environ, {name: "override"}, clear=True),
            ):
                with self.assertRaises(ValueError):
                    helper.runtime_env(self.cfg)
        with patch.dict(
            os.environ,
            {"PYTHONPATH": "/wrong", "SGLANG_ENABLE_SPEC_V2": "1"},
            clear=True,
        ):
            env = helper.runtime_env(self.cfg)
            self.assertEqual(
                env["PYTHONPATH"], str(Path(self.cfg["source_dir"]) / "python")
            )
            self.assertEqual(env["HF_HUB_OFFLINE"], "1")
            self.assertNotIn("SGLANG_ENABLE_SPEC_V2", env)

    def test_source_identity_detects_modified_and_extra_files(self):
        source = self.root / "source"
        source.mkdir()
        target = source / "entry.py"
        target.write_text("original\n")
        helper.write_json(
            self.root / "source-manifest.json",
            {
                "commit": helper.SOURCE_COMMIT,
                "files": {
                    "entry.py": {"kind": "file", "sha256": helper.digest(target)}
                },
            },
        )
        self.assertTrue(helper.source_identity(source)["ok"])
        target.write_text("modified\n")
        (source / "extra.py").write_text("extra\n")
        result = helper.source_identity(source)
        self.assertFalse(result["ok"])
        self.assertEqual(result["mismatches"], ["entry.py"])
        self.assertEqual(result["extra_files"], ["extra.py"])

    def test_git_source_requires_full_explicit_matching_commit(self):
        source, commit = self.git_source()
        for invalid in (None, "main", commit[:12], "g" * 40):
            with self.subTest(commit=invalid), self.assertRaises(ValueError):
                helper.source_identity(source, invalid)
        valid = helper.source_identity(source, commit.upper())
        self.assertTrue(valid["ok"])
        self.assertEqual(valid["mode"], "git")
        self.assertEqual(valid["repository_root"], str(source.resolve()))
        self.assertEqual(valid["baseline_relation"], "different_commit")
        mismatch = helper.source_identity(source, "0" * 40)
        self.assertFalse(mismatch["ok"])
        self.assertEqual(mismatch["commit"], commit)
        self.assertEqual(mismatch["expected_commit"], "0" * 40)

    def test_git_source_rejects_unstaged_staged_and_hidden_production_changes(self):
        source, commit = self.git_source()
        name = "python/sglang/__init__.py"
        (source / name).write_text("VERSION = 2\n")
        for staged in (False, True):
            if staged:
                self.git(source, "add", name)
            result = helper.source_identity(source, commit)
            self.assertFalse(result["ok"])
            self.assertEqual(result["modified_files"], [name])
        self.git(source, "reset", "--hard", "HEAD")
        self.git(source, "update-index", "--assume-unchanged", name)
        (source / name).write_text("VERSION = 3\n")
        result = helper.source_identity(source, commit)
        self.assertFalse(result["ok"])
        self.assertEqual(result["hidden_index_entries"], [name])

    def test_git_source_rejects_untracked_and_ignored_importable_files(self):
        source, commit = self.git_source()
        for name in (
            "python/sglang/new.py",
            "python/sglang/ignored.py",
            "python/sglang/orphan.pyc",
            "python/sglang/__pycache__/extra.py",
            "shadow.py",
            "extension.so",
        ):
            with self.subTest(name=name):
                file = source / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(b"test fixture")
                result = helper.source_identity(source, commit)
                self.assertFalse(result["ok"])
                self.assertEqual(result["extra_files"], [name])
                file.unlink()

    def test_git_source_preserves_scoped_safe_directory_environment(self):
        source, commit = self.git_source()
        with patch.dict(
            os.environ,
            {
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "safe.directory",
                "GIT_CONFIG_VALUE_0": str(source),
                "GIT_DIR": str(self.root / "wrong-repository"),
                "GIT_INDEX_FILE": str(self.root / "wrong-index"),
            },
        ):
            self.assertIn(
                str(source),
                helper.git_output(
                    source, "config", "--get-all", "safe.directory"
                ).splitlines(),
            )
            self.assertTrue(helper.source_identity(source, commit)["ok"])

    def test_dirty_git_preflight_stops_before_runtime_imports(self):
        source, commit = self.git_source()
        self.cfg["source_commit"] = commit
        (source / "python/sglang/__init__.py").write_text("VERSION = 2\n")
        with patch.object(helper, "run_probe") as probe:
            self.assertEqual(helper.preflight(self.cfg, self.root), 1)
            probe.assert_not_called()
        report = json.loads((self.root / "preflight.json").read_text())
        self.assertEqual(report["checks"]["source"]["commit"], commit)
        self.assertFalse(report["checks"]["source"]["ok"])

    def test_command_cli_records_real_git_commit_without_npu(self):
        _, commit = self.git_source()
        self.cfg["source_commit"] = commit
        config = self.root / "target-only.local.json"
        helper.write_json(config, self.cfg)
        out = self.root / "command-evidence"
        result = subprocess.run(
            [
                sys.executable,
                str(Path(helper.__file__).resolve()),
                "command",
                "--config",
                str(config),
                "--out",
                str(out),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads((out / "launch-plan.json").read_text())
        self.assertEqual(plan["source_commit"], commit)
        self.assertEqual(plan["source"]["commit"], commit)
        self.assertEqual(plan["source"]["baseline_relation"], "different_commit")
        self.assertIn("sglang.launch_server", plan["argv"])

    def test_git_source_allows_external_evidence_and_regular_caches(self):
        source, commit = self.git_source()
        for name in (
            "evidence/report.json",
            "target-only.local.json",
            "cache/result.json",
        ):
            file = self.root / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("{}")
        (source / "README.md").write_text("Local notes\n")
        (source / "local-report.json").write_text("{}")
        cache = source / "python/sglang/__pycache__"
        cache.mkdir()
        (cache / "__init__.cpython-312.pyc").write_bytes(b"test fixture")
        self.assertTrue(helper.source_identity(source, commit)["ok"])

    def test_git_worktree_supported_but_nested_source_root_rejected(self):
        source, commit = self.git_source()
        linked = self.root / "linked-worktree"
        self.git(source, "worktree", "add", "--detach", str(linked), commit)
        self.assertTrue((linked / ".git").is_file())
        self.assertTrue(helper.source_identity(linked, commit)["ok"])
        with self.assertRaisesRegex(ValueError, "repository root"):
            helper.source_identity(source / "python", commit)

    def test_git_source_rejects_external_production_symlink(self):
        source, _ = self.git_source()
        external = self.root / "external.py"
        external.write_text("VERSION = 99\n")
        name = "python/sglang/external.py"
        (source / name).symlink_to(external)
        self.git(source, "add", name)
        self.git(source, "commit", "-qm", "External link fixture")
        result = helper.source_identity(source, self.git(source, "rev-parse", "HEAD"))
        self.assertFalse(result["ok"])
        self.assertEqual(result["unsafe_symlinks"], [name])

    def test_config_validates_source_commit(self):
        path = self.root / "config.json"
        for invalid in (None, "main", "abcd", "x" * 40):
            helper.write_json(path, dict(self.cfg, source_commit=invalid))
            with self.subTest(commit=invalid), self.assertRaises(ValueError):
                helper.config_from(path)
        helper.write_json(
            path, dict(self.cfg, source_commit=helper.SOURCE_COMMIT.upper())
        )
        self.assertEqual(
            helper.config_from(path)["source_commit"], helper.SOURCE_COMMIT
        )

    def test_metadata_cannot_pass_without_quantization_and_tokenizer(self):
        model = self.root / "model"
        model.mkdir()
        helper.write_json(
            model / "config.json", {"architectures": ["GlmMoeDsaForCausalLM"]}
        )
        # This test exercises file presence only, as the real preflight states.
        (model / "model.safetensors").write_bytes(b"test fixture")
        self.assertFalse(helper.metadata_check(model)["ok"])
        helper.write_json(model / "quant_model_description.json", {})
        (model / "tokenizer.json").write_text("{}")
        self.assertTrue(helper.metadata_check(model)["ok"])
        helper.write_json(
            model / "model.safetensors.index.json",
            {"weight_map": {"w": "missing.safetensors"}},
        )
        self.assertFalse(helper.metadata_check(model)["ok"])

    def test_smoke_rejects_wrong_model_device_spec_and_graph(self):
        helper.verify_server(self.server_info(), self.cfg)
        for override in (
            {"model_path": "/wrong"},
            {"device": "cuda"},
            {"tp_size": 16},
            {"speculative_algorithm": "NEXTN"},
            {"cuda_graph_config": {}},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                helper.verify_server(dict(self.server_info(), **override), self.cfg)

    def test_smoke_records_only_normal_completions(self):
        source_dir, commit = self.git_source()
        self.cfg["source_commit"] = commit
        source = helper.source_identity(source_dir, commit)
        evidence = self.root / "launch"
        evidence.mkdir()
        helper.write_json(
            evidence / "launch-plan.json",
            {"config": self.cfg, "source_commit": commit, "source": source},
        )
        helper.write_json(
            evidence / "preflight.json",
            {"preflight_ok": True, "checks": {"source": source}},
        )
        response = {
            "text": "output",
            "meta_info": {"completion_tokens": 1, "finish_reason": {"type": "length"}},
        }
        ok_dir = self.root / "ok"
        ok_dir.mkdir()
        with patch.object(
            helper,
            "request_json",
            side_effect=[None, self.server_info(), response, response, response],
        ):
            helper.smoke("http://127.0.0.1:30000", ok_dir, self.cfg, evidence)
        self.assertEqual(
            json.loads((ok_dir / "smoke-summary.json").read_text())["status"],
            "SMOKE_ONLY_PASS",
        )
        summary = json.loads((ok_dir / "smoke-summary.json").read_text())
        self.assertEqual(summary["source_commit"], commit)
        self.assertEqual(summary["source"]["baseline_relation"], "different_commit")
        failed = copy.deepcopy(response)
        failed["meta_info"]["finish_reason"]["type"] = "abort"
        bad_dir = self.root / "bad"
        bad_dir.mkdir()
        with patch.object(
            helper, "request_json", side_effect=[None, self.server_info(), failed]
        ):
            with self.assertRaises(ValueError):
                helper.smoke("http://127.0.0.1:30000", bad_dir, self.cfg, evidence)
        self.assertFalse((bad_dir / "smoke-summary.json").exists())

    def test_smoke_rejects_evidence_for_another_commit_before_http(self):
        source_dir, commit = self.git_source()
        self.cfg["source_commit"] = commit
        source = helper.source_identity(source_dir, commit)
        evidence = self.root / "launch"
        evidence.mkdir()
        helper.write_json(
            evidence / "launch-plan.json",
            {"config": self.cfg, "source_commit": commit, "source": source},
        )
        helper.write_json(
            evidence / "preflight.json",
            {
                "preflight_ok": True,
                "checks": {"source": dict(source, commit=helper.SOURCE_COMMIT)},
            },
        )
        with patch.object(helper, "request_json") as request:
            with self.assertRaises(ValueError):
                helper.smoke("http://127.0.0.1:30000", self.root, self.cfg, evidence)
            request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
