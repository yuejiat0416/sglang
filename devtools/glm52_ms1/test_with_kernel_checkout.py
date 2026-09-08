"""CPU tests of package selection; dummy libraries never exercise an NPU."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("with_kernel_checkout.py")
SPEC = importlib.util.spec_from_file_location("with_kernel_checkout", SCRIPT)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)

COMMAND = r"""
import importlib
import json
import multiprocessing as mp
from pathlib import Path
import sys

def inspect_package():
    import sgl_kernel_npu as package
    from sgl_kernel_npu.norm import other_norm
    from sgl_kernel_npu import other_helper
    module = importlib.import_module('sgl_kernel_npu.norm.split_qkv_rmsnorm_rope')
    lib = Path(package.__file__).parent / 'lib/libsgl_kernel_npu.so'
    return {
        'marker': module.MARKER,
        'module': module.__file__,
        'package': package.__file__,
        'library': str(lib.resolve()),
        'library_contents': package.LIBRARY_CONTENTS,
        'norm_helper': str(Path(other_norm.__file__).resolve()),
        'root_helper': str(Path(other_helper.__file__).resolve()),
    }

def worker(queue):
    queue.put(inspect_package())

if __name__ == '__main__':
    parent = inspect_package()
    # A subsequent checkout edit must not change the frozen service copy.
    Path(sys.argv[2]).write_text('MARKER = "changed-after-launch"\n')
    context = mp.get_context('spawn')
    queue = context.Queue()
    process = context.Process(target=worker, args=(queue,))
    process.start()
    child = queue.get(timeout=15)
    process.join(timeout=15)
    assert process.exitcode == 0
    Path(sys.argv[1]).write_text(json.dumps({'parent': parent, 'spawn': child}))
"""


class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.site = self.root / "installed"
        self.package = self.site / "sgl_kernel_npu"
        (self.package / "norm").mkdir(parents=True)
        (self.package / "utils").mkdir()
        (self.package / "lib").mkdir()
        (self.package / "__init__.py").write_text(
            "from pathlib import Path\n"
            "LIBRARY_CONTENTS = (Path(__file__).parent / 'lib/libsgl_kernel_npu.so').read_text()\n"
        )
        (self.package / "lib/libsgl_kernel_npu.so").write_text("dummy-installed-binary")
        (self.package / "norm/__init__.py").write_text("")
        (self.package / "utils/__init__.py").write_text("")
        (self.package / "utils/triton_utils.py").write_text(
            "def get_device_properties():\n"
            "    raise AssertionError('device initialization must not run')\n"
        )
        (self.package / "other_helper.py").write_text("MARKER = 'root-helper'\n")
        (self.package / "norm/other_norm.py").write_text("MARKER = 'norm-helper'\n")
        original = "from sgl_kernel_npu.utils.triton_utils import get_device_properties\nMARKER = 'installed'\n"
        (self.package / "norm" / helper.SOURCE.name).write_text(original)
        self.repo = self.root / "kernel-repo"
        self.source = self.repo / helper.SOURCE
        self.source.parent.mkdir(parents=True)
        self.source.write_text(original.replace("'installed'", "'candidate'"))
        self.git("init", "--quiet")
        self.git("add", ".")
        self.git(
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.com",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        )
        self.state = self.root / "state"
        self.env = os.environ.copy()
        self.env["PYTHONPATH"] = str(self.site)
        self.env["PYTHONDONTWRITEBYTECODE"] = "1"

    def git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.repo), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def run_helper(self, command, env=None, repo=None):
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--kernel-repo",
                str(repo or self.repo),
                "--state-dir",
                str(self.state),
                "--",
                *command,
            ],
            cwd=self.root,
            env=env or self.env,
            capture_output=True,
            text=True,
            timeout=40,
        )

    def manifests(self):
        return [
            json.loads(path.read_text())
            for path in sorted(self.state.glob("*/manifest.json"))
        ]

    def snapshot(self):
        return {
            str(p.relative_to(self.site)): p.read_bytes()
            for p in self.site.rglob("*")
            if p.is_file()
        }

    def test_fresh_import_and_spawn_use_frozen_candidate(self):
        before = self.snapshot()
        command_file = self.root / "command.py"
        command_file.write_text(COMMAND)
        output = self.root / "command.json"
        result = self.run_helper(
            [sys.executable, str(command_file), str(output), str(self.source)]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        observations = json.loads(output.read_text())
        report = self.manifests()[0]
        self.assertEqual(report["status"], "READY_TO_EXEC_COMMAND")
        self.assertEqual(report["command_result"], "NOT_OBSERVED_AFTER_EXEC")
        self.assertEqual(report["kernel_git"]["head"], self.git("rev-parse", "HEAD"))
        self.assertEqual(report["kernel_git"]["status_porcelain"], "")
        self.assertEqual(observations["parent"], observations["spawn"])
        for observation in observations.values():
            self.assertEqual(observation["marker"], "candidate")
            self.assertEqual(observation["module"], report["candidate_frozen"])
            self.assertEqual(observation["library"], str(self.package / helper.LIBRARY))
            self.assertEqual(observation["library_contents"], "dummy-installed-binary")
            self.assertEqual(
                observation["root_helper"], str(self.package / "other_helper.py")
            )
            self.assertEqual(
                observation["norm_helper"], str(self.package / "norm/other_norm.py")
            )
        self.assertEqual(before, self.snapshot())
        self.assertFalse(list(self.site.rglob("__pycache__")))
        self.assertFalse(Path(report["candidate_frozen"]).is_symlink())
        self.assertEqual(
            report["candidate_sha256"],
            helper.sha256(Path(report["candidate_frozen"]).read_bytes()),
        )
        self.assertEqual(
            report["import_check"]["helper_resolved"], str(self.package / helper.HELPER)
        )
        self.assertEqual(Path(report["run_dir"]).stat().st_mode & 0o777, 0o700)

    def test_repeated_runs_get_new_directories_and_record_dirty_source(self):
        self.source.write_text(self.source.read_text() + "# uncommitted candidate\n")
        for _ in range(2):
            result = self.run_helper([sys.executable, "-c", "pass"])
            self.assertEqual(result.returncode, 0, result.stderr)
        reports = self.manifests()
        self.assertEqual(len(reports), 2)
        self.assertNotEqual(reports[0]["run_dir"], reports[1]["run_dir"])
        for report in reports:
            self.assertIn(str(helper.SOURCE), report["kernel_git"]["status_porcelain"])
            self.assertEqual(
                report["child_pythonpath"],
                report["overlay"] + os.pathsep + str(self.site),
            )

    def test_find_spec_does_not_execute_installed_package(self):
        (self.package / "__init__.py").write_text(
            "raise RuntimeError('must not import package')\n"
        )
        code = (
            "import runpy, sys; ns=runpy.run_path(sys.argv[1]); "
            "print(ns['installed_package_path']()); "
            "assert 'sgl_kernel_npu' not in sys.modules; assert 'torch' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-S", "-c", code, str(SCRIPT)],
            env=self.env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(self.package), result.stdout)

    def test_missing_binary_fails_before_command_and_keeps_evidence(self):
        (self.package / helper.LIBRARY).unlink()
        marker = self.root / "should-not-exist"
        result = self.run_helper(
            [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"]
        )
        self.assertEqual(result.returncode, 1)
        self.assertFalse(marker.exists())
        report = self.manifests()[0]
        self.assertEqual(report["status"], "SETUP_OR_EXEC_FAILED")
        self.assertIn("binary library is missing", report["error"])
        self.assertTrue((Path(report["run_dir"]) / "traceback.txt").is_file())

    def test_norm_init_reexports_candidate_from_overlay(self):
        (self.package / "norm/__init__.py").write_text(
            "# An ordinary package initializer may expose this public symbol.\n"
            "from .split_qkv_rmsnorm_rope import MARKER as EXPORTED\n"
        )
        result = self.run_helper(
            [
                sys.executable,
                "-c",
                "from sgl_kernel_npu.norm import EXPORTED; print('EXPORTED=' + EXPORTED)",
            ]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("EXPORTED=candidate", result.stdout)

    def test_bad_candidate_import_keeps_stderr_and_does_not_exec(self):
        self.source.write_text("raise RuntimeError('candidate-import-failure')\n")
        result = self.run_helper(
            [sys.executable, "-c", "raise AssertionError('must not exec')"]
        )
        self.assertEqual(result.returncode, 1)
        report = self.manifests()[0]
        self.assertEqual(report["status"], "SETUP_OR_EXEC_FAILED")
        self.assertIn(
            "candidate-import-failure",
            (Path(report["run_dir"]) / "import.stderr.txt").read_text(),
        )

    def test_redirected_import_is_rejected_even_with_python_optimization(self):
        init = self.package / "__init__.py"
        init.write_text(init.read_text() + f"__path__ = [{str(self.package)!r}]\n")
        env = dict(self.env, PYTHONOPTIMIZE="1")
        result = self.run_helper([sys.executable, "-c", "pass"], env=env)
        self.assertEqual(result.returncode, 1)
        report = self.manifests()[0]
        self.assertEqual(report["status"], "SETUP_OR_EXEC_FAILED")
        self.assertIn(
            "Candidate import selected:",
            (Path(report["run_dir"]) / "import.stderr.txt").read_text(),
        )

    def test_exec_failure_is_recorded_but_child_failure_is_not_success(self):
        result = self.run_helper([str(self.root / "nonexistent-command")])
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.manifests()[0]["status"], "SETUP_OR_EXEC_FAILED")
        result = self.run_helper([sys.executable, "-c", "raise SystemExit(7)"])
        self.assertEqual(result.returncode, 7)
        self.assertIn("READY_TO_EXEC_COMMAND", {r["status"] for r in self.manifests()})
        self.assertTrue(
            all(
                r["command_result"] == "NOT_OBSERVED_AFTER_EXEC"
                for r in self.manifests()
            )
        )

    def test_help_works_without_device_packages(self):
        result = subprocess.run(
            [sys.executable, "-S", str(SCRIPT), "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--kernel-repo", result.stdout)


if __name__ == "__main__":
    unittest.main()
