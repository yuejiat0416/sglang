#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Start the existing recipe with an explicit, private proposal observer.

Temporary sync branch only. The current service must first be stopped by its
owner. This wrapper does not stop processes, install packages or change recipes.
"""

import argparse
import importlib.util
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dspark_proposal_snapshot import ENV, file_hash, require, write_json

DEFAULT_STATE = Path("/home/tyj/glm52-ms1")


def prepare(state, scripts):
    require(
        ENV not in os.environ and "GLM52_CONTEXT_SNAPSHOT_CONFIG" not in os.environ,
        "Nested snapshot launch is not supported",
    )
    # Do not silently mask another startup hook. This check does not import it.
    require(
        importlib.util.find_spec("sitecustomize") is None,
        "Existing sitecustomize needs explicit integration",
    )
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    run = state / "evidence" / f"proposal-snapshot-{run_id}"
    boot = run / "bootstrap"
    boot.mkdir(parents=True)
    observer = scripts / "dspark_proposal_snapshot.py"
    for filename in (observer.name, "dspark_context_snapshot.py"):
        (boot / filename).write_bytes((scripts / filename).read_bytes())
    (boot / "sitecustomize.py").write_text(
        "from dspark_proposal_snapshot import install\ninstall()\n"
    )
    repo = scripts.parents[1]
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    config = {
        "run_dir": str(run),
        "run_id": run_id,
        "rid": "ms1-proposal-" + uuid.uuid4().hex,
        "cache_salt": "ms1-proposal-cache-" + uuid.uuid4().hex,
        "max_bytes": 64 * 1024**2,
        "repo": str(repo),
        "git_head": head,
        "observer_sha256": file_hash(observer),
        "client_sha256": file_hash(scripts / "probe_proposal_snapshot.py"),
        "reference_sha256": file_hash(scripts / "dspark_local_reference.py"),
        "git_status": subprocess.check_output(
            ["git", "status", "--porcelain=v1"], cwd=repo, text=True
        ),
        "launcher_sha256": file_hash(scripts / "with_proposal_snapshot.py"),
        "scope": "first proposal, layer0, rank0, before output projection",
        "tool_hashes": {
            name: file_hash(scripts / name)
            for name in (
                "dspark_proposal_snapshot.py",
                "with_proposal_snapshot.py",
                "probe_proposal_snapshot.py",
                "dspark_attention_reference.py",
                "dspark_context_snapshot.py",
                "probe_context_snapshot.py",
                "dspark_local_reference.py",
                "probe_quarot_vocab.py",
                "collect_acceptance_trace.py",
            )
        },
    }
    write_json(run / "config.json", config)
    write_json(
        state / "proposal-snapshot-current.json", {"config": str(run / "config.json")}
    )
    env = os.environ.copy()
    env[ENV] = str(run / "config.json")
    env["PYTHONPATH"] = str(boot) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    # Preserve the established diagnostic settings, explicitly for this launch.
    env.update(
        SGLANG_NPU_GLM_DSPARK_QUAROT="original",
        SGLANG_ENABLE_FAST_INPUT_LOGPROBS="0",
        SGLANG_DSPARK_DEBUG_DUMP="core,reqs",
        MODE="dspark",
        GRAPH="0",
        MS1_STATE=str(state),
    )
    return config, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent
    config, env = prepare(args.state_dir.expanduser().resolve(), scripts)
    print(f"Proposal snapshot evidence: {config['run_dir']}", flush=True)
    print(
        "Start original static/eager recipe; use probe_proposal_snapshot.py in a second terminal when ready.",
        flush=True,
    )
    os.execvpe("bash", ["bash", str(scripts / "single_dspark_static.sh")], env)


if __name__ == "__main__":
    main()
