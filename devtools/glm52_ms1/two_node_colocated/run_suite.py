#!/usr/bin/env python3
"""Print the five-mode plan or run a selected client stage on an existing server."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from config import MODES, load_config

HERE = Path(__file__).resolve().parent


def commands(args, cfg):
    prefix = [sys.executable]
    config = ["--config", str(args.config.resolve())]
    data = (
        Path(args.data_dir)
        if args.data_dir
        else Path(cfg["state"]) / "datasets/two-node-colocated"
    )
    if args.stage in {"smoke", "accuracy"}:
        datasets = ("gsm8k",) if args.stage == "smoke" else ("gsm8k", "gpqa")
        return [
            prefix
            + [
                str(HERE / "bench_accuracy.py"),
                args.mode,
                *config,
                "--fixture",
                str(data / f"{dataset}.json"),
                "--limit",
                "1" if args.stage == "smoke" else str(args.limit),
                "--max-tokens",
                "64" if args.stage == "smoke" else str(args.max_tokens),
                "--concurrency",
                "1",
            ]
            for dataset in datasets
        ]
    command = prefix + [
        str(HERE / "bench_prefix.py"),
        "quick" if args.stage == "prefix" else "load",
        *config,
        "--mode",
        args.mode,
        "--cache-hit",
        args.cache_hit,
    ]
    if args.stage == "load":
        command += [
            "--num-prompts",
            str(args.num_prompts),
            "--concurrency",
            str(args.concurrency),
        ]
        if args.duration_seconds is not None:
            command += ["--duration-seconds", str(args.duration_seconds)]
    return [command]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, default="dspark-eager")
    parser.add_argument(
        "--stage", choices=("smoke", "accuracy", "prefix", "load"), default="smoke"
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--cache-hit", choices=("all", "0", "50", "90"), default="all")
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--duration-seconds", type=float)
    args = parser.parse_args()
    if args.limit < 0 or min(args.max_tokens, args.num_prompts, args.concurrency) < 1:
        parser.error(
            "limit must be >=0; token/count/concurrency options must be positive"
        )
    cfg = load_config(args.config)
    if args.action == "plan":
        print(
            "# Candidate plan only. The owner switches modes on BOTH nodes; this tool does not restart them."
        )
        for mode in (
            "target-eager",
            "target-graph",
            "dspark-eager",
            "dspark-graph",
            "nextn-graph",
        ):
            print(
                f"\n# {mode}: start rank0 and rank1 in separate server terminals, then wait for readiness."
            )
            for rank in (0, 1):
                print(
                    shlex.join(
                        [
                            sys.executable,
                            str(HERE / "launch.py"),
                            mode,
                            "--config",
                            str(args.config.resolve()),
                            "--rank",
                            str(rank),
                            "--execute",
                        ]
                    )
                )
            args.mode = mode
            for stage in ("smoke", "accuracy", "prefix", "load"):
                args.stage = stage
                print(
                    f"# {stage}: review results before the next stage; load duration remains unfrozen."
                )
                for command in commands(args, cfg):
                    print(shlex.join(command))
        return 0
    selected = commands(args, cfg)
    # Missing GPQA must be reported before the accuracy stage sends GSM traffic.
    for command in selected:
        if "--fixture" in command:
            path = Path(command[command.index("--fixture") + 1])
            if (
                not path.is_file()
                or json.loads(path.read_text()).get("status") != "DATASET_PREPARED"
            ):
                print(f"BLOCKED: prepare the local fixture first: {path}")
                return 2
    for command in selected:
        print(shlex.join(command), flush=True)
        result = subprocess.call(command)
        if result != 0:
            return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
