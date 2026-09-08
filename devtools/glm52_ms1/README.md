# Temporary target-only validation tools

This directory supports the temporary `sync/glm52-dspark-ms1` development branch.
It is separate from the SGLang feature commits intended for upstream review.

`start_container.sh` contains the team's A3 Docker command with each device
listed explicitly. Run it on the host with Bash; it only creates an interactive
container, without model loading or preflight. For the second node, set
`CONTAINER_NAME=tyj-glm52-ms1-target-node1` when invoking it. The image and host
mounts are visible in the script; after entering the container, change directory
to `/home/tyj/glm52` before using the commands below.

The helper provides environment preflight, a launch command preview, foreground
launch, and three short raw generation requests. It does not install dependencies
or download model weights. Use an existing Ascend environment and local weights.

## Configure a run

Copy `target-only.example.json` to a directory outside the checkout. Set the
absolute `source_dir`, the exact full Git `source_commit` selected for testing,
the container-visible model path, image reference, rendezvous address, and
network interfaces. Run both nodes at the same reviewed commit. Git mode rejects
source changes instead of silently testing a different production tree.

The initial recipe is two nodes with TP32/DP8, ModelSlim target weights, BF16
computation, and prefill/decode graphs disabled. Each node needs 16 visible NPU
logical devices. This is a diagnostic configuration, not a performance result.
Coordinate device and port use before starting a model on a shared server.

```bash
python devtools/glm52_ms1/ms1_target_only.py preflight \
  --config /absolute/path/target-only.local.json --node-rank 0 \
  --out /absolute/path/evidence/node0-preflight-01

python devtools/glm52_ms1/ms1_target_only.py command \
  --config /absolute/path/target-only.local.json --node-rank 0 \
  --out /absolute/path/evidence/node0-command-01

python devtools/glm52_ms1/ms1_target_only.py launch \
  --config /absolute/path/target-only.local.json --node-rank 0 \
  --out /absolute/path/evidence/node0-launch-01
```

Use rank 1 and separate evidence directories on the second node. `launch` sets
the source import path and offline environment, repeats preflight, then runs the
server in the foreground. Use Ctrl-C to stop this launch. Command preview alone
does not set the runtime environment in another shell.

When node 0 is ready, run in another terminal in the same environment:

```bash
python devtools/glm52_ms1/ms1_target_only.py smoke \
  --config /absolute/path/target-only.local.json \
  --launch-evidence /absolute/path/evidence/node0-launch-01 \
  --base-url http://127.0.0.1:30000 \
  --out /absolute/path/evidence/node0-smoke-01
```

`SMOKE_ONLY_PASS` means that the three basic requests returned nonempty text with
normal completion metadata. It does not establish task accuracy, DSpark support,
performance, or milestone completion. Keep full configuration and logs in the
execution environment, and correlate the checked endpoint with the launched
process.

Stop the running model before pulling another commit into the same checkout.
After `git pull --ff-only`, verify the delivered SHA, update `source_commit` in
the local configuration, and use fresh evidence directories. Do not commit local
configuration, proxy credentials, caches, or run logs.

## Test the helper

```bash
python devtools/glm52_ms1/test_ms1_target_only.py -v
```

These tests exercise the helper and real temporary Git repositories. Model/NPU
probe and HTTP test doubles do not count as actual model or device validation.
