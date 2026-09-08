# Temporary MS1 validation tools

This directory supports the temporary `sync/glm52-dspark-ms1` development branch.
It is separate from the SGLang feature commits intended for upstream review.

## Native 192 experiment (single NPU)

`probe_native192.py` tests the installed `sgl_kernel_npu` fused kernel with real
192-element blocks. `native192_probe_ops.py` contains three small compiler probes.
Neither file is imported by SGLang. The tool does not modify the installed host
function, install packages, change compiler defaults, start distributed workers,
or load a model. Its baseline is the agreed Triton-Ascend **3.2.2** image
(`triton.__version__` may correctly report **3.2.0**).

In the existing container, use the same initialized CANN/ATB environment as the
working target-only service. Select an available logical NPU; if your target
service occupies the whole machine, stop your service first. From the checkout:

```bash
cd /home/tyj/glm52/sglang
python3 devtools/glm52_ms1/probe_native192.py
```

The default is logical NPU 0. Use `--device 1` (or another available index) to
change it. These are indices in the process's visible device list. The first
run includes JIT compilation and can take several minutes. No second machine is
needed. Results are saved automatically under
`/home/tyj/glm52-ms1/evidence/native192-<UTC-time>-<pid>/`, outside Git.

The first command runs these checks in order and stops at the first failure:

1. An existing **128-dimensional public-host call** as an environment control.
2. Actual-length 192 load/store, reshape/reduction, and 96+96 slice/rotation.
3. The installed full fusion, with `Q_BLOCK_SIZE=KV_BLOCK_SIZE=192` and the
   original grid formula. It uses BF16 Q/K/V and norm weights, FP32 sin/cos,
   full NeoX RoPE, no bias, and epsilon `1e-5`. Nine token/head combinations,
   a guaranteed second row-loop iteration, zero and near-zero input are checked.

The script calls the underlying kernel directly for the experiment. The public
host's current power-of-two assertion stays intact. Q/K are compared with an
independent CPU FP32 formula without intermediate BF16 rounding; V must be
bitwise identical. The diagnostic absolute tolerance `5e-2` comes from the
existing kernel's test, **not** a new GLM model-quality acceptance threshold.
Per-output error metrics are saved for review.

After the first command passes, optionally run graph validation and diagnostic
timing. This reruns the eager checks before capturing one `[8, 4, 192]` case:

```bash
python3 devtools/glm52_ms1/probe_native192.py --graph --benchmark
```

Graph checks use one side stream, stable tensor addresses, and two changed-input
replays, poisoning output buffers before each replay. Capture uses
`auto_dispatch_capture=True`, as in the current SGLang NPU graph backend. Event
timings exclude JIT/capture, but include gaps on the measured stream; they are
diagnostic measurements, not an end-to-end speedup or performance acceptance.

Result interpretation:

- `EAGER_DIAGNOSTIC_PASS_GRAPH_NOT_RUN`: eager checks passed; graph was not run.
- `EAGER_AND_GRAPH_DIAGNOSTIC_PASS`: eager plus this single-operator graph passed.
- `FAILED`: inspect `active_stage`, the metrics, and `traceback.txt`.
- `RUNNING` left behind after a crash/interruption: incomplete evidence, never a pass.

Return `report.json`; on failure also return `traceback.txt` and the terminal
error. `run.log`, the installed kernel source, and the actual `arange` source are
saved alongside them. Versions, selected-device properties, compiler metadata
and source hashes identify what actually ran. A source different from the local
audit is recorded, not mislabeled as the audited commit. No automatic fallback
to 2x128 is performed: first identify whether a failure is environmental,
compiler-related, numerical, or specific to graph execution.

New reports record `kernel_function_ast_schema` with the AST fingerprint. This
schema ignores source locations and empty `type_params` fields added in Python
3.12; nonempty type parameters and calculation changes remain significant.
`kernel_function_source_sha256` separately hashes the saved UTF-8 source after
dedenting, normalizing line endings to LF and keeping one final newline, so
comments remain available for audit. Older reports without the schema used
Python-version-dependent AST dumps; their hashes must not be compared directly
with new ones. Existing reports are not rewritten.

Local tool tests (CPU PyTorch; no Triton/NPU installation required):

```bash
python3 devtools/glm52_ms1/test_probe_native192.py -v
```

These tests do not count as NPU evidence. This experiment does not establish
SGLang DSpark, actual TP/DP execution, A5 support, or milestone completion. The
temporary tools stay on the sync branch and are not promoted as feature code.

## Existing container and target-only tools

`start_container.sh` contains the team's A3 Docker command with each device
listed explicitly. Run it on the host with Bash; it only creates an interactive
container, without model loading or preflight. For the second node, set
`CONTAINER_NAME=tyj-glm52-ms1-target-node1` when invoking it. The image and host
mounts are visible in the script; after entering the container, change directory
to `/home/tyj/glm52/sglang` before using the commands below.

The script mounts the host's entire `/home` at `/home`, following the team's
container command. Host and container paths are identical: the target is at
`/home/weights/GLM-5.2-w8a8`, and personal configuration, cache, and logs remain
under `/home/tyj/glm52-ms1`. The separate `/workspace/weight` alias is no longer
created. Driver, firmware, and the other system mounts are unchanged.

An existing container retains the mounts selected when it was created; pulling
this script or restarting that container does not update them. After pulling the
new version on the host, create a new container without deleting the old one:

```bash
cd /home/tyj/glm52/sglang
CONTAINER_NAME=tyj-glm52-ms1-home-node0 bash devtools/glm52_ms1/start_container.sh
```

Use this new name with `docker exec` and `docker inspect`. In an existing local
configuration, change `model_path` to `/home/weights/GLM-5.2-w8a8` and update
`source_commit` to the reviewed commit being run. Copying the example with
`cp -n` will not update an existing configuration. Files already stored under
the host-mounted `/home/tyj` remain available in the new container; files saved
only inside the old container remain there.

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
The container itself can be used on one node, but this helper's launch recipe
still requires two nodes. The mount update does not add a single-node model
launch recipe.

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
