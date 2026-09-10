# Temporary colocated validation tools

These tools belong to the personal sync branch. They are not production changes
or the formal SGLang/kernel PR test suite. The owner runs all NPU commands.

## Single-node entry points

- [GSM8K five-mode guide](GSM8K_MODES.md): target eager/graph, DSpark static
  eager/graph, NEXTN graph, using the same ten GSM8K questions.
- [GSP 131072/1024 guide](GSP_PREFIX.md): separate 0/50/90% prefix-cache cases,
  exact token lengths, actual cache counts and per-request speculative A/P/N.
- `single_dspark_static.sh`: local server launcher with owner-supplied model,
  service and working-directory parameters.
- `start_container.sh`: optional container creation with owner-supplied image,
  container name and cache directory; explicit NPU devices and core host mounts.
- `ms1_target_only.py` / `target-only.example.json`: legacy target-only
  configuration/preflight entry point. Populate the local example before use.

## Two-node entry points

[Two-node colocated guide](two_node_colocated/README.md) covers paired environment
checks, the same five modes, separate GSM8K/GPQA samples, target-only accuracy
comparison, DP-aware prefix-cache/load cases, evidence inventory and packaging.
All node addresses, NICs, personal directories, model paths and service names
belong in the owner's local configuration. Example placeholders are not deployable.

## Scope and results

Scripts collect evidence; successful collection is not model, graph, performance
or formal acceptance qualification. Keep failures, truncated outputs, actual
cache counts, per-request counters, both node launch logs and model provenance.
The current short dataset checks use ten questions each; GPQA requires a local
dataset supplied by the owner. No automatic download, SSH or server restart is
performed by a benchmark client.

The archive contains an explicit allowlist of serving-test tools, dependencies
and licenses. It excludes local configurations, weights, GPQA data and evidence.
Older diagnosis tools may remain elsewhere in this checkout; they are not part
of this sanitized serving-test archive. Historical collaboration records are
maintained separately in the local project records.
