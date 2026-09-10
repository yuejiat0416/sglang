"""Configuration only: no network, model import, or remote action."""

import copy
import ipaddress
import json
import re
from pathlib import Path

MODES = ("dspark-eager", "dspark-graph", "target-eager", "target-graph", "nextn-graph")


def expected_algorithm(mode):
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    return (
        "DSPARK"
        if mode.startswith("dspark-")
        else "EAGLE"
        if mode == "nextn-graph"
        else None
    )


def positive_int(value):
    return type(value) is int and value > 0


def require_local_value(value, field):
    """Do not turn a shareable template into an executable local configuration."""
    if not isinstance(value, str) or not value.strip() or "__FILL_" in value:
        raise ValueError(f"Fill {field} in a private local configuration")


def validate_config(raw):
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a JSON object")
    cfg = copy.deepcopy(raw)
    if cfg.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    for key in ("repo", "kernel_repo", "state", "target_model", "draft_model"):
        require_local_value(cfg.get(key), key)
        if not Path(cfg[key]).is_absolute():
            raise ValueError(f"{key} must be an absolute container path")
    cfg.setdefault("tokenizer", cfg["target_model"])
    require_local_value(cfg["tokenizer"], "tokenizer")
    if not Path(cfg["tokenizer"]).is_absolute():
        raise ValueError("tokenizer must be an absolute local path")
    for key in ("python", "served_model_name"):
        require_local_value(cfg.get(key), key)
    for key in (
        "port",
        "dist_port",
        "tp_size",
        "dp_size",
        "nnodes",
        "npus_per_node",
        "context_length",
        "max_prefill_tokens",
        "max_running_requests",
    ):
        if not positive_int(cfg.get(key)):
            raise ValueError(f"{key} must be a positive integer")
    if cfg["nnodes"] != 2 or cfg["npus_per_node"] != 16 or cfg["tp_size"] != 32:
        raise ValueError(
            "This temporary deployment profile covers two nodes / 16 NPUs each / TP32"
        )
    if cfg["tp_size"] % cfg["dp_size"]:
        raise ValueError("tp_size must be divisible by dp_size")
    if cfg["max_running_requests"] < cfg["dp_size"]:
        raise ValueError(
            "max_running_requests must be at least dp_size; the current runtime divides this limit across attention DP workers"
        )
    if cfg["context_length"] < 132098:
        raise ValueError(
            "context_length must be at least 132098 for the current 131072/1024 scheduler bound; KV capacity still needs runtime checking"
        )
    for key in ("port", "dist_port"):
        if cfg[key] > 65535:
            raise ValueError(f"{key} must be a valid TCP port")
    if cfg["port"] == cfg["dist_port"]:
        raise ValueError("HTTP port and distributed initialization port must differ")
    chunk = cfg.get("chunked_prefill_size")
    if not (chunk == -1 or positive_int(chunk)) or type(chunk) is not int:
        raise ValueError("chunked_prefill_size must be -1 or a positive integer")
    maximum = cfg.get("max_total_tokens")
    if maximum is not None and not positive_int(maximum):
        raise ValueError(
            "max_total_tokens must be null (automatic sizing) or a positive integer"
        )
    fraction = cfg.get("mem_fraction_static")
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (float, int))
        or not 0 < fraction < 1
    ):
        raise ValueError("mem_fraction_static must be between 0 and 1")
    sizes = cfg.get("graph_batch_sizes")
    if (
        not isinstance(sizes, list)
        or not sizes
        or not all(positive_int(x) for x in sizes)
    ):
        raise ValueError(
            "graph_batch_sizes must be a nonempty list of positive integers"
        )
    if sizes != sorted(set(sizes)):
        raise ValueError("graph_batch_sizes must be unique and ascending")
    nodes = cfg.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != 2:
        raise ValueError("nodes must contain ranks 0 and 1")
    if not all(isinstance(n, dict) and type(n.get("rank")) is int for n in nodes):
        raise ValueError("Each node needs an integer rank")
    if sorted(n["rank"] for n in nodes) != [0, 1]:
        raise ValueError("nodes must contain each of ranks 0 and 1 exactly once")
    cfg["nodes"] = sorted(nodes, key=lambda n: n["rank"])
    for node in cfg["nodes"]:
        require_local_value(node.get("host"), f"rank {node['rank']} host")
        try:
            ip = ipaddress.IPv4Address(node.get("host", ""))
        except (ipaddress.AddressValueError, TypeError):
            raise ValueError(
                f"Fill rank {node['rank']} host with its actual IPv4 address"
            ) from None
        if ip.is_unspecified or ip.is_loopback or ip.is_multicast:
            raise ValueError("Two-node host must be a non-loopback unicast address")
        for key in ("hccl_socket_ifname", "gloo_socket_ifname"):
            require_local_value(node.get(key), f"rank {node['rank']} {key}")
            if (
                not isinstance(node.get(key), str)
                or not re.fullmatch(r"[A-Za-z0-9_.:-]+", node[key])
                or node[key] == "lo"
            ):
                raise ValueError(
                    f"Fill rank {node['rank']} {key} with its actual non-loopback NIC name"
                )
    if cfg["nodes"][0]["host"] == cfg["nodes"][1]["host"]:
        raise ValueError("The two node hosts must differ")
    cfg["base_url"] = f"http://{cfg['nodes'][0]['host']}:{cfg['port']}"
    cfg["dist_init_addr"] = f"{cfg['nodes'][0]['host']}:{cfg['dist_port']}"
    cfg["requested_max_running_requests_per_dp"] = (
        cfg["max_running_requests"] // cfg["dp_size"]
    )
    cfg["profile_status"] = "CANDIDATE_NOT_NPU_VALIDATED"
    return cfg


def load_config(path):
    return validate_config(json.loads(Path(path).read_text()))


def node_config(cfg, rank):
    if type(rank) is not int or rank not in (0, 1):
        raise ValueError("rank must be 0 or 1")
    return next(n for n in cfg["nodes"] if n["rank"] == rank)
