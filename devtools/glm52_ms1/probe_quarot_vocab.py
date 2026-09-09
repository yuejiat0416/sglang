# SPDX-License-Identifier: Apache-2.0
"""Compare sampled checkpoint vocab rows on CPU; never import a model/runtime.

This temporary diagnostic tests predefined coordinate hypotheses, not inference
correctness. Only selected rows and the two rotation tensors are read. NumPy is
loaded after the standalone process's BLAS thread limits have been configured.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

DTYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2}
THREAD_ENV = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
MAX_SAMPLES = 256
_np = None


def load_numpy(threads=1):
    global _np
    if _np is None:
        if threads < 1:
            raise ValueError("threads must be positive")
        for key in THREAD_ENV:
            os.environ[key] = str(threads)
        import numpy

        _np = numpy
    return _np


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_bytes(stream, offset, length):
    stream.seek(offset)
    value = stream.read(length)
    if len(value) != length:
        raise ValueError(f"Truncated data at offset {offset}: expected {length} bytes")
    return value


def read_json_file(path):
    data = path.read_bytes()
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value, {"path": str(path), "sha256": sha256(data), "content": value}


class TensorFile:
    """Header plus bounded payload reads, following the project's header audit.

    No safetensors/torch loader is used: a tensor's absolute payload offset is
    8 + header_length + data_offsets[0]. Evidence hashes cover read regions only.
    """

    def __init__(self, path):
        self.path = Path(path).resolve(strict=True)
        with self.path.open("rb") as stream:
            stat = os.fstat(stream.fileno())
            self._identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            self.header_length = int.from_bytes(read_bytes(stream, 0, 8), "little")
            if not 0 < self.header_length <= min(256 * 1024 * 1024, stat.st_size - 8):
                raise ValueError(f"Invalid safetensors header length: {self.path}")
            raw = read_bytes(stream, 8, self.header_length)
        self.header = json.loads(raw)
        if not isinstance(self.header, dict):
            raise ValueError(f"Expected a safetensors header object: {self.path}")
        self.data_start = 8 + self.header_length
        self.reads = []
        self._evidence = {
            "path": str(self.path),
            "file_size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "header_length": self.header_length,
            "header_sha256": sha256(raw),
            "data_start": self.data_start,
            "reads": self.reads,
            "full_checkpoint_hash": "NOT_COMPUTED",
        }

    def evidence(self):
        return self._evidence

    def matrix_info(self, key):
        info = self.header.get(key)
        if not isinstance(info, dict):
            raise ValueError(f"Missing tensor {key!r} in {self.path}")
        shape, offsets, dtype = (
            info.get("shape"),
            info.get("data_offsets"),
            info.get("dtype"),
        )
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(type(n) is not int or n <= 0 for n in shape)
        ):
            raise ValueError(f"Expected a positive 2D matrix for {key}: {shape}")
        if dtype not in DTYPE_BYTES:
            raise ValueError(f"Unsupported vocab/rotation dtype {dtype!r}: {key}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(n) is not int for n in offsets)
            or not 0 <= offsets[0] <= offsets[1]
            or offsets[1] - offsets[0] != math.prod(shape) * DTYPE_BYTES[dtype]
            or self.data_start + offsets[1] > self._identity[2]
        ):
            raise ValueError(f"Invalid shape/dtype/payload offsets for {key}")
        return info

    def _check_identity(self, stream):
        stat = os.fstat(stream.fileno())
        current = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if current != self._identity:
            raise ValueError(f"Source file changed during diagnosis: {self.path}")

    def read_tensor(self, key, rows=None):
        np = load_numpy()
        info = self.matrix_info(key)
        shape, dtype = info["shape"], info["dtype"]
        start, end = info["data_offsets"]
        rows = None if rows is None else list(rows)
        if rows is not None and (
            not rows or any(type(i) is not int or not 0 <= i < shape[0] for i in rows)
        ):
            raise ValueError(f"Invalid row selection for {key}: {rows}")
        row_bytes = shape[1] * DTYPE_BYTES[dtype]
        spans = (
            [(self.data_start + start, end - start)]
            if rows is None
            else [(self.data_start + start + i * row_bytes, row_bytes) for i in rows]
        )
        record = {"key": key, **info, "rows": rows, "ranges": []}
        self.reads.append(record)
        chunks = []
        with self.path.open("rb") as stream:
            self._check_identity(stream)
            for offset, length in spans:
                raw = read_bytes(stream, offset, length)
                record["ranges"].append(
                    {"offset": offset, "length": length, "sha256": sha256(raw)}
                )
                chunks.append(raw)
            self._check_identity(stream)
        raw = chunks[0] if len(chunks) == 1 else b"".join(chunks)
        if dtype == "BF16":
            bits = np.frombuffer(raw, dtype="<u2").astype("<u4")
            np.left_shift(bits, 16, out=bits)
            values = bits.view("<f4")
        else:
            values = np.frombuffer(raw, dtype={"F32": "<f4", "F16": "<f2"}[dtype])
            values = values.astype(np.float32, copy=False)
        return values.reshape((shape[0] if rows is None else len(rows), shape[1]))


class Checkpoint:
    def __init__(self, path, registry=None):
        self.path = Path(path).resolve(strict=True)
        if not self.path.is_dir():
            raise ValueError(f"Expected checkpoint directory: {self.path}")
        self.config, self.config_evidence = read_json_file(self.path / "config.json")
        self.files = {}
        self.registry = registry if registry is not None else []
        self.indices = []
        for index in sorted(self.path.glob("*.safetensors.index.json")):
            value, evidence = read_json_file(index)
            if not isinstance(value.get("weight_map"), dict):
                raise ValueError(f"Missing weight_map in {index}")
            self.indices.append(evidence)

    def get_file(self, path):
        path = Path(path).resolve(strict=True)
        if not path.is_relative_to(self.path):
            raise ValueError(f"Weight path is outside checkpoint directory: {path}")
        if path not in self.files:
            reader = TensorFile(path)
            self.files[path] = reader
            self.registry.append(reader.evidence())
        return self.files[path]

    def find_tensor(self, aliases):
        matches = set()
        if self.indices:
            for index in self.indices:
                for key in aliases:
                    filename = index["content"]["weight_map"].get(key)
                    if filename is not None:
                        if not isinstance(filename, str):
                            raise ValueError(f"Invalid shard for {key}")
                        matches.add(((self.path / filename).resolve(), key))
        else:
            # Headers only. Never use suffix matching that could select MTP.
            for path in sorted(self.path.glob("*.safetensors")):
                reader = self.get_file(path)
                matches.update(
                    (reader.path, key) for key in aliases if key in reader.header
                )
        if len(matches) != 1:
            raise ValueError(
                f"Expected one unambiguous tensor from {aliases} in {self.path}; "
                f"found {sorted((str(p), k) for p, k in matches)}"
            )
        path, key = matches.pop()
        reader = self.get_file(path)
        reader.matrix_info(key)
        return reader, key


def select_token_ids(vocab_size, configs, explicit=None, count=32):
    if type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError("Invalid vocab size")
    sources = {}
    if explicit is not None:
        ids = (
            [int(x.strip()) for x in explicit.split(",")]
            if isinstance(explicit, str)
            else list(explicit)
        )
        if not ids or any(type(i) is not int or not 0 <= i < vocab_size for i in ids):
            raise ValueError("Explicit token IDs must be within the shared vocab")
        ids = list(dict.fromkeys(ids))
        sources = {str(i): ["explicit"] for i in ids}
    else:
        if not 1 <= count <= MAX_SAMPLES:
            raise ValueError(f"samples must be between 1 and {MAX_SAMPLES}")
        n = min(count, vocab_size)
        ids = [i * (vocab_size - 1) // (n - 1) if n > 1 else 0 for i in range(n)]
        sources = {str(i): ["evenly_spaced"] for i in ids}
        for ci, config in enumerate(configs):
            for label, subconfig in (
                (str(ci), config),
                (
                    f"{ci}.transformer_layer_config",
                    config.get("transformer_layer_config", {}),
                ),
            ):
                if not isinstance(subconfig, dict):
                    continue
                for key in (
                    "mask_token_id",
                    "bos_token_id",
                    "eos_token_id",
                    "pad_token_id",
                ):
                    values = subconfig.get(key)
                    values = values if isinstance(values, list) else [values]
                    for value in values:
                        if type(value) is int and 0 <= value < vocab_size:
                            sources.setdefault(str(value), []).append(
                                f"config[{label}].{key}"
                            )
        ids = sorted(int(i) for i in sources)
    if len(ids) > MAX_SAMPLES:
        raise ValueError(
            f"This row-sampling diagnostic supports at most {MAX_SAMPLES} unique IDs"
        )
    return {"ids": ids, "sources": sources}


def compare_rows(reference, candidate, ids):
    np = load_numpy()
    if (
        reference.shape != candidate.shape
        or reference.ndim != 2
        or len(ids) != len(reference)
    ):
        raise ValueError("Comparison requires aligned 2D matrices and row IDs")
    rows = []
    for token_id, ref, cand in zip(ids, reference, candidate):
        finite = bool(np.isfinite(ref).all() and np.isfinite(cand).all())
        item = {"id": int(token_id), "finite": finite}
        names = (
            "reference_norm",
            "candidate_norm",
            "norm_ratio",
            "cosine",
            "relative_l2",
            "max_abs_error",
        )
        item.update(dict.fromkeys(names))
        if finite:
            # Double precision reductions on tiny sampled rows, not full Q/R.
            ref, cand = ref.astype(np.float64), cand.astype(np.float64)
            rn, cn = float(np.linalg.norm(ref)), float(np.linalg.norm(cand))
            item.update(
                reference_norm=rn,
                candidate_norm=cn,
                max_abs_error=float(np.max(np.abs(cand - ref))),
            )
            if rn > 0:
                item.update(
                    norm_ratio=cn / rn,
                    relative_l2=float(np.linalg.norm(cand - ref)) / rn,
                )
            if rn > 0 and cn > 0:
                item["cosine"] = float(np.clip(np.dot(ref, cand) / (rn * cn), -1, 1))
        rows.append(item)
    summary = {}
    for key in names:
        values = [row[key] for row in rows if row[key] is not None]
        summary[key] = (
            {
                "count": len(values),
                "min": min(values),
                "median": float(np.median(values)),
                "max": max(values),
            }
            if values
            else {"count": 0, "min": None, "median": None, "max": None}
        )
    return {
        "rows": rows,
        "summary": summary,
        "needs_review": any(
            not row["finite"] or row["cosine"] is None or row["relative_l2"] is None
            for row in rows
        ),
    }


def evaluate_candidates(
    target_embedding, draft_embedding, target_head, draft_head, q, r, ids
):
    np = load_numpy()
    width = draft_embedding.shape[1]
    if (
        any(
            x.shape != draft_embedding.shape
            for x in (target_embedding, target_head, draft_head)
        )
        or q.shape != (width, width)
        or r.shape != (width, width)
    ):
        raise ValueError("Vocab samples and Q/R must share the same hidden dimension")
    with np.errstate(over="ignore", invalid="ignore"):
        rotated_head = draft_head @ q
        return {
            "embedding": {
                "raw": compare_rows(target_embedding, draft_embedding, ids),
                "rotated": compare_rows(target_embedding, draft_embedding @ q, ids),
            },
            "head": {
                "raw": compare_rows(target_head, draft_head, ids),
                "rotated": compare_rows(target_head, rotated_head, ids),
                "rotated_norm": compare_rows(target_head, rotated_head @ r, ids),
            },
        }


def git_head():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def run_probe(args, report):
    if args.threads < 1:
        raise ValueError("threads must be positive")
    np = load_numpy(args.threads)
    report["versions"]["numpy"] = np.__version__
    report["thread_environment"] = {key: os.environ.get(key) for key in THREAD_ENV}
    target, draft = (
        Checkpoint(path, report["files"]) for path in (args.target, args.draft)
    )
    report["inputs"] = {"target": str(target.path), "draft": str(draft.path)}
    report["configs"] = {
        "target": target.config_evidence,
        "draft": draft.config_evidence,
    }
    report["indices"] = {
        label: [
            {
                "path": index["path"],
                "sha256": index["sha256"],
                "weight_map_entries": len(index["content"]["weight_map"]),
            }
            for index in checkpoint.indices
        ]
        for label, checkpoint in (("target", target), ("draft", draft))
    }
    references = {
        "target_embedding": target.find_tensor(("model.embed_tokens.weight",)),
        "target_head": target.find_tensor(("lm_head.weight",)),
        "draft_embedding": draft.find_tensor(
            ("embed_tokens.weight", "model.embed_tokens.weight")
        ),
        "draft_head": draft.find_tensor(("lm_head.weight", "model.lm_head.weight")),
    }
    shapes = {
        label: reader.matrix_info(key)["shape"]
        for label, (reader, key) in references.items()
    }
    shape = shapes["target_embedding"]
    if any(other != shape for other in shapes.values()):
        raise ValueError(f"Vocab matrix shapes differ: {shapes}")
    report["samples"] = select_token_ids(
        shape[0], [target.config, draft.config], args.token_ids, args.samples
    )
    ids = report["samples"]["ids"]
    description_path = target.path / "quant_model_description.json"
    description, description_evidence = read_json_file(description_path)
    # Keep the report usable: the full description/index can contain every
    # target layer/expert. Fingerprint them, but only emit relevant entries.
    report["quant_description"] = {
        "path": description_evidence["path"],
        "sha256": description_evidence["sha256"],
        "selected": {
            key: description[key]
            for key in (
                "model_quant_type",
                "model.embed_tokens.weight",
                "lm_head.weight",
                "metadata",
                "optional",
                "is_rot_used",
            )
            if key in description
        },
    }
    q_relative = description["optional"]["quarot"]["rotation_map"]["global_rotation"]
    if not isinstance(q_relative, str):
        raise ValueError("Invalid global_rotation path in quant description")
    q_file = target.get_file(target.path / q_relative)
    r_file = target.get_file(target.path / "rot.safetensors")
    for reader, key in ((q_file, "global_rotation"), (r_file, "rot.weight")):
        if reader.matrix_info(key)["shape"] != [shape[1], shape[1]]:
            raise ValueError(f"{key} dimensions do not match vocab hidden size")
    arrays = {}
    for label, (reader, key) in references.items():
        print(f"READ {label}: {len(ids)} rows", flush=True)
        arrays[label] = reader.read_tensor(key, rows=ids)
    print("READ Q and R (CPU only)", flush=True)
    q = q_file.read_tensor("global_rotation")
    r = r_file.read_tensor("rot.weight")
    report["matrix_finite"] = {
        "Q": bool(np.isfinite(q).all()),
        "R": bool(np.isfinite(r).all()),
    }
    print("COMPARE raw / rotated / rotated_norm", flush=True)
    report["comparisons"] = evaluate_candidates(**arrays, q=q, r=r, ids=ids)
    probes = np.random.default_rng(0).standard_normal((8, shape[1])).astype(np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        report["q_roundtrip"] = compare_rows(probes, (probes @ q) @ q.T, list(range(8)))
    report["q_roundtrip"]["row_meaning"] = (
        "Eight synthetic probes, seed=0; not token IDs or a full orthogonality proof"
    )
    review = (
        not all(report["matrix_finite"].values())
        or report["q_roundtrip"]["needs_review"]
        or any(
            item["needs_review"]
            for group in report["comparisons"].values()
            for item in group.values()
        )
    )
    report["status"] = (
        "NUMERICAL_REVIEW_REQUIRED" if review else "COORDINATE_COMPARISON_COLLECTED"
    )
    return 2 if review else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", type=Path, default=Path("/workspace/weight/GLM-5.2-w8a8")
    )
    parser.add_argument(
        "--draft", type=Path, default=Path("/workspace/weight/GLM-5.2-DSpark-NPU-0805")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/tyj/glm52-ms1/evidence"),
        help="Evidence parent; each run creates a new subdirectory",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=32,
        help=f"Evenly spaced IDs before special IDs; total limited to {MAX_SAMPLES}",
    )
    parser.add_argument(
        "--token-ids", help="Comma-separated explicit IDs; overrides automatic sampling"
    )
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out / f"quarot-vocab-{stamp}-{uuid.uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=False)
    print(f"Evidence: {out}", flush=True)
    started = time.monotonic()
    report = {
        "status": "STARTED",
        "inputs": {"target": str(args.target), "draft": str(args.draft)},
        "requested_sampling": {"samples": args.samples, "token_ids": args.token_ids},
        "requested_threads": args.threads,
        "versions": {"python": platform.python_version()},
        "files": [],
        "notes": [
            "Row-vector hypotheses: E_t ~ E_d Q; W_t ~ W_d Q or (W_d Q) R.",
            "R = Q.T diag(original target norm gamma) Q is an export hypothesis, not proved by the header.",
            "Stored BF16 R adds rounding error; no custom cosine/error pass threshold is applied.",
            "Source hashes cover recorded regions, not the entire target/draft checkpoint.",
            "No runtime loading, hidden-to-FC, full-vocab logits, acceptance or performance validation.",
        ],
    }
    code = 1
    try:
        report["git_head"] = git_head()
        report["runner_sha256"] = sha256(Path(__file__).read_bytes())
        code = run_probe(args, report)
    except Exception as exc:
        report.update(status="FAILED", error=str(exc), traceback=traceback.format_exc())
        print(f"FAILED: {exc}", file=sys.stderr)
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (out / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(report["status"], flush=True)
    for group, candidates in report.get("comparisons", {}).items():
        for label, item in candidates.items():
            summary = item["summary"]
            print(
                f"{group}.{label}: median cosine={summary['cosine']['median']} relative_l2={summary['relative_l2']['median']} norm_ratio={summary['norm_ratio']['median']}"
            )
    print(
        "Diagnostic data only; this does not establish a correct model or an acceptance-rate improvement."
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
