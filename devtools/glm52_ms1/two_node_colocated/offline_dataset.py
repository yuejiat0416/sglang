"""Offline fixtures and conservative answer extraction for temporary diagnostics.

GSM8K uses the existing ten-question fixture unless a local JSONL is supplied.
GPQA requires a locally supplied Diamond CSV/JSONL; this module never downloads
data or authenticates a file's claimed split. Send only each case's ``messages``
to the service. Gold answers and source records remain in the local fixture.
"""

import argparse
import csv
import hashlib
import io
import json
import random
import re
from decimal import Decimal
from fractions import Fraction
from pathlib import Path


DATASETS = ("gsm8k", "gpqa")
GSM_PROMPT_SUFFIX = (
    "\n\nSolve the problem. End your response with a separate line in the format "
    "#### <number>."
)
GPQA_PROMPT_SUFFIX = (
    "\n\nChoose the single best answer. End your response with a separate line "
    "in the format Answer: A, Answer: B, Answer: C, or Answer: D."
)
NUMBER = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?\Z")
DECIMAL_ONLY = re.compile(r"[+-]?\.\d+\Z")
GSM_MARKER = re.compile(
    r"(?im)^[ \t]*(?:\*\*)?(?:####[ \t]*|final[ \t]+answer(?:\*\*)?"
    r"[ \t]*(?::|=|is\b)[ \t]*)([^\r\n]*)"
)
GPQA_MARKER = re.compile(
    r"(?im)^[ \t]*(?:\*\*)?(?:final[ \t]+)?answer(?:\*\*)?[ \t]*:[ \t]*([^\r\n]*)"
)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _strip_format(value):
    value = value.strip()
    # Remove presentation wrappers only, never arbitrary words or code.
    for _ in range(3):
        before = value
        for left, right in (("**", "**"), ("`", "`"), ("\\(", "\\)"), ("$", "$")):
            if (
                value.startswith(left)
                and value.endswith(right)
                and len(value) > len(left + right)
            ):
                value = value[len(left) : -len(right)].strip()
        if value == before:
            break
    return value


def _number(value):
    """Normalize only a complete numeric literal, using exact rational equality."""
    if not isinstance(value, str) or len(value) > 256:
        return None
    value = _strip_format(value)
    if value.startswith("$"):
        value = value[1:].strip()
    if value.endswith("."):
        value = value[:-1]
    latex = re.fullmatch(r"\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}", value)
    if latex:
        value = f"{latex.group(1)}/{latex.group(2)}"
    pieces = value.split("/")
    if len(pieces) > 2:
        return None
    normalized = []
    for piece in pieces:
        piece = piece.strip()
        if not (NUMBER.fullmatch(piece) or DECIMAL_ONLY.fullmatch(piece)):
            return None
        normalized.append(Fraction(Decimal(piece.replace(",", ""))))
    if len(normalized) == 2:
        if normalized[1] == 0:
            return None
        result = normalized[0] / normalized[1]
    else:
        result = normalized[0]
    return str(result)


def _boxed_candidates(text):
    candidates = []
    for match in re.finditer(r"\\boxed[ \t]*\{", text):
        start, depth, cursor = match.end(), 1, match.end()
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        value = text[start : cursor - 1] if depth == 0 else None
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", cursor)
        line = text[line_start : line_end if line_end >= 0 else len(text)]
        if re.search(r"\bor\b", line, re.IGNORECASE):
            # Do not turn "boxed(18) or boxed(19)" into an asserted answer.
            value = None
        candidates.append((match.start(), "boxed", value))
    return candidates


def _extract(dataset, text):
    if not isinstance(text, str):
        return None, {"method": None, "reason": "Response text is not a string"}
    if dataset == "gsm8k":
        candidates = [
            (m.start(), "explicit_final_marker", m.group(1))
            for m in GSM_MARKER.finditer(text)
        ]
        candidates.extend(_boxed_candidates(text))
        if not candidates:
            return None, {
                "method": None,
                "reason": "No ####, boxed or Final answer marker",
            }
        _, method, raw = max(candidates, key=lambda item: item[0])
        extracted = _number(raw)
    else:
        candidates = list(GPQA_MARKER.finditer(text))
        if not candidates:
            return None, {"method": None, "reason": "No explicit Answer: A-D line"}
        raw = candidates[-1].group(1)
        value = _strip_format(raw).strip()
        match = re.fullmatch(r"(?:\(([A-D])\)|([A-D]))[.!]?", value, re.IGNORECASE)
        extracted = (match.group(1) or match.group(2)).upper() if match else None
        method = "last_explicit_answer_line"
    return extracted, {
        "method": method,
        "raw": raw,
        "reason": None
        if extracted is not None
        else "Last marked answer is ambiguous or unsupported; no fallback guessing",
    }


def score_response(dataset, answer, text, finish_reason):
    """Score an explicitly marked final answer without executing generated content.

    ``answer`` is the prepared case's gold string. Length-truncated, aborted and
    unparsed responses have ``correct=None``; callers must not drop these rows
    when reporting counts over the requested fixture. An apparent matching
    answer in a truncated response is retained as evidence, never a pass.
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if dataset == "gsm8k":
        expected = _number(answer)
        if expected is None and isinstance(answer, str):
            expected, _ = _extract(dataset, answer)
    else:
        expected = (
            answer.upper()
            if isinstance(answer, str) and answer.upper() in "ABCD" and len(answer) == 1
            else None
        )
    if expected is None:
        raise ValueError("Gold answer must be an unambiguous prepared dataset answer")
    extracted, extraction = _extract(dataset, text)
    finish_type = (
        finish_reason.get("type") if isinstance(finish_reason, dict) else finish_reason
    )
    truncated = finish_type == "length"
    issues = []
    if truncated:
        status = "TRUNCATED"
        issues.append(
            "Output reached its length limit; not counted as a correct completed answer"
        )
    elif finish_type != "stop":
        status = "INVALID_FINISH"
        issues.append("Missing or unsuccessful completion finish reason")
    elif extracted is None:
        status = "UNPARSED"
        issues.append(extraction["reason"])
    else:
        status = "CORRECT" if extracted == expected else "INCORRECT"
    return {
        "status": status,
        "correct": extracted == expected
        if status in {"CORRECT", "INCORRECT"}
        else None,
        "extracted_answer": extracted,
        "expected_answer": expected,
        "answer_matches": extracted == expected if extracted is not None else None,
        "truncated": truncated,
        "finish_reason": finish_reason,
        "issues": issues,
        "extraction": extraction,
    }


def _required_text(row, key, row_number):
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Row {row_number}: missing nonempty text field {key!r}")
    return value.strip()


def _load_rows(path, dataset):
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    source = {"path": str(path.resolve()), "sha256": _sha(raw), "bytes": len(raw)}
    if path.suffix.lower() == ".csv":
        if dataset != "gpqa":
            raise ValueError(
                "GSM8K input supports official JSONL or the bundled JSON fixture"
            )
        rows = list(csv.DictReader(io.StringIO(text)))
        source["format"] = "gpqa_csv"
    elif path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        source["format"] = f"{dataset}_jsonl"
    elif dataset == "gsm8k" and path.suffix.lower() == ".json":
        document = json.loads(text)
        if not isinstance(document, dict) or not isinstance(
            document.get("cases"), list
        ):
            raise ValueError(
                "GSM8K JSON requires a cases list; official raw input may be JSONL"
            )
        rows = document["cases"]
        source["format"] = "gsm8k_bundled_json"
        source["upstream_source_url"] = document.get("source_url")
        source["upstream_source_sha256"] = document.get("source_sha256")
    else:
        raise ValueError(
            "Expected local .csv/.jsonl for GPQA or .jsonl/.json for GSM8K"
        )
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Every source row must be an object")
    return rows, source


def _gpqa_row(row, index):
    if "Question" in row:
        question = _required_text(row, "Question", index)
        correct = _required_text(row, "Correct Answer", index)
        incorrect = []
        for i in range(1, 4):
            names = (f"Incorrect Answer {i}", f"Incorrect Answer{i}")
            found = [name for name in names if name in row]
            if len(found) != 1:
                raise ValueError(
                    f"Row {index}: require exactly one field for incorrect answer {i}"
                )
            incorrect.append(_required_text(row, found[0], index))
    else:
        question = _required_text(row, "question", index)
        correct = _required_text(row, "correct_answer", index)
        incorrect = row.get("incorrect_answers")
        if (
            not isinstance(incorrect, list)
            or len(incorrect) != 3
            or any(
                not isinstance(value, str) or not value.strip() for value in incorrect
            )
        ):
            raise ValueError(
                f"Row {index}: incorrect_answers must contain three nonempty strings"
            )
        incorrect = [value.strip() for value in incorrect]
    options = [correct, *incorrect]
    if len(set(options)) != 4:
        raise ValueError(f"Row {index}: GPQA options must be distinct")
    return question, options


def prepare_dataset(dataset, input_path=None, *, limit=10, seed=42):
    """Prepare first ``limit`` rows in source order; only GPQA options shuffle.

    GPQA CSV supports Question, Correct Answer, Incorrect Answer 1/2/3 (also
    their no-space-before-digit variants). JSONL supports the same fields or
    question/correct_answer/incorrect_answers (a three-string list). Supply the
    intended Diamond file; a filename is not independent proof of dataset split.
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if type(limit) is not int or limit <= 0 or type(seed) is not int:
        raise ValueError(
            "limit must be a positive integer and seed an integer, not bool"
        )
    fixture = {
        "schema_version": 1,
        "status": "BLOCKED",
        "dataset": dataset,
        "requested_subset": "diamond" if dataset == "gpqa" else "test",
        "limit": limit,
        "seed": seed,
        "source": None,
        "selection": "first limit rows in source order; GPQA answer options shuffled with fixed seed",
        "prompt_protocol": {
            "id": f"{dataset}-zero-shot-explicit-final-v1",
            "gold_answers_sent": False,
            "description": "Question-only zero-shot with explicit final-answer format; GPQA includes shuffled options. No demonstrations or gold rationale.",
            "comparison_limit": "This prompt protocol differs from the earlier question-only GSM8K acceptance run.",
        },
        "cases": [],
        "issues": [],
        "limits": [
            "No downloads. GPQA Diamond provenance/split depends on the supplied local source; SHA records exactly the bytes used.",
            "Ten questions are a diagnostic sample, not a full benchmark or statistical proof of no accuracy regression.",
            "Send only case.messages. answer and source records are local scoring data.",
        ],
    }
    if input_path is None:
        if dataset == "gpqa":
            fixture["issues"].append(
                "GPQA Diamond local input is missing; provide --input path/to/gpqa_diamond.csv or .jsonl"
            )
            return fixture
        path = Path(__file__).resolve().parent.parent / "gsm8k10.json"
    else:
        path = Path(input_path).expanduser()
    if not path.is_file():
        fixture["source"] = {"path": str(path.resolve()), "sha256": None}
        fixture["issues"].append(f"Local dataset file does not exist: {path}")
        return fixture
    rows, source = _load_rows(path, dataset)
    fixture["source"] = source
    source["row_count"] = len(rows)
    if len(rows) < limit:
        raise ValueError(
            f"Requested {limit} rows, but local source contains only {len(rows)}"
        )
    rng, seen_ids = random.Random(seed), set()
    for index, row in enumerate(rows[:limit]):
        rid = row.get("id", f"{dataset}-row-{index:04d}")
        if not isinstance(rid, str) or not rid or rid in seen_ids:
            raise ValueError(f"Row {index}: id must be nonempty and unique")
        seen_ids.add(rid)
        if dataset == "gsm8k":
            question = _required_text(row, "question", index)
            raw_answer = _required_text(row, "answer", index)
            answer, _ = _extract(dataset, raw_answer)
            if answer is None:
                answer = _number(raw_answer)
            if answer is None:
                raise ValueError(
                    f"Row {index}: GSM8K gold answer has no unambiguous numeric result"
                )
            content = question + GSM_PROMPT_SUFFIX
            extra = {}
        else:
            question, original = _gpqa_row(row, index)
            order = list(range(4))
            rng.shuffle(order)
            choices = {
                "ABCD"[i]: original[source_index]
                for i, source_index in enumerate(order)
            }
            answer = "ABCD"[order.index(0)]
            content = (
                question
                + "\n\n"
                + "\n".join(f"{letter}. {choice}" for letter, choice in choices.items())
                + GPQA_PROMPT_SUFFIX
            )
            extra = {
                "choices": choices,
                "option_source_order": [
                    "correct_answer" if i == 0 else f"incorrect_answer_{i}"
                    for i in order
                ],
            }
        fixture["cases"].append(
            {
                "id": rid,
                "source_row_index": index,
                "question": question,
                "question_sha256": _sha(question.encode("utf-8")),
                "prompt_sha256": _sha(content.encode("utf-8")),
                "messages": [{"role": "user", "content": content}],
                "answer": answer,
                **extra,
            }
        )
    fixture["status"] = "DATASET_PREPARED"
    return fixture


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--dataset", choices=DATASETS, required=True)
    prepare.add_argument("--input", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--limit", type=int, default=10)
    prepare.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        result = prepare_dataset(
            args.dataset, args.input, limit=args.limit, seed=args.seed
        )
    except (ValueError, OSError) as exc:
        result = {
            "status": "DATASET_PREPARATION_FAILED",
            "dataset": args.dataset,
            "cases": [],
            "issues": [str(exc)],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(result["status"])
    print(f"Fixture: {args.output}")
    for issue in result.get("issues", []):
        print(issue)
    return (
        0
        if result["status"] == "DATASET_PREPARED"
        else 2
        if result["status"] == "BLOCKED"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
