# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for the single-node model-card sample runner."""

import json

import pytest

import run_modelcard_50_single as runner


def source_rows():
    return {
        "gsm8k": [
            {"question": f"gsm question {i}", "answer": str(i)} for i in range(60)
        ],
        "math500": [
            {"problem": f"math problem {i}", "unique_id": str(i)}
            for i in range(60)
        ],
        "aime2025": [
            {"problem": f"aime problem {i}", "id": str(i)} for i in range(30)
        ],
        "mbpp": [
            {"task_id": i, "text": f"code task {i}", "test_list": [f"assert f() == {i}"]}
            for i in range(60)
        ],
        "humaneval": [
            {"task_id": f"HumanEval/{i}", "prompt": f"def f_{i}():\n    pass"}
            for i in range(60)
        ],
        "mt_bench": [
            {"prompt_id": i, "prompt": [f"turn one {i}", f"turn two {i}"]}
            for i in range(60)
        ],
        "swe_bench": [
            {
                "instance_id": f"issue-{i}",
                "repo": "owner/repo",
                "problem_statement": f"fix issue {i}",
                "hints_text": "",
            }
            for i in range(60)
        ],
    }


def test_fixed_without_replacement_samples_and_aime_source_limit():
    first = runner.build_bundle(source_rows())
    second = runner.build_bundle(source_rows())
    assert first == second
    assert first["datasets"]["aime2025"]["actual_samples"] == 30
    for name in set(runner.DATASET_ORDER) - {"aime2025"}:
        cases = first["datasets"][name]["cases"]
        assert len(cases) == len({case["id"] for case in cases}) == 50
    sent = json.dumps(
        [case["messages"] for case in first["datasets"]["gsm8k"]["cases"]]
    )
    assert '"answer"' not in sent


def test_short_non_aime_source_is_rejected():
    rows = source_rows()
    rows["mbpp"] = rows["mbpp"][:49]
    with pytest.raises(ValueError, match="mbpp"):
        runner.build_bundle(rows)


def test_histogram_reconstructs_model_card_positions_and_length(monkeypatch):
    monkeypatch.setattr(runner, "MODE", "dspark-eager")
    histogram = [2, 3, 4, 5, 6, 4, 3, 2, 1]
    response = {
        "choices": [{"meta_info": {"spec_correct_drafts_histogram": histogram}}],
        "sglext": {
            "spec_tokens_details": {"spec_correct_drafts_histogram": histogram}
        },
    }
    rounds = sum(histogram)
    accepted = sum(index * count for index, count in enumerate(histogram))
    result = runner.modelcard_acceptance(
        [response],
        {"verify_rounds": rounds, "accepted_drafts": accepted},
        "gsm8k",
    )
    assert result["accept_length"] == 1 + accepted / rounds
    assert result["position_acceptance"][0] == (rounds - histogram[0]) / rounds
    assert result["position_acceptance"][7] == histogram[8] / rounds


def test_model_card_position_rates_explain_reported_accept_length():
    for positions, length in runner.MODEL_CARD.values():
        assert 1 + sum(value / 100 for value in positions) == pytest.approx(
            length, abs=0.006
        )
