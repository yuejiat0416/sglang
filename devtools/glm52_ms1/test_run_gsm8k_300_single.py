# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the direct GSM8K 300-question entry."""

import json

import run_gsm8k_300_single as runner


def test_prepare_uses_first_300_official_test_rows(monkeypatch, tmp_path):
    source = tmp_path / "gsm8k.jsonl"
    source.write_text(
        "".join(
            json.dumps({"question": f"q{i}", "answer": f"work\n#### {i}"}) + "\n"
            for i in range(301)
        )
    )
    monkeypatch.setattr(runner, "SOURCE", source)
    monkeypatch.setattr(runner, "STATE", tmp_path / "state")
    fixture = runner.prepare_fixture()
    data = json.loads(fixture.read_text())
    assert len(data["cases"]) == 300
    assert data["cases"][0]["question"] == "q0"
    assert data["cases"][-1]["question"] == "q299"


def test_single_node_config_is_direct_and_matches_current_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "STATE", tmp_path)
    cfg = runner.single_node_config()
    assert (cfg["nnodes"], cfg["tp_size"], cfg["dp_size"]) == (1, 16, 1)
    assert cfg["base_url"] == f"http://{runner.HOST}:{runner.PORT}"
    assert cfg["evidence_scope"] == "single-node-accuracy"
