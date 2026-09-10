"""Check the actual distributable, including metadata and its allowlist."""

import hashlib
import json
import tarfile

import pytest

import build_archive as bundle


def test_archive_contains_only_declared_files_without_host_metadata(tmp_path):
    output = tmp_path / "arbitrary-local-name.tar.gz"
    result = bundle.build(output)
    raw = output.read_bytes()
    assert raw[:3] == b"\x1f\x8b\x08"
    assert raw[3] == 0  # No original filename or other optional gzip fields.
    assert raw[4:8] == b"\0\0\0\0"
    assert result["sha256"] == hashlib.sha256(raw).hexdigest()
    expected = {
        str(path.relative_to(bundle.REPO)): path for path in bundle.selected_files()
    }
    with tarfile.open(output) as archive:
        assert set(archive.getnames()) == set(expected) | {
            "colocated-tools-manifest.json"
        }
        for member in archive.getmembers():
            assert member.isfile()
            assert not member.name.startswith("/")
            assert ".." not in member.name.split("/")
            assert (member.uid, member.gid, member.mtime) == (0, 0, 0)
            assert (member.uname, member.gname, member.pax_headers) == ("", "", {})
        manifest = json.load(archive.extractfile("colocated-tools-manifest.json"))
        assert "git_status" not in manifest
        assert set(manifest["files"]) == set(expected)
        for name, source in expected.items():
            data = archive.extractfile(name).read()
            assert data == source.read_bytes()
            assert manifest["files"][name] == hashlib.sha256(data).hexdigest()
    second = tmp_path / "different-name.tar.gz"
    bundle.build(second)
    assert output.read_bytes() == second.read_bytes()
    with pytest.raises(FileExistsError):
        bundle.build(output)


def test_local_config_and_extra_python_are_excluded(tmp_path, monkeypatch):
    directory = tmp_path / "suite"
    directory.mkdir()
    for name in bundle.SUITE_FILES:
        (directory / name).write_text("public example")
    for name in bundle.HELPERS:
        (directory.parent / name).write_text("public helper")
    for name in ("config.json", "local_settings.py", "server.log", "results.json"):
        (directory / name).write_text("private local data")
    monkeypatch.setattr(bundle, "HERE", directory)
    selected = bundle.selected_files()
    assert len(selected) == len(bundle.SUITE_FILES) + len(bundle.HELPERS)
    assert directory / "local_settings.py" not in selected
    (directory / "launch.py").unlink()
    (directory / "launch.py").symlink_to(directory / "config.json")
    with pytest.raises(ValueError, match="symbolic links"):
        bundle.selected_files()
