from hashlib import sha256
import json

import pytest

import src.analysis.explore_prepared as runner


def test_cli_preserves_input_fingerprints_and_writes_before_optional_figures(monkeypatch, tmp_path):
    image_path, caption_path = tmp_path / "images.json", tmp_path / "captions.json"
    image_path.write_text('{"source":"images"}', encoding="utf-8")
    caption_path.write_text('{"source":"captions"}', encoding="utf-8")
    output = tmp_path / "report.json"
    expected = {"population": {"derived_training_image_count": 3, "validation_image_count": 2}}
    calls = []
    def build(directory, *, image_report, caption_report):
        calls.append((directory, image_report, caption_report))
        return expected
    monkeypatch.setattr(runner, "build_prepared_training_report", build)
    runner.main(["--prepared-dir", str(tmp_path / "prepared"), "--image-report", str(image_path), "--caption-report", str(caption_path), "--output", str(output)])
    actual = json.loads(output.read_text(encoding="utf-8"))
    assert calls == [(tmp_path / "prepared", {"source": "images"}, {"source": "captions"})]
    assert actual["input_report_sha256"] == {"images": sha256(image_path.read_bytes()).hexdigest(), "captions": sha256(caption_path.read_bytes()).hexdigest()}


def test_invalid_integration_does_not_replace_existing_report(monkeypatch, tmp_path):
    image_path, caption_path = tmp_path / "images.json", tmp_path / "captions.json"
    image_path.write_text("{}", encoding="utf-8")
    caption_path.write_text("{}", encoding="utf-8")
    output = tmp_path / "report.json"
    output.write_bytes(b"prior valid report")
    def fail(*args, **kwargs):
        raise ValueError("Invalid preparation")
    monkeypatch.setattr(runner, "build_prepared_training_report", fail)
    with pytest.raises(ValueError, match="Invalid preparation"):
        runner.main(["--prepared-dir", str(tmp_path), "--image-report", str(image_path), "--caption-report", str(caption_path), "--output", str(output)])
    assert output.read_bytes() == b"prior valid report"
