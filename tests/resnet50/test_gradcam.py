from types import SimpleNamespace
from PIL import Image
import pytest
import torch
from torch import nn
from src.resnet50 import gradcam as explanation
import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import numpy as np
import src.resnet50.gradcam as gallery
from src.resnet50.gradcam import SELECTED_TARGETS, resolve_gallery_targets


class SpatialClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(3, 2, 1, bias=False)
        self.head = nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            self.features.weight.zero_()
            self.features.weight[0, 0, 0, 0] = 1
            self.features.weight[1, 1, 0, 0] = 1
            self.head.weight.copy_(torch.eye(2))
            self.head.bias.zero_()
        self.encoder = SimpleNamespace(layer4=[self.features])
        self.eval().requires_grad_(False)

    def forward(self, images):
        return self.head(self.features(images).mean(dim=(2, 3)))


def spatial_input():
    images = torch.zeros(1, 3, 4, 4)
    images[0, 0, :2, :2] = 2
    images[0, 1, 2:, 2:] = 4
    return images


def test_each_class_highlights_its_own_known_region_with_frozen_parameters():
    model, images = SpatialClassifier(), spatial_input()
    state = {key: value.clone() for key, value in model.state_dict().items()}
    for class_id, region in [(0, (slice(0, 2), slice(0, 2))), (1, (slice(2, 4), slice(2, 4)))]:
        heatmap, logits, selected = explanation.gradcam(model, images, model.features, class_id=class_id)
        expected = torch.zeros(4, 4)
        expected[region] = 1
        torch.testing.assert_close(heatmap, expected, rtol=0, atol=0)
        assert selected == class_id and logits.shape == (1, 2)
        assert not heatmap.requires_grad and not logits.requires_grad
    _, _, selected = explanation.gradcam(model, images, model.features)
    assert selected == 1
    assert not images.requires_grad and images.grad is None
    assert all(parameter.grad is None for parameter in model.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)


def test_uses_logit_gradient_even_when_softmax_is_saturated():
    model, images = SpatialClassifier(), spatial_input()
    before, _, _ = explanation.gradcam(model, images, model.features, class_id=0)
    with torch.no_grad():
        model.head.bias.copy_(torch.tensor([1000., -1000.]))
    with torch.no_grad():
        after, _, _ = explanation.gradcam(model, images, model.features, class_id=0)
    torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_no_positive_support_returns_finite_zero_map():
    model = SpatialClassifier()
    with torch.no_grad():
        model.head.weight.neg_()
    heatmap, _, _ = explanation.gradcam(model, spatial_input(), model.features, class_id=0)
    assert torch.isfinite(heatmap).all() and torch.count_nonzero(heatmap) == 0


@pytest.mark.parametrize("class_id", [-1, True, 2])
def test_invalid_class_and_failure_do_not_leave_hooks(class_id):
    model = SpatialClassifier()
    before = len(model.features._forward_hooks)
    with pytest.raises(ValueError, match="class_id"):
        explanation.gradcam(model, spatial_input(), model.features, class_id=class_id)
    assert len(model.features._forward_hooks) == before


def test_unused_target_layer_is_rejected_and_hook_removed():
    model, unused = SpatialClassifier(), nn.Identity()
    with pytest.raises(ValueError, match="exactly once"):
        explanation.gradcam(model, spatial_input(), unused)
    assert len(unused._forward_hooks) == 0


def test_training_and_inference_modes_cannot_silently_disable_gradcam():
    model = SpatialClassifier()
    with torch.inference_mode(), pytest.raises(ValueError, match="autograd"):
        explanation.gradcam(model, spatial_input(), model.features)
    model.train()
    with pytest.raises(ValueError, match="eval"):
        explanation.gradcam(model, spatial_input(), model.features)


def test_cli_renders_metadata_and_refuses_to_overwrite(tmp_path, monkeypatch, capsys):
    model = SpatialClassifier()
    config = {
        "data": {"image_size": 8, "geometry": "fit", "normalization": "imagenet"},
        "data_provenance": {"class_names": ["Red bird", "Green bird"]},
    }
    monkeypatch.setattr(explanation, "load_classifier", lambda *args, **kwargs: (model, config))
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"stand-in for loader")
    image = tmp_path / "bird.png"
    with Image.new("RGB", (9, 5), (200, 80, 20)) as rgb:
        rgb.save(image)
    output = tmp_path / "maps" / "example.png"
    args = ["--image", str(image), "--checkpoint", str(checkpoint), "--output", str(output),
            "--class-id", "1", "--true-class-id", "1"]
    assert explanation.main(args) == 0
    import json
    report = json.loads(capsys.readouterr().out)
    assert report == json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert report["explained_class_name"] == "Green bird"
    assert report["correct"] == (report["predicted_class_id"] == 1)
    assert report["weights_updated"] is False
    with Image.open(output) as figure:
        assert figure.format == "PNG" and figure.width > figure.height
    before = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        explanation.main(args)
    assert error.value.code == 1 and output.read_bytes() == before
    assert len(model.features._forward_hooks) == 0


def test_selected_targets_counts():
    """Verify that exactly 8 targets are defined: 3 correct, 3 incorrect, 2 borderline."""
    assert len(SELECTED_TARGETS) == 8
    categories = [t["category"] for t in SELECTED_TARGETS]
    assert categories.count("confident_correct") == 3
    assert categories.count("confident_incorrect") == 3
    assert categories.count("borderline") == 2

    # Verify all test indices are unique
    indices = [t["test_index"] for t in SELECTED_TARGETS]
    assert len(set(indices)) == 8


def test_resolve_gallery_targets_against_evaluation_run():
    """Verify that targets match the actual M5 test evaluation metrics."""
    eval_dir = Path("outputs/resnet50_inaturalist/evaluation/07dd204f3ae141e89e626500c142b6a6")
    manifest_path = Path("preparation/classification_manifest.jsonl")

    if not eval_dir.exists() or not manifest_path.exists():
        pytest.skip("Evaluation artifacts or manifest not present locally")

    manifest_rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    targets = resolve_gallery_targets(eval_dir, manifest_rows)

    assert len(targets) == 8

    # Check confident correct
    correct_targets = [t for t in targets if t.category == "confident_correct"]
    for t in correct_targets:
        assert t.correct is True
        assert t.confidence >= 0.99
        assert t.true_class_id == t.pred_class_id

    # Check confident incorrect
    incorrect_targets = [t for t in targets if t.category == "confident_incorrect"]
    for t in incorrect_targets:
        assert t.correct is False
        assert t.confidence >= 0.95
        assert t.true_class_id != t.pred_class_id

    # Check borderline
    borderline_targets = [t for t in targets if t.category == "borderline"]
    for t in borderline_targets:
        assert t.confidence <= 0.10  # low confidence


@pytest.fixture
def gallery_case(tmp_path, monkeypatch):
    """One synthetic image; no downloads, real checkpoint or GPU required."""
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"synthetic checkpoint identity")
    names = [f"Species {i}" for i in range(200)]
    config = {
        "data": {"prepared_dir": str(tmp_path / "preparation"), "image_size": 8,
                 "geometry": "fit", "normalization": "imagenet"},
        "data_provenance": {"class_names": names, "manifest_sha256": "a" * 64},
    }
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    report = {
        "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
        "data_provenance": deepcopy(config["data_provenance"]),
        "preprocessing": {k: config["data"][k] for k in ("image_size", "geometry", "normalization")},
    }
    (evaluation / "report.json").write_text(json.dumps(report), encoding="utf-8")
    spec = {"category": "confident_correct", "description": "Synthetic example", "test_index": 0}

    def saved_scores(predicted=0, truth=0):
        spec["category"] = "confident_correct" if predicted == truth else "confident_incorrect"
        probabilities = np.full((1, 200), 0.01 / 199)
        probabilities[0, predicted] = 0.99
        np.savez(evaluation / "full_class_scores.npz", probabilities=probabilities,
                 labels=np.array([truth]), image_ids=np.array(["test:0"]))

    saved_scores()
    manifest = [{"partition": "test", "source_row": 0, "class_id": 0,
                 "image_id": "test:0", "file_name": "bird.png"}]
    image_path = tmp_path / "bird.png"
    with Image.new("RGB", (10, 6), "white") as image:
        image.save(image_path)
    model = torch.nn.Module()
    model.encoder = torch.nn.Module()
    model.encoder.layer4 = torch.nn.Sequential(torch.nn.Identity())
    monkeypatch.setattr(gallery, "load_classifier", lambda *a, **k: (model, config))
    monkeypatch.setattr(gallery, "load_classification_manifest", lambda _: manifest)
    monkeypatch.setattr(gallery, "SELECTED_TARGETS", [spec])
    source_calls, rendered = [], []

    def load_source(split):
        source_calls.append(split)
        return SimpleNamespace(class_names=names, dataset=[{
            "label": 0, "file_name": "bird.png", "image": {"path": str(image_path)},
        }])

    monkeypatch.setattr(gallery, "load_cub_split", load_source)
    monkeypatch.setattr(gallery, "render_gradcam_figure",
                        lambda images, heatmap, target, geometry, path: rendered.append(target))
    current = {"predicted": 0}

    def current_gradcam(model, images, layer, *, class_id):
        assert class_id is None  # Heatmap explains this forward pass's prediction.
        probabilities = torch.full((1, 200), 0.2 / 199)
        probabilities[0, current["predicted"]] = 0.8
        return torch.zeros(8, 8), probabilities.log(), current["predicted"]

    monkeypatch.setattr(gallery, "gradcam", current_gradcam)
    return SimpleNamespace(checkpoint=checkpoint, evaluation=evaluation, report=report,
                           output=tmp_path / "gallery", source_calls=source_calls,
                           rendered=rendered, current=current, saved_scores=saved_scores)


@pytest.mark.parametrize("field, message", [
    ("checkpoint_sha256", "Checkpoint does not match"),
    ("data_provenance", "data provenance/class order"),
    ("preprocessing", "preprocessing differ"),
])
def test_gallery_rejects_mismatched_evaluation_before_loading_images(gallery_case, field, message):
    case = gallery_case
    if field == "checkpoint_sha256":
        case.report[field] = "0" * 64
    elif field == "data_provenance":
        case.report[field]["class_names"].reverse()
    else:
        case.report[field]["image_size"] = 224
    (case.evaluation / "report.json").write_text(json.dumps(case.report), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        gallery.generate_gallery(case.checkpoint, str(case.evaluation), case.output, device="cpu")
    assert case.source_calls == []
    assert not case.output.exists()


def test_gallery_rejects_current_prediction_differing_from_saved_result(gallery_case):
    case = gallery_case
    case.current["predicted"] = 1  # Saved evaluation predicts class 0.
    with pytest.raises(ValueError, match="Current prediction differs"):
        gallery.generate_gallery(case.checkpoint, str(case.evaluation), case.output, device="cpu")
    assert case.rendered == []
    assert not (case.output / "gallery_manifest.json").exists()


@pytest.mark.parametrize("predicted", [0, 1])
def test_gallery_figure_json_and_summary_use_current_logits(gallery_case, predicted):
    case = gallery_case
    case.saved_scores(predicted=predicted)
    case.current["predicted"] = predicted
    records = gallery.generate_gallery(case.checkpoint, str(case.evaluation), case.output, device="cpu")
    record = records[0]
    figure_target = case.rendered[0]
    assert record["predicted_class_id"] == record["top5"][0]["class_id"] == predicted
    assert record["explained_class_id"] == predicted
    assert record["confidence"] == pytest.approx(0.8, abs=2e-6)  # Saved confidence was 0.99.
    assert figure_target.confidence == record["confidence"]
    assert figure_target.pred_class_id == record["predicted_class_id"]
    assert figure_target.pred_class_name == record["predicted_class_name"] == f"Species {predicted}"
    assert figure_target.correct == record["correct"] == (predicted == 0)
    assert record["checkpoint_sha256"] == case.report["checkpoint_sha256"]
    assert record["evaluation_run_id"] == str(case.evaluation)
    saved = json.loads((case.output / "gallery_manifest.json").read_text(encoding="utf-8"))[0]
    assert saved == record
    assert "80.00%" in (case.output / "README.md").read_text(encoding="utf-8")


def test_gallery_accepts_identical_checkpoint_copied_to_another_path(gallery_case):
    case = gallery_case
    copied = case.checkpoint.with_name("downloaded_model.pt")
    copied.write_bytes(case.checkpoint.read_bytes())
    records = gallery.generate_gallery(copied, str(case.evaluation), case.output, device="cpu")
    assert records[0]["checkpoint"] == str(copied.resolve())


def test_gallery_allows_extra_evaluation_runtime_settings(gallery_case):
    case = gallery_case
    # Historic reports include training/batching settings alongside preprocessing.
    case.report["preprocessing"].update(batch_size=16, num_workers=0, augmentation="mild")
    (case.evaluation / "report.json").write_text(json.dumps(case.report), encoding="utf-8")
    records = gallery.generate_gallery(case.checkpoint, str(case.evaluation), case.output, device="cpu")
    assert len(records) == 1


def test_gallery_rejects_saved_true_label_differing_from_manifest(gallery_case):
    case = gallery_case
    case.saved_scores(truth=1)
    with pytest.raises(ValueError, match="True label mismatch"):
        gallery.generate_gallery(case.checkpoint, str(case.evaluation), case.output, device="cpu")
    assert case.source_calls == []
