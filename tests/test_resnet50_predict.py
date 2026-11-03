"""Check checkpoint identity, preprocessing, labels and prediction CLI offline."""

import json

from PIL import Image
import pytest
import torch
from torch import nn

from src import resnet50_predict as prediction
from src.data.image_io import load_rgb
from src.data.resnet50_transforms import resnet50_transform


class TinyClassifier(nn.Module):
    """Small substitute; the actual selected ResNet is checked separately."""

    def __init__(self, num_classes, *, weights, dropout, finetune_scope):
        super().__init__()
        assert weights is None  # Prediction must not download pretrained weights.
        self.classifier = nn.Linear(3, num_classes)

    def forward(self, images):
        return self.classifier(images.mean(dim=(2, 3)))


@pytest.fixture
def saved_model(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction, "ResNet50Classifier", TinyClassifier)
    model = TinyClassifier(3, weights=None, dropout=0, finetune_scope="layer4")
    with torch.no_grad():
        model.classifier.weight.copy_(torch.eye(3))
        model.classifier.bias.copy_(torch.tensor([0.1, 0.3, -0.2]))
    config = {
        "model": {"num_classes": 3, "dropout": 0, "finetune_scope": "layer4"},
        "data": {"image_size": 32, "geometry": "fit", "normalization": "imagenet"},
        "data_provenance": {"class_names": ["Zebra Finch", "Blue Jay", "Robin"]},
    }
    payload = {"num_classes": 3, "model_state_dict": model.state_dict(), "config": config}
    checkpoint = tmp_path / "model.pt"
    torch.save(payload, checkpoint)
    image = tmp_path / "bird.png"
    with Image.new("RGBA", (19, 7), (40, 80, 120, 128)) as source:
        source.save(image)
    return checkpoint, image, payload, model


def test_roundtrip_preserves_scores_preprocessing_and_saved_label_order(saved_model):
    checkpoint, image, payload, original = saved_model
    before = checkpoint.read_bytes()
    with load_rgb({"path": str(image)}) as rgb:
        tensor = resnet50_transform(training=False, image_size=32, geometry="fit")(rgb)
    with torch.inference_mode():
        expected = original(tensor.unsqueeze(0)).softmax(1)[0]
    first = prediction.predict_image(image, checkpoint, top_k=3)
    assert first == prediction.predict_image(image, checkpoint, top_k=3)
    assert first["preprocessing"] == payload["config"]["data"]
    assert [row["confidence"] for row in first["predictions"]] == sorted(expected.tolist(), reverse=True)
    for row in first["predictions"]:
        assert row["class_name"] == payload["config"]["data_provenance"]["class_names"][row["class_id"]]
        assert row["confidence"] == pytest.approx(expected[row["class_id"]].item())
    restored, _ = prediction.load_classifier(checkpoint)
    assert not restored.training and all(not p.requires_grad for p in restored.parameters())
    assert checkpoint.read_bytes() == before


@pytest.mark.parametrize("problem", ["labels", "normalization", "geometry", "state"])
def test_invalid_checkpoint_cannot_silently_change_predictions(saved_model, problem):
    checkpoint, _, payload, _ = saved_model
    if problem == "labels":
        payload["config"]["data_provenance"]["class_names"] = ["same"] * 3
    elif problem == "normalization":
        payload["config"]["data"]["normalization"] = "unknown"
    elif problem == "geometry":
        del payload["config"]["data"]["geometry"]
    else:
        del payload["model_state_dict"]["classifier.bias"]
    torch.save(payload, checkpoint)
    with pytest.raises((ValueError, RuntimeError)):
        prediction.load_classifier(checkpoint)


def test_nonfinite_scores_are_rejected(saved_model):
    checkpoint, image, payload, _ = saved_model
    payload["model_state_dict"]["classifier.bias"][0] = float("nan")
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="nonfinite"):
        prediction.predict_image(image, checkpoint, top_k=3)


def test_cli_writes_json_and_does_not_overwrite_files(saved_model, tmp_path, capsys):
    checkpoint, image, _, _ = saved_model
    output = tmp_path / "prediction.json"
    args = ["--image", str(image), "--checkpoint", str(checkpoint), "--top-k", "3", "--output", str(output)]
    assert prediction.main(args) == 0
    assert json.loads(capsys.readouterr().out) == json.loads(output.read_text(encoding="utf-8"))
    before = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        prediction.main(args)
    assert error.value.code == 2 and output.read_bytes() == before


def test_missing_image_and_invalid_top_k_fail_explicitly(saved_model):
    checkpoint, image, _, _ = saved_model
    with pytest.raises(FileNotFoundError):
        prediction.predict_image(image.with_name("missing.png"), checkpoint)
    for top_k in (0, True, 4):
        with pytest.raises(ValueError, match="top_k"):
            prediction.predict_image(image, checkpoint, top_k=top_k)


def test_cuda_request_does_not_silently_fall_back_to_cpu(saved_model, monkeypatch):
    checkpoint, _, _, _ = saved_model
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="unavailable"):
        prediction.load_classifier(checkpoint, device="cuda")
