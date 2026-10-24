"""Thin pretrained integration keeps prepared membership and native source pixels."""
import importlib
import json
from pathlib import Path
import shutil

import pytest
import torch


def data_api():
    return importlib.import_module("src.training.data")


def model_config(architecture="efficientnet_b3"):
    return importlib.import_module("src.models.pretrained").ModelConfig(architecture)


def test_default_preprocessing_retains_all_content_and_normalizes_once():
    from PIL import Image
    api = data_api()
    image = Image.new("RGB", (8, 4), (255, 0, 0))
    image.paste((0, 0, 255), (4, 0, 8, 4))
    tensor = api.PreprocessingConfig().build_transform(training=False)(image)
    assert tensor.dtype == torch.float32 and tensor.shape == (3, 224, 224)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    assert torch.allclose(tensor[:, 0, 0], (torch.ones(3) - mean) / std)
    assert torch.allclose(tensor[:, 56, 0], (torch.tensor([1., 0., 0.]) - mean) / std)
    assert torch.allclose(tensor[:, 167, 223], (torch.tensor([0., 0., 1.]) - mean) / std)


@pytest.mark.parametrize("kwargs", [
    {"image_size": True}, {"image_size": 0}, {"image_size": 16.5},
    {"normalization": False}, {"normalization": ((0, 0), (1, 1, 1))},
    {"normalization": ((0, 0, 0), (1, 0, 1))},
    {"normalization": ((True, 0, 0), (1, 1, 1))},
    {"normalization": ((float("nan"), 0, 0), (1, 1, 1))},
])
def test_invalid_preprocessing_is_rejected_at_configuration(kwargs):
    with pytest.raises(ValueError, match="image_size|normalization"):
        data_api().PreprocessingConfig(**kwargs)


def test_metadata_separates_full_fit_baseline_from_b3_native_crop():
    config = data_api().PreprocessingConfig()
    metadata = config.describe(model_config())
    assert json.loads(json.dumps(metadata)) == metadata
    assert metadata["baseline"]["geometry"] == "fit"
    assert metadata["baseline"]["image_size"] == 224
    assert metadata["baseline"]["padding_rgb"] == [255, 255, 255]
    assert metadata["baseline"]["train_horizontal_flip_probability"] == 0.5
    assert metadata["baseline"]["validation_augmentation"] is False
    assert metadata["native_weights"]["weight_id"] == "EfficientNet_B3_Weights.IMAGENET1K_V1"
    assert metadata["native_weights"]["preprocessing"]["crop_size"] == [300]
    assert metadata["native_weights"]["applied"] is False


def test_prepared_loaders_are_lazy_and_preserve_mapping_membership_and_batches(audited_prepared, pinned_source):
    api = data_api()
    prepared, _, _ = audited_prepared
    loaders = api.make_pretrained_loaders(prepared, model_config(), source=pinned_source,
        preprocessing=api.PreprocessingConfig(image_size=4, normalization=None), batch_size=511)
    assert set(loaders) == {"train", "val"}
    assert pinned_source.dataset.reads == []
    assert len(loaders["train"].dataset) == 4794 and len(loaders["val"].dataset) == 1200
    train_rows, val_rows = (loaders[name].dataset.rows for name in ("train", "val"))
    assert {r["image_id"] for r in train_rows}.isdisjoint(r["image_id"] for r in val_rows)
    batches = list(loaders["train"])
    assert [len(batch["image_id"]) for batch in batches] == [511] * 9 + [195]
    actual = {identifier: label for batch in batches for identifier, label in zip(batch["image_id"], batch["label"].tolist())}
    assert actual == {row["image_id"]: row["class_id"] for row in train_rows}
    assert all(set(batch) == {"image", "label", "image_id"} for batch in batches)
    assert all(batch["label"].dtype == torch.int64 for batch in batches)


@pytest.mark.parametrize("normalization,expected", [
    (None, [1., 0., 0.]), (((.5, .25, .75), (.5, .25, .5)), [1., -1., -1.5]),
])
def test_validation_pixels_and_normalization_are_deterministic(audited_prepared, pinned_source, normalization, expected):
    api = data_api()
    loaders = api.make_pretrained_loaders(audited_prepared[0], model_config(), source=pinned_source,
        preprocessing=api.PreprocessingConfig(image_size=16, normalization=normalization), batch_size=2)
    first, second = next(iter(loaders["val"])), next(iter(loaders["val"]))
    assert first["image_id"] == second["image_id"]
    assert torch.equal(first["image"], second["image"])
    assert torch.allclose(first["image"][0, :, 4, 0], torch.tensor(expected))
    assert first["image"].shape == (2, 3, 16, 16)
    train_transform = loaders["train"].dataset.transform
    val_transform = loaders["val"].dataset.transform
    from PIL import Image
    from io import BytesIO
    with Image.open(BytesIO(audited_prepared[2])) as image:
        val = val_transform(image)
        torch.manual_seed(0)
        flipped = train_transform(image)
        torch.manual_seed(1)
        unflipped = train_transform(image)
    assert torch.equal(flipped, val.flip(-1)) and torch.equal(unflipped, val)


def test_sampler_seed_matches_without_resplitting(audited_prepared, pinned_source):
    api = data_api()
    kwargs = {"source": pinned_source, "preprocessing": api.PreprocessingConfig(image_size=4, normalization=None)}
    a = api.make_pretrained_loaders(audited_prepared[0], model_config(), seed=42, **kwargs)
    b = api.make_pretrained_loaders(audited_prepared[0], model_config(), seed=42, **kwargs)
    c = api.make_pretrained_loaders(audited_prepared[0], model_config(), seed=43, **kwargs)
    assert list(a["train"].sampler) == list(b["train"].sampler)
    assert list(b["train"].sampler) != list(c["train"].sampler)
    assert a["train"].dataset.rows == c["train"].dataset.rows
    assert list(a["val"].sampler) == list(range(1200))


@pytest.mark.parametrize("mutation", ["missing", "tampered", "summary"])
def test_invalid_preparation_fails_before_any_source_fetch(tmp_path, audited_prepared, monkeypatch, mutation):
    api = data_api()
    prepared = tmp_path / "prepared"
    if mutation != "missing":
        shutil.copytree(audited_prepared[0], prepared)
        if mutation == "tampered":
            path = prepared / "classification_manifest.jsonl"
            path.write_bytes(path.read_bytes() + b"\n")
        else:
            (prepared / "preparation.json").write_text('{"complete":false}', encoding="utf-8")
    def no_fetch(*args, **kwargs):
        pytest.fail("invalid preparation must fail before external source fetch")
    monkeypatch.setattr("src.data.source.load_dataset", no_fetch)
    with pytest.raises((ValueError, FileNotFoundError)):
        api.make_pretrained_loaders(prepared, model_config())


def test_implicit_source_opens_only_official_train(audited_prepared, pinned_source, monkeypatch):
    from datasets import ClassLabel, Image, Value
    pinned_source.dataset.features = {
        "image": Image(), "label": ClassLabel(names=list(pinned_source.class_names)),
        "description": Value("string"), "file_name": Value("string"), "id": Value("int64")}
    pinned_source.dataset.cast_column = lambda *args, **kwargs: pinned_source.dataset
    def external_fetch(repo, *, split, revision, streaming):
        assert repo == "Multimodal-Fatima/CUB_train" and split == "train"
        assert revision == "bd3988995e6db193427255abdebd522afe24ce76" and streaming is False
        return pinned_source.dataset
    monkeypatch.setattr("src.data.source.load_dataset", external_fetch)
    loaders = data_api().make_pretrained_loaders(audited_prepared[0], model_config(), batch_size=1)
    assert next(iter(loaders["val"]))["image"].shape == (1, 3, 224, 224)


@pytest.mark.parametrize("mutation", ["source_split", "mapping"])
def test_source_mismatch_is_rejected(audited_prepared, pinned_source, mutation):
    from src.data.source import _SOURCES
    if mutation == "source_split":
        pinned_source.spec = _SOURCES["test"]
    else:
        pinned_source.class_names = tuple(reversed(pinned_source.class_names))
    with pytest.raises(ValueError, match="source|mapping"):
        data_api().make_pretrained_loaders(audited_prepared[0], model_config(), source=pinned_source)


@pytest.mark.parametrize("kwargs", [{"model_config": {}}, {"batch_size": True}, {"num_workers": -1}, {"seed": -1}, {"pin_memory": 1}, {"preprocessing": {}}])
def test_invalid_loader_or_model_configuration_precedes_network(tmp_path, monkeypatch, kwargs):
    def no_fetch(*args, **kwargs):
        pytest.fail("invalid configuration must fail before external source fetch")
    monkeypatch.setattr("src.data.source.load_dataset", no_fetch)
    arguments = {"model_config": model_config(), **kwargs}
    with pytest.raises(ValueError):
        data_api().make_pretrained_loaders(tmp_path / "missing", **arguments)
