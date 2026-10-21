import pytest
from datasets import ClassLabel, Dataset, Features, Image, Value

import src.data.source as source


TRAIN_REVISION = "bd3988995e6db193427255abdebd522afe24ce76"
TEST_REVISION = "82447a42c5e05c7fe21cd36accfdf399148e8668"


def synthetic_dataset(*, names=None, label=0, description=" First  bird. \nSecond caption\n"):
    if names is None:
        names = [f"Bird {index}" for index in range(200)]
    features = Features(
        {
            "image": Image(),
            "description": Value("string"),
            "label": ClassLabel(names=names),
            "file_name": Value("string"),
            "id": Value("int64"),
        }
    )
    return Dataset.from_dict(
        {
            "image": [{"bytes": b"not decoded image bytes", "path": None}],
            "description": [description],
            "label": [label],
            "file_name": ["bird.jpg"],
            "id": [7],
        },
        features=features,
    )


def test_schema_accepts_real_feature_types_and_rejects_missing_or_wrong_fields():
    features = synthetic_dataset().features
    assert len(source.validate_source_schema(features)) == 200

    missing_id = Features({name: feature for name, feature in features.items() if name != "id"})
    with pytest.raises(ValueError, match="id"):
        source.validate_source_schema(missing_id)

    wrong_label = Features({**features, "label": Value("int64")})
    with pytest.raises(ValueError, match="ClassLabel"):
        source.validate_source_schema(wrong_label)


def test_schema_requires_200_classes():
    features = synthetic_dataset(names=[f"Bird {index}" for index in range(199)]).features
    with pytest.raises(ValueError, match="200"):
        source.validate_source_schema(features)


def test_loader_pins_revision_and_exposes_metadata_without_decoding(monkeypatch):
    original = synthetic_dataset()
    calls = []

    def fake_load_dataset(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return original

    monkeypatch.setattr(source, "load_dataset", fake_load_dataset)
    loaded = source.load_cub_split("train")
    image, captions = next(loaded.iter_records())

    assert calls == [
        ("Multimodal-Fatima/CUB_train", {"split": "train", "revision": TRAIN_REVISION, "streaming": False})
    ]
    assert original.features["image"].decode is True
    assert loaded.dataset.features["image"].decode is False
    assert image.image_id == f"hf:Multimodal-Fatima/CUB_train@{TRAIN_REVISION}:train:0"
    assert (image.source_id, image.file_name, image.class_id, image.class_name) == (7, "bird.jpg", 0, "Bird 0")
    assert [(caption.caption_index, caption.clean_caption) for caption in captions] == [
        (0, "First bird."),
        (1, "Second caption"),
    ]


def test_pair_rejects_same_sized_class_mapping_in_different_order(monkeypatch):
    names = [f"Bird {index}" for index in range(200)]
    swapped = names.copy()
    swapped[0], swapped[1] = swapped[1], swapped[0]

    def fake_load_dataset(repo_id, **kwargs):
        return synthetic_dataset(names=swapped if repo_id.endswith("CUB_test") else names)

    monkeypatch.setattr(source, "load_dataset", fake_load_dataset)
    with pytest.raises(ValueError, match="class mapping"):
        source.load_cub_sources(streaming=True)


def test_invalid_label_is_rejected_when_record_is_read(monkeypatch):
    monkeypatch.setattr(source, "load_dataset", lambda repo_id, **kwargs: synthetic_dataset(label=None))
    loaded = source.load_cub_split("train")
    with pytest.raises(ValueError, match="label"):
        next(loaded.iter_records())


def test_unknown_split_is_rejected_before_loading():
    with pytest.raises(ValueError, match="source_split"):
        source.load_cub_split("validation")
