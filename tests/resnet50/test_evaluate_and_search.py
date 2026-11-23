import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.resnet50.evaluate import evaluate_loader, metrics_from_confusion
from copy import deepcopy
import src.resnet50.search as search


def test_metrics_treat_rows_as_truth_and_include_missing_predictions():
    matrix = torch.tensor([[2, 0, 0], [1, 1, 0], [0, 2, 0]])
    metrics = metrics_from_confusion(matrix)
    assert metrics["accuracy"] == 0.5
    assert metrics["macro_precision"] == pytest.approx((2/3 + 1/3 + 0)/3)
    assert metrics["macro_recall"] == pytest.approx((1 + 0.5 + 0)/3)
    assert metrics["macro_f1"] == pytest.approx((0.8 + 0.4 + 0)/3)
    assert metrics["per_class"][2]["support"] == 2


def samples():
    return [
        {"image": torch.tensor([3., 0.]), "label": 0, "image_id": "a"},
        {"image": torch.tensor([0., 3.]), "label": 1, "image_id": "b"},
        {"image": torch.tensor([0., 2.]), "label": 0, "image_id": "c"},
    ]


def test_evaluation_counts_final_partial_batch_without_updates():
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    original = model.weight.detach().clone()
    metrics, matrix, predictions = evaluate_loader(
        model, DataLoader(samples(), batch_size=2), device=torch.device("cpu"),
        num_classes=2, expected_samples=3,
    )
    assert not model.training and model.weight.grad is None
    torch.testing.assert_close(model.weight, original, rtol=0, atol=0)
    assert metrics["samples"] == 3 and metrics["correct"] == 2
    assert metrics["accuracy"] == pytest.approx(2/3)
    assert matrix.tolist() == [[1, 1], [0, 1]]
    assert [row["image_id"] for row in predictions] == ["a", "b", "c"]


def test_evaluation_rejects_missing_or_repeated_images():
    kwargs = dict(device=torch.device("cpu"), num_classes=2, expected_samples=4)
    with pytest.raises(ValueError, match="Incomplete"):
        evaluate_loader(nn.Identity(), DataLoader(samples(), batch_size=2), **kwargs)
    repeated = samples() + [samples()[0]]
    with pytest.raises(ValueError, match="Repeated"):
        evaluate_loader(nn.Identity(), DataLoader(repeated, batch_size=2), **kwargs)


def test_nonfinite_predictions_are_rejected():
    class InvalidModel(nn.Module):
        def forward(self, x):
            return x * float("nan")
    with pytest.raises(ValueError, match="Invalid"):
        evaluate_loader(InvalidModel(), DataLoader(samples(), batch_size=3),
                        device=torch.device("cpu"), num_classes=2, expected_samples=3)


def example():
    return {
        "train": (torch.tensor([[-3., -1.], [-2., 1.], [1., 0.], [3., 2.], [-1., 4.], [1., 5.]]),
                  torch.tensor([0, 0, 1, 1, 2, 2])),
        "val": (torch.tensor([[-2.5, 0.], [2., 1.], [0., 4.]]), torch.tensor([0, 1, 2])),
    }


def test_random_seed_replays_candidates_without_changing_grid():
    first = search.candidates(8, 42)
    assert first == search.candidates(8, 42)
    changed = search.candidates(8, 7)
    assert first["grid"] == changed["grid"] and first["random"] != changed["random"]
    assert all(len(values) == 8 for values in first.values())
    assert all(search.LOWER <= value <= search.UPPER for values in first.values() for value in values)


def test_both_methods_get_identical_features_budget_and_fitting_policy(monkeypatch):
    features = example()
    before = deepcopy(features)
    real_fit = search.fit_discriminant_head
    calls = []

    def spy(train_x, train_y, val_x, val_y, **kwargs):
        assert train_x is features["train"][0] and val_x is features["val"][0]
        assert train_y is features["train"][1] and val_y is features["val"][1]
        assert kwargs["calibrate"] is False and kwargs["device"] == torch.device("cpu")
        calls.append(kwargs["shrinkage"])
        return real_fit(train_x, train_y, val_x, val_y, **kwargs)

    monkeypatch.setattr(search, "fit_discriminant_head", spy)
    settings = search.candidates(2, 42)
    trials = list(search.run_trials(features, settings, 3))
    assert [(row["method"], row["trial"]) for row, _ in trials] == [
        ("grid", 1), ("random", 1), ("random", 2), ("grid", 2)]
    assert sorted(calls) == sorted(settings["grid"] + settings["random"])
    for row, _ in trials:
        assert row["seconds"] >= 0
        for key in ("val_accuracy", "val_macro_precision", "val_macro_recall", "val_macro_f1"):
            assert row[key] == 1.0
    for partition in features:
        for original, actual in zip(before[partition], features[partition]):
            torch.testing.assert_close(original, actual, rtol=0, atol=0)


def test_summary_uses_validation_score_then_f1_then_earlier_trial():
    rows = []
    for method in ("grid", "random"):
        for trial, accuracy, f1 in [(1, .8, .7), (2, .8, .75), (3, .8, .75)]:
            rows.append(dict(method=method, trial=trial, val_accuracy=accuracy, val_macro_f1=f1, seconds=2.))
    report = search.summarize(rows)
    for method in report:
        assert report[method]["best"]["trial"] == 2
        assert report[method]["trials"] == 3 and report[method]["fit_score_seconds"] == 6


def test_cache_with_test_features_is_rejected_before_source_loading(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"partitions": {"train": (), "val": (), "test": ()}, "image_ids": {}}, path)
    with pytest.raises(ValueError, match="train and val only"):
        search.load_features(path, tmp_path / "source_not_opened.pt")


@pytest.mark.parametrize("trials,seed", [(1, 42), (True, 42), (8, -1), (8, True)])
def test_invalid_search_settings_fail_before_fitting(trials, seed):
    with pytest.raises(ValueError):
        search.candidates(trials, seed)


def test_existing_output_directory_is_preserved(tmp_path):
    original = tmp_path / "keep.txt"
    original.write_text("important", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        search.main(["--features", "not-loaded.pt", "--source-checkpoint", "not-loaded.pt", "--output-dir", str(tmp_path)])
    assert error.value.code == 2 and original.read_text(encoding="utf-8") == "important"
