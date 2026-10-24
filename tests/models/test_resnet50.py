import torch
from torch import nn
from src.models.resnet50 import ResNet50Classifier


def test_model_output():
    model = ResNet50Classifier(weights=None)
    model.eval()

    images = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        predictions = model(images)

    assert predictions.shape == (2, 200)


def test_training():
    model = ResNet50Classifier(weights=None)
    model.set_stage("head")
    model.train()

    images = torch.randn(2, 3, 64, 64)
    labels = torch.tensor([0, 199])

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    before = model.classifier.weight.detach().clone()

    optimizer.zero_grad()
    predictions = model(images)
    loss = loss_fn(predictions, labels)

    assert torch.isfinite(loss).item()

    loss.backward()
    optimizer.step()

    after = model.classifier.weight

    assert not torch.equal(before, after)


def test_save_load(tmp_path):
    model = ResNet50Classifier(weights=None)
    model.eval()

    images = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        original_predictions = model(images)

    checkpoint_path = tmp_path / "model.pth"
    torch.save(model.state_dict(), checkpoint_path)

    new_model = ResNet50Classifier(weights=None)
    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    new_model.load_state_dict(state_dict)
    new_model.eval()

    with torch.no_grad():
        new_predictions = new_model(images)

    torch.testing.assert_close(original_predictions, new_predictions)