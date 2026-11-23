"""ResNet-50 classifier wrapper, transforms, iNaturalist loading, and single-image prediction."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Literal

from PIL import Image
import torch
from torch import Tensor, nn
from torchvision import transforms as T
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms import InterpolationMode

from src.data.image_io import load_rgb
from src.data.transforms import IMAGENET_NORMALIZATION, classification_transform


TrainingStage = Literal["head", "finetune"]
FineTuneScope = Literal["layer4", "last_block"]
INATURALIST_WEIGHTS = "INAT2021_FROM_SCRATCH"


class ResNet50Classifier(nn.Module):
    """ResNet-50 classifier with frozen encoder BatchNorm running stats."""

    def __init__(
        self, num_classes: int = 200, *,
        weights: ResNet50_Weights | None = ResNet50_Weights.IMAGENET1K_V2,
        dropout: float = 0.0,
        finetune_scope: FineTuneScope = "layer4",
    ):
        super().__init__()

        if (
            isinstance(num_classes, bool)
            or not isinstance(num_classes, int)
            or num_classes < 2
        ):
            raise ValueError("num_classes must be an integer greater than one")

        if (
            isinstance(dropout, bool) or not isinstance(dropout, (int, float))
            or not 0 <= dropout < 1
        ):
            raise ValueError("dropout must be a finite probability in [0, 1).")
        if finetune_scope not in ("layer4", "last_block"):
            raise ValueError("finetune_scope must be 'layer4' or 'last_block'.")
        self._finetune_scope = finetune_scope

        backbone = resnet50(weights=weights)
        self.feature_dim = backbone.fc.in_features

        self.encoder = nn.Sequential(
            OrderedDict(
                (name, layer)
                for name, layer in backbone.named_children()
                if name not in ("avgpool", "fc")
            )
        )
        self.pool = backbone.avgpool
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self.set_stage("head")

    @property
    def stage(self) -> TrainingStage:
        return self._stage

    @property
    def finetune_scope(self) -> FineTuneScope:
        return self._finetune_scope

    def _finetune_module(self) -> nn.Module:
        if self.finetune_scope == "last_block":
            return self.encoder.layer4[-1]
        return self.encoder.layer4

    def set_stage(self, stage: TrainingStage):
        """Switch between 'head' (encoder frozen) and 'finetune' (chosen scope unfrozen)."""
        if stage not in ("head", "finetune"):
            raise ValueError("stage must be 'head' or 'finetune'")

        self.encoder.requires_grad_(False)
        if stage == "finetune":
            self._finetune_module().requires_grad_(True)
        self.classifier.requires_grad_(True)

        self._stage = stage
        self.zero_grad(set_to_none=True)
        self.train(self.training)

    def train(self, mode: bool = True) -> ResNet50Classifier:
        """Apply training mode while keeping frozen layers and BatchNorm running stats in eval."""
        super().train(mode)
        self.encoder.eval()

        if mode and self.stage == "finetune":
            module = self._finetune_module()
            module.train()
            for layer in module.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()

        return self

    def forward_features(self, images: Tensor) -> Tensor:
        """Return spatial features: [B, 2048, 7, 7] for 224x224 inputs."""
        return self.encoder(images)

    def encode(self, images: Tensor) -> Tensor:
        """Return pooled visual embeddings shaped [B, 2048]."""
        features = self.forward_features(images)
        return self.pool(features).flatten(start_dim=1)

    def forward(self, images: Tensor) -> Tensor:
        """Return raw logits shaped [B, num_classes]."""
        return self.classifier(self.dropout(self.encode(images)))


def load_inaturalist_encoder(
    model: ResNet50Classifier,
    checkpoint_path: str | Path,
    expected_sha256: str,
) -> None:
    """Verify SHA-256 and load the iNat2021 ResNet-50 encoder weights, dropping the 10k head."""
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", expected_sha256,
    ):
        raise ValueError("pretrained_sha256 must contain 64 hexadecimal characters.")
    path = Path(checkpoint_path)
    with path.open("rb") as stream:
        digest = sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        if digest.hexdigest() != expected_sha256.lower():
            raise ValueError("iNaturalist checkpoint SHA-256 does not match the configuration.")
        stream.seek(0)
        checkpoint = torch.load(stream, map_location="cpu", weights_only=True)

    if not isinstance(checkpoint, Mapping) or not isinstance(
        checkpoint.get("state_dict"), Mapping,
    ):
        raise ValueError("iNaturalist checkpoint must contain a state_dict mapping.")
    state = {}
    for key, value in checkpoint["state_dict"].items():
        if not isinstance(key, str) or not isinstance(value, Tensor):
            raise ValueError("iNaturalist state_dict must map names to tensors.")
        name = key.removeprefix("module.")
        if name in state:
            raise ValueError(f"Duplicate iNaturalist parameter: {name}.")
        state[name] = value

    expected = model.encoder.state_dict()
    expected_shapes = {name: tensor.shape for name, tensor in expected.items()}
    expected_shapes.update({
        "fc.weight": (10000, model.feature_dim), "fc.bias": (10000,),
    })
    if state.keys() != expected_shapes.keys():
        raise ValueError("iNaturalist checkpoint keys do not match a complete ResNet-50.")
    for name, shape in expected_shapes.items():
        if state[name].shape != shape:
            raise ValueError(f"Unexpected iNaturalist parameter shape: {name}.")
    model.encoder.load_state_dict(
        {name: state[name] for name in expected}, strict=True,
    )


@dataclass(frozen=True)
class RandomFitAndPad:
    """Fit the full image inside a white square at a random scale in [min_scale, 1] and position."""

    size: int
    min_scale: float = 0.85

    def __post_init__(self) -> None:
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 1:
            raise ValueError("size must be a positive integer.")
        if (
            isinstance(self.min_scale, bool) or not isinstance(self.min_scale, (int, float))
            or not math.isfinite(self.min_scale) or not 0 < self.min_scale <= 1
        ):
            raise ValueError("min_scale must be finite and in (0, 1].")

    def __call__(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            raise ValueError("RandomFitAndPad expects EXIF-corrected RGB input from load_rgb.")
        fraction = self.min_scale + (1 - self.min_scale) * torch.rand(()).item()
        longest = max(1, min(self.size, math.floor(self.size * fraction + 0.5)))
        factor = longest / max(image.size)
        width = max(1, min(self.size, math.floor(image.width * factor + 0.5)))
        height = max(1, min(self.size, math.floor(image.height * factor + 0.5)))
        left = torch.randint(self.size - width + 1, ()).item()
        top = torch.randint(self.size - height + 1, ()).item()
        canvas = Image.new("RGB", (self.size, self.size), (255, 255, 255))
        with image.resize((width, height), Image.Resampling.BILINEAR) as resized:
            canvas.paste(resized, (left, top))
        return canvas


def resnet50_transform(
    *, training: bool, image_size: int = 224, geometry: str = "fit",
    augmentation: str = "baseline",
) -> T.Compose:
    """Build ResNet-50 preprocessing with optional 'mild' or 'spatial' training augmentation."""
    if augmentation not in ("baseline", "mild", "spatial"):
        raise ValueError("augmentation must be 'baseline', 'mild' or 'spatial'.")
    if augmentation == "spatial" and geometry != "fit":
        raise ValueError("Spatial augmentation requires fit geometry to preserve image content.")
    base = classification_transform(
        training=training, image_size=image_size, geometry=geometry,
        normalization=IMAGENET_NORMALIZATION,
    )
    if not training or augmentation == "baseline":
        return base

    if augmentation == "spatial":
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(brightness=0.15, contrast=0.15)], p=0.8),
            T.RandomRotation(
                degrees=10, interpolation=InterpolationMode.BILINEAR,
                expand=True, fill=(255, 255, 255),
            ),
            RandomFitAndPad(image_size),
            T.PILToTensor(), T.ConvertImageDtype(torch.float32),
            T.Normalize(*IMAGENET_NORMALIZATION),
        ])

    operations = list(base.transforms)
    tensor_index = next(
        index for index, operation in enumerate(operations)
        if isinstance(operation, T.PILToTensor)
    )
    operations[tensor_index:tensor_index] = [
        T.RandomApply([T.ColorJitter(brightness=0.15, contrast=0.15)], p=0.8),
        T.RandomRotation(
            degrees=10, interpolation=InterpolationMode.BILINEAR,
            expand=True, fill=(255, 255, 255),
        ),
        T.Resize(
            (image_size, image_size), interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
    ]
    return T.Compose(operations)


def load_classifier(
    checkpoint_path: str | Path, *, device: str = "cpu",
) -> tuple[ResNet50Classifier, dict]:
    """Load a saved ResNet-50 checkpoint and its preprocessing metadata."""
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'.")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Expected a project ResNet-50 checkpoint mapping.")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint is missing its saved configuration.")
    data, model_config = config.get("data"), config.get("model")
    provenance = config.get("data_provenance")
    if not all(isinstance(section, dict) for section in (data, model_config, provenance)):
        raise ValueError("Checkpoint requires data, model and data_provenance metadata.")
    if config.get("architecture", "resnet50") != "resnet50":
        raise ValueError("Only the ordinary ResNet-50 architecture is supported.")
    num_classes = checkpoint.get("num_classes")
    names = provenance.get("class_names")
    if (
        isinstance(num_classes, bool) or not isinstance(num_classes, int)
        or num_classes < 2 or not isinstance(names, list)
        or len(names) != num_classes
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != num_classes
        or model_config.get("num_classes", num_classes) != num_classes
    ):
        raise ValueError("Saved class names must match the classifier's ordered classes.")
    if data.get("normalization") != "imagenet":
        raise ValueError("Unsupported saved normalization; expected 'imagenet'.")
    resnet50_transform(
        training=False, image_size=data.get("image_size"), geometry=data.get("geometry"),
    )
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint is missing model_state_dict.")
    model = ResNet50Classifier(
        num_classes=num_classes, weights=None,
        dropout=model_config.get("dropout", 0),
        finetune_scope=model_config.get("finetune_scope", "layer4"),
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, config


@torch.inference_mode()
def predict_image(
    image_path: str | Path, checkpoint_path: str | Path, *,
    device: str = "cpu", top_k: int = 5,
) -> dict:
    """Classify one local image using a saved ResNet-50 checkpoint."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    image_path = Path(image_path).resolve(strict=True)
    if not image_path.is_file():
        raise ValueError("image_path must be a local image file.")
    checkpoint_path = Path(checkpoint_path).resolve(strict=True)
    model, config = load_classifier(checkpoint_path, device=device)
    names = config["data_provenance"]["class_names"]
    if top_k > len(names):
        raise ValueError(f"top_k must not exceed the number of classes ({len(names)}).")
    data = config["data"]
    transform = resnet50_transform(
        training=False, image_size=data["image_size"], geometry=data["geometry"],
    )
    with load_rgb({"path": str(image_path)}) as rgb:
        images = transform(rgb).unsqueeze(0).to(device)
    logits = model(images)
    if logits.shape != (1, len(names)) or not torch.isfinite(logits).all():
        raise ValueError("Model returned invalid or nonfinite class logits.")
    scores, indices = logits.softmax(dim=1)[0].topk(top_k)
    return {
        "image": str(image_path), "checkpoint": str(checkpoint_path),
        "device": device,
        "preprocessing": {key: data[key] for key in ("image_size", "geometry", "normalization")},
        "predictions": [
            {"class_id": index, "class_name": names[index], "confidence": score}
            for index, score in zip(indices.cpu().tolist(), scores.cpu().tolist())
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Local bird image.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Saved project model.pt.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, help="Optional new JSON output path.")
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be a positive integer.")
    if args.output and args.output.exists():
        parser.error("--output already exists; choose a new file.")
    torch.set_num_threads(2)
    try:
        result = predict_image(
            args.image, args.checkpoint, device=args.device, top_k=args.top_k,
        )
        content = json.dumps(result, indent=2, allow_nan=False)
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(content + "\n")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Prediction failed: {error}\n")
    print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
