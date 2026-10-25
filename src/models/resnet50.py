"""ResNet-50 classification with a reusable visual encoder."""
from __future__ import annotations

from collections import OrderedDict
from typing import Literal

from torch import Tensor, nn
from torchvision.models import ResNet50_Weights, resnet50


TrainingStage = Literal["head", "finetune"]
FineTuneScope = Literal["layer4", "last_block"]


class ResNet50Classifier(nn.Module):
    """Classify RGB images and expose spatial and pooled visual features.

    Inputs are preprocessed floating-point tensors shaped [B, 3, H, W].
    Forward returns raw class logits. Encoder BatchNorm running statistics
    remain fixed in both training stages.
    """

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

    def set_stage(self, stage: TrainingStage) :
        """Train the head alone, or the head plus the chosen encoder scope.

        Call this at a stage boundary and rebuild the optimizer afterward.
        Fine-tuning includes the chosen scope's BatchNorm affine parameters, while its
        running statistics stay fixed.
        """
        if stage not in ("head", "finetune"):
            raise ValueError("stage must be 'head' or 'finetune'")

        self.encoder.requires_grad_(False)
        if stage == "finetune":
            self._finetune_module().requires_grad_(True)
        self.classifier.requires_grad_(True)

        self._stage = stage
        self.zero_grad(set_to_none=True)
        self.train(self.training)

    def train(self, mode: bool = True) -> ResNet50Classifier: #mode is True for training 
        """Apply training mode while preserving the encoder freezing policy."""
        super().train(mode)
        self.encoder.eval()

        if mode and self.stage == "finetune":
            module = self._finetune_module()
            module.train()
            for layer in module.modules():
                if isinstance(layer, nn.BatchNorm2d): 
                    layer.eval() # keep running stats fixed

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
