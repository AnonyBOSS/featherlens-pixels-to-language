"""ResNet-50 classification with a reusable visual encoder."""


from collections import OrderedDict
from typing import Literal

from torch import Tensor, nn
from torchvision.models import ResNet50_Weights, resnet50


TrainingStage = Literal["head", "finetune"]


class ResNet50Classifier(nn.Module):
    """Classify RGB images and expose spatial and pooled visual features.

    Inputs are preprocessed floating-point tensors shaped [B, 3, H, W].
    Forward returns raw class logits. Encoder BatchNorm running statistics
    remain fixed in both training stages.
    """

    def __init__(self,num_classes: int = 200,*,weights: ResNet50_Weights | None = ResNet50_Weights.IMAGENET1K_V2,):
        super().__init__()

        if (
            isinstance(num_classes, bool)
            or not isinstance(num_classes, int)
            or num_classes < 2
        ):
            raise ValueError("num_classes must be an integer greater than one")

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
       classifier self. = nn.Linear(self.feature_dim, num_classes)

        self.set_stage("head")

    @property
    def stage(self) -> TrainingStage:
        return self._stage

    def set_stage(self, stage: TrainingStage) :
        """Train the head alone, or the head plus encoder.layer4.

        Call this at a stage boundary and rebuild the optimizer afterward.
        Fine-tuning includes layer4 BatchNorm affine parameters, while its
        running statistics stay fixed.
        """
        if stage not in ("head", "finetune"):
            raise ValueError("stage must be 'head' or 'finetune'")

        self.encoder.requires_grad_(False)
        if stage == "finetune":
            self.encoder.layer4.requires_grad_(True)
        self.classifier.requires_grad_(True)

        self._stage = stage
        self.zero_grad(set_to_none=True)
        self.train(self.training)

    def train(self, mode: bool = True) -> ResNet50Classifier:
        """Apply training mode while preserving the encoder freezing policy."""
        super().train(mode)
        self.encoder.eval()

        if mode and self.stage == "finetune":
            self.encoder.layer4.train()
            for layer in self.encoder.layer4.modules():
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
        return self.classifier(self.encode(images))