"""Optional ResNet-50 augmentation layered over the shared EDA preprocessing."""

from dataclasses import dataclass
import math

from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from src.data.transforms import IMAGENET_NORMALIZATION, classification_transform


@dataclass(frozen=True)
class RandomFitAndPad:
    """Fit the whole image at random scale/position without cropping.

    Torch controls all random draws for worker and seed replay. Aspect ratio
    is retained within pixel rounding; the resized image stays in the canvas.
    """

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
    """Keep validation unchanged; optionally augment training images gently.

    Mild augmentation runs after the shared fit/crop. Spatial augmentation
    rotates native images before fitting and varies size/placement without a
    crop; it requires fit geometry. Neither explicitly jitters hue/saturation.
    """
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
