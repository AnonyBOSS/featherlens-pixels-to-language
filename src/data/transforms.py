"""Classification image transforms; no parameters are fitted from source data."""
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from PIL import Image

import torch
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


IMAGENET_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


@dataclass(frozen=True)
class _FitAndPad:
    size: int

    def __call__(self, image: Image.Image) -> Image.Image:
        scale = self.size / max(image.size)
        width = max(1, min(self.size, math.floor(image.width * scale + 0.5)))
        height = max(1, min(self.size, math.floor(image.height * scale + 0.5)))
        canvas = Image.new('RGB', (self.size, self.size), (255, 255, 255))
        with image.resize((width, height), resample=Image.Resampling.BILINEAR) as resized:
            canvas.paste(resized, ((self.size - width) // 2, (self.size - height) // 2))
        return canvas


def classification_transform(
    *, training: bool, image_size: int = 224, geometry: Literal["fit", "crop"] = "fit",
    normalization: tuple[Sequence[float], Sequence[float]] | None = IMAGENET_NORMALIZATION,
    strong_augment: bool = False,
) -> T.Compose:
    """Build train or deterministic evaluation preprocessing.

    EXIF-corrected RGB input comes from load_rgb. The default geometry='fit'
    scales the entire image into a centered white square without cropping;
    aspect ratio is retained within integer pixel rounding. Training adds a
    0.5 horizontal flip (and affine/color/erasing augmentations when
    ``strong_augment=True``). geometry='crop' retains the original shorter-side
    resize and random(train)/center(eval) crop profile. Output is float32 CHW.
    normalization=None keeps [0,1]; otherwise provide (three means, three stds).
    The ImageNet default is an external constant, never estimated from CUB.
    """
    if not isinstance(training, bool):
        raise ValueError('training must be bool')
    if not isinstance(strong_augment, bool):
        raise ValueError('strong_augment must be bool')
    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size <= 0:
        raise ValueError('image_size must be a positive integer')
    if normalization is not None:
        try:
            mean, std = normalization
            if len(mean) != 3 or len(std) != 3:
                raise ValueError
            for value in (*mean, *std):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError
            if any(value <= 0 for value in std):
                raise ValueError
        except (TypeError, ValueError) as error:
            raise ValueError('normalization must be None or (three finite means, three positive stds)') from error
    if geometry not in ('fit', 'crop'):
        raise ValueError("geometry must be 'fit' or 'crop'")
    if geometry == 'fit':
        operations = [_FitAndPad(image_size)]
    else:
        resize = math.floor(image_size * 256 / 224 + 0.5)
        operations = [T.Resize(resize, interpolation=InterpolationMode.BILINEAR, antialias=True)]
        operations.append(T.RandomCrop(image_size) if training else T.CenterCrop(image_size))
    if training:
        operations.append(T.RandomHorizontalFlip(p=0.5))
        if strong_augment:
            operations.extend([
                T.RandomAffine(
                    degrees=15,
                    translate=(0.08, 0.08),
                    scale=(0.88, 1.12),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=(255, 255, 255),
                ),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            ])
    operations.extend([T.PILToTensor(), T.ConvertImageDtype(torch.float32)])
    if normalization is not None:
        operations.append(T.Normalize(tuple(mean), tuple(std)))
    if training and strong_augment:
        operations.append(T.RandomErasing(p=0.2, scale=(0.02, 0.12), value=0.0))
    return T.Compose(operations)
