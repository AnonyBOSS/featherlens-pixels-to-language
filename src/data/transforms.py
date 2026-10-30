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


@dataclass(frozen=True)
class RandomStretch:
    """Randomly stretch or compress image width and height independently.

    Applies anisotropic scaling along the horizontal and vertical axes to simulate
    aspect-ratio variation, camera focal length stretching, and bird posture changes.
    """

    p: float = 0.5
    scale_x: tuple[float, float] = (0.85, 1.18)
    scale_y: tuple[float, float] = (0.85, 1.18)
    fill: tuple[int, int, int] = (255, 255, 255)

    def __call__(self, image: Image.Image) -> Image.Image:
        if float(torch.rand(1).item()) > self.p:
            return image
        sx = float(torch.empty(1).uniform_(self.scale_x[0], self.scale_x[1]).item())
        sy = float(torch.empty(1).uniform_(self.scale_y[0], self.scale_y[1]).item())
        orig_w, orig_h = image.size
        new_w = max(1, int(round(orig_w * sx)))
        new_h = max(1, int(round(orig_h * sy)))
        resized = image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (orig_w, orig_h), self.fill)
        canvas.paste(resized, ((orig_w - new_w) // 2, (orig_h - new_h) // 2))
        return canvas


@dataclass(frozen=True)
class RandomRadialDistortion:
    """Random barrel or pincushion optical lens distortion via radial polynomial mapping.

    Simulates optical camera lens aberrations (radial bowing outwards or inwards)
    using torch.nn.functional.grid_sample.
    """

    p: float = 0.4
    distortion_scale: tuple[float, float] = (-0.15, 0.15)
    fill: float = 1.0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if float(torch.rand(1).item()) > self.p:
            return tensor
        k = float(torch.empty(1).uniform_(self.distortion_scale[0], self.distortion_scale[1]).item())
        if abs(k) < 1e-4:
            return tensor
        c, h, w = tensor.shape
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, dtype=tensor.dtype, device=tensor.device),
            torch.linspace(-1.0, 1.0, w, dtype=tensor.dtype, device=tensor.device),
            indexing="ij",
        )
        r2 = x * x + y * y
        distortion = 1.0 + k * r2
        x_dist = x * distortion
        y_dist = y * distortion
        grid = torch.stack((x_dist, y_dist), dim=-1).unsqueeze(0)
        out = torch.nn.functional.grid_sample(
            tensor.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0)
        mask = (x_dist < -1.0) | (x_dist > 1.0) | (y_dist < -1.0) | (y_dist > 1.0)
        out[:, mask] = self.fill
        return out


@dataclass(frozen=True)
class RandomAugmentMix:
    """Randomly selects and applies a mix/combination of augmentations.

    For each input image, samples a randomized combination of k transformations
    from the candidate pool of geometric, optical, and perspective operations.
    """

    ops: tuple[object, ...]
    num_ops: tuple[int, int] = (1, 3)

    def __call__(self, image: Image.Image) -> Image.Image:
        if not self.ops:
            return image
        min_k, max_k = self.num_ops
        k = int(torch.randint(min_k, max_k + 1, (1,)).item())
        indices = torch.randperm(len(self.ops))[:k]
        for idx in indices:
            image = self.ops[idx.item()](image)
        return image


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
        operations: list[object] = [_FitAndPad(image_size)]
    else:
        resize = math.floor(image_size * 256 / 224 + 0.5)
        operations = [T.Resize(resize, interpolation=InterpolationMode.BILINEAR, antialias=True)]
        operations.append(T.RandomCrop(image_size) if training else T.CenterCrop(image_size))
    if training:
        operations.append(T.RandomHorizontalFlip(p=0.5))
        if strong_augment:
            pool: list[object] = [
                # Translation:
                T.RandomAffine(
                    degrees=0,
                    translate=(0.08, 0.08),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=(255, 255, 255),
                ),
                # Rotation:
                T.RandomRotation(
                    degrees=(-16, 16),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=(255, 255, 255),
                ),
                # Stretching:
                RandomStretch(p=1.0, scale_x=(0.88, 1.14), scale_y=(0.88, 1.14), fill=(255, 255, 255)),
                # Shearing:
                T.RandomAffine(
                    degrees=0,
                    shear=(-10, 10, -10, 10),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=(255, 255, 255),
                ),
                # Combined geometric affine:
                T.RandomAffine(
                    degrees=(-12, 12),
                    translate=(0.06, 0.06),
                    shear=(-8, 8, -8, 8),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=(255, 255, 255),
                ),
                # Optical variations ("go crazy"):
                T.RandomAdjustSharpness(sharpness_factor=1.4, p=1.0),
                T.RandomAutocontrast(p=1.0),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
            ]
            if image_size >= 4:
                # Lens perspective distortion:
                pool.append(
                    T.RandomPerspective(distortion_scale=0.20, p=1.0, fill=(255, 255, 255))
                )
            operations.append(RandomAugmentMix(ops=tuple(pool), num_ops=(1, 3)))
    operations.extend([T.PILToTensor(), T.ConvertImageDtype(torch.float32)])
    if training and strong_augment:
        # Radial optical lens distortion:
        operations.append(RandomRadialDistortion(p=0.35, distortion_scale=(-0.15, 0.15), fill=1.0))
    if normalization is not None:
        operations.append(T.Normalize(tuple(mean), tuple(std)))
    if training and strong_augment:
        # Cutout / erasing:
        operations.append(T.RandomErasing(p=0.20, scale=(0.02, 0.12), value=0.0))
    return T.Compose(operations)
