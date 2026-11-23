# Classification Models and Evaluation

FeatherLens includes three classification tracks on the `4,794 / 1,200 / 5,794` CUB-200-2011 split:

1. **ConvNeXt-Tiny + ConvNeXt-V2 Base ensemble** (main classifier — `src/models/pretrained.py`, `src/training/`, `src/evaluation/`)
2. **iNaturalist2021 ResNet-50** (`src/resnet50/`, `configs/resnet50/`)
3. **Scratch CNN** — see [`docs/classification_training_progress.md`](classification_training_progress.md) (`configs/cls.yaml`, `bin/{train,tune,test}`).

---

## 1. ConvNeXt Ensemble (90.30% Test Top-1)

### Supported pretrained adapters

`src/models/pretrained.py` (`ModelConfig`, `create_pretrained_model`) supports four CNN backbones (`stride=32`) with a separated `head` (dropout + 200-class linear layer) and `forward_features()` returning `{"spatial", "pooled"}`:

| Architecture | Default weights | Channels | Partial fine-tune layers | Grad-CAM target |
| --- | --- | ---: | --- | --- |
| `convnext_tiny` | `ConvNeXt_Tiny_Weights.IMAGENET1K_V1` | 768 | `encoder[6]`, `encoder[7]`, `encoder_norm` | `encoder.7` |
| `efficientnet_b3` | `EfficientNet_B3_Weights.IMAGENET1K_V1` | 1,536 | `encoder[7]`, `encoder[8]` | `encoder.8` |
| `regnet_y_3_2gf` | `RegNet_Y_3_2GF_Weights.IMAGENET1K_V2` | 1,512 | `encoder.trunk_output.block4` | `encoder.trunk_output.block4` |
| `convnextv2_base` | `timm/convnextv2_base.fcmae_ft_in22k_in1k_384` | 1,024 | `encoder.stages[3]`, `encoder_norm` | `encoder.stages.3` |

ConvNeXt-Tiny also supports `facebook/convnext_tiny.in22k_ft_in1k_384` (revision `32c741ba106f9e062174ef829354a16719c3dcac`). `convnextv2_base` is pinned to revision `bea9dc139d5981aff6ce4d6d9690086e45ad8b56` (`timm==1.0.30`, `safetensors==0.8.0`; weights carry a CC-BY-NC-4.0 license).

### Ensemble protocol and measured results

The delivered bundle (`classifier/bundle.json` with `tiny.pt`, `base.pt`, and `.sha256` sidecars) uses full-image white padding and ImageNet normalization `(0.485, 0.456, 0.406)` / `(0.229, 0.224, 0.225)`:

- **Tiny (`tiny.pt`):** 448×448, CUDA FP16 microbatch 8 (`float32` on CPU).
- **Base (`base.pt`):** 384×384, FP32 encoder microbatch 4 + CPU FP32 Ridge head (its train-only score scale is already folded into the saved head weights — do not rescale again).
- Predictions average both components' softmax probabilities **50/50** on the original unflipped image:

| Partition | Images | Top-1 | Macro precision | Macro recall | Macro F1 | Top-5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 1,200 | 90.9167% | 91.9721% | 90.9167% | 90.6110% | 98.4167% |
| Official test | 5,794 | 90.3003% | 90.6041% | 90.3735% | 90.1246% | 98.1878% |

A separate validation-only horizontal-flip TTA check reached `91.00%` top-1 (not the default protocol; no test run was performed). `94%` was not reached.

### CLI and encoder reuse

```powershell
# Single-image prediction & Grad-CAM (use --device cuda if available)
python -m src.cli predict --bundle classifier/bundle.json --image path/to/bird.jpg --device cpu
python -m src.cli gradcam --bundle classifier/bundle.json --image path/to/bird.jpg --component tiny --output outputs/cam-tiny.png --device cpu
python -m src.cli gradcam --bundle classifier/bundle.json --image path/to/bird.jpg --component base --output outputs/cam-base.png --device cpu

# Validation / official test evaluation
python -m src.cli evaluate --bundle classifier/bundle.json --prepared-dir preparation --partition val --batch-size 8 --device cuda --output-dir outputs/validation-check
python -m src.cli evaluate --bundle classifier/bundle.json --prepared-dir preparation --partition test --allow-official-test --batch-size 8 --device cuda --output-dir outputs/test-check

# Training runner (or run notebooks/train_convnext.ipynb on Colab/Kaggle GPU)
python -m src.cli train --task classification --config configs/pretrained_acceptance.json --prepared-dir preparation --run-dir outputs/experiments/convnext-run --device cuda --tracking mlflow
```

`run_training(..., resume=True)` recovers from `latest.pt` (optimizer, scaler, scheduler, and RNG state saved per completed epoch). `initial_head` optionally loads a frozen-probe head on the same backbone and manifest before training.

Reusing one exported component for downstream feature extraction:

```python
import torch
from src.data.image_io import load_rgb
from src.evaluation.inference import load_component
from src.training.data import PreprocessingConfig

model, metadata = load_component("classifier/tiny.pt", device="cpu")
norm = metadata["preprocessing"]["normalization"]
recipe = PreprocessingConfig(
    image_size=metadata["preprocessing"]["image_size"],
    normalization=None if norm is None else (norm["mean"], norm["std"]),
)
with load_rgb({"path": "path/to/bird.jpg"}) as image:
    batch = recipe.build_transform(training=False)(image).unsqueeze(0)
with torch.no_grad():
    features = model.forward_features(batch)
# Tiny at 448 -> spatial [N, 768, 14, 14], pooled [N, 768]
# Base at 384 -> spatial [N, 1024, 12, 12], pooled [N, 1024]
```

---

## 2. iNaturalist ResNet-50 (87.54% Test Top-1)

Code lives in `src/resnet50/` with YAML configs in `configs/resnet50/` (`inaturalist.yaml` default, `imagenet.yaml`, `regularized.yaml`, `spatial.yaml`).

For iNaturalist2021 pretraining, download `inat2021_supervised_large_from_scratch.pth.tar` (from the NEWT benchmark archive) and place it at `outputs/pretrained/inat2021_supervised_large_from_scratch.pth.tar`. The runner verifies its SHA-256 against `configs/resnet50/inaturalist.yaml`:
`e845bd2512b7728c390523afc8696dafedf2675c93e9d7508a48b416cda8e8d9`.

### Workflow commands

```powershell
# 1. Batch check & 224px training (3 head epochs + up to 10 layer4+head epochs)
python -m src.resnet50.train --config configs/resnet50/inaturalist.yaml --check-batch
python -m src.resnet50.train --config configs/resnet50/inaturalist.yaml --train

# 2. Refit regularized LDA head at 320px (archived encoder SHA-256: a6ed5697a0a7555292e84ab2219848428cda7a18eb43fe9e02dff43c88eeb221)
python -m src.resnet50.head_refit --source-checkpoint <best.pt> --method discriminant --shrinkage 0.05 --image-size 320 --batch-size 16 --train-horizontal-flip --device cuda

# 3. Prediction, single/gallery Grad-CAM, and evaluation
python -m src.resnet50.model --image path/to/bird.jpg --checkpoint <model.pt> --device cpu --top-k 5
python -m src.resnet50.gradcam --image path/to/bird.jpg --checkpoint <model.pt> --output outputs/gradcam/bird.png --device cpu
python -m src.resnet50.gradcam gallery --checkpoint <model.pt> --output-dir outputs/gradcam/gallery --device cpu
python -m src.resnet50.evaluate --checkpoint <model.pt> --device cuda --output-dir outputs/resnet50_inaturalist/evaluation

# 4. Grid vs. random LDA shrinkage comparison (feature cache SHA-256: b01132aaeb6b902bc284d3dee924093b424de5a44296941509187106099263c3)
python -m src.resnet50.search --features <finetuned_features.pt> --source-checkpoint <best.pt> --output-dir outputs/resnet50_inaturalist/search_rerun --trials 8 --seed 42
```

### Measured 320px checkpoint results and LDA shrinkage search

| Split | Images | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 4,794 | 98.06% | 98.17% | 98.06% | 98.06% |
| Validation | 1,200 | 87.92% | 89.61% | 87.92% | 87.91% |
| Official test | 5,794 | 87.54% | 88.29% | 87.65% | 87.62% |

Note: these are saved single-seed checkpoint scores where overfitting remains, validation was repeatedly reused during head exploration, and earlier test scores were inspected.

The grid vs. random search study tuned LDA shrinkage in `[0.03, 0.95]` (`8` trials each, seed `42`, frozen 320px original-view features only, 2 CPU threads):

| Method | Trials | Best shrinkage | Val accuracy | Val macro F1 | Fit + score time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Grid | 8 | 0.049147 | 87.9167% | 87.8934% | 12.80s |
| Random | 8 | 0.435032 | 87.9167% | 87.8052% | 11.87s |
