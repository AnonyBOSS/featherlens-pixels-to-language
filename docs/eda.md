# Exploratory Data Analysis (CUB-200-2011)

Statistical EDA covers all **5,994 official training images** and **59,940 captions**, plus a comparison of the derived **4,794 train / 1,200 validation** splits. Official test images (`5,794`) are excluded from EDA.

## Reproducing reports and figures

All EDA code lives under `src/eda/`. Run commands from the repository root after setting `HF_HUB_OFFLINE` / `HF_DATASETS_OFFLINE` only if the training dataset is already cached locally:

```powershell
$reports = "outputs/eda/reports"
$figures = "outputs/eda/figures"

python -m src.eda.explore --full-training --split train --output "$reports/train-full.json"
python -m src.eda.plots --report "$reports/train-full.json" --output-dir $figures
python -m src.eda.images --output "$reports/train-images-full.json" --figures-dir $figures
python -m src.eda.captions --output "$reports/train-caption-details-full.json" --figures-dir $figures
python -m src.eda.prepared --prepared-dir preparation --image-report "$reports/train-images-full.json" --caption-report "$reports/train-caption-details-full.json" --output "$reports/prepared-training-eda.json" --figures-dir $figures
```

To regenerate PNG/PDF figure groups directly from saved JSON reports (reopening only the few gallery images):

```python
import json
from pathlib import Path
from src.data.source import load_cub_split
from src.eda.detail_plots import generate_caption_detail_figures, generate_prepared_eda_figures
from src.eda.image_plots import generate_image_figures
from src.eda.images import prepare_gallery
from src.eda.plots import generate_training_figures

reports = Path("outputs/eda/reports")
out = Path("outputs/eda/figures")
read = lambda name: json.loads((reports / name).read_text(encoding="utf-8"))

generate_training_figures(read("train-full.json"), out)
images = read("train-images-full.json")
generate_image_figures(images, out, image_getter=prepare_gallery(images, load_cub_split("train")))
generate_caption_detail_figures(read("train-caption-details-full.json"), out)
generate_prepared_eda_figures(read("prepared-training-eda.json"), out)
```

Generated reports (`train-full.json`, `train-images-full.json`, `train-caption-details-full.json`, `prepared-training-eda.json`) and figures (PNG + PDF) are written under `outputs/eda/` and ignored by Git.

## Summary of findings (5,994 official training images)

### Class balance and image decoding
- **Class balance:** All 200 species appear (`194` classes with 30 images, `6` with 29; mean `29.97`, max/min ratio `1.0345`). The six 29-image classes are *Common Raven*, *Nelson Sharp tailed Sparrow*, *Bank Swallow*, *Artic Tern*, *Red cockaded Woodpecker*, and *House Wren* (dataset spelling preserved).
- **Decoding coverage:** `5,994 / 5,994` decoded cleanly (`5,990` RGB, `4` grayscale `L` converted to RGB, `3,004` with an embedded ICC key, `0` alpha/transparency). EXIF orientation values were absent in `5,502`, `0` in `490`, and `1` in `2`, with no orientation-swapped dimensions.

### Native image geometry and appearance
Equal-image means measured on EXIF-corrected native RGB arrays (not transformed/padded tensors):

| Metric | Min | Median | p95 | Max | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Width (px) | 121 | 500 | 500 | 500 | — |
| Height (px) | 120 | 375 | 500 | 500 | — |
| Aspect ratio (`W/H`) | 0.4880 | 1.3333 | 1.5337 | 3.5971 | — |
| Megapixels | 0.01946 | 0.17850 | 0.23650 | 0.25000 | 0.17940 |
| Brightness (`mean Y`) | 0.0823 | 0.4834 | 0.7032 | 0.9577 | 0.4876 |
| Contrast (`std Y`) | 0.0381 | 0.1737 | 0.2767 | 0.4166 | 0.1779 |
| Saturation (`mean HSV S`) | 0.0000 | 0.2945 | 0.6128 | 0.9537 | 0.3140 |

- Per-image R/G/B means average to `[0.4856, 0.4994, 0.4324]` with average per-image standard deviations `[0.1817, 0.1811, 0.1927]`.
- Notable extremes reflect background as much as the bird: row `1196` (Olive-sided Flycatcher, max brightness `0.9577`, large pale sky), row `4270` (Caspian Tern, min brightness `0.0823`), row `4259` (Caspian Tern, min contrast `0.0381`), row `281` (Red-winged Blackbird, max contrast `0.4166`), row `726` (Pelagic Cormorant, grayscale saturation `0`), and row `464` (Painted Bunting, max saturation `0.9537`).

### Captions and vocabulary (59,940 captions)
- **Length & vocabulary:** Every training image has `10` nonempty captions. Word counts average `15.2504` (`median 14`, `p95 24`, `p99 32`, range `1–65`; shortest is `"this"` at row `1171`, caption `7`); character counts average `77.4612` (`median 70`, `p95 129`, `p99 168`, range `4–332`). Across `914,107` tokens there are `4,701` distinct word types, including `2,308` singletons (`49.10%` of types, `0.2525%` of tokens).
- **Repetition:** `50,477` unique cleaned strings leave `9,463` repeated occurrences (`15.7875%`): `639` extra occurrences occur within the same image (`583` images, `9.7264%`), and `8,824` occur across different images (`1,534` generic descriptions; top shared caption appears `601` times across `501` images).
- **Lexicon & diversity:** Under the `caption-terms-v1` lexicon, color terms appear in `97.494%` of captions (`58,438`, all `5,994` images), anatomy terms in `97.227%` (`58,278`, all images), and direction/position terms in `0.868%` (`520` captions / `482` images; `left` appears 3 times, `right` 20 times). Median per-image TTR is `0.3351` (`p05 0.2739`, `p95 0.4000`), and median pairwise Jaccard across an image's 45 caption pairs is `0.3047` (`p05 0.2383`, `p95 0.3850`). By species mean length, Fish Crow is shortest (`12.9133` words) and European Goldfinch is longest (`17.3433` words).

### Derived train vs. validation check

| Equal-image mean | Train (`n=4,794`) | Validation (`n=1,200`) |
| --- | ---: | ---: |
| Native brightness | 0.4868 | 0.4910 |
| Native contrast | 0.1774 | 0.1800 |
| Native saturation | 0.3137 | 0.3154 |
| Mean caption words | 15.2484 | 15.2582 |
| Per-image TTR | 0.3350 | 0.3360 |
| Per-image mean Jaccard | 0.3079 | 0.3064 |

**Practical takeaways:** Fit any dataset-level pixel stats or token vocabularies strictly on the `4,794` train subset (not the full `5,994`), account for the duplicate Pigeon Guillemot pair kept in `val`, keep aspect-preserving white padding since dimensions vary widely (`0.49–3.60` aspect ratio), and note that contrast (`std(Y)`) does not measure blur/sharpness.
