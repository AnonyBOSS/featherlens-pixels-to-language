# Official pretrained GPU preflight — Stage 1B1

This is the historical October 25–26, 2026 cloud-preflight workflow and evidence
record. Use the [pretrained handoff](pretrained_handoff.md) for current inference
setup, the selected ensemble and measured results, and the
[training reference](pretrained_cnn_foundation.md#classification-training-and-recovery)
for training/recovery. The initial handoff's CPU-only environment and pending
training status below describe that stage, not the project's current capabilities.

The [preflight notebook](../notebooks/pretrained/01_gpu_preflight.ipynb) calls
`src.training.runtime.run_gpu_preflight` for ConvNeXt-Tiny, EfficientNet-B3 and
RegNetY-3.2GF. It checks genuine official torchvision weights, reusable features,
finite loss/backward gradients, fine-tuning policies, supported AMP, memory and
resident-batch inference throughput. It performs **zero optimizer updates**.

## Returned evidence and AMP correction

| Evidence from the returned Colab runs | Result |
| --- | --- |
| All three exact official checkpoints and retained backbone states | Verified |
| All models, frozen/partial/full, FP32 train backward + validation inference | 18/18 passed |
| First run: EfficientNet-B3 and RegNetY-3.2GF BF16 | 12/12 passed on a T4 without native BF16 |
| First run: ConvNeXt-Tiny BF16 | 6/6 failed; incorrect automatic precision selection |
| All models/policies, FP32 and explicit FP16 rerun | 36/36 passed |
| Corrected automatic precision, targeted ConvNeXt-Tiny | 12/12 FP32/FP16 passed; native BF16 false |
| Targeted report, ZIP/inventory and storage readback | Verified; PR 13 merged as `6ad1adfe47f339414693c2034658a32bc6df2ca6` |
| GPU memory and short resident-batch FP32 inference measurements | Recorded, not training benchmarks |
| Fresh-session training restoration and Kaggle offline packaging | Not established by these preflight runs |

The first run used clean commit `d2b221630becc0b9d11363f031248dec633c175e`,
Tesla T4 (compute capability 7.5), Python 3.13.16, Torch 2.11.0+cu130,
Torchvision 0.26.0+cu130 and cuDNN 92700. Canonical metadata and representative
train/validation pixels matched; no official test pixels were accessed. All
artifact hashes/backbone comparisons passed. Overall 30/36 checks passed; the
six ConvNeXt BF16 failures covered both partitions and all policies. Storage was
not requested, so this first run correctly left acceptance incomplete.

The original automatic selection used `torch.cuda.is_bf16_supported()` with its
default `including_emulation=True`. Representable BF16 tensors do not establish
native BF16 convolution support. Automatic selection now asks for
`including_emulation=False`, choosing FP16 on this T4 and BF16 with native support.
Explicit BF16 on an emulation-only device is unsupported; failed checks are never
silently retried at another precision. Runtime facts retain `bfloat16_supported`
and add `bfloat16_native_supported`. See the
[PyTorch 2.11 capability implementation](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/cuda/__init__.py).

Preserve the first failed report and its independently checked ZIP/inventory:

- JSON SHA-256: `893c653c2bb6148b0f3aff2591eeccd560b1ae0498bcd7d440aa9c570ba6e046`.
- Evidence ZIP SHA-256: `ea491df7b74a4b45828069d0841c8561bab9a9caf9884b4357812c4af55f0b4d`.

The all-model rerun explicitly used `amp="float16"`; the later clean correction
run targeted ConvNeXt with `amp="auto"`. Both passed the coverage listed above;
the correction's report, inventory and storage readback were verified before PR 13
merged. No Torch/CUDA reinstall was needed; source/weight caches were preserved.
Inference timings covered only ten images after warm-up, so they cannot rank
architectures or establish training budgets. Generated evidence stays outside Git.

At the historical preflight stage the local environment was CPU-only, without
cached official weights; these GPU claims came from returned cloud evidence.
CPU preflight exports `pending_gpu` without fetching images or weights. CPU tests,
configured identities, synthetic checkpoints and mocked CUDA calls alone cannot
establish genuine pretrained/GPU execution. Focused CPU-safe import/config/cache/
comparison/report tests were recorded in the PR verification report.

The October 26 training handoff next proposed the bounded
[ConvNeXt acceptance notebook](../notebooks/pretrained/02_train_convnext.ipynb):
two frozen epochs, batch 8, accumulation 2, FP16, seed 42 and workers 0. Genuine
training metrics, updates, checkpoints and fresh-session restoration were then
pending. That historical status does not supersede later canonical CUDA results;
use the current handoff and individual experiment artifacts for those claims.

## First run: online Colab GPU

1. In [Google Colab](https://colab.research.google.com/), use **File → Upload
   notebook** to open `notebooks/pretrained/01_gpu_preflight.ipynb` from the published
   checkout. This works when Colab cannot browse the private repository.
2. Select **Runtime → Change runtime type → GPU**, save and connect. In section 1,
   inspect actual Python, Torch/Torchvision, CUDA and device facts. Fix runtime
   selection if CUDA is absent. Preserve a working accelerator stack; do not install
   `requirements-classification.txt` or the Windows CPU wheel index here.
3. Before project imports, set `CLOUD="colab"` and the **full published Stage 1B1
   commit SHA** as `PUBLISHED_COMMIT` in section 2. Default `REPO_DIR` is
   `/content/featherlens`. Use the exact published revision, not a branch name or
   Stage 1A baseline. The helper checks out/verifies it, rejects modified tracked
   source and records actual HEAD. A clean supplied checkout must include `.git`;
   an ordinary source ZIP lacks required provenance.
4. For a private repo, keep `PRIVATE_REPOSITORY=True` and enter a fine-grained
   token with repository **Contents: read** access at the hidden `getpass` prompt.
   Keep it out of cells, URLs and shell commands. Temporary askpass/environment
   credentials are cleaned and the remote is credential-free. Public access uses
   `False`. After a failed clone, explicitly remove the incomplete clone or choose
   another `REPO_DIR` before retrying.
5. Section 3 inspects dependencies and runs strict `pip check`. If listed non-Torch
   packages are missing, review them, set `INSTALL_MISSING_DATA_DEPS=True` and rerun.
   It uses the current interpreter, installs only missing non-Torch dependencies,
   preserves existing versions and records final versions. Review unrelated global
   conflicts before targeted repairs. An incompatible Torch/Torchvision pair needs
   a deliberate accelerator-compatible repair/restart; do not upgrade everything.
6. Upload canonical `preparation.json` and `classification_manifest.jsonl` through
   **Files → Upload** to `/content/classification-v1`; set `PREPARED_DIR` there.
   Alternatively upload the small handoff ZIP, set
   `PREPARED_ZIP=Path("/content/<filename>.zip")` and use an empty `PREPARED_DIR`.
   The helper reads only the two known entries, refusing duplicates/overwrites.
   Metadata supplies no images; do not fabricate local image directories.
7. Optional persistence: set `MOUNT_GOOGLE_DRIVE=True` in section 4, authorize the
   mount and choose a fresh `PERSISTENT_DIR`, such as
   `Path("/content/drive/MyDrive/featherlens/preflight-<unique-run>")`. Preparation,
   output and persistence paths must be separate and not contain each other.
   Supply paths before section 8. A mounted Drive destination is required for
   complete storage acceptance; `None` intentionally leaves storage pending.
8. Keep `OFFLINE=False` (set in section 2 before Hugging Face imports), initial
   `batch_size=2` and `num_workers=0`. Default output is a unique ignored
   `outputs/gpu-preflight/<run-id>` directory. No GPU capacity is assumed; changes
   are recorded. Run sections 4–7 in order: section 5 validates canonical metadata,
   section 6 displays configured official identities, and section 8 later loads
   pinned **official train** pixels with saved train/validation membership.
9. Run section 8 once, then 9–12. Models run sequentially with resource release.
   Retain all failures. After out-of-memory, preserve evidence and choose a fresh
   output directory for an intentional smaller-batch rerun.
10. Inspect status, failures, architecture/policy/precision coverage, checkpoint
    hashes and device facts. Download **both `preflight-report.json` and
    `preflight-evidence.zip`** from printed `OUTPUT_DIR` via Files, or set
    `DOWNLOAD_COLAB_EVIDENCE=True` in section 12 and rerun it. The ZIP includes
    hashed `evidence-inventory.json`, not weights/images. Return both files even
    for failed/pending runs. Clear notebook outputs before sharing the notebook.
11. Reconnect in a fresh session and verify the same Drive artifacts before
    claiming durability; immediate write/readback proves current-session access only.

The online path needs network access and session disk for pinned train data and
three official checkpoints. Local tests promise no duration, memory budget or GPU type.

## Canonical data and preprocessing

`validate_preflight_preparation` runs the full preparation validator and requires
manifest SHA-256 `a8bd05e048febf32df6d84a6e1bc16eddf6416be723e12dfc89621044962736b`,
seed 42, **4,794 train / 1,200 validation / 5,794 official-test metadata rows**,
no exclusions and unchanged ordered 200 classes. Preparation/class-map hashes
are recorded too. Metadata validity does not establish source accessibility.

Only `Multimodal-Fatima/CUB_train` at
`bd3988995e6db193427255abdebd522afe24ce76` supplies pixels through the existing
data/source interfaces. Validation belongs to this official-training source;
mapping is checked before image use. No new audit/split or official test-image
fetch occurs; test provenance remains metadata.

EXIF-corrected full RGB images fit white 224×224 with one ImageNet normalization.
Flips are train-only; native crop recipes stay descriptive. The runner checks
200-logit/spatial/pooled shapes and frozen/partial/full policies. Backward/gradient
checks use train batches only; validation uses `eval()` and `inference_mode()`.
An AMP backward check remains a zero-update preflight, not training.

## Kaggle online and offline preparation

For **online Kaggle**, import the same notebook, enable **Settings → Accelerator
→ GPU** and **Internet**, and set `CLOUD="kaggle"`. Writable root is
`/kaggle/working`; attached datasets under `/kaggle/input/<dataset-name>/` are
read-only. Point `PREPARED_DIR` to attached canonical metadata or extract its two
files into working storage. Clone the exact published commit with section 2, or
attach a clean checkout including `.git` and set `REPO_DIR`. A read-only checkout
needs a fresh writable `OUTPUT_DIR`, such as `/kaggle/working/preflight-<unique-run>`.
Follow the Colab steps with these paths. **Save Version** retains session outputs;
download via Output and verify after reopening the saved version. Two working
directories do not provide independent durable storage.

**Internet-disabled Kaggle remains an unverified staging path.** Before disabling
internet, prepare the following in a compatible Linux cloud environment:

- Clean published source with `.git`, plus both canonical metadata files. Keep
  source, metadata and outputs separate.
- Only the exact official-training snapshot, including data shards/repository
  metadata for non-streaming random access. Populate a new dataset-only cache:

  ```python
  from huggingface_hub import snapshot_download
  snapshot_download(repo_id="Multimodal-Fatima/CUB_train", repo_type="dataset",
                    revision="bd3988995e6db193427255abdebd522afe24ce76")
  ```

  The offline runner resolves with `local_files_only=True`, then loads the local
  path without remote metadata. Preserve snapshot/blob layout; set `HF_CACHE_DIR`
  in section 2 to a writable cache copy before section 3 imports. Test `OFFLINE=True`
  in a fresh disconnected environment. Do not stage test pixels, invented image
  layouts, a personal `HF_HOME`, unrelated caches, tokens or credentials. This
  snapshot staging has not been verified by the retained preflight evidence.
- Three exact official checkpoint filenames from section 6, copied from a verified
  Torch Hub cache into `<hub-root>/checkpoints/`. Call `torch.hub.set_dir(<hub-root>)`
  before section 8. Filenames/hash prefixes and full SHA-256 remain checked; missing
  or corrupt weights fail without random fallback.
- Compatible installed CUDA Torch/Torchvision and required data/transitive packages.
  Inspect versions and `pip check`; any offline wheels must be a deliberately
  compatible Linux set. The Windows CPU environment/wheelhouse is unsuitable.
  No offline wheel bundle is supplied or verified.

Set `OFFLINE=True` in section 2 **before** Hugging Face imports/loading in a fresh
session. Source schema/rows, metadata and weight integrity still need verification.
Online Colab does not depend on finishing this packaging.

## Evidence interpretation and return

Reports record actual Git HEAD/dirty status, allowlisted versions, CUDA/device
facts, metadata identities, requested/supported coverage, official identity/full
hash, retained-backbone comparison, head/features, policies/precisions, loss/gradients,
peak memory, batch size, failures and persistence readback. No environment or
credential dump is included. Existing outputs are never silently overwritten.

Throughput is warmed, synchronized GPU inference on one resident representative
batch, excluding downloads and data-loading time. Backward peak memory is separate.
Architecture subsets cannot prove all-model coverage; unsupported/skipped precision
or partial failures are not passes. Read `acceptance_complete`/`acceptance_pending`
alongside diagnostic `status`: an all-model pass can still lack required AMP/storage
acceptance. Review the JSON and inventory before declaring cloud verification;
fresh-session durability needs a later read of the same artifacts. Retain failed
reports alongside successful reruns and keep training claims tied to training evidence.
