# Stage 1B-1: official pretrained GPU preflight

The [notebook](../notebooks/pretrained/01_gpu_preflight.ipynb) runs the existing
ConvNeXt-Tiny, EfficientNet-B3 and RegNetY-3.2GF adapters through
`src.training.runtime.run_gpu_preflight`. It checks genuine official Torchvision
weights, features, finite loss/backward gradients, fine-tuning policies, supported
AMP, memory and resident-batch inference throughput. It performs no optimizer
updates, tuning, model selection, or substantive training.

| Evidence | Verified state after returned Colab runs |
| --- | --- |
| Shared verification code, thin notebook and setup instructions | COMPLETED |
| Focused CPU-safe import, config, cache, comparison and report tests | TESTED locally; see PR verification report for counts |
| All three exact official checkpoints and retained backbone states | VERIFIED in returned Colab evidence |
| All three models, frozen/partial/full, FP32 training backward and validation inference | VERIFIED: 18/18 checks passed |
| First run: EfficientNet-B3 and RegNetY-3.2GF BF16 execution | 12/12 checks passed; this T4 has no native BF16 support |
| First run: ConvNeXt-Tiny BF16 execution on this T4 | FAILED: 6/6 checks; incorrect automatic precision selection |
| All three models, frozen/partial/full, FP32 and explicit FP16 | VERIFIED: 36/36 checks passed in the returned rerun |
| Corrected automatic precision on T4 | VERIFIED: targeted ConvNeXt-Tiny 12/12 FP32/FP16 checks; native BF16 false |
| Targeted automatic run evidence ZIP/inventory and storage readback | VERIFIED; PR 13 merged as `6ad1adfe47f339414693c2034658a32bc6df2ca6` |
| GPU memory and short resident-batch FP32 inference measurements | Recorded in returned evidence; not training benchmarks |
| Fresh-session training restoration and Kaggle offline packaging | PENDING |

Configured weight names, synthetic checkpoints, mocked CUDA calls and CPU smoke
tests do not establish genuine pretrained/GPU execution. The local environment is
CPU-only with no official cached weights; cloud claims above come from the user's
returned evidence, not local tests. A CPU preflight exports `pending_gpu`,
without fetching source images or weights. Missing checks and failures remain
explicit in evidence; a successful notebook cell alone is insufficient.

## First Colab findings and AMP correction

The returned run used clean commit
`d2b221630becc0b9d11363f031248dec633c175e`, a Tesla T4 (compute capability 7.5),
Python 3.13.16, Torch 2.11.0+cu130, Torchvision 0.26.0+cu130 and cuDNN 92700.
Canonical preparation matched, representative train/validation pixels loaded,
and official test pixels were not accessed. All three official artifact hashes
and retained backbone comparisons passed. Overall, 30 of 36 checks passed; all
six failures were ConvNeXt BF16 checks, across both partitions and all policies.
Storage was not requested, so acceptance correctly remained incomplete.

The original automatic selection called `torch.cuda.is_bf16_supported()` with its
default `including_emulation=True`. The T4 can represent BF16 tensors, but that
does not establish native BF16 convolution support. Automatic selection now asks
for `including_emulation=False`: it chooses FP16 on this T4 and BF16 when native
support exists. Explicit BF16 on an emulation-only device is reported unsupported;
failed checks are never silently retried in a different precision. Runtime facts
retain the broad `bfloat16_supported` flag and add `bfloat16_native_supported`.
See [PyTorch's capability implementation](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/cuda/__init__.py).

Preserve the first failed report. Its JSON SHA-256 is
`893c653c2bb6148b0f3aff2591eeccd560b1ae0498bcd7d440aa9c570ba6e046`;
the evidence ZIP SHA-256 is
`ea491df7b74a4b45828069d0841c8561bab9a9caf9884b4357812c4af55f0b4d`.
The ZIP inventory and standalone report were independently checked. Generated
evidence stays outside Git. The measured inference times cover only ten images
after warm-up; they cannot establish architecture rankings or training budgets.

The subsequent all-three-model rerun explicitly selected `amp="float16"` and
passed 36/36 FP32/FP16 checks. A later clean published correction run targeted
ConvNeXt-Tiny with `amp="auto"` and passed 12/12 checks, selecting FP16 with native
BF16 support false. Its report, evidence ZIP/inventory and storage readback were
verified before PR 13 merged as `6ad1adfe47f339414693c2034658a32bc6df2ca6`.
The earlier failures remain part of the record. These are zero-update preflight
proofs, not training metrics or fresh-session training recovery proofs.
No Torch/CUDA reinstall was needed; source and weight caches were preserved.

## Next: first canonical training acceptance

Use [02_train_convnext.ipynb](../notebooks/pretrained/02_train_convnext.ipynb) on
online Colab GPU at its full published training PR SHA. It reuses the tested
bootstrap/metadata patterns below, mounts Drive, loads the acceptance JSON and
calls `run_training` once. The initial ConvNeXt-Tiny run freezes two epochs with
physical batch 8, accumulation 2, FP16, seed 42 and workers 0. Batch 8 is a
conservative choice from T4 preflight headroom; actual resource use is measured
by the notebook, with no duration promise. Existing CUDA packages and global
pixel/weight caches are preserved. Do not install CPU requirements in Colab.

Keep the stable Drive directory
`/content/drive/MyDrive/featherlens/experiments/convnext-frozen-acceptance-v1` and
set `RESUME=True` explicitly to recover with the same code/config/preparation.
Return `run_summary.json` and `evidence.zip`, retaining the entire experiment,
checkpoints, integrity sidecars and `mlruns` separately. See
[training and recovery](pretrained_cnn_foundation.md#classification-training-and-recovery)
for exact outputs, CLI/API and limitations. Expected updates are 600 only if all
steps succeed; actual history is authoritative. **Real canonical training metrics,
updates, cloud checkpoints and fresh-session restoration remain PENDING.**

## First run: online Colab GPU

1. Open [Google Colab](https://colab.research.google.com/), choose **File → Upload
   notebook**, and select `notebooks/pretrained/01_gpu_preflight.ipynb` from the
   published PR checkout. This also works when Colab cannot browse the private repo.
2. Choose **Runtime → Change runtime type**, select a **GPU** hardware accelerator,
   and save. Connect the runtime. Availability/quota determines the actual device.
3. Run section 1. Inspect Python, Torch/Torchvision versions, CUDA availability and
   actual device. If CUDA is absent, fix the runtime selection before seeking GPU
   evidence. Preserve a working CUDA stack; do not install
   `requirements-classification.txt` or the documented Windows CPU wheel index.
4. In section 2 set `CLOUD="colab"`, paste the **full published Stage 1B-1 commit
   SHA** from the PR into `PUBLISHED_COMMIT`, and leave `REPO_DIR` as
   `/content/featherlens` unless supplying a checkout. Do not enter a branch name,
   the Stage 1A baseline, or an invented future SHA. Run the cell before project
   imports. It checks out and verifies that exact commit and rejects modified
   tracked source. The report records the actual Git HEAD.
5. For the private repo, keep `PRIVATE_REPOSITORY=True` and enter a fine-grained
   GitHub token with repository **Contents: read** access at the hidden `getpass`
   prompt. Never paste it into a cell, repository URL or shell command. The
   temporary askpass helper and its subprocess environment are cleaned after the
   clone; the Git remote contains only the credential-free HTTPS URL. For public
   access set the flag to false. A failed clone leaves an incomplete directory;
   remove that incomplete clone explicitly or select a new `REPO_DIR` before retrying.
   An alternative is uploading a clean published checkout including `.git` and
   pointing `REPO_DIR` there; an ordinary source ZIP without Git provenance is
   insufficient.
6. Run section 3. It inspects installed dependencies and runs `pip check`. Only if
   the listed non-Torch packages are missing, review them, set
   `INSTALL_MISSING_DATA_DEPS=True`, and rerun. Installation uses the current
   interpreter and only missing non-Torch direct dependencies. Existing versions
   are preserved and final versions are recorded. An incompatible Torch/Torchvision
   pair requires a deliberate accelerator-compatible repair and runtime restart;
   the notebook never blindly reinstalls it.
   The global `pip check` is strict: unrelated preinstalled-package conflicts can
   also stop this cell. Review the reported conflicts before making targeted
   environment changes; do not automatically upgrade the entire cloud environment.
7. Upload the canonical `preparation.json` and
   `classification_manifest.jsonl` through the **Files → Upload** control into
   `/content/classification-v1`, then set `PREPARED_DIR` to that directory. Or
   upload the small handoff ZIP, set `PREPARED_ZIP=Path("/content/<filename>.zip")`
   and use an empty `PREPARED_DIR`. The helper reads only those two known entries
   and refuses duplicate entries or overwrites. The ZIP supplies metadata, **not
   images**. Do not fabricate local image directories.
8. For optional Drive persistence, set `MOUNT_GOOGLE_DRIVE=True` in section 4 and
   authorize the Drive mount. Set `PERSISTENT_DIR` to a new directory such as
   `Path("/content/drive/MyDrive/featherlens/preflight-<unique-run>")`. Supply paths
   before section 8. Preparation, output and persistent directories must be
   separate and must not contain one another. Leave `OFFLINE=False` for this first
   run (`OFFLINE` is set in section 2 before Hugging Face imports). The default output is a unique ignored `outputs/gpu-preflight/<run-id>`
   directory within the repo. Keep `batch_size=2` and `num_workers=0` initially;
   any later change is recorded, and no GPU capacity is assumed.
   Set a fresh mounted Drive `PERSISTENT_DIR` when seeking complete storage
   acceptance; leaving it `None` intentionally keeps storage verification pending.
9. Run sections 4–7 in order. Section 5 validates canonical metadata before source
   access. Section 6 only displays configured official identities. The runtime
   loads pinned Hugging Face **official train** separately when section 8 executes;
   train/validation representatives use the saved membership. No new audit or
   split is performed, and no official test images are loaded.
10. Run section 8 once, then sections 9–12. Models execute sequentially, with
    resource release between them. Failures are preserved. If a batch exhausts
    memory, retain its failure evidence, then choose a fresh output directory for
    a deliberate smaller-batch rerun; do not erase the failed report.
11. Read the top-level status, failures, architecture coverage, checkpoint hashes,
    device facts, and every policy/precision check. Download **both
    `preflight-report.json` and `preflight-evidence.zip`** from the printed
    `OUTPUT_DIR` using Colab's Files pane. Alternatively set
    `DOWNLOAD_COLAB_EVIDENCE=True` in section 12 and rerun that cell. Return those
    two files for review. `evidence-inventory.json` supplies hashes and is included
    in the compact bundle. No weights or images belong in the return files.
12. Clear notebook outputs before sharing a saved notebook. Verify any Drive
    copies after reconnecting in a fresh session if claiming durability. Successful
    immediate write/readback proves only access in the current session.

The online path needs network access and enough session disk space for pinned
train data and the three official checkpoints. No runtime duration, memory budget
or cloud GPU type is promised by local tests.

## Canonical data and preprocessing

`validate_preflight_preparation` uses the existing full preparation validator and
requires manifest SHA-256
`a8bd05e048febf32df6d84a6e1bc16eddf6416be723e12dfc89621044962736b`, seed 42,
4,794 train / 1,200 validation / 5,794 official test metadata rows, no exclusions,
and the unchanged ordered 200-class mapping. It records preparation and ordered
class-map hashes too. Metadata validation is separate from source accessibility.

The only image source requested is `Multimodal-Fatima/CUB_train` at
`bd3988995e6db193427255abdebd522afe24ce76`, accessed through the existing source
and data interfaces. Validation images are members of this same official-training
source. The report verifies source mapping before image use. Test provenance
remains in metadata; the official test image dataset is not fetched.

Full-image EXIF-corrected RGB fitting into white 224×224 is unchanged. ImageNet
normalization runs once, training flips remain train-only, and native weight crop
recipes are descriptive metadata. The runner records expected/actual 200-logit,
spatial and pooled shapes and checks frozen/partial/full policies without optimizer
updates. Backward and gradient checks use training batches only; validation uses
`eval()` and `inference_mode()` for forward, loss and feature checks. An AMP backward
check does not constitute a training run.

## Kaggle online and offline preparation

For an **online Kaggle run**, create a notebook, import the same `.ipynb`, choose
**Settings → Accelerator → GPU** and enable **Internet**. Set `CLOUD="kaggle"`;
the writable root is `/kaggle/working`, and attached datasets are read-only under
`/kaggle/input/<dataset-name>/`. Set `PREPARED_DIR` to the attached canonical
metadata directory, or extract its two known files into `/kaggle/working`.
Clone the exact published source commit using section 2 or attach a clean Git
checkout including `.git` and point `REPO_DIR` to it. For a read-only attached
checkout, set `OUTPUT_DIR` to a fresh `/kaggle/working/preflight-<unique-run>`
directory. Follow the remaining Colab steps with those path changes. Kaggle
`/kaggle/working` is session storage until **Save Version** saves the notebook
outputs; download from the Output pane and verify them after reopening that saved
version. Setting a second directory there is not independent durable storage.

An **internet-disabled Kaggle run is a separate, unverified staging path**. Before
turning internet off, a compatible Linux cloud environment must stage:

- A clean source checkout at the published commit, with `.git` provenance, and
  the two canonical preparation files. Keep source, metadata and outputs separate.
- Only the exact pinned **official train** dataset snapshot, including its data
  shards and repository metadata needed for non-streaming random access. Populate
  a new dataset-only Hugging Face cache online with
  `huggingface_hub.snapshot_download(repo_id="Multimodal-Fatima/CUB_train",
  repo_type="dataset", revision="bd3988995e6db193427255abdebd522afe24ce76")`.
  Offline, the runner resolves that snapshot with `local_files_only=True`, then
  loads it from its local path; it does not ask the Hub for remote metadata.
  Preserve the cache's snapshot/blob layout and test it with `OFFLINE=True` in a
  fresh disconnected environment before claiming it works. Do not stage official
  test pixels, invent image directory layouts, or copy a complete personal
  `HF_HOME`, tokens, unrelated caches or credentials. Set `HF_CACHE_DIR` in
  section 2 to an appropriate writable copy of this dataset-only cache, before
  section 3 imports Hugging Face. This snapshot staging has not been executed here.
- The three **exact official checkpoint filenames** printed by section 6, copied
  from a verified Torch Hub cache into `<hub-root>/checkpoints/`. Set
  `torch.hub.set_dir(<hub-root>)` before section 8. Preserve the official filenames;
  the verifier checks the supported filename hash prefix and records the full
  SHA-256. Missing or corrupt files fail rather than falling back to random weights.
- A compatible CUDA Torch/Torchvision stack already installed, plus required data
  dependencies and their transitive dependencies for the cloud Python/Linux
  platform. Inspect versions and `pip check`. If offline packages are necessary,
  stage only a deliberately prepared compatible Linux wheel set; the Windows CPU
  virtual environment/wheelhouse is unsuitable. No bundled offline wheel set is
  provided or verified here.

Set `OFFLINE=True` in section 2 before importing/loading Hugging Face source data in a fresh
session; the runner also requests local-only source loading and offline weight
lookup. Metadata, cached source schema/rows and weight integrity still require
verification. Online Colab execution does not depend on completing this packaging.

## Evidence interpretation and return

The report records actual Git commit/dirty status, allowlisted runtime versions,
CUDA/device facts, canonical metadata identities, requested and supported coverage,
official artifact identity/full SHA-256, retained-backbone comparison, head and
feature checks, policy/precision statuses, loss/gradients, peak memory, measured
batch size, failures and persistence readback. No environment or credential dump
is included. The compact ZIP contains report/inventory evidence, not model weights
or images. Existing run files are not silently overwritten.

Throughput means warmed, synchronized GPU inference on one resident representative
batch; it excludes download and end-to-end data-loading time. Backward peak memory
is reported separately. A subset of architectures cannot establish all-three-model
coverage. Unsupported or skipped precision and partial failures are not passes.
Read `acceptance_complete` and `acceptance_pending` as well as diagnostic `status`;
an all-model diagnostic pass without required AMP or storage evidence can still
leave acceptance pending. Immediate readback cannot establish fresh-session durability.

Return `preflight-report.json` and `preflight-evidence.zip` even when the run fails
or reports pending. Review the JSON before declaring any model cloud-verified.
Fresh-session storage durability requires a later read of the same artifacts.
Preflight evidence has now been returned and reviewed as described above; the
next experiment is the bounded Stage 1B2 training notebook, whose genuine
training evidence remains pending.
