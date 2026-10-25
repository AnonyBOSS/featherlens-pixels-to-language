# Stage 1B-1: official pretrained GPU preflight

The [notebook](../notebooks/pretrained/01_gpu_preflight.ipynb) runs the existing
ConvNeXt-Tiny, EfficientNet-B3 and RegNetY-3.2GF adapters through
`src.training.runtime.run_gpu_preflight`. It checks genuine official Torchvision
weights, features, finite loss/backward gradients, fine-tuning policies, supported
AMP, memory and resident-batch inference throughput. It performs no optimizer
updates, tuning, model selection, or substantive training.

| Evidence | State before a manual cloud run |
| --- | --- |
| Shared verification code, thin notebook and setup instructions | COMPLETED |
| Focused CPU-safe import, config, cache, comparison and report tests | TESTED locally; see PR verification report for counts |
| ConvNeXt-Tiny `ConvNeXt_Tiny_Weights.IMAGENET1K_V1` actual checkpoint/GPU | PENDING |
| EfficientNet-B3 `EfficientNet_B3_Weights.IMAGENET1K_V1` actual checkpoint/GPU | PENDING |
| RegNetY-3.2GF `RegNet_Y_3_2GF_Weights.IMAGENET1K_V2` actual checkpoint/GPU | PENDING |
| Actual GPU/AMP execution, measured memory and throughput | PENDING; no local measurements |
| Fresh-session persistence and Kaggle offline packaging | PENDING |

Configured weight names, synthetic checkpoints, mocked CUDA calls and CPU smoke
tests do not establish genuine pretrained/GPU execution. The local environment is
CPU-only with no official cached weights. A CPU preflight exports `pending_gpu`,
without fetching source images or weights. Missing checks and failures remain
explicit in evidence; a successful notebook cell alone is insufficient.

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
Fresh-session storage durability requires a later read of the same artifacts. The
smallest next step is one online Colab GPU run at the published commit, then review
its genuine evidence before proposing Stage 1B-2 or a first training run.
