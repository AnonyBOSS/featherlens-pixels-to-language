# Stage 1B-1 pretrained GPU preflight implementation plan

**Authority:** User approval attachment 6a6130c6-92f1-4d3f-94a1-6c1db0662a2e.
**Base:** latest main `41f7686830d8940166b9e718cdb235c127292f15`.
**Branch:** `feat/pretrained-gpu-preflight`, isolated managed worktree.

## Global constraints

- Only ConvNeXt-Tiny V1, EfficientNet-B3 V1, RegNetY-3.2GF V2 official Torchvision weights.
- Existing Stage 1A adapters and Task 1.1 source/manifests/transforms stay unchanged.
- Canonical manifest SHA a8bd05e048febf32df6d84a6e1bc16eddf6416be723e12dfc89621044962736b;
  seed42, 4794/1200/5794, 200 ordered classes, no resplitting or test-image access.
- Baseline full-image fit/white224/one ImageNet normalization and train-only flips.
- No optimizer updates, trainer, tracking backend, recovery engine, CLI, tuning,
  model selection, CAM rendering, extra architecture or teammate implementation.
- No weights/data/logs/caches/environments committed; no secrets in notebook/output.
- Local runtime is CPU-only and has no official cached weights. Actual official/GPU
  execution remains PENDING unless genuinely performed; tests never download weights.
- Publish reviewed, tested commits and PR; no merge or Stage 1B-2.

## Task 1 — Official weight verification

Own `src/training/weights.py`, `tests/training/test_preflight_weights.py`.

Interfaces:
- `resolve_weight_info(config: ModelConfig) -> dict`: pinned qualified ID, official
  URL, filename, supported filename hash prefix, Torch/Torchvision versions.
- `prepare_official_weight(config, *, offline=False, progress=True) -> dict`:
  Torch Hub checkpoints directory; cached file reused, no duplicate download;
  missing offline file fails before network. Use supported Torch Hub download
  facilities with official hash prefix; compute/record full SHA-256 and bytes.
  Existing corrupt checkpoints fail clearly, without random fallback.
- `verify_backbone_state(model, state_dict) -> dict`: exact shape/dtype/value match
  of all retained parameters and buffers, including ConvNeXt encoder_norm.
  Reject missing/unexpected non-head state; exclude only native 1000-class head.
- `build_verified_pretrained_model(config, *, offline=False, progress=True)` returns
  `(model, proof_dict)`. Use existing factory with load_weights=True after cache
  preparation; strict weights-only CPU state reading for independent comparison;
  prove 200-class trainable replacement head. No implementation of another registry.

TDD: offline ID/cache/download failures; real adapter + synthetic native checkpoint
comparisons including buffer corruption, missing/extra keys and shape/dtype mismatch.
Mock only external download/cache seams; never label synthetic fixture as official evidence.

## Task 2 — Runtime, GPU checks and evidence

Own `src/training/runtime.py`, `tests/training/test_gpu_preflight.py`.

Interfaces:
- `PreflightConfig(prepared_dir, output_dir, persistent_dir=None, architectures=(three),
  batch_size=2, num_workers=0, seed=42, offline=False, warmup_batches=2,
  measured_batches=5, amp="auto")`: validated simple dataclass; no experiment framework.
- `collect_runtime(repo_dir=None) -> dict`: allowlisted versions and actual CUDA/device
  facts; Git commit/dirty status if available, no environment/credentials dump.
- `run_gpu_preflight(config) -> dict`: complete structured report plus evidence export.
  Canonical metadata validated before source fetch. If CUDA is absent, return pending_gpu,
  never CPU fallback/pass; weights and GPU checks remain not_run. No weight download locally.

With CUDA: load representative existing train/val batches only, verify source mapping,
use Task 1 verified builder, execute models sequentially, release objects/graphs between.
For frozen/partial/full: actual FP32 and supported AMP training forward/backward;
validation uses eval/inference_mode without backward. Check finite loss and unscaled
training gradients, meaningful permitted gradients/frozen masks, logits/spatial/pooled
shapes. No optimizer.step; a temporary optimizer solely for AMP unscaling is permitted.
Respect the existing adapter train()/BN policies. Record AMP overflows/failures honestly.
Time warmed, synchronized GPU inference on a resident representative batch; label that
measurement accurately, separately from backward memory and end-to-end data loading.
Record peak allocated/reserved memory per check and measured batch size/device.

Report: schema/status, Git/version/device facts, canonical manifest/class-map identities,
explicit requested/supported architecture coverage, verified artifact identities/digests,
per-model/per-policy/per-precision outcomes, shapes/loss/gradient/memory/throughput,
failures and persistence write/read evidence. Preserve partial failures; no false pass.
Write JSON and a compact evidence ZIP with hashes, containing no weights/images/tokens.
Storage readback does not prove fresh-session durability; document that limit explicitly.
Do not silently overwrite a previous run report; output/persistent paths must preserve inputs.

TDD: CPU pending/no downloads, config/input/source failures, report export/readback,
testable CPU smoke primitives clearly labelled CPU, and mocked orchestration with no GPU
claims. Tests must not treat mocked device calls as executed cloud verification.

## Task 3 — Thin notebook and portable documentation

Own `notebooks/pretrained/01_gpu_preflight.ipynb`, `docs/gpu_preflight.md`, README.md,
and `tests/notebooks/test_gpu_preflight_notebook.py` for meaningful structure/import/config
verification. Use stdlib notebook JSON; no new authoring dependencies.

Twelve clear notebook sections follow user brief: runtime, repo, dependencies, source,
manifest/classes, weights, model/features, GPU forward/backward, AMP, measurements,
persistent storage, summary/export. Code invokes shared modules, no copied model/trainer.
Prioritize online Colab. Explicit Colab/Kaggle paths; no assumption of GPU type/quota.
Private-repo setup must not embed tokens in code, command arguments, logs or Git remotes.
Inspect/preserve compatible installed CUDA Torch/Torchvision; no blanket CPU requirement
installation. Install missing non-Torch data dependencies only deliberately and record
versions. Document offline staging without claiming it was run.
Prepared ZIP contains metadata only; optionally extract only its two known preparation
files, validate canonical identity, access pinned HF train data separately. Never make
another split or load official test images. User can mount/upload prepared files.
No copying complete HF_HOME/credentials, Windows wheelhouse or invented data files.

Give exact manual Colab steps, published-commit checkout guidance, expected evidence ZIP/
JSON filenames to return, and COMPLETED/TESTED/PENDING evidence table. Actual official
weights/GPU execution is pending locally. No scientific results or runtime guesses.

## Task 4 — Verification, independent review and publication

Root coordinates task gates and final whole-branch review. Run full regression suite with
cached real-data opt-in (existing 614 baseline), no network-dependent weights. Exercise
actual local CPU-pending report and compile/smoke notebook setup safely. Verify dependency
compatibility, ignored artifacts, no secrets and whitespace. Create incremental commits,
push and open a PR that distinguishes code readiness from pending manual cloud evidence.
Stop after PR and verification report. Preserve unrelated files and existing worktrees.

## Review focus

- All retained tensors, including buffers/norm, are verified against file contents.
- Missing offline files and incompatible checkpoints never lead to random fallback.
- No test, CPU fallback or metadata flag is represented as executed official/GPU proof.
- FP16 gradients are inspected after proper unscaling; no optimizer update occurs.
- Partial failure evidence, storage limits and architecture coverage are explicit.
- Notebook bootstrap/private access and evidence avoid credentials and unsafe extraction.
