# Classification Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans after the user approves this proposal. Steps use checkbox syntax for tracking. Implementation and isolated dependency installation are approved. On October 22, 2026 the user authorized the Stage 3 commit, push and PR creation after final verification. Merge still requires separate user approval.

**Status:** Approved on October 21, 2026 with the user refinements recorded below. Both checkpoints implemented and verified; final publication suite 280 passed in 70.00s. The user approved publication on October 22, 2026, subject to final Git, documentation, dependency and test checks. Merge remains unapproved.

**Goal:** Deliver a reproducible classification-ready CUB pipeline for the October 24 internal deadline.

**Architecture:** Read the completed Stage 2 audit into a generated preparation manifest. Apply explicit eligibility decisions and a group-constrained class-stratified split, then expose lazily decoded images through a small PyTorch Dataset/DataLoader interface. Preserve immutable source records and keep generated decisions separate.

**Tech stack:** Existing Python 3.11+ source contract, Hugging Face datasets, Pillow, fsspec, NumPy, PyTorch, torchvision, pytest. Test a compatible torch/torchvision pair on CPU before recording exact dependency versions; GPU tuning belongs to training.

**Spec:** User request of October 21, 2026, summarized below; [source contract](../../data_contract.md); [audit contract](../../integrity_audit.md).

## Request and proposed choices

The user requested classification eligibility/cleaning, reproducible stratified train/validation splitting, grouping the exact-duplicate Pigeon Guillemot records, preprocessing and training-only augmentation, reusable PyTorch integration, tests, README updates, and classification-team handoff.

Approved implementation settings:

- Retain both valid Pigeon Guillemot records and constrain them to one partition.
- Validation fraction 0.20, seed 42, per-class target `floor(n * 0.20 + 0.5)`.
- User-approved refinement after training previews: default full-image aspect-preserving fit within a centered white 224 x 224 canvas for train/val/test; training-only flip probability 0.5. Original resize/crop profile remains explicit via geometry="crop". Normalization follows image preparation.
- Configurable normalization: `normalization=None` preserves [0,1]; a `(mean, std)` pair supplies model-specific constants. ImageNet channel mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)` as the documented default preprocessing profile. These are external constants, not estimates from CUB validation/test. The Dataset accepts caller-supplied transforms for later model-specific profiles.
- Batch size 32; `num_workers=0` and `pin_memory=False` by default; expose both settings.
- Implement sequentially in this chat, with independent review at the end, to keep coupled interfaces consistent and control usage.

## Global constraints

- Internal deadline: October 24.
- User approved implementation in two checkpoints: verify A before proceeding to B.
- Do not merge any PR automatically. PR #2 merged at its reviewed head 6e4305e. The synchronized integrity audit is open as PR #4 after the user requested the branch rename to feat/integrity-audit; original PR #3 retains its review history.
- Initial branch strategy used isolated `feat/classification-data-pipeline` on tested audit head `eb09838f07e9bd4b31292acce1a0584c778c11c9` while PR #4 was pending. That foundation has since merged and Stage 3 is now based on main `51f422d976e34e87a9bb962a2598c3385d2eab2b`. Commit, push and PR creation are authorized; merge still requires separate approval.
- Keep source datasets and Stage 1 identity/caption contracts immutable.
- Preserve original class mapping, official source split, source row, and image_id.
- Official test images remain in the official test partition; preparation must not fit normalization, thresholds, or augmentation settings on them.
- Invalid captions alone never exclude an otherwise valid classification image.
- Preserve the EDA teammate's ownership of descriptive statistics, vocabulary analysis, and EDA visualizations.
- Keep generated manifests, caches, decoded images, and environments ignored by Git.
- pHash, extensive near-duplicate analysis, model training, and shared CLI expansion are outside this stage.
- Commit, push, or merge only with explicit authorization for Stage 3.

## Review focus

1. Incomplete, stale, mixed-revision, or malformed audit bundles must fail before publishing a usable preparation.
2. Two copies of one image must never cross derived partitions; conflicting-label groups require review.
3. Missing labels/decode failures exclude classification rows, while caption-only issues do not.
4. A source image that becomes unreadable after the audit must fail with its stable ID, never disappear silently from a batch.
5. Windows worker spawning and seeded augmentation must preserve row membership and avoid unpickleable closures.

## File map

| File | Responsibility |
| --- | --- |
| `src/data/preparation.py` | Audit validation, eligibility policy, group-aware stratification, preparation manifests and provenance |
| `src/data/image_io.py` | Shared source-image access and orientation/RGB normalization |
| `src/data/image_integrity.py` | Reuse shared internals while retaining its public inspection API and hash results |
| `src/data/transforms.py` | Explicit train/evaluation preprocessing profile |
| `src/data/classification.py` | Map-style Dataset and loader construction |
| `tests/data/test_preparation.py` | Eligibility, coverage, provenance, group/split invariants |
| `tests/data/test_image_io.py` | Decoding, ownership, orientation/transparency, failure handling |
| `tests/data/test_transforms.py` | Tensor and deterministic evaluation behavior |
| `tests/data/test_classification.py` | Row identity, batching, worker and seed behavior |
| `docs/classification_data.md`, `README.md`, `requirements-classification.txt` | Verified setup, API, policies, limitations and team handoff |

## Task 1: Preparation policy and split manifest

**Interfaces:** `prepare_classification(audit_dir: str | Path, output_dir: str | Path, *, val_fraction: float = 0.20, seed: int = 42) -> dict[str, Any]` and `load_classification_manifest(prepared_dir: str | Path) -> list[dict[str, Any]]`.

- [x] Add failing synthetic tests: reject incomplete coverage, missing/duplicate source rows, inconsistent counts, mismatched pinned revisions, inconsistent group membership, and malformed flags. Assert that no complete preparation is published.
- [x] Add failing policy tests: image/class and noncaption metadata errors exclude a row with explicit reason codes; caption-only errors preserve classification eligibility, including `metadata.missing_description` with `metadata_valid=false`; unrecognized flags and conflicting-label or cross-official-split groups stop preparation pending review. Reject nonfinite/out-of-range validation fractions and seeds outside the integer range 0 through 2**32-1 (including bool).
- [x] Add failing split tests: same seed and inputs yield identical sorted membership; changing input order does not change membership; every eligible official train row occurs in exactly one of train/val; official test rows stay test; group members stay together; every class has nonempty train and val where feasible; impossible grouped stratification fails clearly.
- [x] Run `python -m pytest tests/data/test_preparation.py -q` and confirm the intended failures.
- [x] Implement eligibility gates using successful image decoding, a valid original class reference, pinned row identity, and explicit error-code decisions. Do not use aggregate `metadata_valid` or `audit_status` as a blanket exclusion: `metadata.missing_description` and `caption.*` flags alone are nonfatal for classification. Other known metadata/image errors exclude with reasons. Treat `duplicate.exact` as an allowed warning when group membership and labels agree; other noncaption warnings require a reviewed policy. Preserve audit and source unchanged.
- [x] Build groups from `exact_group_ids`; treat other rows as singleton groups. Verify each group has one class and one official source split.
- [x] Within each class, sort groups by stable ID, seed their order, and select whole groups whose total size is nearest the validation target. Use a small reachable-count/subset-sum table; choose equal-distance solutions deterministically from the seeded order. Choose among counts strictly between zero and the class total, requiring nonempty train/val per represented class; fail clearly if no whole-group choice can satisfy that.
- [x] Publish `classification_manifest.jsonl` for all source rows, with `image_id`, original provenance/class references, eligibility, decision codes, audit flags, group ID and `partition` (`train`, `val`, `test`, or `excluded`). Publish `preparation.json` last, with source revisions, audit-input hashes, class mapping, policy/split versions, settings, counts, and completeness.
- [x] Use the Stage 2 incomplete-marker/final-summary publication pattern. Implement `load_classification_manifest` to reject incomplete preparations and mismatched manifest hashes/counts before returning rows; test a seeded stale-complete summary and interrupted publication.
- [x] Re-run focused tests. With the current audited source and proposed defaults, verify 4,794 train + 1,200 val + 5,794 official test, and identical membership for Pigeon Guillemot source rows 1712 and 1731. These counts were measured and verified in the pinned-source preparation.

**Checkpoint A verified:** 213 combined tests passed. Real preparation: 4,794 train, 1,200 val, 5,794 test, zero exclusions; 6 validation images per class. Both Pigeon rows 1712/1731 are in validation.

The subset-sum work is small for CUB (29-30 rows per class), retains exact target counts when feasible, and avoids splitting duplicate groups or accepting a poorly stratified global group shuffle.

## Task 2: Shared image access

**Public interface:** `load_rgb(reference: Mapping[str, Any]) -> PIL.Image.Image`; returns a fully loaded, independent RGB image owned and closed by the caller.

- [x] Add failing tests for bytes/path precedence, path-read failures, EXIF orientation, grayscale/palette/alpha inputs, truncated/corrupt content, and returned-image lifetime after closing its encoded source.
- [x] Run the new test file and confirm failures.
- [x] Extract common fetch/normalization internals from the auditor into `image_io.py`. Keep the existing bounded spool for path reads. Share the versioned EXIF/RGB/white-alpha rules rather than implementing a second normalization policy.
- [x] Keep `inspect_image` signatures, statuses, error distinctions, encoded hashes, pixel hashes, and source metadata unchanged. `load_rgb` must not compute hashes during every training access.
- [x] Re-run image I/O and existing image/audit tests. Compare a fresh cached audit's fingerprints with the saved Stage 2 audit if the shared internals change.
- [x] Preserve explicit decoder errors; the Dataset will add stable identity context rather than skipping failed rows.

## Task 3: Classification transforms

**Approved interface:** `classification_transform(*, training: bool, image_size: int = 224, geometry="fit", normalization=IMAGENET_NORMALIZATION)`.

- [x] Observe failing tests before implementation; require finite float32 CHW output, deterministic evaluation, configurable normalization, and train-only randomness.
- [x] Preserve the original resize-shorter-side/crop profile as `geometry="crop"`.
- [x] Inspect training-only previews; crops removed heads at source rows 1429 and 305. Obtain user approval before changing geometry.
- [x] Implement approved `geometry="fit"`: resize the full image within a square, retaining aspect ratio within integer pixel rounding; center on white padding. Use the same geometry in all partitions and flip only training.
- [x] Test portrait, landscape, square, and extreme aspect ratios; opposite-edge features, white padding, centering, configurable normalization and explicit float32 even with a changed default torch dtype.
- [x] Verify all four geometry/training combinations are pickleable and inspect fresh real training previews.

Full-image fitting preserves bird content but can reduce object resolution in wide or tall images. No preprocessing parameter is fitted from official test data. The immutable preparation records partition membership; model-specific geometry/normalization remain separate caller settings recorded in the handoff.

## Task 4: Dataset and DataLoader interface

**Interfaces:**

- `CubClassificationDataset(source: CubSource, manifest: Sequence[Mapping[str, Any]], *, partition: str, transform: Callable)`.
- `make_classification_loaders(datasets: Mapping[str, Dataset], *, batch_size: int = 32, num_workers: int = 0, seed: int = 42, pin_memory: bool = False) -> dict[str, DataLoader]`.

- [x] Add failing tests for exact manifest/source identity matching, wrong-source rejection, class-map/label consistency, sample schema, batch tensor dtype/shape, deterministic evaluation and seeded first-epoch order. Test incomplete-preparation rejection through `load_classification_manifest` in Task 1; the Dataset validates supplied rows but cannot itself certify an external summary from a bare sequence.
- [x] Add a runtime-read-failure test asserting that the error identifies `image_id` and source row; no silent `None`/skip behavior.
- [x] Add a Windows-spawn-compatible two-worker smoke test using top-level definitions; keep zero workers as the documented default.
- [x] Run focused tests and confirm failures.
- [x] Implement a lazy map-style Dataset using the indexed pinned source and manifest rows. Load one image per requested sample, apply its assigned transform, and return `{"image": tensor, "label": class_id, "image_id": stable_id}`. Class IDs must remain original zero-based integers in [0,199]; default collation must produce int64 labels and a list of IDs.
- [x] Require manifest membership to agree with source row/revision/class. The normal consumer path uses `load_classification_manifest` from Task 1 before supplying rows to the Dataset.
- [x] Use shuffle only for training and `drop_last=False`. Give train/val/test separate torch Generators (default seeds 42/43/44), so validation iteration cannot advance the training sampler RNG. Use a top-level worker initializer that seeds Python/NumPy from each worker's torch seed modulo 2**32. The loader factory must not reset global RNGs; the handoff startup recipe explicitly seeds Python, NumPy and torch once before iteration for zero-worker augmentation. Test that validation iteration does not change the training sampler order.
- [x] Load/create an official test Dataset only when the consumer explicitly requests evaluation; normal train/val construction uses the official train source alone.
- [x] Avoid caching all decoded images, CUDA operations in workers, hidden downloads on import, or rehashing on each sample.
- [x] Re-run tests and verify repeatability for a fixed run/iteration sequence within the recorded environment. Do not promise bit-identical results across arbitrary PyTorch versions or GPU kernels.

## Task 5: Real-data verification and handoff

- [x] Pin and validate a compatible torch/torchvision pair for the supported Python environment, retaining the existing EDA dependency contract. Report the actual Python/CPU/GPU scope tested.
- [x] Run all repository tests, including the EDA suite if PR #2 has landed, and `git diff --check`.
- [x] Generate the complete preparation from the verified audit; report measured eligibility/exclusion counts, per-class train/val counts, official-test preservation, and duplicate-group isolation.
- [x] Verify the entire membership manifest without image decoding; run representative real train/val batches and the two-worker smoke. Report measured loading time and memory scope.
- [x] Run one tiny-model forward/backward smoke using batch tensors and cross-entropy targets; this is an interface check, not a delivered model/training implementation.
- [x] Update README only after those checks pass. Document installation, explicit defaults, audit/preparation prerequisites, a short loader example, batch schema, class mapping, seed behavior, error handling, and generated artifact paths.
- [x] Prepare the classification-team handoff with the immutable manifest, source revisions, policy/config snapshot, class mapping, API example, test results, and limitations; publication was approved on October 22, subject to final verification. The EDA teammate can consume the same eligible IDs without rebuilding integrity logic.
- [x] Obtain user review before committing/pushing Stage 3; approval received October 22, 2026. Do not merge automatically.

## Delivery order and acceptance

Deliver Task 1 first so partition membership is fixed and available to the classification team; then shared decoding/transforms, then DataLoader integration. Reserve October 24 for integration fixes and handoff. Optional pHash and elaborate tuning must not displace these gates.

Acceptance: every source row has an explicit preparation decision; no eligible ID appears in multiple partitions; no exact group crosses train/val/test; all 200 classes are represented in train/val; validation/test transforms are deterministic; tensors/labels work in a classification loss; tests and actual real-data results are recorded.

Suggested eventual Stage 3 commit message: `feat(data): prepare reproducible CUB classification datasets`.

## References

- [Torchvision transforms](https://docs.pytorch.org/vision/stable/transforms.html)
- [Torchvision ImageNet preprocessing example](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- [PyTorch Dataset/DataLoader and worker seeding](https://docs.pytorch.org/docs/stable/data.html)

## Final integration state — October 22, 2026

- Shared image adapter published separately with user approval: PR #5, branch `feat/shared-image-access`, commit `09cf81fd515e121c446b0f2509d5c992b8f3522a`; 162 tests passed and all 5,994 training views match saved audit pixels. Stage 3 uses the identical adapter.
- The user authorized merging PR #4 and PR #5. Their documentation conflicts were resolved in PR #4 head `a059664a238e33d047c6002fdde432ecc861699a`; combined foundation suite 182 passed in 5.39s. At that historical checkpoint both awaited formal collaborator approval; the subsequent merged state is recorded below.
- Protect main ruleset `25575979` is active and applies only to `main`; administrators have no bypass. No CI/signature/linear-history requirement was added.
- Stage 3 remains on isolated `feat/classification-data-pipeline` with no Stage 3 commit or push. Its worktree retains the classification changes while foundation integration completes.
- Final classification/source/audit/EDA suite: 278 passed in 70.19s, including opt-in real-data integration. Generated outputs remain ignored.

## Foundation merged and Stage 3 preserved — October 22, 2026

The user removed the external-approval requirement: Protect main remains active with required approving reviews set to zero, while PRs, resolved conversations, force-push/deletion blocks and no administrator bypass remain. PR #5 merged as `753031edd6533db9508d85509b8bb2a14038ae6c`; PR #4 merged as `51f422d976e34e87a9bb962a2598c3385d2eab2b`. These merges were explicitly authorized.

The classification branch was fast-forwarded to that merged main without creating a Stage 3 commit. All 17 pre-sync working files were backed up and verified byte-identical after restoration; shared paths already present in main drop out of the classification diff. The remaining changes are the classification implementation and documentation, ready for user review. The verified source implementation is unchanged from the 278-test run.

## Final review and verification — October 22, 2026

- Confirmed current branch HEAD, origin/main and live GitHub main all equal `51f422d976e34e87a9bb962a2598c3385d2eab2b`; Stage 3 has no commits ahead.
- Independent reviews covered audit preparation/splitting, image/PyTorch interfaces, documentation and Git hygiene.
- Reproduced and fixed the unexplained `not_checked` acceptance gap. Two public-API regressions distinguish rejection of an uninspected image from a properly reported malformed source row.
- Added `pytest.ini` with `testpaths = tests` after the bare repository test command collected ignored safety-backup copies. No backups were removed.
- Fresh shared-decoder audit accounted for 11,788 rows in 29.001s, with zero errors; image, caption and exact-group manifests were byte-identical to immutable Stage 2.
- Preparation twice plus validation took 2.542s; both preparations were byte-identical. Counts: 4,794 train, 1,200 validation, 5,794 official test, zero excluded; all 200 classes represented.
- Train counts: 194 classes have 24 rows, six classes have 23. All classes have six validation rows. Official test class counts remain 11–30.
- No stable-ID, encoded-hash or pixel-hash overlap across partitions. Both duplicate Pigeon Guillemot rows remain in validation.
- Real 32-image batches, deterministic evaluation, seeded training repetition, Windows two-worker pixel/ID parity and finite tiny-model forward/backward passed.
- Inspected 12 originals and 24 fit/flip training views; important visible bird features remain. Dataset source assets and audit fingerprints unchanged.
- Entire combined suite: **280 passed in 65.06s**, including real-data opt-in. All 53 dependency packages compatible; offline dry run would change no package.
- Updated the professional README and standalone classification handoff, including executable Windows-safe training and final-evaluation examples.
- Generated evidence, manifests, per-class CSV and previews remain under ignored `outputs/stage3-final-2026-10-22/`.
- This checkpoint preceded publication approval. The user subsequently authorized the Stage 3 commit, push and PR creation after final checks; merging remains unapproved.

## Approved publication checks — October 22, 2026

The user approved commit `feat(data): prepare reproducible CUB classification datasets`, push of `feat/classification-data-pipeline`, and PR creation against `main`. No merge is authorized.

Final scope is 14 intentional files. README and handoff use portable repository-relative paths. The final development-environment suite passed 280 tests in 70.00s. A separate standard Python 3.12 venv installed 53 dependencies from cache, passed pip/uv compatibility checks (54 packages including pip), and passed all 280 tests from a clean 59-file source export in 76.08s. The full cold online pip install route was not repeated; official CPU-index version availability was verified separately. Direct dependencies are pinned, with the transitive-lock limitation documented.

Generated audits, manifests, reports, source exports, fresh environments and logs remain under ignored outputs. Publication uses only the 14 reviewed files; automatic merge and new feature/model work remain excluded.
