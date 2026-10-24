# Pretrained CNN foundation implementation plan

**Goal:** Implement and publish approved Task 1.2 Stage 1A, without training.

**Authority:** User Stage 1A approval, 2026-10-24. Three architectures only:
ConvNeXt-Tiny, EfficientNet-B3, RegNetY-3.2GF. Existing data contracts are binding.

**Base:** `c979dc268c4191d938690c572f48ff9134cf9383`; branch
`feat/pretrained-cnn-foundation` in an isolated managed worktree.

## Global constraints

- Exactly 200 logits; canonical class ordering, stable IDs and seed-42 partitions.
- Preserve Task 1.1 modules and the sample contract `image`, `label`, `image_id`.
- Default full-image fit, white centered padding, 224 square, train-only flips,
  explicit configurable normalization once. Native weight transforms are metadata.
- No network weights or GPU experiments in Stage 1A. Clearly distinguish configured
  weights from loaded weights. Real weight loading remains available and fails loudly.
- Frozen, partial, full policies must survive `train()`/`eval()` and correctly handle
  frozen normalization buffers. Feature extraction must preserve autograd when requested.
- No trainer, tracker, search, checkpoint engine, training CLI, notebook, CAM rendering,
  extra architecture, ResNet50, scratch CNN or Milestone 2 implementation.
- No new dependencies, generated data, weights, environment or cache in commits.
- TDD for meaningful behavior; full regression suite; independent code review; PR only.

## Public interface decisions

Task 1 owns `src/models/pretrained.py` and `src/models/finetuning.py` plus model tests.

- `ModelConfig(architecture, weights=None, num_classes=200, fine_tuning="frozen",
  head_dropout=0.0, freeze_bn_stats=True)` is a validated immutable dataclass.
  `weights=None` resolves the architecture's explicit pinned enum, not random initialization.
  Reject unsupported classes/architecture/weight/policy and invalid dropout/types.
- `get_model_spec(architecture)` exposes the exact qualified weight ID, native recipe
  (resize, crop, interpolation, mean/std), encoder channels/stride and CAM path.
  Keep the spec small and serializable; no generic registry framework.
- `create_pretrained_model(config, *, load_weights=True, progress=True)` returns a
  `PretrainedClassifier`. Offline construction uses `load_weights=False` explicitly;
  record that weights were not loaded and never silently fall back after download failure.
- `forward(images)` returns `[B,200]`; `forward_features(images)` returns a dictionary
  with `spatial` NCHW and `pooled` `[B,D]`. `head` is separate from the encoder.
- `set_fine_tuning(mode, *, freeze_bn_stats=None)` updates policy and parameter masks;
  head remains trainable. Partial unfreezes the final stage including its transition
  and final projection/encoder normalization as appropriate for each architecture.
- `gradcam_target_layer` resolves an existing spatial module without registering it twice.
- Native TV heads: retain ConvNeXt's encoder-side final normalization; EfficientNet's
  pooled embedding is before head dropout; derive/verify input dimensions from native heads.

Task 2 owns `src/training/data.py` plus integration tests.

- `PreprocessingConfig(image_size=224, normalization=IMAGENET_NORMALIZATION)` validates
  input and builds existing full-fit transforms; no crop default or tensor reprocessing.
- `make_pretrained_loaders(prepared_dir, model_config, *, preprocessing=None,
  source=None, batch_size=32, num_workers=0, seed=42, pin_memory=False)` validates the
  complete existing preparation, then constructs train/val datasets/loaders through
  Task 1.1 APIs. Optional injected pinned `CubSource` enables offline tests. If absent,
  open `load_cub_split("train")` only. No test-source fetch, resplitting or silent skipping.
- Expose a small JSON-serializable preprocessing description showing baseline settings
  separately from `get_model_spec`'s native weight recipe. Metadata must not imply that
  B3's native crop recipe is the baseline. Validate model config before network effects.

Task 3 owns `docs/pretrained_cnn_foundation.md`, minimal README updates and final review.

## Task 1 — Model adapters and policies

1. Write focused tests that fail because the adapter is absent.
2. Implement three TV wrappers with explicit weight resolution and a 200-class head.
3. Test 224-pixel output and feature shapes, independent head-free feature extraction,
   meaningful gradients, normalization buffers, repeated mode transitions and CAM hooks.
4. Test offline construction, invalid configs and real-loader dispatch/failure propagation
   without downloading. Disclose that loader delegation is not real-weight verification.
5. Run model tests; review implementation against scope and interface.

## Task 2 — Existing data integration

1. Write failing tests for preprocessing and prepared train/val loading.
2. Implement the thin data helper without changing Task 1.1.
3. Test tensor dtype/range/one normalization, geometry/content, train-only augmentation,
   deterministic validation, IDs/labels, malformed preparation, and no implicit test fetch.
4. Use valid synthetic preparation from existing tests rather than bypassing its validator.
5. Run integration/data regressions; review against contract.

## Task 3 — Documentation, combined verification and publication

1. Document public APIs, exact weights/native recipes, shapes/stages, freeze semantics,
   preprocessing, offline construction, real-loading limitations and dependency setup.
2. Update README status only for implemented and verified functionality.
3. Run combined repository tests; optionally exercise cached real train/val with no weights.
4. Independently review the complete diff, resolve blockers, check ignored artifacts and
   whitespace; commit meaningful model/data/documentation changes.
5. Push the feature branch and create a detailed PR against main; attach it to this chat.
   Do not merge or start Stage 1B.

## Review focus

- Freeze-policy changes must not leak stale gradients or BatchNorm running-stat updates.
- Feature extraction and frozen-backbone CAM must retain an enabled gradient path.
- Offline tests must not download or report real pretrained-weight execution.
- Metadata must distinguish 224 full-fit from native weight center-crop profiles.
- Data integration must validate preparation before fetching and preserve class/ID/order.

## Acceptance evidence

Fresh baseline and combined test counts, per-model shapes and trainability summaries,
source-contract regression checks, independent review, exact commits and PR URL.
Cloud/real pretrained verification remains an explicit Stage 1B limitation.
