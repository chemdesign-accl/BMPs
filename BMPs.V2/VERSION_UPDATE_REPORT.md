# BMPs Version Update Report

This update refines the previous implementation used in Digital Discovery,
2025, 4, 3320-3338. The previous version already supported classification and
regression; this version focuses on preprocessing reuse, safer molecule
handling, normalization options, message-passing comparison, class-imbalance
handling, and improved node-level importance visualization.

## Concise Change Summary

- Replaced sampler-based class balancing in the DataLoader with
  `BCEWithLogitsLoss(pos_weight=negatives/positives)`.
- Added persistent processed-molecule caching keyed by canonical SMILES and
  preprocessing settings, with labels reattached when cached graphs are loaded.
- Added multi-CPU molecule preprocessing and safer invalid-molecule skipping.
- Reworked stereoisomer/conformer generation: undefined stereocenters are now
  enumerated, multiple ETKDGv3 conformers are generated per candidate, MMFF is
  used to optimize them, and the lowest-energy optimized conformer is selected.
- Improved graph indexing for molecules where explicit hydrogens could remain
  after conformer handling, by mapping RDKit atom indices onto heavy-atom
  feature rows before edge construction.
- Added an optional unique-occupancy buried-volume mode that can report the
  fraction of local grid points covered at least once, instead of double-counting
  overlapping neighboring atom volumes.
- Added `graphnorm`, `layernorm`, and `none` as normalization options alongside
  the existing BatchNorm-style path.
- Added configurable `message_passing_steps` and optional automatic comparison
  of message-passing depths.
- Changed node-level importance from a learned 1D node score followed by simple
  per-molecule min-max scaling to raw message-derived scores with `log1p` plus
  robust min-max scaling.
- Coupled ABMP/ABMP+SN node-importance scores to the same edge-attention weights
  used in the message-passing update.
- Updated molecule image outputs so one dominant atom does not collapse the
  remaining atoms toward zero, making secondary important regions easier to see.
- Added regression target standardization so continuous targets can be learned
  on a stable scale while still reporting values on the original scale.
- Added raw-logit/probability diagnostics for classification debugging.
- Added cache/version/dimension checks to reduce stale-cache and empty-feature
  failures.

## Data Processing Changes

The previous workflow recomputed molecule graph objects repeatedly. The updated
workflow can store processed molecules in a cache and reload them in later runs.
Cached graph features are keyed by molecular structure and preprocessing
settings. The current label is attached when the molecule is loaded, which keeps
labels aligned even when invalid molecules are skipped.

Invalid inputs are now handled more explicitly:

- non-string or missing SMILES are skipped.
- RDKit parse failures are skipped.
- failed stereoisomer/conformer generation returns a skipped molecule instead
  of crashing later.
- invalid edge indices are detected before batching.

The stereochemistry and conformer protocol changed substantially. Previously,
each molecule was embedded once with hydrogens, optimized once with MMFF, and
then assigned 3D stereochemistry from that single optimized structure. The new
protocol first checks for undefined atom stereochemistry and enumerates those
unassigned stereocenters up to the configured `max_isomers`. For each candidate
stereoisomer, multiple ETKDGv3 conformers are generated, MMFF is set up, each
valid conformer is minimized, and the lowest-energy optimized conformer across
all successful candidates is selected. 3D stereochemistry is then assigned from
that selected conformer before hydrogens are removed for heavy-atom graph
construction. This makes the final molecular graph features depend on a selected
low-energy stereoisomer/conformer instead of a single initial embedding.

The previous implementation was already removing hydrogens after conformer
generation and constructing features for heavy atoms. However, some molecules
with explicit hydrogens could still leave RDKit atom indices that did not match
the heavy-atom feature rows. The edge builder now creates an explicit mapping
from RDKit atom indices to heavy-atom feature indices and skips bonds involving
hydrogens. This keeps `edge_index` aligned with the feature matrix and addresses
the earlier class of errors where an edge index could exceed the number of
atom-feature rows.

Buried-volume calculation now includes an optional unique-occupancy mode. In
this mode, each atom-centered grid point is counted only once if it is covered
by one or more neighboring van der Waals spheres. The possible benefit is a more
physical local-occlusion fraction that is less inflated by overlapping or
redundant neighboring atom volumes, which may make steric features more
comparable across crowded molecular environments.

## Training Changes

For classification, class imbalance is now handled through the loss:

```text
BCEWithLogitsLoss(pos_weight = number_of_negative_labels / number_of_positive_labels)
```

This replaces the previous DataLoader sampler-based balancing approach. The new
behavior keeps mini-batch sampling simpler and moves the imbalance correction
into the binary logit objective.

For regression, target standardization can be enabled during training. This is
useful when continuous targets have large offsets or broad numerical ranges:
the model optimizes a centered/scaled target, while predictions and metrics can
still be inverse-transformed back to the original target scale for
interpretation.

## Architecture Changes

The core bidirectional message-passing structure remains the same, but the new
version adds:

- selectable normalization: `batchnorm`, `graphnorm`, `layernorm`, or `none`.
- configurable message-passing depth.
- optional depth comparison where classification is selected by AUROC and
  regression is selected by MAE.
- step-specific edge/node modules when more than one message-passing step is
  used.

The current tested baseline still generally favors `message_passing_steps = 1`;
deeper message passing is available for comparison but is not assumed to improve
every dataset.

## Node-Level Importance Changes

The node-level importance computation was changed to avoid over-concentrated
atom maps.

Previous behavior:

- each node block produced a learned 1D node score using a linear layer followed
  by sigmoid.
- during evaluation, scores were min-max scaled independently within each
  molecule.
- a single high-scoring atom could stretch the molecule-level min-max range and
  make most other atoms appear close to zero.

Updated behavior:

- raw message-derived node scores are extracted from the message vectors.
- for ABMP and ABMP+SN, those scores are weighted by the same attention weights
  used in the architecture message update.
- negative scores are clipped to zero for visualization.
- scores are compressed with `log1p`.
- robust min-max scaling is applied using quantiles.
- high-score and low-score atom masks are exported.

Expected image change:

- the strongest atom is still highlighted.
- secondary important atoms remain visible.
- color maps become smoother and less binary.
- low-scoring regions are easier to identify as candidate sites for
  bioisostere-replacement search.

The exported node-score files now include raw score, transformed score,
normalized score, high-score flag, and low-score flag.

## Evaluation And Logging Changes

The evaluation flow already returned performance metrics in the previous
version. The update keeps that behavior but adds more diagnostic information for
debugging model behavior:

- classification logs raw logits and probabilities before thresholding.
- threshold selection can be calibrated automatically.
- train-mode/eval-mode diagnostics can be used to inspect dropout-related
  behavior.
- preprocessing logs report cache use and skipped molecules without changing
  label alignment.

## Cross-Validation And Prediction Changes

Cross-validation now reuses cached processed molecules where possible and can
apply the same preprocessing choices as the regular trainer.

Prediction output is more explicit:

- classification output includes raw logit, probability, threshold, and class.
- regression output can include both standardized/raw predictions and
  inverse-transformed values.

## Practical Notes For This Version

- Keep `message_passing_steps = 1` as the baseline unless automatic comparison
  shows a clear improvement.
- Use the cache for repeated runs, cross-validation, and prediction, but clear
  it after changing feature definitions or preprocessing settings.
- Use low-scoring atom regions from the node-importance output as candidate
  sites for bioisostere replacement while preserving high-scoring regions as the
  likely conserved core.
