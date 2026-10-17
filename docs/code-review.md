# Code review and validation scope

Silhouette K candidates are bounded by available distinct training feature rows; label columns are excluded, tiny classes remain valid and class-count reporting includes one-cluster cases. Feature-only centroids and augmentation distances now share raw feature coordinates. CNN predictions preserve numeric category names. `runTraining.m` still uses a fixed ten-fold loop; `getCNN.m` retains the smaller-network branch for five-feature inputs.

## Changed behavior and acceptance

The fixes preserve original source attribution and the documented experiment entry points. They change previously erroneous edge cases and may change reported results. Stored outputs were not rewritten or relabelled as new results.

Run in a clean MATLAB session with only this repository on the path:

```matlab
results = runtests('tests');
assertSuccess(results);
```

The test fixtures are synthetic. `FixedClassifier`, where supplied, implements `predict` without fitting a model and is on the path only for the test session. Majority ties choose the smallest numeric label. Empty or invalid votes fail explicitly. ANN vocabularies are learned from training labels, not held-out labels. PSO thresholds retain the repository's setting, with one shared nonempty mask policy for objective scoring and final return. Random selection, where used, has at most pool-size times elimination-budget attempts.

## Executed MATLAB validation

Actual CNN training and original numeric-label prediction, bounded silhouette clustering, voting/centroid utilities, source loading, and 96x96 RGB image layout checks.

The historical getCIFAR helper expects 96x96 RGB images (27648 column-major values per row), not 32x32 CIFAR inputs. cifarImages transposes sample rows before reshaping, preventing pixels from different images from mixing. The legacy image driver loads named fields explicitly, routes with feature-only centroids and batches classify calls by model. The large 96x96 image experiment itself has not been trained end to end.

See [validation.md](validation.md) for the exact run, release and boundaries.

## Remaining validation

- Compatibility beyond the tested MATLAB R2024b/toolbox release.
- Full training runs with the documented datasets, preserving training/validation/test separation.
- Numerical quality, random seeds, paper-method agreement, runtime and memory consumption.
- Optional historical optimizer/diversity/alternate-driver files are not a validated library.

The regression/smoke suite passed in MATLAB R2024b using MathWorks GitHub Actions. Synthetic fixtures check implementation behavior; full research experiments, historical accuracy and performance benchmarks have not been reproduced.
