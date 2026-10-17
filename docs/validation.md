# Executed validation

Date: 17 October 2026. [Successful GitHub Actions run 39000792279](https://github.com/zohaibjan/PatternRecognition/actions/runs/39000792279) at commit `35ca22e06d26c718ef7c1732c1315b3295cd1253`.

- MATLAB R2024b on Ubuntu 22.04.
- 16 tests passed; zero failed or incomplete.
- 20 root function files loaded/compiled by MATLAB without executing the experiment entry scripts.
- Installed toolboxes: Statistics and Machine Learning, Deep Learning, Global Optimization, Optimization (dependency), and Parallel Computing.

## What was executed

Actual CNN training and original numeric-label prediction, bounded silhouette clustering, voting/centroid utilities, source loading, and 96x96 RGB image layout checks.

Toolbox tests fit real models and/or call real solvers. Deterministic predict-compatible fixtures are also used to isolate selection and voting behavior. Smoke tests use small synthetic datasets and bounded solver/epoch budgets; default research budgets remain available.

## Changes and behavior

The historical getCIFAR helper expects 96x96 RGB images (27648 column-major values per row), not 32x32 CIFAR inputs. cifarImages transposes sample rows before reshaping, preventing pixels from different images from mixing. The legacy image driver loads named fields explicitly, routes with feature-only centroids and batches classify calls by model. The large 96x96 image experiment itself has not been trained end to end.

## Reproduce

Use a clean MATLAB session with only this repository on the path:

```matlab
results = runtests('tests');
assertSuccess(results);
```

The workflow runs tests on MATLAB changes and pull requests, and supports manual dispatch. It has read-only repository permissions and performs no deployment.

## Limits

These tests do not execute every function branch or reproduce a paper's full datasets, published accuracy, runtime or memory results. Experiment scripts and unsupported auxiliary paths still need their documented inputs/dependencies. Root function loading is a compilation check, not full code coverage. Consult [code-review.md](code-review.md) for repository-specific limitations. Historical result CSVs were not regenerated or relabelled.
