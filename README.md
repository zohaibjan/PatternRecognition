# Multiple Strong and Balanced Clusters: Ensemble of Deep Learners

MATLAB research implementation of clustered deep learners and ensemble selection.

## Research reference and attribution

Jan, Z. M. and Verma, B. (2020). **Multiple Strong and Balanced Clusters based Ensemble of Deep Learners.** Pattern Recognition, article 107420.

Please cite the relevant publication and retain source author notices. Existing dataset and third-party terms still apply; this cleanup does not grant a new licence.

## Data and environment

Extract `datasets.zip` into the repository root, preserving the `DTE` and `P-Data` names. The active workflow uses `DTE/<name>/<name>-CV-tr-<fold>.mat` (`dtrX`, `dtrY`) and matching `-CV-ts-` files (`dtsX`, `dtsY`). Check that the configured dataset, currently `shuttle`, is present; otherwise select a supplied dataset.

**MATLAB dependencies:** Statistics and Machine Learning Toolbox; Deep Learning Toolbox.

No exact original MATLAB/toolbox version is recorded. Use MATLAB rather than assuming GNU Octave compatibility for the learner APIs.

## Start here

1. Open this repository as the MATLAB current folder. Use a separate session/path for each research repository: helper filenames overlap across projects.
2. Supply or extract the data described above. Check feature/label row counts and class labels.
3. Edit `Problem` in `mainProgram.m` (or `dataSetNames.m` where used), and start with one dataset and one repeat.
4. Run the synthetic utility checks, then inspect `mainProgram.m` before starting the full experiment:

```matlab
results = runtests('tests');
assertSuccess(results);
rng(42, 'twister');
mainProgram;
```

The seed is a starting point for recording runs; nested seed resets, parallel workers and toolbox versions can still affect reproducibility. `runMultiple.m` performs 50 full driver repetitions and is not the quick-start command.

## Workflow and output

`mainProgram.m` chooses the dataset; `runTraining.m` reads ten pre-created folds and forms a validation holdout. `generateClustersV2.m` and `balanceClusters.m` construct training subsets; `getCNN.m` trains learners; `ensembleSelection.m`, `majVote.m` and `fusion.m` evaluate combinations.

`saveResults.m` appends rows to `results.csv` in the current folder and writes the header only when creating the file. Driver-level CSV writes are sequential to avoid concurrent append races. Archive old output before a new configuration; rows do not contain a complete environment/configuration record. Committed CSV files are historical outputs, not freshly validated benchmarks.

## Review notes and reproducibility limits

Silhouette K candidates are bounded by available distinct training feature rows; label columns are excluded, tiny classes remain valid and class-count reporting includes one-cluster cases. Feature-only centroids and augmentation distances now share raw feature coordinates. CNN predictions preserve numeric category names. `runTraining.m` still uses a fixed ten-fold loop; `getCNN.m` retains the smaller-network branch for five-feature inputs.

## Runtime follow-up — October 2026

The historical getCIFAR helper expects 96x96 RGB images (27648 column-major values per row), not 32x32 CIFAR inputs. cifarImages transposes sample rows before reshaping, preventing pixels from different images from mixing. The legacy image driver loads named fields explicitly, routes with feature-only centroids and batches classify calls by model. The large 96x96 image experiment itself has not been trained end to end.

## Correctness review — October 2026

This pass fixes implementation defects; it does not validate the associated research results. Changes can alter model pools, predictions and measured scores. Re-run the experiments before comparing corrected code with historical CSVs or published results.

- Label conversion uses numeric category names, not categorical indices.
- Voting rejects empty/nonfinite inputs; a prediction failure is reported rather than counted as a fabricated class-one vote.
- CNN output categories retain their numeric label names, including zero and nonconsecutive labels.
- Synthetic checks cover normalization, CSV append behavior and the new prediction/selection boundaries. Additional cluster checks run where applicable.

See [the code review](docs/code-review.md) for repository-specific changes, tests and remaining limitations.

**Validation:** 16 regression/smoke tests passed in MATLAB R2024b on Linux through [GitHub Actions](https://github.com/zohaibjan/PatternRecognition/actions/workflows/matlab-tests.yml). Tests include real toolbox execution on synthetic data. Full research datasets and published accuracy results have not been reproduced. See [validation evidence and limits](docs/validation.md).

[Profile and project directory](https://github.com/zohaibjan)
