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

`runTraining.m` currently uses a fixed ten-fold loop. The historical centroid calculation includes the label column while `fusion.m` compares feature-only rows; review that dimension/scale mismatch before claiming an end-to-end reproduction. Also review categorical prediction conversion and all-tied ensemble scores. These scientific conventions have not been silently replaced. The previously missing five-feature CNN branch is now covered by the smaller-network branch.

The cleanup aligns function/file names, restores visible warnings, makes CSV handles close on errors, and tests constant-feature normalisation plus CSV append behaviour. MATLAB experiments and the new utility tests have **not** been executed in the cleanup environment, which has no MATLAB installation. Static parsing is not a reproduction of the paper.

[Profile and project directory](https://github.com/zohaibjan)
