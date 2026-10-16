# Multiple Strong and Balanced Clusters: Ensemble of Deep Learners

MATLAB research implementation accompanying:

Jan, Z. M. and Verma, B. (2020). **Multiple Strong and Balanced Clusters based Ensemble of Deep Learners.** Pattern Recognition, article 107420.

Please cite the paper when using this work.

## Run the research workflow

1. Extract `datasets.zip` into the repository root.
2. Keep the extracted `DTE` and `P-Data` directory names unchanged.
3. Open `mainProgram.m` in MATLAB and review the `Problem` dataset selection.
4. Install the MATLAB toolboxes required by the chosen learners, then run `mainProgram`.

## Code map

| Files | Purpose |
|---|---|
| `mainProgram.m`, `runTraining.m` | Experiment setup and training |
| `generateClusters.m`, `generateClustersV2.m`, `balanceClusters.m` | Cluster generation and balancing |
| `getCNN.m`, `getCNNPred.m` | Neural model construction and prediction |
| `ensembleSelection.m`, `fusion.m`, `majVote.m` | Selection and prediction combination |
| `saveResults.m` | Result export |

## Reproducibility

This repository preserves a historical research workflow. Verify the selected data, toolbox versions and evaluation settings before attempting to reproduce published results. The documentation cleanup has not rerun the experiments.

[Profile and project directory](https://github.com/zohaibjan)
