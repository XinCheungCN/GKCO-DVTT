# Dataset preparation

The datasets used in the paper are not redistributed in this repository.

Prepare one local directory per dataset with the following files:

```text
data/<dataset_name>/
├── train_x.npy
├── train_y.npy
├── validation_x.npy
├── validation_y.npy
├── test_x.npy
└── test_y.npy
```

Feature arrays must have shape `[N, L]`, and label arrays must have shape `[N]`.

Paper sample lengths:

- Axial-flow pump: `L = 1024`
- BJTU-RAO: `L = 512`

The pipeline concatenates the three splits only for unsupervised graph optimization, while DVTT supervision is controlled by explicit train/validation/test masks derived from the original split boundaries.
