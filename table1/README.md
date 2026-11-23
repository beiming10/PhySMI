# Table 1 reproduction

Training and evaluation code, configurations, and numerical source data for Table 1. The four application models in `shared_models/` are unchanged.

## Download

The [Table 1 release](https://github.com/beiming10/PhySMI/releases/tag/table1-v1.0) contains:

- 500 eight-channel training observations.
- 50 test observations, five-component references, and eight-channel NNLS results.
- The complete 1,708-image single-stain library.
- The common initialization and five trained checkpoints at 45,000 updates (40 epochs).

From the repository root:

```sh
python -m pip install -r table1/requirements.txt
python table1/reproduce.py
```

The single command downloads the test data and all five trained checkpoints, evaluates every checkpoint and the NNLS baseline on the same 50 pairs, and writes `table1_results/Table1.csv`, `per_pair.csv`, and full-precision JSON files. It selects CUDA when available; add `--device cpu` to use a CPU. `--data` and `--output` select the local directories. No training is needed to reproduce the checkpoint results.

For training, download the additional training observations and full single-stain library:

```sh
python table1/download.py --output ./table1_data
```

The evaluator also accepts `--save-components` to save reconstructed component TIFF stacks. The supplied NNLS images reproduce the stored baseline; `--recompute-nnls` solves nonnegative least squares again from the eight-channel observations.

## Training

```sh
python table1/train.py --configuration full --data ./table1_data --output ./training_runs/table1_full --device cuda:0
```

Repeat with the configuration names below to train the other rows. Training starts from the common released initialization. The default budget is 45,000 generator updates. `--steps 1` provides a short execution check in a separate output directory.

| Configuration | Table 1 row | Training branches | Reconstruction | Shared prior | Cross-branch denoising |
|---|---|---|---:|---:|---:|
| `reconstruction_only` | Reconstruction only | A | 2000 | 0 | 0 |
| `reconstruction_cross_denoising` | Reconstruction + cross-branch | A+B | 2000 | 0 | 1000 |
| `reconstruction_image_prior` | Reconstruction + image prior | A+B | 2000 | 1 | 0 |
| `full` | Full model | A+B | 2000 | 1 | 1000 |
| `adversarial_prior_only` | Adversarial prior only | A | 0 | 1 | 0 |

The two-branch configurations retain the component-consistency term with weight 1. The full and reconstruction-plus-prior configurations differ only in the cross-observation denoising weight. All neural evaluations use branch A alone. One discriminator is shared across the five output components.

To evaluate a newly trained checkpoint:

```sh
python table1/evaluate.py --configuration full --checkpoint ./training_runs/table1_full/checkpoint_040.pt --data ./table1_data --output ./table1_full_results --device cuda:0
```

## Data and numerical settings

`dataset.json` lists the training, prior, and test assignments with relative paths and checksums. Test references are used by evaluation, not the training loss. Files retain their original numerical values.

The native observation order is 561, 551, 542, 532, 521, 510, 499, and 488 nm. Branch A selects 561, 532, and 488 nm; branch B selects 551, 542, and 521 nm. Output order is mitochondria, nucleus, lipid droplets, microtubules, and actin. Each configuration includes its matching fixed response matrix.

Training converts raw values to float32 and divides by 2500. There is no background subtraction, per-image normalization, clipping, or rescaling of individual components. A 128-pixel grid with stride 64 gives nine patches per 256-pixel training image. A batch contains four patches, giving 1,125 updates per epoch. The configurations preserve the shared initialization, epoch-wise image order, augmentation seeds, and cosine-scheduler horizon of 225,000 updates.

Evaluation multiplies predictions by 2500 and uses a fixed intensity range of 5000 for PSNR and SSIM. PSNR and SSIM are calculated for each component and then averaged within each test pair. Spectral angle is calculated across the five-component vector at each pixel and averaged over the full field. Table values are means over the 50 pairs. The PSNR standard deviation is the sample standard deviation across those pairs. No display contrast or spatial postprocessing enters the metrics.

The recorded numerical environment used Python 3.11 and PyTorch 2.10.0 with CUDA 12.9. `requirements.txt` records the numerical-library versions. CPU and GPU floating-point calculations can differ slightly. `expected_results.csv` and `per_pair_reference.csv` provide the full-precision recorded scores.

The spectral-attention backbone builds on [MST++](https://github.com/caiyuanhao1998/MST-plus-plus). Its copyright and licence notice is retained in [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt).
