# Fig. 6 self-supervised training

Train the five-organelle cell and 13-fluorophore E. coli models using our laboratory's single-stain images and the target spectral responses. The workflow includes image-library download, simulation, training, validation, and inference. The pretrained weights in `shared_models/weights/` are unchanged.

## Install

From the repository root:

```sh
python -m pip install -r fig6/requirements.txt
```

Use Python 3.11 or later and a PyTorch build suitable for your CPU or CUDA device. The numerical environment used for these experiments was Python 3.13.2, PyTorch 2.10.0 with CUDA 12.9, NumPy 2.4.2 and SciPy 1.17.1.

## Prepare and train

The preparation command downloads the [Fig. 6 texture library](https://github.com/beiming10/PhySMI/releases/tag/fig6-training-v1.0), then generates the simulation splits. It checks the downloaded files against `library.json`. The download is shared by both models.

Five-organelle cells:

```sh
python fig6/prepare.py --model cells --data fig6_data
python fig6/train.py --model cells --data fig6_data --output training_runs/fig6_cells --device cuda:0
```

E. coli:

```sh
python fig6/prepare.py --model ecoli --data fig6_data
python fig6/train.py --model ecoli --data fig6_data --output training_runs/fig6_ecoli --device cuda:0
```

Preparation uses four CPU threads by default. `--workers 1` reduces memory and CPU use. Add `--offline` if the library has already been downloaded. Each simulation dataset occupies approximately 1–2 GB after compression. Preparation and training refuse to overwrite existing output directories.

| Setting | Cells | E. coli |
|---|---|---|
| Measured inputs at inference | 3 integrated emission bands | 5 disjoint integrated emission bands |
| Component outputs | 5 | 13 |
| Reconstruction observations during training | 28 | 83 |
| Training / validation / test samples | 2,048 / 40 / 64 | 1,024 / 64 / 132 |
| Batch size | 2 | 2 |
| Training patch | 128 × 128 | 128 × 128 |
| Generator updates | 40,000 | 20,000 |
| Initial generator-only updates | 2,000 | 1,000 |
| Learning-rate schedule | 20k from 2×10⁻⁴ to 4×10⁻⁵, then 20k to 8×10⁻⁶ | 20k from 2×10⁻⁴ to 4×10⁻⁵ |

The cell model's second phase continues the generator, shared discriminator, optimizers and random-number streams without resetting them. Training recipes are in `configs/`. Input bands, component order, response matrices and observation-likelihood tables are loaded from `shared_models/configs/` and `shared_models/calibration/`.

For cells, the three measured inputs are accompanied by an internally generated constant identifying branch A or B.

## Simulation

The release contains 1,108 nonnegative float32 single-stain texture crops, each 256 × 256 pixels. They come from our laboratory's five-structure image library. `library.json` supplies file checksums, source-cluster identifiers and the source assignments used by each simulation. Cell validation and test simulations use source groups held out from training; the bacterial simulation uses separate training, validation and test source groups.

For cells, each spectral component can combine textures from any library class. Natural, punctate, filamentous and diffuse textures are recomposed with independent image-prior donors. Mixtures include ordinary, weak, single-component, paired, absent-component, blank and overlapping cases. Spatial degradation precedes measurement generation. The 28-channel target and the two branch inputs use independently drawn noise. Individual channels are rounded and censored to 8 bits before the selected channels are summed into the three inputs. No clipping or renormalization is applied after summation.

For bacteria, single-stain texture crops modulate simulated rod fields with single, partially overlapping, colocated, separated, sparse and blank component configurations. The simulation includes intensity imbalance, spatial degradation and quantization-aware amplitude selection. The released 83-observation response and noise calibration produces the 23-channel multiexcitation view and four additional excitation views. The five inference inputs are sums of disjoint bins within the first 23 channels. All 13 component outputs remain enabled.

Training files contain only noisy observations, branch inputs where required, independent unlabelled image priors, and numerical noise parameters. Component maps are used to form the simulations, but are saved only for held-out evaluation and are not passed to the optimizer. There is no component-map loss or NNLS teacher. `orders.npz` supplies the original two-sample training schedules; `simulation.json` supplies the simulation settings and held-out scene definitions.

## Checkpoints and evaluation

The trainer saves complete resumable checkpoints and a loss history. Validation produces component-wise numerical tables without changing the training state. The final training step also evaluates the held-out test set. Checkpoints can be evaluated separately:

```sh
python fig6/evaluate.py --model cells --data fig6_data --checkpoint training_runs/fig6_cells/checkpoint_040000.pt --output outputs/cells_test.csv --device cuda:0
python fig6/evaluate.py --model ecoli --data fig6_data --checkpoint training_runs/fig6_ecoli/checkpoint_020000.pt --output outputs/ecoli_test.csv --device cuda:0
```

Omit `--checkpoint` to evaluate the corresponding published model. The CSV records all component outputs, reference and output RMS, RMSE, and the fitted response slope for each component. Components absent from a reference have no fitted slope; their output RMS reports false output.

To resume in a new output directory:

```sh
python fig6/train.py --model cells --data fig6_data --resume training_runs/fig6_cells/checkpoint_020000.pt --output training_runs/fig6_cells_continued --device cuda:0
```

## Apply a trained model

The checkpoints are compatible with the public inference entry point:

```sh
python shared_models/infer.py --model cells --weights training_runs/fig6_cells/checkpoint_040000.pt --input TestDATA/external_cells/NIH3T3_field33_three_bands.tif --mode raw --output outputs/cells.tif --device cuda:0
python shared_models/infer.py --model ecoli --weights training_runs/fig6_ecoli/checkpoint_020000.pt --input TestDATA/ecoli/Ecoli_mixture_five_bands.tif --mode raw --output outputs/ecoli.tif --device cuda:0
```

The external test files contain only the three or five inference bands. They are not used as training textures or multicolor training observations. Their source attribution is given in [TestDATA](../TestDATA/README.md).

## Short execution check

Use a separate data directory for a small execution test:

```sh
python fig6/prepare.py --model cells --data fig6_smoke --train-count 4 --val-count 2 --test-count 2
python fig6/train.py --model cells --data fig6_smoke --output training_runs/cells_smoke --stop-after 2 --validate-every 0 --device cpu
```

This checks the data and training interfaces; the reported model uses the full sample counts and update budget above. A CPU execution is useful for this check; the full training runs use a CUDA GPU.

Software checks can be run with:

```sh
python -m unittest discover -s fig6 -p test_training.py -v
```
