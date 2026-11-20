# Shared-prior PhySMI models

Four configurations share the feature backbone and an unlabelled image prior. Each has its own response matrix and generator weights.

| Configuration | Application | Inputs | Outputs |
|---|---|---:|---:|
| `static` | Laboratory fixed cells and structure–spectrum reassignment | 3 excitation channels | 5 |
| `live` | Laboratory four- and five-colour live cells | 3 excitation channels | 5 |
| `cells` | NIH-3T3 five-organelle data | 3 integrated emission bands | 5 |
| `ecoli` | 13-dye E. coli data | 5 integrated emission bands | 13 |

The files in `weights/` contain generator parameters. Keep each weight file with its corresponding configuration and network code.

## Inference

The following commands run from the repository root after installing `shared_models/requirements.txt`.

```sh
python shared_models/infer.py --model static --input TestDATA/data_561_532_488_0001.tif --mode raw --output outputs/static.tif
python shared_models/infer.py --model live --input TestDATA/live_cells/four_colour_cell01_cycle0001.tif --mode raw --panel 1,1,1,1,0 --output outputs/live_four.tif
python shared_models/infer.py --model live --input TestDATA/live_cells/five_colour_selected_cycle0073.tif --mode raw --panel 1,1,1,1,1 --offset 255,256,295 --output outputs/live_five.tif
python shared_models/infer.py --model cells --input TestDATA/external_cells/NIH3T3_field33_three_bands.npy --mode raw --output outputs/cells.tif
python shared_models/infer.py --model ecoli --input TestDATA/ecoli/Ecoli_mixture_five_bands.tif --mode raw --output outputs/ecoli.tif
```

Input axes are `C,Y,X` or `T,C,Y,X`. Laboratory inputs are ordered 561, 532 and 488 nm. Raw inputs are prepared using the offsets and scale in the configuration; `--offset` overrides the selected-channel offsets. Normalized inputs must already contain the selected bands and must have completed this preparation.

The fixed-cell model uses all five output channels without a staining-panel input. The live-cell model takes the experimental panel through `--panel`: four-colour data use `1,1,1,1,0`, and five-colour data use `1,1,1,1,1`. All five output images are retained.

The external-cell model uses 128-pixel tiles with 64-pixel overlap and a positive Hann window. E. coli uses non-overlapping 512-pixel tiles. These settings match the native-image application routines. They can be overridden with `--tile-size` and `--tile-overlap`; `--tile-size 0` uses a full frame.

Outputs are float32 TIFF or NPY arrays with `T,K,Y,X` axes. A JSON sidecar gives component names, units and the weight identifier. No display contrast, intensity renormalization or output suppression is applied.

## Run the example collection

```sh
python shared_models/run_examples.py --list
python shared_models/run_examples.py --model static --output outputs/static_examples
python shared_models/run_examples.py --output outputs/all_examples
```

Each example is paired with its own input mode, model and acquisition settings. The batch runner loads each model once and writes separate component stacks. It does not overwrite an existing output directory.

`examples/` also contains small normalized input arrays for interface testing. The single-channel files in `examples/own_single_stain/` form a small laboratory texture-library subset for the training example.

## Self-supervised training

Run from `shared_models/`:

```sh
python make_example.py --model static --library examples/own_single_stain --output examples/static_batches
python train.py --model static --data examples/static_batches --output training_runs/static --steps 4 --device cpu
```

Training batches provide `observed`, `input_a` and an independently sampled `prior`; paired-branch models also use `input_b`. Live-cell batches provide `panel`, and external-cell batches provide `sigma_DN` and `shot_alpha`. Full reconstruction observations contain eight laboratory channels, 28 external-cell channels or 83 calibrated E. coli observations. The inference-only files in TestDATA are not complete training batches.

The trainer supports fresh initialization and fine-tuning on prepared observation batches, with spectral reconstruction and one shared image discriminator. Component GT and NNLS images are not training inputs. The compact example generator illustrates the batch interface using independent single-stain textures. Acquisition-specific simulations use their calibrated noise and intensity distributions.

## Software tests

From the repository root:

```sh
python -m unittest discover -s shared_models/tests -v
```

The tests check all four model interfaces, input preparation, full-frame and overlap inference, and TIFF/NPY output. Numerical calibration tables are in `calibration/`; component order, band selections and input scales are in `configs/`.
