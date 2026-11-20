# Shared-prior PhySMI models

Four spectral configurations use the same feature backbone and a shared,
unlabelled image prior. Each configuration has its own response matrix and
generator weights.

| Configuration | Application | Measured inputs | Component outputs |
|---|---|---:|---:|
| `static` | Laboratory fixed cells and structure–spectrum exchange | 3 excitation channels | 5 |
| `live` | Laboratory four- and five-colour live cells | 3 excitation channels | 5 |
| `cells` | External NIH-3T3 five-organelle spectral data | 3 integrated emission inputs | 5 |
| `ecoli` | External 13-dye E. coli spectral data | 5 integrated emission inputs | 13 |

The files in `weights/` contain the generator parameters only. The earlier
implementation in the repository root remains available separately.

## Install

Run from this directory with Python 3.11 or later:

```sh
python -m pip install -r requirements.txt
```

Install a PyTorch build appropriate for the chosen CPU or CUDA device.

## Inference

Arrays have `C,Y,X` or `T,C,Y,X` axes. In normalized mode, supply the selected
inputs in the order listed in the configuration. The output is a float32
`T,K,Y,X` array; its JSON sidecar gives the component names and units. No display
window, spatial filter or output suppression is applied.

```sh
python infer.py --model static --input examples/static_input.npy --output outputs/static.npy --device cpu
python infer.py --model live --input input.npy --output outputs/live_four.npy --panel 1,1,1,1,0 --device cpu
python infer.py --model cells --input input.npy --output outputs/cells.npy --device cpu
python infer.py --model ecoli --input input.npy --output outputs/ecoli.npy --device cpu
```

For raw laboratory images, add `--mode raw`. Full eight-channel stacks are
selected into A channels automatically, in the order 561, 532 and 488 nm. Raw
three-channel stacks must already follow that order. Subtraction and scaling
follow the selected configuration. The five-colour live acquisition has its own
offsets, given in `configs/live.json`; pass its three selected offsets with
`--offset` when processing that acquisition. For the external configurations,
raw mode integrates the listed native emission bins and divides by the stated
unit. The source acquisition must already use the response-calibration units.

The fixed-cell model uses the three acquired input images and retains all five
outputs, including non-target signal. It does not require a staining-panel
vector. Its signal-conditioned readout is included in the supplied generator.
The live-cell model receives the experimental staining panel through `--panel`;
use `1,1,1,1,0` for four-colour data and `1,1,1,1,1` for five-colour data.

## Self-supervised training interface

`train.py` accepts observation-only NPZ batches. Required keys are `observed`,
`input_a` and `prior`. Paired-branch configurations additionally use `input_b`;
the live-cell configuration uses `panel`. Cell-panel batches include `sigma_DN`
and `shot_alpha`. Arrays may contain a batch axis or one sample. `prior` is an
independently sampled, unlabelled image pool in normalized abundance units,
processed on the same spatial grid as the observations. It is not the source
component image used to generate that observation.

Eight-channel observation and input arrays are normalized by the configured
unit. Cell-panel `observed` arrays are quantized count-domain observations;
their A/B inputs are normalized. E. coli batches contain 83 normalized
observations and five integrated normalized inputs. The calibrated likelihood
tables are included in `calibration/`.

```sh
python make_example.py --model static --library examples/own_single_stain --output examples/static_batches
python train.py --model static --data examples/static_batches --output training_runs/static --steps 4 --device cpu
```

The compact example generator illustrates the data interface. It is not the
acquisition-specific sampler used to obtain the supplied weights. The trainer
supports observation-based fine-tuning or fresh initialization, uses no
component GT or NNLS teacher, and never reads evaluation references. Starting
from published generator weights creates fresh optimizers; it is fine-tuning,
not restoration of a historical optimizer trajectory. Supply independently
prepared batches and their actual acquisition calibration for a scientific
training comparison.

`gan=False` selects an observation-only batch. All output channels share one
discriminator. The adversarial gradient
is capped at 10% of the observation-gradient norm, followed by generator
gradient clipping at 1. The four observation objectives and their numerical
constants are implemented in `physmi/losses.py` and `physmi/cells_loss.py`.

## Single-component evaluation

```sh
python evaluate.py prediction_KYX.npy reference_KYX.npy --active 1 --output statistics.json
```

The evaluation retains every output. It reports non-target signal over the
entire FOV, target-intensity retention and the spatial regression of each
output on the single-component reference. A non-target background floor and
a copied source texture are therefore visible as different quantities.

## Files

Run the CPU interface tests with `python -m unittest discover -s tests`.
The example directory includes five laboratory single-stain textures and four
simulated inputs with their expected outputs. These five textures are an
example subset, not the complete single-stain image library.

`configs/` defines component order, measured bins, response matrices, input
units, offsets and selected training constants. `physmi/` contains the network,
conditioning layers and objectives. `weights/` contains the four corresponding
generators. A configuration and weight file must be used together. Colour
assignments and manuscript display contrast are presentation choices and are
not applied by inference.
