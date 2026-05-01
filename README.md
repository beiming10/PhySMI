# PhySMI: Spectral Unmixing with Hybrid-SelfSupervised Adversarial Learning

A PyTorch implementation of PhySMI, a dual-branch hybrid-self-supervised adversarial unmixing framework with self-supervised denoising for hyperspectral image analysis.

## Overview

PhySMI is designed for spectral unmixing and denoising tasks, particularly in fluorescence microscopy imaging. It employs a dual-branch architecture that combines:

- **Self-supervised reconstruction loss** for accurate spectral decomposition
- **Self-supervised denoising** via the STC (Signal-to-Texture Contrasting) library
- **GAN-based adversarial training** for enhanced texture preservation
- **Branch consistency regularization** for robust dual-branch fusion

### Dual-Branch Input Channels

- **Branch A**: Channels [0, 3, 7] (561nm, 532nm, 488nm excitation)
- **Branch B**: Channels [1, 2, 4] 

## Installation
Still actively organizing this part of the content

## Dataset Structure

The dataset consists of hyperspectral TIFF stacks where each file contains multiple channels. The data loader automatically splits each stack into Target (full spectrum) and two input subsets (Branch A and Branch B) based on channel indices.

```
data_root/
├── Train/
│   ├── sample001.tif    # Full hyperspectral stack (8 channels, uint16)
│   ├── sample002.tif
│   └── ...
└── Valid/
    ├── sample001.tif
    ├── sample002.tif
    └── ...
```

### Data Loading Process

1. Each TIFF stack contains `hyper_frames` channels (e.g., 8 channels)
2. The full stack is used as `Target`
3. Channels specified by `input_indices_a` are extracted as Branch A input
4. Channels specified by `input_indices_b` are extracted as Branch B input
5. Random patches of size `patch_size` are cropped with stride `stride`
6. Data augmentation includes random rotation and flipping

## Quick Start

### Training

```bash
python train_PIANet_D_Single_Dual.py --config config_PIANet_D_488_532_561_dual_single.json
```

### Inference

The `testPIANetD_single_dual.py` script supports single/batch inference with multiple output modes.

#### Basic Usage

```bash
python testPIANetD_single_dual.py --config config_PIANet_D_488_532_561_dual_single.json --model generator_best.pth --input input_image.tif --output results/
```

#### Output Modes

| Mode | Description |
|------|-------------|
| `dual` | Fusion of Branch A and Branch B outputs |
| `single_a` | Only Branch A output |
| `single_b` | Only Branch B output |

#### Full Command Options

```bash
python testPIANetD_single_dual.py \
  --config config_file.json \
  --model model_checkpoint.pth \
  --input input.tif \
  --output output_dir/ \
  --device cuda \
  --output-mode dual \
  --no-visualize
```

#### Output Files

The inference produces:
- `{filename}_recon.tif` - Reconstructed hyperspectral image
- `{filename}_abundances.tif` - Abundance maps for each endmember
- `{filename}_comparison.png` - Visual comparison
- `{filename}_all_abundances.png` - All endmember visualization

## Configuration

Key parameters in `config_*.json`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `hyper_frames` | Total number of spectral channels in input |
| `num_endmembers` | Number of spectral endmembers |
| `input_indices_a` | Branch A channel indices | [0, 3, 7] |
| `input_indices_b` | Branch B channel indices | [1, 2, 4] |
| `patch_size` | Training patch size | 128 |
| `batch_size` | Batch size | 4 |
| `end_epoch` | Training epochs | 200 |
| `scalefactor` | Data normalization factor |
| `stride` | Patch cropping stride | 64 |
| `step` | Channel stepping for overlapping frames | 4 |
| `unfreeze_endmembers` | Whether to fine-tune endmember dictionary | false |

### Loss Function Weights

| Parameter | Description | Default |
|-----------|-------------|---------|
| `lambda_recon` | Reconstruction loss weight | 
| `lambda_gan` | GAN adversarial loss weight | 
| `lambda_denoise` | Self-supervised denoising weight  
| `lambda_contrastive` | Contrastive loss weight | 
| `lambda_abundance_consistency` | Branch consistency weight | 

## Pre-trained Endmembers

The repository includes pre-defined spectral endmembers for five subcellular structures:

| Index | Structure | Spectral Profile |
|-------|-----------|------------------|
| 0 | Mitochondria | ATTO542 |
| 1 | Nuclei | SybrGold |
| 2 | Lipid | LipidSpot 488 |
| 3 | Microtube | CF514 |
| 4 | Actin | YF633 |

Endmember file: `endmembers_SF_542_SybrGold_488_514_633.csv`

## Evaluation Metrics

The framework tracks the following metrics during training and validation:

- **MRAE** (Mean Relative Absolute Error)
- **RMSE** (Root Mean Square Error)
- **PSNR** (Peak Signal-to-Noise Ratio)

## Output Files

Training generates the following outputs in `outf/`:

```
experiment_directory/
├── generator_best.pth           # Best model checkpoint
├── generator_epoch_*.pth        # Per-epoch checkpoints
├── validation_outputs/           # Visual validation results
│   ├── sample_***_recon.tif
│   └── sample_***_abundances.tif
└── train.log                    # Training log
```

## Citation

If this work is useful for your research, please cite:

```

## License

MIT License

## Acknowledgments

- SpeAtten architecture based on [MST](https://github.com/caiyuanhao1998/MST-plus-plus)
