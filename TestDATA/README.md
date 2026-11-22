# Test inputs

This collection contains input images for the four public models. Laboratory
fixed-cell, single-stain and live-cell files have three channels in the order
**561, 532, 488 nm**. External NIH-3T3 inputs contain three integrated emission
bands. E. coli contains five integrated emission bands.

| Examples | Files | Model |
|---|---:|---|
| Fixed-cell acquisitions | 8 | `static` |
| Structure–spectrum reassignment inputs | 2 | `static` |
| Experimental single-stain fields | 5 | `static` |
| Four-colour live-cell snapshots | 4 | `live` |
| Five-colour live-cell snapshot | 1 | `live` |
| NIH-3T3 five-organelle fields | 4 | `cells` |
| E. coli mixed field | 1 | `ecoli` |

Run from the repository root:

```sh
python shared_models/run_examples.py --list
python shared_models/run_examples.py --output outputs/test_data --device cpu
```

Use `--model static`, `--model live`, `--model cells` or `--model ecoli` to select
an application. An individual example can be selected with `--id Mito`, for
example. The runner uses the settings in [examples.json](examples.json),
including the different acquisition offsets for the five-colour live cell.

All supplied input images are TIFF files. Acquired inputs retain
acquisition-unit intensities; single-stain inputs are means of repeated
acquisitions. The structure–spectrum reassignment TIFF files are
normalized observations constructed from acquired cell images. No contrast
stretch is applied to these test inputs. All five fixed-cell outputs remain
enabled when a single-stain field is processed.

## External data attribution

The four NIH-3T3 examples derive from Leonel Malacrida's
[Phasor-based multi-harmonic unmixing for in-vivo hyperspectral imaging](https://doi.org/10.5281/zenodo.13625087).
The native emission observations were integrated into three bands using the
selections in `shared_models/configs/cells.json`.

The E. coli example derives from Ruogu Wang's
[Multi-view spectral images](https://doi.org/10.5281/zenodo.13988378).
Five integrated bands were extracted from the calibrated 8-bit working
observations, with a central 1024×1024 crop. The selections are given in
`shared_models/configs/ecoli.json`. Bands are summed, not averaged or stretched.

Both external datasets are distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), which also applies to
these derived external inputs. The other examples are provided by the PhySMI
authors. Component outputs and the original full spectral stacks are not
included in this test collection; inference generates the output images.
