# NNWS

NNWS is an open, testable implementation of **Neural Network Weighting
Stacking** for seismic cross-correlation functions (CCFs). Each CCF is passed
independently through an MLP,

```text
N → 512 → 256 → 128 → 64 → 16 → 1,
```

whose sigmoid output is its stacking weight.

## Installation

Create an environment with Python 3.10 or newer, then install the package:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

To reproduce the publication figures:

```bash
python -m pip install -e ".[figures]"
nnws-figures --root . --figures all
```

The repository's `Figures.ipynb` provides the same three figure workflows in
notebook form. It reads the versioned data under `syn_datasets/` and published
weights under `syn_results/`; figure reproduction therefore does not retrain a
stochastic model.

## Metrics and reproducibility

The training SNR loss uses both regions outside the signal window as noise.
The figure metric `reported_snr` uses signal-window RMS divided by the RMS of
a remote positive-lag tail window. The RMSR_SS comparison combines the
pre-signal and remote-tail windows. All SNR-related calculations use RMS
ratios; they are exposed as `training_rms_ratio`, `reported_snr`, and
`selective_rms_ratio` so their noise-window definitions remain explicit.

Run the test suite with:

```bash
python -m unittest discover -s tests -v
```

Data preprocessing (instrument correction, filtering, one-bit normalization,
spectral whitening, windowing, and CCF construction) is outside this package.
Inputs must already be preprocessed consistently with the experiment being
reproduced.

## Repository layout

- `src/nnws/`: model, losses, training, metrics, and figure APIs
- `tests/`: formula, shape-contract, and numerical tests
- `Figures.ipynb`: deterministic reproduction of the three included figures
- `syn_datasets/`, `syn_results/`: synthetic inputs and published weights

## License and citation

Code is available under the MIT License. See `CITATION.cff` for citation
metadata. Research data may carry provenance requirements from its original
source; verify those requirements before redistribution.
