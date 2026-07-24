# NNWS

NNWS is an open, testable implementation of **Neural Network Weighting
Stacking** for seismic cross-correlation functions (CCFs). Each CCF is passed
independently through an MLP,

```text
N → 512 → 256 → 128 → 64 → 16 → 1,
```

whose sigmoid output is its stacking weight.

This release makes two formula-level corrections to the earlier research
script:

1. The stack is the normalized weighted mean
   `S = sum(w_i * d_i) / sum(w_i)`, not an unnormalized weighted sum.
2. The symmetry loss is consistently defined and named as anti-symmetric RMS
   divided by symmetric-component RMS.

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

## Minimal use

```python
import numpy as np
from nnws import TrainingConfig, apply_stacking_network, train

ccfs = np.load("ccfs.npy")  # shape [M, N], with odd N
result = train(
    ccfs,
    tau=8,
    tmin=64,
    tmax=69,
    sampling_rate=1.0,
    config=TrainingConfig(seed=0),
)
stack, weights = apply_stacking_network(result.model, ccfs)
```

The model accepts one CCF matrix with shape `[M, N]` or `[1, M, N]`.
`sampling_rate` is expressed in Hz and defaults to `1.0`; `lag`, `tau`,
`tmin`, and `tmax` are expressed in seconds. The legacy sampling-interval
argument `delta` remains supported and must satisfy
`delta = 1 / sampling_rate` when both are supplied. A trained model has a
fixed `N`; the default training parameters are 200 epochs, Adam with learning
rate `0.001`, and loss weights `0.8:0.2:0.01`.

## Metrics and reproducibility

The training SNR loss uses both regions outside the signal window as noise.
The figure metric historically called `ratio4` uses a remote positive-lag tail
window, the comprehensive figure reports peak amplitude over remote-tail RMS,
and the RMSR_SS comparison combines the pre-signal and remote-tail windows.
They are intentionally exposed as `training_rms_ratio`,
`reported_tail_rms_ratio`, `reported_peak_rms_ratio`, and
`selective_rms_ratio` so the definitions cannot be confused.

Run the test suite with:

```bash
pytest
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
