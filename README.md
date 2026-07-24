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

## Repository layout

- `src/nnws/`: model, losses, training, metrics, and figure APIs
- `tests/`: formula, shape-contract, and numerical tests
- `Figures.ipynb`: deterministic reproduction of the three included figures
- `syn_datasets/`, `syn_results/`: synthetic inputs and published weights

## License and citation

Code is available under the MIT License. See `CITATION.cff` for citation
metadata. Research data may carry provenance requirements from its original
source; verify those requirements before redistribution.
