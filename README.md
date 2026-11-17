# second-viewing

A preregistered test of whether seeing a novel image once more changes a human medial temporal lobe neuron's later response in proportion to its category tuning.

Paper: https://ryuzokijima.com/research/second-viewing/ · Registration: https://osf.io/z56gw

| File | Contents |
|---|---|
| `preregistration.md` | Hypothesis, analysis, decision rule |
| `run.py` | Locked analysis |
| `analysis.py` | Functions used by `run.py` and `calibrate.py` |
| `simulate.py` | Synthetic data with known effects |
| `calibrate.py` | False-positive rate and power on synthetic data |
| `calibration/` | One line per synthetic dataset, as reported in the preregistration |
| `load_001187.py`, `load_000004.py` | Loaders for DANDI 001187 and 000004 |
| `examined.py` | Encoding-phase numbers disclosed in the preregistration |
| `typicality.py` | Builds `inputs/typicality.csv` |
| `inputs/` | Image set of each 001187 session; image typicality |
| `results/` | Results of the registered analysis (`001187.json`, `000004.json`) and of the exploratory analyses (`extra_*.json`); figures |
| `extra/` | Exploratory analyses and figures, added after the results |
| `OSF_REGISTRATION_URL.txt`, `POSTED_001187_SHA.txt` | The registration, and where the SHA-256 of `results/001187.json` was posted |

## Run

```
pip install -r requirements.txt
dandi download DANDI:001187/0.240904.2347 DANDI:000004/0.220126.1852
export DATA=$PWD
DRYRUN=1 python run.py
```

`DRYRUN=1 python run.py` (and `DRYRUN=1 python run.py 000004`) run on synthetic spike counts for both phases, without neuron exclusion, and write `results/*_DRYRUN.json`. Without it, `run.py` reads test-phase spikes only when `OSF_REGISTRATION_URL.txt` holds the registration and `UNBLIND=after-registration`. `python run.py 000004` runs after the 001187 result is posted.

Each line in `calibration/` records its synthetic dataset as `world` and `seed`; `python calibrate.py MODE 'WORLD:SEED'` reproduces it.

## Data

- Daume, J., Cheng, S. & Rutishauser, U. (2024). Data for: Daume et al. (2024) "Persistent activity during working memory maintenance predicts long-term memory formation in the human hippocampus". DANDI archive, version 0.240904.2347. https://doi.org/10.48324/dandi.001187/0.240904.2347
- Chandravadia, N. et al. (2022). A NWB-based dataset and processing pipeline of human single-neuron activity during a declarative memory task. DANDI archive, version 0.220126.1852. https://doi.org/10.48324/dandi.000004/0.220126.1852

Both are licensed CC-BY 4.0.

