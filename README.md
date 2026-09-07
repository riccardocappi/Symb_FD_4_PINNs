# Integrating Data-Driven Symbolic Fundamental Diagrams into Physics-Informed Neural Networks for Traffic State Estimation


## Models

All models are trained on velocity observations $u(x,t)$ sampled at a few sensor
locations, and evaluated on the held-out points of the full space–time grid.

| Name | `--config_path` | Flags | Description |
|---|---|---|---|
| **PUNN** | `conf/punn.yml` | – | Purely data-driven MLP baseline. No physics loss. |
| **PINN-LWR** | `conf/pinn.yml` | – | PINN with a Greenshields FD whose $(v_{\max}, \rho_{\max})$ are pre-fitted by least squares on the training pairs $(\rho, v)$ and then frozen. |
| **PINN-LWR (end-to-end)** | `conf/pinn.yml` | `--end_to_end` | Same, but $(v_{\max}, \rho_{\max})$ are `nn.Parameter`s trained jointly with the network. |
| **PINN-FDL** | `conf/pinn_fd.yml` | `--learn_fd` | FD replaced by a second MLP, regularised towards concavity over $[\rho_a, \rho_b]$ (weight `xi`). Black-box FD baseline. |
| **PINN-SR** *(ours)* | `conf/symb_fd.yml` | – | FD discovered by PySR under a Greenshields template, converted to a `torch` module via `sympytorch`, and frozen inside the PINN. |
| **PINN-SR (poly)** | `conf/symb_fd_poly.yml` | `--poly_fit` | Ablation: instead of symbolic regression, a Greenshields-family polynomial of degree 2 or 3 is curve-fitted (`power` is a searched hyper-parameter). |

The network maps $(x, t) \mapsto \rho$. Velocity predictions are obtained as
$v = Q(\rho) / \rho$, and for the symbolic/polynomial FDs the density is squashed
into the physically admissible range with $\rho = \rho_{\max}\,\sigma(\cdot)$.
`--ablation` disables this $\rho_{\max}$ prior.

---

## Repository layout

```
main.py                     entry point: parses CLI args, dispatches to an Experiment
train_loop.py               training loop, early stopping, seeding
conf/                       YAML configs (model name, device, Optuna search space)
models/
  pinn.py                   MLP, PINN interface, LWR_NN (Greenshields / learned / symbolic FD)
  punn.py                   PUNN — data-only baseline
experiments/
  experiment.py             base class: data handling, Optuna study, test evaluation, heatmaps
  experiment_pinn.py        PINN-LWR and PINN-FDL
  experiment_punn.py        PUNN
  experiment_symb_fd.py     PINN-SR: PySR fit, rho_max extraction, poly-fit ablation
utils/
  data_utils.py             NGSIM loading, virtual-sensor sampling, LHS collocation points
  utils.py                  Greenshields fitting, PySR wrapper, equation parsing, sympy→torch
data/I80/                   NGSIM I-80 velocity & density matrices (raw .txt)
data/torch_data/            cached, pre-sampled datasets (.pth) for reproducible runs
```

## Installation

Requires Python 3.12, and CUDA 11.8 for the pinned GPU wheels.

```bash
conda create -n symb_fd python=3.12
conda activate symb_fd

pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
```

On a CPU-only machine, drop both index flags and remove the `+cu118` local version
tags from `requirements.txt` (and set `device: "cpu"` in the config files).

PySR runs a Julia backend: `juliacall` ships a private Julia install, and the
`SymbolicRegression.jl` environment is built automatically on the first
instantiation of `PySRRegressor` (this can take a few minutes).

---

## Usage

```bash
python main.py --config_path=./conf/symb_fd.yml \
               --study_name=my_run \
               --seed=42 \
               --n_trials=100 \
               --scale \
               --data_name=I80_075_seed42.pth \
               --load_data
```

### Command-line arguments

| Argument | Default | Meaning |
|---|---|---|
| `--config_path` | `./conf/pinn.yml` | YAML config; its `model_name` selects the experiment class |
| `--vel_path` / `--dens_path` | 4 pm I-80 files | Raw NGSIM velocity / density matrices |
| `--n_trials` | `20` | Number of Optuna trials (ignored for `grid_search`) |
| `--method` | `optuna` | `optuna` (TPE) or `grid_search` |
| `--study_name` | `test` | Study name; also the results sub-folder |
| `--process_id` | `0` | Distinguishes parallel workers sharing one study |
| `--val_perc` | `0.2` | Validation fraction (tail of the shuffled training set) |
| `--seed` | `42` | Global seed (torch, numpy, random, deterministic algorithms) |
| `--scale` | off | Min-max scale the $(x,t)$ inputs to $[-1, 1]$ |
| `--sample_ratio` | `0.75` | Fraction of sensor cells kept as observations |
| `--load_data` / `--data_name` | off / `test` | Load / save the cached dataset in `data/torch_data/` |
| `--learn_fd` | off | **PINN-FDL**: FD is a black-box MLP (use `conf/pinn_fd.yml`) |
| `--end_to_end` | off | **PINN-LWR**: learn Greenshields $(v_{\max}, \rho_{\max})$ jointly |
| `--poly_fit` | off | **SYMB-FD**: polynomial FD fit instead of PySR (use `conf/symb_fd_poly.yml`) |
| `--pysr_ms` | `loss` | PySR equation selection criterion: `loss` or `score` |
| `--ablation` | off | Remove the $\rho_{\max}$ physics prior on the density output |
| `--split_out` | off | Two-headed network predicting velocity and density separately |

### Hyper-parameter search

Model selection is delegated to Optuna. The search space lives in the `search_space`
block of each YAML config:

```yaml
model_name: "SYMB-FD"
device: "cuda"
opt: "Adam"
criterion: "MSE"
storage: "journal"
search_space:
  n_hidden_neurons: [16, 64]        # int range
  n_hidden_layers: [2, 5]           # int range
  activation_function: ["relu", "tanh", "leaky_relu", "softplus"]
  mu1: [0.1, 1.0]                   # data-fidelity loss weight
  mu2: [0.1, 1.0]                   # PDE-residual loss weight
  epochs: [1000, 3000]
  patience: [200, 500]
  batch_size: [256, 512, 1400]
  lr: [0.0001, 0.01]                # log-uniform
```

Trials are persisted through an Optuna `JournalStorage`
(`optuna_journal_storage.log`), so several processes can contribute to the same
study concurrently and a run can be resumed after an interruption. Set
`storage: "sqlite"` to use `optuna_study.db` instead.

---

## Outputs

Each run writes to `./trained_models/<MODEL_NAME>/<study_name>/<process_id>/`:

| File | Content |
|---|---|
| `ckpt.pth` | Best checkpoint: weights, per-epoch losses, test metrics, predictions |
| `best_params.json` | Best hyper-parameters found by Optuna |
| `config.yml` | Copy of the config used |
| `symb_fds/params_loss.json` | *(SYMB-FD)* discovered equation and its fitted parameters |
| `figs/heatmap.png` | Estimated space–time velocity field |
| `optuna_logs/optuna_logs.txt` | Optuna + test-metric logs |

Reported test metrics — computed on grid points **not** used for training — are MSE,
MAE, RMSE and relative RMSE.

Results of the runs reported in the paper are kept in the `trained_models_*`
directories (git-ignored).
