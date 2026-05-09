# PID-SACA

Reference implementation for **PID-Controlled Constrained RL for Hub-based Joint Pricing, Dispatching, and Routing with Service Guarantees**.

PID-SACA studies hub-based mobility service control with three coupled decisions: destination-level pricing, seat allocation/dispatching, and route execution. The implementation models service quality through Order Response Rate (ORR) and uses a PID-controlled Lagrangian multiplier to stabilize constrained SAC training.

This repository is organized as a paper-facing release: core methods live under `src/`, reproducibility entry points live under `experiments/`, and generated artifacts are ignored by git.

## Project Layout

```text
src/pid_saca/
  algorithms/                  Main PID-SACA method and comparison baselines
  routing/                     Attention routing model, data generator, and weights
  paths.py                     Repository-relative path definitions

experiments/                   Training, sensitivity, stability, and ablation entry points
scripts/                       Lightweight command-line utilities
preprocessing/                 Optional station-selection helpers for raw coordinate data
data/stations_30/              Station coordinates used by the released experiments
docs/                          Release notes and auxiliary documentation
legacy/                        Ignored local scripts not used in the paper-facing release
```

Key released files:

```text
src/pid_saca/algorithms/pid_saca.py          PID-controlled Lagrangian SAC method
src/pid_saca/algorithms/saca.py              Spatial-attention SAC baseline
src/pid_saca/algorithms/saca_baseline.py     MLP-style SAC baseline
src/pid_saca/algorithms/grc_elg.py           GRC-ELG-style experimental baseline
src/pid_saca/algorithms/jdrl_pomo.py         JDRL-style PPO baseline
src/pid_saca/routing/model/                  Shared attention-router weights and stats
```

## Environment

Recommended:

- Python 3.10+
- PyTorch installed for your CUDA or CPU environment

Install the package and common dependencies:

```bash
pip install -e .
```

Optional preprocessing dependencies:

```bash
pip install -r requirements-optional.txt
```

## Quick Start

Train PID-SACA:

```bash
python scripts/train_pid_saca.py
```

Run the main comparison baselines sequentially:

```bash
python experiments/train_all.py
```

Generate comparison plots from prepared logs:

```bash
python scripts/plot_comparison.py
```

Training logs are written under `log/` by default. Plotting helpers read logs from their configured folders and write figures under `figure/`. These folders are generated outputs and are ignored by git.

## Experiments

The released experiment entry points are:

```text
experiments/train_all.py                 Main baseline comparison launcher
experiments/stability.py                 PID-vs-vanilla stability helper
experiments/sensitivity_delta.py         Target-ORR sensitivity helper
experiments/sensitivity_capacity.py      Capacity sensitivity helper
experiments/ablation_saca_orr.py         ORR-related ablation variant
experiments/ablation_wo_pricing.py       No-pricing ablation variant
experiments/ablation_wo_route.py         No-route-feedback ablation variant
experiments/summarize_ablation.py        Ablation log summarizer
experiments/summarize_pid_ablation.py    PID ablation log summarizer
```

## Data And Weights

- The released training configuration uses `data/stations_30/bus_stations.txt`.
- Runtime route feedback uses `src/pid_saca/routing/model/best_model.pth` and `src/pid_saca/routing/model/normalization_stats.pt`.
- Shared paths are centralized in `src/pid_saca/paths.py` so scripts can be launched from the repository root without relying on the old flat-file layout.
- Raw ride-order data is not included. The preprocessing helpers are provided for users who have compatible coordinate data.

## Release Scope

The public release keeps paper-facing code in `src/`, `experiments/`, `scripts/`, `preprocessing/`, `data/`, and `docs/`. Local exploratory visualizations, old route-only prototypes, draft case-study scripts, temporary outputs, and generated figures/logs are excluded through `.gitignore`.

## LLM Usage

For the LLM usage statement, see [docs/LLM_USAGE.md](./docs/LLM_USAGE.md).

## Citation

If you find this code helpful in your research, please consider citing our paper:

```bibtex
@inproceedings{du2026pidsaca,
  title={PID-Controlled Constrained RL for Hub-based Joint Pricing, Dispatching, and Routing with Service Guarantees},
  author={Du, Pengfei and Gao, Yucen and Yang, Xiaochun and Wang, Bin},
  booktitle={Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2026}
}
```
