# SESiL: Social, Evolutionary supported Learning

![SESiL Concept Figure](images/overview.png)

This is the official implementation of **SESiL** from our paper:
**[SESiL: Social, Evolutionary supported Learning](https://dl.acm.org/doi/abs/10.65109/LPWP5477) @ AAMAS 2026**.\
Our codebase is built on the one for **ZipIt! Merging Models from Different Tasks without Training @ ICLR 2024**.\
Check their repository for more details_[GitHub](https://github.com/gstoica27/ZipIt).

### Installation
Create a virtual environment and install the dependencies:
```bash
conda create -n sesil python=3.7
conda activate sesil
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### Usage

Everything runs through `run.sh`, which exposes exactly three knobs:

| knob | meaning |
|---|---|
| `--dataset` | which environment: `cifar10` or `cifar100` |
| `--budget` | total training budget: generations for SESiL, epochs for the baseline |
| `--seeds` | comma-separated seeds |

```bash
# SESiL on CIFAR-10, 25 generations, one seed
bash run.sh --dataset cifar10 --budget 25 --seeds 0

# CIFAR-100, 40 generations, three seeds
bash run.sh --dataset cifar100 --budget 40 --seeds 0,1,2

# the learning-based baseline classifier, 100 epochs
bash run.sh --dataset cifar10 --budget 100 --seeds 0 --method baseline
```

Pretrain happens automatically -- there is no manual copying of an initial
population. If one already exists for the configured shape it is reused.

**Everything else lives in `config.py`.** Merge operator, selection rule,
population size, classes per model, all hyper-parameters: edit them there, in
one place. They are still exposed as CLI flags, so a one-off sweep can override
one without editing the file:

```bash
bash run.sh --dataset cifar10 --budget 25 --seeds 0 --merger zipit --selection guided
```

**Methods.** There are two: `sesil` (the evolutionary pipeline) and `baseline`
(a conventionally trained classifier). The merge operator and mate-selection
rule are *arguments to SESiL*, not separate methods:

| flag | values |
|---|---|
| `--merger` | `zipit`, `permute`, `wavg` |
| `--selection` | `bidirectional`, `breed`, `guided`, `hard` |

**Resuming.** A run that was interrupted continues with `--start-gen N`. Pass
`--force-pretrain` to rebuild the initial population, or `--pretrain-only` to
stop after it.

**Outputs.** Each run writes to
`results/<exp-name>/<dataset>/<merger>_<selection>/seed<N>/` containing
`config.json` (every setting the run used), `results.csv` (one row per
individual per generation, with `Generation`, `Stage` and `Seed` columns), and
`checkpoints/gen_N/` holding each generation's population.

**Layout.**

```
run.sh              dataset, budget, seeds -> dispatches to a method
run_sesil.sh        SESiL launcher
run_baseline.sh     learning-based classifier baseline
config.py           EVERY setting for every method, in one place
main.py             entry point: pretrain -> evolution
sesil/
  pretrain.py       creates the initial population
  evolution.py      the generation loop
  selection.py      mating score and the four selection strategies
  merge.py          crossover via ModelMerge
  mutation.py       finetune
  fitness.py        per-individual and multi-task evaluation
  population.py     on-disk population layout and naming
  registry.py       merger / selection lookup tables
  data.py           raw dataset loaders
```

## Citation

If you use SESiL or this codebase in your work, please cite:
```
@inproceedings{zhao2026sesil,
author = {Zhao, Tianshu and Rabinovich, Zinovi},
title = {SESiL: Social, Evolutionary Supported Learning},
year = {2026},
publisher = {International Foundation for Autonomous Agents and Multiagent Systems},
doi = {10.65109/LPWP5477},
booktitle = {Proceedings of the 25th International Conference on Autonomous Agents and Multiagent Systems},
pages = {2160–2168},
numpages = {9}
}
```

