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

Everything runs through `run.sh`. Pretrain happens automatically -- there is no
manual copying of an initial population.

```bash
# SESiL with the default merger (permutation), one seed
bash run.sh --seeds 0

# Three seeds, ZipIt! merging, guided mate selection, 40 generations
bash run.sh --seeds 0,1,2 --merger zipit --selection guided --generations 40

# The learning-based baseline classifier
bash run.sh --seeds 0 --method baseline --baseline-epochs 100
```

**Methods.** There are two: `sesil` (the evolutionary pipeline) and `baseline`
(a conventionally trained classifier). The merge operator and the mate-selection
rule are *arguments to SESiL*, not separate methods:

| flag | values |
|---|---|
| `--merger` | `zipit`, `permute`, `wavg` |
| `--selection` | `bidirectional`, `breed`, `guided`, `hard` |

**Configuration.** Every knob lives in `config.py`, grouped by stage (common /
pretrain / evolution / merging / selection / baseline). Anything in there can be
set from the command line, so no Python needs editing to change an experiment.
Run `python main.py --help` to see them all. Per-method defaults live at the top
of `run_sesil.sh` and `run_baseline.sh`; shared settings (seeds, budget,
dataset, population) live at the top of `run.sh`.

**Pipeline.** `main.py` runs pretrain -> evolution. Pretrain is skipped when a
population already exists at the resolved path; pass `--force-pretrain` to
rebuild it, or `--pretrain-only` to stop after it. A run that was interrupted
resumes with `--start-gen N`.

**Outputs.** Each run writes to
`results/<exp-name>/<merger>_<selection>/seed<N>/` containing `config.json`, a
`results.csv` with one row per individual per generation, and `checkpoints/gen_N/`
holding each generation's population.

**Layout.**

```
run.sh              seeds, budget, dataset; dispatches to a method
run_sesil.sh        SESiL: merger + selection + their hyper-parameters
run_baseline.sh     learning-based classifier baseline
config.py           every setting for every method, in one place
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
  data.py           raw CIFAR loaders
```

The original per-method scripts under `training_scripts/` are kept for
reference; they are superseded by the above.

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

