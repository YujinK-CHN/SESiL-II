"""
Single configuration file for every method implemented in SESiL-II.

Only three things are meant to change from the command line, because only three
things change between runs of an experiment suite:

    --dataset   which environment to run in   (cifar10 / cifar100)
    --budget    total training budget, in epoch-equivalents (see sesil/budget.py)
    --seed      random seed

    bash run.sh --dataset cifar100 --budget 500 --seeds 0,1,2

Everything else -- merge operator, selection rule, population shape, all the
hyper-parameters -- is set by the defaults below. Edit them here. They are still
exposed as CLI flags so a one-off sweep can override one without editing the
file, but run.sh does not pass them, so this file is the single source of truth.

Groups:
    run         the three knobs above, plus method and output locations
    model       backbone
    pretrain    how the initial population is created
    evolution   the SESiL generation loop
    merging     hyper-parameters of the merge operator
    certificate what an agent is licensed to train on
    selection   hyper-parameters of mate selection
    baseline    the learning-based classifier baseline
"""

import argparse
import os


# --------------------------------------------------------------------------- #
# Environments.  Keyed by --dataset.
#
# 'loader_name' must match a variable in datasets/configs.py.  Note the
# CIFAR-100 entry there is called 'cifar50' -- a name inherited from ZipIt!,
# where it meant "50 classes per model".  It is the full CIFAR-100 dataset
# (num_classes: 100).  This indirection lets --dataset use the obvious name.
# --------------------------------------------------------------------------- #
DATASET_PRESETS = {
    'cifar10': {
        'loader_name': 'cifar10',
        'num_classes': 10,
    },
    'cifar100': {
        'loader_name': 'cifar50',
        'num_classes': 100,
    },
}


def _add_run_config(parser):
    """The three knobs run.sh exposes, plus method and where output goes."""
    group = parser.add_argument_group('run')

    group.add_argument('--dataset', type=str, default='cifar10',
                       choices=sorted(DATASET_PRESETS.keys()),
                       help='Which environment to run in.')
    group.add_argument('--budget', type=float, default=100,
                       help='Total training budget, in EPOCH-EQUIVALENTS: one unit '
                            'is a single backprop pass over the full training set. '
                            'This is the common currency that makes SESiL and the '
                            'baseline comparable -- see sesil/budget.py. For the '
                            'baseline it is simply the epoch count; for SESiL it is '
                            'spent on pretrain plus per-generation mutation, and the '
                            'run stops when it is exhausted.')
    group.add_argument('--seed', type=int, default=0,
                       help='Random seed for torch/numpy/random.')

    group.add_argument('--method', type=str, default='sesil',
                       choices=['sesil', 'baseline'],
                       help="'sesil' is the evolutionary pipeline; 'baseline' is "
                            'the learning-based classifier it is compared against.')
    group.add_argument('--exp-name', type=str, default='check',
                       help='Identifier for this experiment; names the output folder.')
    group.add_argument('--device', type=str, default=None,
                       help="'cuda', 'cpu', or unset to auto-detect.")
    group.add_argument('--output-root', type=str, default='./results',
                       help='Root for all run artefacts (checkpoints + csv).')
    group.add_argument('--data-dir', type=str, default='./data',
                       help='Root holding the raw dataset downloads.')


def _add_model_config(parser):
    group = parser.add_argument_group('model')
    group.add_argument('--model-name', type=str, default='resnet20',
                       help="Backbone family, e.g. 'resnet20' or 'vgg11'.")
    group.add_argument('--model-width', type=int, default=4,
                       help='Width multiplier; resnet20 + width 4 -> resnet20x4.')
    group.add_argument('--eval-type', type=str, default='logits',
                       choices=['logits', 'clip'],
                       help='Head type used for evaluation.')


def _add_pretrain_config(parser):
    group = parser.add_argument_group('pretrain')
    group.add_argument('--pop-size', type=int, default=10,
                       help='Number of individuals in the initial population.')
    group.add_argument('--classes-per-model', type=int, default=3,
                       help='How many classes each initial individual is trained on.')
    group.add_argument('--pretrain-epochs', type=int, default=20,
                       help='Epochs per individual during pretrain.')
    group.add_argument('--pretrain-batch-size', type=int, default=500)
    group.add_argument('--pretrain-workers', type=int, default=8)
    group.add_argument('--population-dir', type=str, default=None,
                       help='Where the initial population lives. Defaults to a path '
                            'derived from the dataset/model/population settings.')
    group.add_argument('--force-pretrain', action='store_true', default=False,
                       help='Re-run pretrain even if a population already exists.')
    group.add_argument('--pretrain-only', action='store_true', default=False,
                       help='Create the initial population and stop.')


def _add_evolution_config(parser):
    group = parser.add_argument_group('evolution')
    group.add_argument('--merger', type=str, default='permute',
                       choices=['zipit', 'permute', 'wavg'],
                       help='Which merge operator crossover uses.')
    group.add_argument('--selection', type=str, default='bidirectional',
                       choices=['bidirectional', 'breed', 'guided', 'hard'],
                       help='Which mate-selection strategy to use.')
    group.add_argument('--individual-budget', type=float, default=0.25,
                       help='Training budget granted to ONE agent in ONE generation, '
                            'in the same epoch-equivalent unit as --budget. Each agent '
                            'is finetuned on exactly this many sample-presentations '
                            'drawn from its own inherited labels, so every agent gets '
                            'the same compute regardless of how many classes it covers. '
                            'Values below 1.0 mean sub-epoch training, which is how you '
                            'buy resolution: halving it doubles the number of '
                            'generations for the same --budget.')
    group.add_argument('--no-mutation', action='store_true', default=False,
                       help='Skip the finetune step entirely.')
    group.add_argument('--start-gen', type=int, default=0,
                       help='Resume from this generation. 0 starts from the initial '
                            'population; N>0 reads generation N from the run folder.')
    group.add_argument('--max-generations', type=int, default=1000,
                       help='Safety cap. The real stopping condition is --budget; '
                            'this only prevents an unbounded loop if the per-generation '
                            'cost is tiny.')


def _add_merging_config(parser):
    group = parser.add_argument_group('merging')
    group.add_argument('--stop-node', type=int, default=21,
                       help='Partial-zipping depth; None merges the whole network.')
    group.add_argument('--merge-alpha', type=float, default=0.0001,
                       help="ZipIt! alpha ('a'), Section 4.3 of the ZipIt! paper.")
    group.add_argument('--merge-beta', type=float, default=0.075,
                       help="ZipIt! beta ('b'), Section 4.3 of the ZipIt! paper.")
    group.add_argument('--reduce-ratio', type=float, default=0.5,
                       help='Feature-space reduction ratio for the merge.')
    group.add_argument('--merging-metrics', type=str, default='covariance,mean',
                       help='Comma-separated alignment metrics to accumulate.')


def _add_certificate_config(parser):
    """Proficiency certificates -- what an agent is licensed to train on.

    Agents have no names. Every generation the population is evaluated on the
    whole label space and ranked per class; the rule below decides who is
    certified in what. See sesil/certificate.py.
    """
    group = parser.add_argument_group('certificate')
    group.add_argument('--certify-top-frac', type=float, default=0.3,
                       help='Fraction of the population that may hold a certificate '
                            'in any one class. Proficiency is relative: an agent must '
                            'be in this top slice of its peers on a class to be '
                            'licensed to train on it.')
    group.add_argument('--certify-floor', type=float, default=0.5,
                       help='Absolute accuracy a certificate also requires, whatever '
                            'the ranking says. Without it, the top slice of a '
                            'uniformly incompetent population still gets certified.')


def _add_selection_config(parser):
    group = parser.add_argument_group('selection')
    group.add_argument('--weight-extra', type=float, default=1.0,
                       help='Weight on certified classes the mate has and the chooser lacks.')
    group.add_argument('--weight-common', type=float, default=0.1,
                       help='Weight on certified classes both already hold.')
    group.add_argument('--max-retries', type=int, default=100,
                       help='Attempts to find reciprocated pairs before giving up.')
    group.add_argument('--breed-key', type=str, default='Joint',
                       help='Fitness field the breed/guided/hard strategies rank on.')


def _add_baseline_config(parser):
    group = parser.add_argument_group('baseline')
    group.add_argument('--baseline-mode', type=str, default='scratch',
                       choices=['scratch', 'finetune'],
                       help="'scratch' trains one classifier on --baseline-classes; "
                            "'finetune' extends a checkpoint onto --finetune-classes.")
    group.add_argument('--baseline-classes', type=str, default=None,
                       help='Comma-separated class ids, or unset for the full dataset.')
    group.add_argument('--finetune-classes', type=str, default=None,
                       help='Comma-separated class ids to add in finetune mode.')
    group.add_argument('--baseline-load-path', type=str, default=None,
                       help='Checkpoint to start finetune mode from.')
    group.add_argument('--baseline-batch-size', type=int, default=500)


def get_config():
    """Build the argument parser shared by every method."""
    parser = argparse.ArgumentParser(
        description='SESiL-II: social, evolutionary supported learning.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_run_config(parser)
    _add_model_config(parser)
    _add_pretrain_config(parser)
    _add_evolution_config(parser)
    _add_merging_config(parser)
    _add_certificate_config(parser)
    _add_selection_config(parser)
    _add_baseline_config(parser)
    return parser


# --------------------------------------------------------------------------- #
# Derived values
# --------------------------------------------------------------------------- #

def arch_name(args):
    """Canonical backbone string, e.g. 'resnet20x4' or 'vgg11_w4'."""
    if args.model_name.startswith('vgg'):
        return f'{args.model_name}_w{args.model_width}'
    return f'{args.model_name}x{args.model_width}'


def parse_int_list(value):
    """'0,1,2' -> [0, 1, 2]; None/''/'None' -> None."""
    if value is None or str(value).strip() in ('', 'None', 'none'):
        return None
    return [int(v) for v in str(value).split(',') if v.strip() != '']


def default_population_dir(args):
    """Where the initial population lives when --population-dir is not given.

    e.g. ./checkpoints/cifar10_C3_P10/resnet20x4/initial

    Keyed by dataset/classes/population so two different population shapes never
    collide, and so a population is reused across runs that share that shape.
    """
    tag = f'{args.dataset}_C{args.classes_per_model}_P{args.pop_size}'
    return os.path.join('./checkpoints', tag, arch_name(args), 'initial')


def run_dir(args):
    """Root for this run's artefacts, kept separate per method/config/seed."""
    if args.method == 'sesil':
        leaf = f'{args.merger}_{args.selection}'
    else:
        leaf = f'baseline_{args.baseline_mode}'
    return os.path.join(args.output_root, args.exp_name, args.dataset, leaf,
                        f'seed{args.seed}')


def build_raw_config(args):
    """Assemble the dict that utils.prepare_experiment_config() expects.

    This replaces the old per-experiment modules under configs/ -- the settings
    now come from the flags above instead of from a checked-in Python file.
    """
    preset = DATASET_PRESETS[args.dataset]
    return {
        'dataset': {
            'name': preset['loader_name'],
        },
        'model': {
            'name': arch_name(args),
            'dir': args.population_dir,
            'bases': [],
        },
        'merging_fn': 'match_tensors_zipit',   # replaced per-call by the registry
        'eval_type': args.eval_type,
        'merging_metrics': [m.strip() for m in args.merging_metrics.split(',') if m.strip()],
        'device': args.device,
    }


def resolve(args):
    """Fill in values that depend on other flags. Call once after parsing."""
    import torch

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.num_classes = DATASET_PRESETS[args.dataset]['num_classes']
    args.arch = arch_name(args)

    # One budget knob, in epoch-equivalents (see sesil/budget.py).
    #
    # For the baseline the conversion is the identity: one epoch over the full
    # training set IS one epoch-equivalent.
    #
    # For SESiL there is no fixed generation count -- each generation costs a
    # different amount depending on how many labels the population covers, so
    # the loop spends the budget and stops when it runs out.
    args.baseline_epochs = int(args.budget)

    if args.population_dir is None:
        args.population_dir = default_population_dir(args)
    args.run_dir = run_dir(args)

    return args
