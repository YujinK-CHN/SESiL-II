"""
Single configuration file for every method implemented in SESiL-II.

Only three things are meant to change from the command line, because only three
things change between runs of an experiment suite:

    --dataset   which environment to run in   (cifar10 / cifar100)
    --budget    total training budget, in epoch-equivalents (see sesil/budget.py)
    --seed      random seed

    bash run.sh --dataset cifar100 --budget 500 --seeds 0,1,2

Everything else -- merge operator, certification rule, population shape, all the
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


def optional_int(value):
    """An int, or None for 'none' / 'null' / '' / a negative number.

    argparse's type=int cannot express "no stop node", so --stop-node could
    never actually select a full merge from the command line even though the
    code supports it.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ('none', 'null', ''):
        return None
    parsed = int(text)
    return None if parsed < 0 else parsed


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
                       help="'cpu', an explicit 'cuda:N', or unset to auto-detect and "
                            'claim a GPU via --gpus. An explicit value is always '
                            'respected as given.')
    group.add_argument('--gpus', type=str, default=None,
                       help="Comma-separated GPU indices this run may use, e.g. '0,1,3'. "
                            'Each process claims the least-loaded one through lock files '
                            'in results/.gpu_locks, so several seeds launched together '
                            'spread across devices instead of stacking on cuda:0. Unset '
                            '(or the SESIL_GPUS environment variable) means every visible '
                            'GPU. Ignored when --device names a device explicitly.')
    group.add_argument('--output-root', type=str, default='./results',
                       help='Root for all run artefacts (checkpoints + csv).')
    group.add_argument('--data-dir', type=str, default='./data',
                       help='Root holding the raw dataset downloads.')
    group.add_argument('--eval-interval', type=float, default=5.0,
                       help='Epoch-equivalents between external evaluation points. '
                            'Evaluation fires on a WATERMARK over spent budget, not '
                            'every N iterations, so SESiL (which advances in '
                            'generations) and the baseline (which advances in epochs) '
                            'land on the same x-values and can be plotted on one axis. '
                            'See sesil/evaluator.py.')
    group.add_argument('--val-fraction', type=float, default=0.1,
                       help='Fraction of the TRAINING set held out for validation. '
                            'Certification ranks agents on this split, which is what '
                            'drives mating and licensing -- so it sits inside the '
                            'optimisation loop. The test set is touched only by the '
                            'external evaluator, keeping reported accuracy honest. '
                            'Set to 0 to disable, which reintroduces the bias.')


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
    """Pretrain runs in two phases, as in SEMCS.

        phase A   train ONE backbone on all classes, typically self-supervised
        phase B   copy it to every agent, then finetune each on its own subset

    --phase-a-ratio splits the pretrain budget between them. The point of the
    shared backbone is not only better features: agents finetuned from a common
    initialisation stay in the same loss basin, so alignment is close to
    identity and merging destroys far less than it does between independently
    initialised models.
    """
    group = parser.add_argument_group('pretrain')
    group.add_argument('--pop-size', type=int, default=10,
                       help='Number of individuals in the initial population.')
    group.add_argument('--classes-per-model', type=int, default=3,
                       help='How many classes each initial individual is trained on.')
    group.add_argument('--pretrain-mode', type=str, default='none',
                       choices=['none', 'ssl'],
                       help="How the initial population is built. 'none' is the "
                            "original SESiL behaviour -- every agent trained from "
                            "scratch on its own class subset -- kept as the default so "
                            "it stays available as a baseline / ablation. 'ssl' runs "
                            "the two-phase scheme: one shared backbone trained on all "
                            "classes (phase A), then per-agent finetuning (phase B).")
    group.add_argument('--pretrain-budget', type=float, default=10.0,
                       help='Budget for the whole pretrain stage, in epoch-equivalents, '
                            'carved out of --budget. Split between phase A and phase B '
                            'by --phase-a-ratio.')
    group.add_argument('--phase-a-ratio', type=float, default=0.7,
                       help='SSL MODE ONLY. Fraction of --pretrain-budget spent on '
                            'phase A (the shared backbone); phase B gets the rest, '
                            'divided equally among agents.')
    group.add_argument('--phase-a-method', type=str, default='rotation',
                       help="SSL MODE ONLY. Objective for the backbone, from the registry "
                            "in sesil/ssl.py. 'rotation' is a self-supervised pretext task "
                            "(predict the rotation applied to an image) and is handed "
                            "images without labels, so it cannot read one. 'supervised' "
                            "trains on all classes WITH labels -- not self-supervised, and "
                            "included as the control that separates 'a shared backbone "
                            "helps' from 'self-supervision helps'.")
    group.add_argument('--phase-a-cost-multiplier', type=float, default=None,
                       help='Epoch-equivalents charged per image in phase A. Leave unset '
                            'and it defaults to the forward passes per image the objective '
                            'itself performs (rotation 4, supervised 1), declared beside '
                            'the method '
                            'in sesil/ssl.py -- so switching methods re-prices phase A '
                            'automatically instead of silently charging the wrong rate. '
                            'Set it explicitly only to override that.')
    group.add_argument('--cluster-k', type=int, default=None,
                       help="SSL MODE, --phase-a-method cluster ONLY. Number of k-means "
                            "clusters used as pseudo-labels. Defaults to the dataset's "
                            "class count. Over-clustering (a multiple of it) is what "
                            "DeepCluster does in practice, on the grounds that several "
                            "tight clusters beat one loose one.")
    group.add_argument('--cluster-rounds', type=int, default=5,
                       help='SSL MODE, --phase-a-method cluster ONLY. How many times to '
                            're-cluster during phase A. The budget is split evenly among '
                            'rounds. More rounds means fresher pseudo-labels but fewer '
                            'gradient steps each.')
    group.add_argument('--backbone-path', type=str, default=None,
                       help='Where the phase-A backbone is cached. Defaults to a path '
                            'beside the population. Reused across runs and charged at its '
                            'recorded creation cost, like the population itself.')
    group.add_argument('--pretrain-batch-size', type=int, default=500)
    group.add_argument('--pretrain-workers', type=int, default=2,
                       help='DataLoader workers per run. Low by default because '
                            'this is PER RUN: several seeds in parallel multiply '
                            'it, and 3 runs x 8 workers is 24 loader processes '
                            'competing for the same cores.')
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
    group.add_argument('--stop-node', type=optional_int, default=21,
                       help="Partial-zipping depth: the graph node the merge stops "
                            "at, beyond which each parent keeps its own head. Pass "
                            "'none' (or a negative value) to merge the whole network "
                            "instead. The two modes produce sibling children "
                            "differently -- see sesil/merge.extract_children.")
    group.add_argument('--merge-bias', type=float, default=0.7,
                       help='FULL-MERGE ONLY (--stop-node none). How far each child '
                            'leans toward its own parent when the merged weights are '
                            'interpolated. A couple yields one child per parent, so '
                            'this is what makes them differ. 0.5 makes both children '
                            'identical; 1.0 makes each child its parent, with none of '
                            'the merge in it. Ignored when a stop node is set, where '
                            'the parents\' separate heads supply the asymmetry.')
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
    group.add_argument('--mate-score', type=str, default='count',
                       choices=['count', 'accuracy', 'rank'],
                       help="What a certified class is worth when scoring a mate. "
                            "'count': all certified classes weigh the same -- the pure "
                            "certificate, and the default, because mate choice decides "
                            "the offspring's inherited certificate, which is the union "
                            "of the parents' certificate SETS. "
                            "'accuracy': weigh by the mate's raw accuracy on the class; "
                            "prefers strong certificate holders but discriminates only "
                            "within an already-selected band and flattens as the "
                            "population saturates. "
                            "'rank': weigh by the mate's population percentile on the "
                            "class; scale-free, so it keeps separating agents late in a "
                            "run. Switching between these is the ablation for whether "
                            "strength-weighting helps at all.")
    group.add_argument('--weight-extra', type=float, default=1.0,
                       help='Weight on certified classes the mate has and the chooser lacks.')
    group.add_argument('--weight-common', type=float, default=0.1,
                       help='Weight on certified classes both already hold.')
    group.add_argument('--max-retries', type=int, default=100,
                       help='Attempts to find reciprocated pairs before giving up.')


def _add_baseline_config(parser):
    group = parser.add_argument_group('baseline')
    group.add_argument('--baseline-init', type=str, default='scratch',
                       choices=['scratch', 'backbone'],
                       help="Where the baseline starts. 'scratch' is random init; "
                            "'backbone' loads the same phase-A backbone SESiL uses and is "
                            "charged for it identically. Running both separates 'the "
                            "backbone helps' from 'evolution helps' -- with only the "
                            "scratch variant you cannot tell which produced a gain.")
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

    e.g. ./checkpoints/cifar10_C3_P10/resnet20x4/seed0/initial

    Keyed by dataset / classes-per-model / population size AND SEED.

    The seed is what makes concurrent runs safe. Without it, every seed of one
    configuration writes its population into the same directory, so seeds
    launched in parallel interleave their writes and corrupt each other -- and
    even run sequentially, later seeds would silently inherit the first seed's
    population, which also means they inherit a population built against a
    different train/validation split.

    It also makes seeds genuinely independent: pretrain randomness varies with
    the seed like everything else, so error bands across seeds cover the whole
    pipeline rather than the evolution stage alone.

    To share one population across seeds deliberately -- an ablation that holds
    pretrain fixed and varies only evolution -- pass --population-dir
    explicitly. Build it once first (`--pretrain-only`) rather than racing
    several runs at it.
    """
    tag = f'{args.dataset}_C{args.classes_per_model}_P{args.pop_size}'
    return os.path.join('./checkpoints', tag, arch_name(args),
                        f'seed{args.seed}', 'initial')


def run_dir(args):
    """Root for this run's artefacts, kept separate per method/config/seed."""
    if args.method == 'sesil':
        leaf = args.merger
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
    """Fill in values that depend on other flags. Call once after parsing.

    Device resolution claims a concrete GPU rather than leaving it as plain
    'cuda'. Several seeds launched in parallel would otherwise all resolve to
    cuda:0 and leave the rest of the machine idle. See sesil/gpu.py.
    """
    from sesil.gpu import resolve_device

    args.device = resolve_device(args)

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
