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
                       choices=['sesil', 'baseline', 'probe'],
                       help="'sesil' is the evolutionary pipeline; 'baseline' is "
                            'the learning-based classifier it is compared against. '
                            "'probe' is not a learning method at all -- it is a "
                            'diagnostic setting that screens every possible pair '
                            'with GLOBA and measures what each merge really '
                            'produced. It shares this flag because it shares the '
                            'launcher and the output layout, but it optimises '
                            'nothing and plot_results.py does not read it.')
    group.add_argument('--exp-name', type=str, default='check',
                       help='Identifier for this experiment; names the output folder.')
    group.add_argument('--device', type=str, default=None,
                       help="'cpu', an explicit 'cuda:N', or unset to auto-detect and "
                            'claim a GPU via --gpus. An explicit value is always '
                            'respected as given.')
    group.add_argument('--gpus', type=str, default=None,
                       help="Comma-separated GPU indices this run may use, e.g. '0,1,3'. "
                            'Each process claims the best free one through lock files '
                            'in results/.gpu_locks, so several seeds launched together '
                            'spread across devices instead of stacking on cuda:0. Unset '
                            '(or the SESIL_GPUS environment variable) means every visible '
                            'GPU. Indices that do not exist on the current machine are '
                            'warned about and dropped, so the same command works on a '
                            'one-GPU laptop and an eight-GPU server. Ignored when '
                            '--device names a device explicitly.')
    group.add_argument('--strict-gpus', action='store_true',
                       help='Fail instead of falling back when --gpus names a GPU this '
                            'machine does not have. Use on a shared server where the '
                            'other cards belong to somebody else and being quietly '
                            'moved onto one would be worse than stopping.')
    group.add_argument('--max-per-gpu', type=int, default=1,
                       help='How many runs of this scheme may share one GPU. Launching '
                            'more than this is refused up front with an explanation, '
                            'rather than becoming an out-of-memory crash minutes into '
                            'a run. Raise it if a card genuinely fits several.')
    group.add_argument('--min-free-gb', type=float, default=1.0,
                       help='Warn when the claimed GPU has less free memory than this. '
                            'Catches a card another process is already filling, which '
                            'the lock files cannot see. Warning only -- the run '
                            'proceeds.')
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
    group.add_argument('--phase-b-freeze', type=float, default=0.0,
                       help='SSL MODE ONLY. Fraction of each agent\'s phase-B '
                            'budget spent with the backbone FROZEN, training only '
                            'the freshly random classifier, before unfreezing for '
                            'the rest. 0.0 (default) is the original scheme: '
                            'everything trains from the first step, and a random '
                            "head's large uninformative gradients flow straight "
                            'into a backbone that cost most of the pretrain '
                            'budget. That matters here beyond feature damage -- '
                            'the backbone is the shared basis that makes merging '
                            'work, so drift away from it is exactly what a later '
                            'merge has to reconcile (measured: 15.2% drift in '
                            '1.25 epochs). Higher values keep agents closer to '
                            'the shared basis and easier to merge, at the cost of '
                            'specialising less. The warmup is charged at the full '
                            'rate even though a head-only backward is cheaper, '
                            'which is conservative.')
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
                       help='BASELINE ONLY, with --baseline-init backbone: directory '
                            'holding a backbone that a SESiL run produced (its '
                            '<run dir>/backbone). Loading it is how the baseline starts '
                            'from identical weights, which is what separates "the '
                            'backbone helped" from "evolution helped". SESiL runs always '
                            'train their own; nothing is ever picked up implicitly.')
    group.add_argument('--pretrain-batch-size', type=int, default=500)
    group.add_argument('--pretrain-workers', type=int, default=2,
                       help='DataLoader workers per run. Low by default because '
                            'this is PER RUN: several seeds in parallel multiply '
                            'it, and 3 runs x 8 workers is 24 loader processes '
                            'competing for the same cores.')
    group.add_argument('--pretrain-only', action='store_true', default=False,
                       help='Create the initial population and stop, without evolving.')


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
    group.add_argument('--stop-node', type=optional_int, default=None,
                       help="Partial-zipping depth: the graph node the merge stops "
                            "at, beyond which each parent keeps its own head. The "
                            "default 'none' merges the WHOLE network. Pass an "
                            'integer (e.g. 21) to merge only up to that node, '
                            'which on resnet20x4 shares about a third of the '
                            'tensors and leaves each child mostly one parent. The '
                            'two modes produce sibling children differently -- see '
                            'sesil/merge.extract_children.')
    group.add_argument('--merge-bias', type=float, default=0.5,
                       help='FULL-MERGE ONLY (--stop-node none). How far each child '
                            'leans toward its own parent when the merged weights are '
                            'interpolated. A couple yields one child per parent, so '
                            'this is what makes them differ. '
                            'NOTE the default 0.5 is the symmetric case: both '
                            'children are then IDENTICAL, each an equal blend of '
                            'both parents. That is what you want when the question '
                            'is how well one model can carry two parents (the '
                            'probe), but it removes sibling diversity from an '
                            'evolution run -- raise it above 0.5 for --method '
                            'sesil. 1.0 makes each child its parent, with none of '
                            'the merge in it. Ignored when a stop node is set, '
                            "where the parents' separate heads supply the "
                            'asymmetry.')
    group.add_argument('--merge-head', type=str, default='average',
                       choices=['average', 'label'],
                       help="How crossover builds the child's classifier. "
                            "'average' (default, the original behaviour) mixes "
                            'both parents row by row. That is destructive here: '
                            'a parent never trained on class c has had that row '
                            'pushed DOWN by training, so it is an anti-detector, '
                            "and averaging it with the other parent's real "
                            'detector cancels part of the signal. Measured on the '
                            'probe data, such rows sit at cosine -0.072 -- opposed, '
                            "not merely unrelated. 'label' instead takes each "
                            'class row whole from whichever parent is certified '
                            'for that class, averaging only rows both or neither '
                            'hold. In the probe this doubled the average merged '
                            'child (0.28 -> 0.58), a larger effect than perfect '
                            'mate selection would buy. Left off by default so the '
                            'two can be compared on equal terms.')
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
    group.add_argument('--mating-mode', type=str, default='certificate',
                       choices=['certificate', 'random', 'globa'],
                       help="What mate choice is based on. 'certificate' scores a "
                            "mate by the classes it is certified on, tuned by "
                            "--cert-with / --weight-extra / --weight-common. "
                            "'random' ignores all of that and pairs uniformly at "
                            'random -- the control arm. Any claim that selection '
                            'helps is a claim about beating this, so it is an '
                            'explicit mode rather than something faked by zeroing '
                            "the weights. 'globa' scores a pair by decomposing "
                            'their task vectors against the phase-A backbone, tuned '
                            'by --globa-with; it needs --pretrain-mode ssl and it '
                            'picks the best-predicted partner rather than sampling.')
    group.add_argument('--globa-with', type=str, default='D_minus',
                       choices=['D_minus', 'D_plus', 'E'],
                       help="GLOBA MODE ONLY. Which cell type scores a pair. "
                            "'D_minus' is opposite-sign overlap -- the two parents "
                            'moved the same structure in opposite directions, which '
                            'is what specialising differently looks like. Over 980 '
                            'measured merges it is the only GLOBA statistic that '
                            'beat random partner choice (22/35 seeds, p=0.032). '
                            "'E' is GLOBA's structural hole, the type its own theory "
                            'rates highest; it is ~42% of the update energy but '
                            'varies by only ~18% of its own size between pairs, and '
                            'across four measured conditions it never beat random. '
                            'Kept so the theory can be tested in evolution, not '
                            'because the probe supported it. '
                            "'D_plus' is same-sign overlap -- the donor moved "
                            'structure the base already moved, the same way, so it '
                            'brings nothing new. It is scored INVERTED (less '
                            'redundancy is better) and is the second strongest '
                            'signal measured, rho -0.155 against merge outcome. It '
                            'only became a distinct option once scoring went '
                            'directional: under symmetric scoring, minimising D_plus '
                            'was arithmetically the same rule as maximising the rest. '
                            'No multi-type option exists: the six fractions sum to '
                            '1, so any sum over a subset is the mirror image of what '
                            'is left out -- summing all types but D_minus is exactly '
                            'minimising D_minus.')
    group.add_argument('--cert-with', type=str, default='count',
                       choices=['count', 'accuracy'],
                       help="CERTIFICATE MODE ONLY. "
                            "What a certified class is worth when scoring a mate. "
                            "'count': all certified classes weigh the same -- the pure "
                            "certificate, and the default, because mate choice decides "
                            "the offspring's inherited certificate, which is the union "
                            "of the parents' certificate SETS. "
                            "'accuracy': weigh by the mate's raw accuracy on the class; "
                            "prefers strong certificate holders but discriminates only "
                            "within an already-selected band and flattens as the "
                            "population saturates. "
                            "A 'rank' mode (population percentile per class) was "
                            "removed: certification already applies rank AND floor to "
                            "decide WHICH classes count, so rank only re-weighted "
                            "inside that set, and among certified holders the range "
                            "compresses to roughly 1.0/0.875/0.75. Measured against "
                            "'count' on real populations it correlated at rho 0.895 -- "
                            "a tie-breaker, not a distinct operator.")
    group.add_argument('--weight-extra', type=float, default=1.0,
                       help='CERTIFICATE MODE ONLY. Weight on certified classes the '
                            'mate has and the chooser lacks.')
    group.add_argument('--weight-common', type=float, default=0.1,
                       help='CERTIFICATE MODE ONLY. Weight on certified classes both '
                            'already hold. The ratio to --weight-extra is what makes '
                            'selection complementarity-seeking (extra > common), '
                            'neutral, or similarity-seeking (common > extra).')
    group.add_argument('--mating-rounds', type=int, default=100,
                       help='How many rounds of mate choice to run. Each round: '
                            'everyone still unpaired picks a partner from those '
                            'still unpaired, reciprocated picks become couples, and '
                            'both leave the pool. Whoever remains at the end is a '
                            'loner. This is really an exploration dial -- a loner '
                            'carries forward and earns one random UNCERTIFIED class, '
                            'which is the only way a class the population has lost '
                            'can return. 1 makes a loner of anyone whose first choice '
                            'did not reciprocate; the default 100 keeps re-matching '
                            'until almost nobody is left, so almost nobody explores. '
                            'Replaces --max-retries, which meant the same thing for '
                            'the sampled modes only.')


def _add_subset_config(parser):
    parser.add_argument('--subset-mode', type=str, default='random',
                        choices=['random', 'disjoint'],
                        help="How each agent's training classes are chosen at "
                             "pretrain. 'random' draws every subset "
                             'independently and leaves overlap to chance -- the '
                             "original scheme, and the default so SESiL's "
                             "behaviour is unchanged. 'disjoint' deals classes "
                             'round-robin so every class is used about equally '
                             'often and pairwise overlap is as low as the '
                             'arithmetic allows. Prefer it for --method probe: '
                             'agents finetuned from one backbone on largely the '
                             'same classes have task vectors that all point the '
                             'same way, which leaves a weight-space predictor '
                             'nothing to discriminate.')


def _add_probe_config(parser):
    """--method probe only. GLOBA's decomposition, used to screen candidate pairs.

    These control the analysis, never the merge: the probe's children are built
    by the configured --merger, exactly as SESiL would build them. GLOBA is only
    ever asked for a prediction here.
    """
    group = parser.add_argument_group('probe')
    group.add_argument('--probe-eta', type=float, default=0.80,
                       help='Energy kept when pruning cells inside the analysed '
                            'subspace. GLOBA reports 0.80 as its modal value. '
                            'Same for both parents, so the decomposition stays '
                            'symmetric under swapping them.')
    group.add_argument('--probe-svd-energy', type=float, default=0.90,
                       help='Symmetric energy truncation defining the analysed '
                            'subspace. 1.0 analyses everything, which is '
                            'degenerate for any layer with out <= in because the '
                            'output-side basis is then an arbitrary rotation.')
    group.add_argument('--probe-basis-energy', type=float, default=0.999,
                       help='Energy kept when re-orthogonalising the concatenated '
                            'singular vectors. 1.0 uses the numerical rank.')
    group.add_argument('--probe-merger', type=str, default='sesil',
                       choices=['sesil', 'globa', 'both'],
                       help="Which operator builds the children the probe scores. "
                            "'sesil' (default) uses --merger, i.e. the operator "
                            "SESiL would really use; GLOBA only predicts. 'globa' "
                            'builds them with GLOBA\'s own recombination instead. '
                            "'both' does each pair twice, so the two operators can "
                            'be compared on identical parents -- which is the only '
                            'way to tell a better merge from an easier pair. GLOBA '
                            'needs no forward passes, so it is the cheaper of the '
                            'two.')
    group.add_argument('--globa-preset', type=str, default='single-full',
                       choices=['average', 'sum', 'orthogonal-full', 'single-full'],
                       help='Type coefficients for GLOBA merging. '
                            "'average' is exactly plain weight averaging and is the "
                            'reference point: a preset that cannot beat it means the '
                            "cell typing bought nothing. 'single-full' keeps whole "
                            'any cell only one parent touched and averages only '
                            'contested ones. Ignored unless --probe-merger uses '
                            'GLOBA.')
    group.add_argument('--globa-head', type=str, default='label',
                       choices=['label', 'average'],
                       help="How GLOBA merging combines the classifier. 'label' "
                            'takes each class row from whichever parent was trained '
                            'on that class, averaging only rows both or neither '
                            'know. Strongly preferred: agents here get a fresh '
                            'random classifier each, so averaging two of them halves '
                            "a trained row into noise. 'average' is the ablation "
                            'that shows how much that costs.')
    group.add_argument('--probe-layer-weighting', type=str, default='energy',
                       choices=['energy', 'uniform'],
                       help="How ~20 per-layer decompositions collapse into one "
                            "number per pair. 'energy' weighs each layer by its "
                            "share of the pair's total update energy, so a large "
                            "late conv counts for more than a tiny early one; "
                            "'uniform' treats every layer alike. Per-layer values "
                            'are kept in the output either way, since the signal '
                            'may not live uniformly across depth.')


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
    _add_subset_config(parser)
    _add_probe_config(parser)
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


def run_dir(args):
    """Root for this run's artefacts, kept separate per method/config/seed."""
    # The classifier head is part of what produced the child, so it has to be
    # part of the path. Without it a --merge-head label run lands on top of the
    # averaged-head run of the same merger and silently destroys it -- the two
    # differ in exactly the variable under test.
    head = '_labelhead' if getattr(args, 'merge_head', 'average') == 'label' else ''

    if args.method == 'sesil':
        leaf = f'{args.merger}{head}'
    elif args.method == 'probe':
        # Named for what actually produced the children, not for --merger,
        # which is left at its default and never used when only GLOBA merges.
        # A folder called probe_permute holding no permute children is a trap.
        globa = f'globa-{args.globa_preset}-{args.globa_head}'
        if args.probe_merger == 'globa':
            # GLOBA builds its own head, so --merge-head does not apply.
            leaf = f'probe_{globa}'
        elif args.probe_merger == 'both':
            leaf = f'probe_{args.merger}{head}+{globa}'
        else:
            leaf = f'probe_{args.merger}{head}'
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

    args.run_dir = run_dir(args)
    # Generation 0 is just the first generation, written into this run's own
    # directory like every later one. Nothing is shared between runs, so
    # concurrent seeds cannot collide and no run inherits another's state.
    args.population_dir = os.path.join(args.run_dir, 'checkpoints', 'gen_0')

    return args
