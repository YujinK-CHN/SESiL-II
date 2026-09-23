"""
Training budget accounting.

SESiL and the learning-based baseline spend compute in structurally different
ways -- generations of population-wide finetuning versus epochs on one model --
so neither "generations" nor "epochs" is a fair unit to compare them on.

The unit used here is the **epoch-equivalent**:

    1 epoch-equivalent = one backpropagation pass over the full training set

Every model in this codebase shares one architecture, so cost per sample is
constant and epoch-equivalents are exactly proportional to training FLOPs.
That makes the arithmetic below both simple and defensible.

Counted against the budget:
    - pretrain            (subset training, charged by actual sample count)
    - mutation            (offspring finetuning, likewise)
    - baseline training

Counted but NOT charged, because they involve no backpropagation:
    - BatchNorm recalibration passes (reset_bn_stats)
    - alignment-metric passes inside the merge (compute_metrics)
    - evaluation passes over the test set

The forward-only work is real compute and SESiL does far more of it than the
baseline, so it is tracked separately and reported. Charging it would need a
forward:backward cost ratio assumption, which belongs in the write-up rather
than baked into the budget.
"""

from collections import defaultdict


# How many full forward passes over the TRAINING set each pipeline step costs.
# These are analytic, not measured: each of these helpers iterates the whole
# loader exactly once, so the count is exact.
#
#   merge: reset_bn_stats on each of the 2 parents,
#          compute_metrics for the alignment,
#          reset_bn_stats on the merged child
FORWARD_TRAIN_PASSES_PER_MERGE = 4
#   loner: reset_bn_stats only
FORWARD_TRAIN_PASSES_PER_LONER = 1
#   population evaluation: reset_bn_stats per individual
FORWARD_TRAIN_PASSES_PER_EVAL = 1


class BudgetTracker:
    """Ledger of training compute, in epoch-equivalents.

    Spend is recorded in raw samples and normalised by the full training set
    size, so a run on a class subset is charged for exactly the data it saw.
    """

    def __init__(self, total, train_set_size, test_set_size=0):
        self.total = float(total)
        self.train_set_size = int(train_set_size)
        self.test_set_size = int(test_set_size)

        self.spent = 0.0                      # epoch-equivalents of backprop
        self.forward_train = 0.0              # epoch-equivalents, forward only
        self.forward_test = 0.0               # test-set passes, forward only
        self.by_stage = defaultdict(float)    # stage -> epoch-equivalents

    # ------------------------------------------------------------------ #
    # Charged against the budget
    # ------------------------------------------------------------------ #

    def spend_samples(self, stage, n_samples, epochs=1):
        """Charge `epochs` passes over `n_samples` training examples."""
        cost = (n_samples * epochs) / self.train_set_size
        self.spent += cost
        self.by_stage[stage] += cost
        return cost

    def spend_epochs(self, stage, epochs):
        """Charge `epochs` passes over the full training set."""
        return self.spend_samples(stage, self.train_set_size, epochs)

    # ------------------------------------------------------------------ #
    # Counted but not charged
    # ------------------------------------------------------------------ #

    def count_forward_train(self, passes=1, n_samples=None):
        """Record forward-only passes over the training set."""
        n = self.train_set_size if n_samples is None else n_samples
        self.forward_train += (n * passes) / self.train_set_size

    def count_forward_test(self, passes=1):
        """Record forward-only passes over the test set."""
        if self.train_set_size:
            self.forward_test += (self.test_set_size * passes) / self.train_set_size

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #

    @property
    def remaining(self):
        return self.total - self.spent

    @property
    def exhausted(self):
        return self.spent >= self.total

    def can_afford(self, cost):
        """Whether `cost` more epoch-equivalents fit in what is left."""
        return self.spent + cost <= self.total

    def summary(self):
        """Serialisable accounting, for config.json and the run log."""
        return {
            'unit': 'epoch-equivalents (one backprop pass over the full training set)',
            'train_set_size': self.train_set_size,
            'budget_total': round(self.total, 3),
            'budget_spent': round(self.spent, 3),
            'budget_remaining': round(self.remaining, 3),
            'spent_by_stage': {k: round(v, 3) for k, v in sorted(self.by_stage.items())},
            'forward_only_train_passes': round(self.forward_train, 3),
            'forward_only_test_passes': round(self.forward_test, 3),
        }

    def report(self):
        """Human-readable accounting."""
        lines = [
            f'budget      : {self.spent:.2f} / {self.total:.2f} epoch-equivalents',
        ]
        for stage, cost in sorted(self.by_stage.items()):
            lines.append(f'  {stage:<10}: {cost:.2f}')
        lines.append(
            f'forward-only: {self.forward_train:.2f} train-set passes, '
            f'{self.forward_test:.2f} test-set-equivalents (not charged)'
        )
        return '\n'.join(lines)


def estimate_pretrain_cost(args):
    """Epoch-equivalents pretrain will consume, before it runs.

    Each individual trains on classes_per_model of num_classes, and CIFAR is
    class-balanced, so the data fraction is exact.
    """
    fraction = args.classes_per_model / args.num_classes
    return args.pop_size * args.pretrain_epochs * fraction


def estimate_generation_cost(args, mean_label_fraction=1.0):
    """Epoch-equivalents one generation of mutation will consume.

    Every individual is finetuned each generation -- offspring from couples and
    loners alike -- so the population size is the multiplier. With mutation
    restricted to inherited labels, `mean_label_fraction` is the average
    |label union| / num_classes across the population, which grows toward 1.0
    as coverage spreads.
    """
    return args.pop_size * args.mutate_epochs * mean_label_fraction
