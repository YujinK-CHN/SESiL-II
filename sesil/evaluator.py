"""
External evaluation -- the plotting surface.

Two problems this solves, both borrowed from how SEMCS handles the same thing.

**Aligned x-axis.** SESiL advances in generations (a few epoch-equivalents
each) and the baseline advances in epochs (one each). Sampling "every N
iterations" would put their curves on incomparable axes. Instead evaluation
fires on a *watermark* over spent budget: whenever cumulative spend crosses the
next multiple of --eval-interval, both methods record a point. The x-values
then coincide by construction, whatever the internal loop looks like.

**Uncontaminated y-axis.** Certification ranks agents on the VALIDATION split
and that ranking drives mating and what each agent may train on, so validation
sits inside the optimisation loop. The evaluator is the only thing that touches
the TEST split, so the numbers plotted are not the numbers optimised against.

Every reduction is recorded at every point; which one is the headline curve is
a plotting decision, not a logging one:

    best_agent_overall   strongest single agent, all classes. The conservative
                         claim: one deployable model against the baseline's one.
    best_agent_balanced  strongest single agent by mean per-class accuracy --
                         differs from the above when classes are imbalanced.
    population_mean      average agent. The honest measure of whether the whole
                         society improved, rather than one lucky member.
    oracle_overall       per class, the population's best agent, averaged. The
                         society's collective capability -- but an ORACLE: no
                         single model achieves it, and realising it would need a
                         router that does not exist here.
    population_best/worst/median, and the full per-class vectors.

For the baseline there is one "agent", so best == mean == oracle. That is
intended: one schema, no special-casing at plot time.
"""

import numpy as np

from sesil.fitness import evaluate_all_classes


class Evaluator:
    """Watermark-driven evaluation against the held-out test split."""

    def __init__(self, test_loader, num_classes, eval_interval, logger,
                 budget, meta=None):
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.eval_interval = float(eval_interval)
        self.logger = logger
        self.budget = budget
        self.meta = dict(meta or {})

        # First watermark is 0.0, so every run records its starting point
        # before any budget is spent -- the curves all begin at the same place.
        self.next_watermark = 0.0
        self.n_points = 0

    # ------------------------------------------------------------------ #

    def due(self):
        """Whether spend has crossed the next watermark."""
        return self.budget.spent >= self.next_watermark

    def maybe_record(self, models, step=None, step_kind=None, **extra):
        """Evaluate and log if a watermark is due; otherwise do nothing.

        `models` is an iterable of plain nn.Modules -- the population for SESiL,
        a one-element list for the baseline.
        """
        if not self.due():
            return None
        record = self.record(models, step=step, step_kind=step_kind, **extra)
        # Advance past every watermark the spend has already crossed, so a
        # single expensive step cannot leave the schedule permanently behind.
        while self.budget.spent >= self.next_watermark:
            self.next_watermark += self.eval_interval
        return record

    def record(self, models, step=None, step_kind=None, final=False, **extra):
        """Evaluate unconditionally and log one aligned point.

        The schema is deliberately identical for every method. SESiL counts in
        generations and the baseline in epochs, so rather than emitting a
        different key for each, both report `step` alongside `step_kind` -- one
        frame, no per-method columns, nothing to reconcile at plot time.
        """
        models = list(models)

        per_class = []
        overall = []
        for model in models:
            pc, ov = evaluate_all_classes(model, self.test_loader, self.num_classes)
            per_class.append(pc)
            overall.append(ov)
            self.budget.count_forward_test(1)

        record = self._reduce(per_class, overall)
        record.update(self.meta)
        record['budget'] = round(self.budget.spent, 4)
        record['step'] = step
        record['step_kind'] = step_kind
        record['final'] = bool(final)
        record['forward_train_passes'] = round(self.budget.forward_train, 3)
        record['n_agents'] = len(models)
        record.update(extra)

        self.logger.log_eval(record)
        self.n_points += 1
        return record

    # ------------------------------------------------------------------ #

    def _reduce(self, per_class, overall):
        """Collapse a population's accuracy matrix into the logged reductions."""
        matrix = np.asarray(per_class, dtype=float)       # [agents, classes]
        overall = np.asarray(overall, dtype=float)        # [agents]

        best_idx = int(np.argmax(overall))
        balanced = matrix.mean(axis=1)                    # per-agent mean over classes
        balanced_idx = int(np.argmax(balanced))
        oracle_per_class = matrix.max(axis=0)             # best agent per class

        return {
            # headline candidates
            'best_agent_overall': float(overall[best_idx]),
            'best_agent_balanced': float(balanced[balanced_idx]),
            'population_mean': float(overall.mean()),
            'oracle_overall': float(oracle_per_class.mean()),
            # spread
            'population_best': float(overall.max()),
            'population_worst': float(overall.min()),
            'population_median': float(np.median(overall)),
            'population_std': float(overall.std()),
            # which agent won, for tracing a lineage back through train.jsonl
            'best_agent_index': best_idx,
            # full vectors, for per-class figures
            'per_class_oracle': [round(v, 5) for v in oracle_per_class.tolist()],
            'per_class_mean': [round(v, 5) for v in matrix.mean(axis=0).tolist()],
            'per_class_best_agent': [round(v, 5) for v in matrix[best_idx].tolist()],
        }
