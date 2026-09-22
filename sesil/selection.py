"""
Mate selection.

All four strategies consume the same population_info -- a list of dicts produced
by sesil.fitness.evaluate_population, each carrying 'Model Name' and the
per-class accuracy vector 'Per Class' -- and return (pairs, loners).

The scoring itself is unchanged from the original SESiL scripts; only the
plumbing around it is new.
"""

import random

import numpy as np


# --------------------------------------------------------------------------- #
# Mating score
# --------------------------------------------------------------------------- #

def known_classes(fitness, threshold=0.5):
    """Indices of the classes this individual is considered to know."""
    return {i for i, acc in enumerate(fitness) if acc > threshold}


def mating_score(fitness_a, fitness_b, mode='threshold', threshold=0.5,
                 weight_extra=1.0, weight_common=0.1):
    """How much A values B as a mate.

    Skills B has that A lacks are worth weight_extra; skills they share are
    worth weight_common.  The score is directional: score(A, B) != score(B, A).
    """
    if mode == 'threshold':
        known_a = known_classes(fitness_a, threshold)
        known_b = known_classes(fitness_b, threshold)

        extra_skills = known_b - known_a
        common_skills = known_a & known_b

        score = weight_extra * sum(fitness_b[i] for i in extra_skills)
        score += weight_common * sum(fitness_b[i] for i in common_skills)

    elif mode == 'soft':
        # No hard cutoff, just weight by differences in accuracy.
        score = 0.0
        for acc_a, acc_b in zip(fitness_a, fitness_b):
            if acc_b > 0:
                if acc_a < threshold:      # A is weak here
                    score += weight_extra * acc_b
                else:                      # A is decent here too
                    score += weight_common * acc_b
    else:
        raise ValueError("mode must be 'threshold' or 'soft'")

    return score


def build_score_matrix(population_info, **kwargs):
    """Directional mating scores: scores[a][b] = how much a wants b."""
    scores = {}
    for model_a in population_info:
        fa = model_a['Per Class']
        scores[model_a['Model Name']] = {}
        for model_b in population_info:
            if model_a is model_b:
                continue
            fb = model_b['Per Class']
            scores[model_a['Model Name']][model_b['Model Name']] = mating_score(fa, fb, **kwargs)
    return scores


def _score_kwargs(args):
    return dict(
        mode=args.score_mode,
        threshold=args.tau,
        weight_extra=args.weight_extra,
        weight_common=args.weight_common,
    )


def probabilistic_choice(score_dict):
    """Pick a mate from score_dict with probability proportional to score."""
    if not score_dict:
        return None
    models = list(score_dict.keys())
    weights = np.array(list(score_dict.values()), dtype=float)
    if weights.sum() == 0:
        return random.choice(models)       # fallback: nobody is attractive
    probs = weights / weights.sum()
    return np.random.choice(models, p=probs)


# --------------------------------------------------------------------------- #
# Strategies.  Signature: fn(population_info, args) -> (pairs, loners)
# --------------------------------------------------------------------------- #

def select_bidirectional(population_info, args):
    """The SESiL default: mutual, probabilistic mate choice.

    Each unpaired individual probabilistically picks a mate; only reciprocated
    picks become couples.  Every couple is recorded twice, so it produces two
    offspring and the population size is preserved.  Individuals nobody
    reciprocated become loners and carry forward unchanged.
    """
    scores = build_score_matrix(population_info, **_score_kwargs(args))

    models = list(scores.keys())
    N = len(models)

    pairs = []
    paired = set()

    for _ in range(args.max_retries):
        choices = {m: probabilistic_choice(scores[m]) for m in models if m not in paired}

        for a, b in choices.items():
            if b is not None and choices.get(b) == a:
                if a not in paired and b not in paired:
                    pairs.append(tuple(sorted((a, b))))
                    pairs.append(tuple(sorted((a, b))))   # two offspring per couple
                    paired.update([a, b])

    loners = [m for m in models if m not in paired]

    assert 2 * (len(pairs) // 2) + len(loners) == N, \
        f'Population size mismatch: {2 * (len(pairs) // 2) + len(loners)} != {N}'

    return pairs, loners


def select_breed(population_info, args):
    """Parent A uniformly at random, parent B by mating score. No loners."""
    scores = build_score_matrix(population_info, **_score_kwargs(args))
    names = list(scores.keys())

    pairs = []
    for _ in range(len(names)):
        a = np.random.choice(names)
        b = _weighted_mate(scores[a], a)
        pairs.append((a, b))

    return pairs, []


def select_guided(population_info, args):
    """Parent A by fitness, parent B by mating score given A.

    Fitness-proportional on --breed-key, so strong individuals reproduce more,
    while who they pair with is still driven by complementary skills.
    """
    scores = build_score_matrix(population_info, **_score_kwargs(args))

    names = [p['Model Name'] for p in population_info]
    perf = np.array([float(p[args.breed_key]) for p in population_info], dtype=float)
    probs = _normalise(perf)

    pairs = []
    for _ in range(len(names)):
        a = np.random.choice(names, p=probs)
        b = _weighted_mate(scores[a], a)
        pairs.append((a, b))

    return pairs, []


def select_hard(population_info, args):
    """Both parents by fitness alone -- skill complementarity is ignored.

    The ablation that isolates how much of SESiL's benefit comes from the
    mating score rather than from merging strong models.
    """

    names = [p['Model Name'] for p in population_info]
    perf = np.array([float(p[args.breed_key]) for p in population_info], dtype=float)
    probs = _normalise(perf)

    pairs = []
    for _ in range(len(names)):
        a = np.random.choice(names, p=probs)
        mates = [m for m in names if m != a]
        if not mates:
            continue
        mate_probs = _normalise(
            np.array([perf[names.index(m)] for m in mates], dtype=float)
        )
        b = np.random.choice(mates, p=mate_probs)
        pairs.append((a, b))

    return pairs, []


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _normalise(values):
    """Clamp negatives and turn into a probability vector."""
    values = np.maximum(values, 0)
    if values.sum() == 0:
        return np.ones_like(values) / len(values)
    return values / values.sum()


def _weighted_mate(score_row, exclude):
    """Pick a mate from score_row, score-proportional, never `exclude`."""
    mates = [m for m in score_row.keys() if m != exclude]
    mate_scores = np.array([score_row[m] for m in mates], dtype=float)
    return np.random.choice(mates, p=_normalise(mate_scores))
