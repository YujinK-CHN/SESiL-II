"""
Proficiency certificates.

An agent has no name and no recorded task id. What it is allowed to train on is
derived, every generation, from how it actually performs: a certificate is the
set of classes an agent has demonstrated proficiency in, relative to the rest of
the population.

This replaces the old scheme where an agent carried a label set inherited from
its parents. A name is a *claim* maintained by bookkeeping; if merging destroyed
a class, the name still asserted the agent knew it, and the error compounded
down the lineage. A certificate is a *measurement*, refreshed from scratch each
generation, so it cannot drift.

Granting rule -- rank AND floor:

    agent is certified on class c  <=>  it is in the top `top_frac` of the
                                        population on c
                                   AND  its accuracy on c exceeds `floor`

The ranking half is SEMFO's competitive design: proficiency is relative, so
there is always pressure to be better than peers rather than merely adequate.
The floor half stops the population certifying itself on classes nobody can do
yet -- without it, in early generations the top 10% of a uniformly hopeless
population still gets certified.
"""

import math


def certify_population(per_class_accuracy, top_frac, floor, num_classes):
    """Grant certificates across a population.

    Args:
        per_class_accuracy: one accuracy vector per agent, each length num_classes
        top_frac: fraction of the population that may be certified on a class
        floor: absolute accuracy a certificate requires regardless of rank
        num_classes: size of the label space

    Returns:
        list of sets, one per agent, holding the class ids it is certified on
    """
    n_agents = len(per_class_accuracy)
    if n_agents == 0:
        return []

    # At least one agent can hold each class, however small the population.
    n_top = max(1, int(math.ceil(top_frac * n_agents)))

    certificates = [set() for _ in range(n_agents)]

    for c in range(num_classes):
        ranked = sorted(
            range(n_agents),
            key=lambda i: per_class_accuracy[i][c],
            reverse=True,
        )
        for agent in ranked[:n_top]:
            if per_class_accuracy[agent][c] > floor:
                certificates[agent].add(c)

    return certificates


def rank_strength(per_class_accuracy, num_classes):
    """Population percentile of each agent on each class.

    strength[i][c] = 1.0 for the best agent on class c, falling linearly to
    near 0 for the worst. This is the same quantity certification thresholds
    on, so `--mate-score rank` and the granting rule share one notion of
    proficiency.

    Scale-free by construction, which is the point: raw accuracies converge as
    the population improves, so weighting by them flattens out late in a run,
    while ranks stay just as informative at generation 100 as at generation 1.

    Note the range is compressed among certified agents -- they are by
    definition the top slice, so with top_frac 0.3 of 10 agents their strengths
    are 1.0, 0.9, 0.8. That is intended: it separates certificate holders
    without letting one dominate.
    """
    n_agents = len(per_class_accuracy)
    strengths = [[0.0] * num_classes for _ in range(n_agents)]
    if n_agents == 0:
        return strengths

    for c in range(num_classes):
        ranked = sorted(
            range(n_agents),
            key=lambda i: per_class_accuracy[i][c],
            reverse=True,
        )
        for position, agent in enumerate(ranked):
            strengths[agent][c] = (n_agents - position) / n_agents

    return strengths


def uniform_strength(num_agents, num_classes):
    """All classes weighted equally -- the pure-certificate scoring mode.

    Mate choice determines the offspring's inherited certificate, which is the
    union of the parents' certificate *sets*; accuracy never enters that union.
    Scoring the set directly makes the objective and the outcome the same
    object.
    """
    return [[1.0] * num_classes for _ in range(num_agents)]


def strength_matrix(mode, per_class_accuracy, num_classes):
    """Per-class weights used by mate selection, chosen by --mate-score."""
    if mode == 'count':
        return uniform_strength(len(per_class_accuracy), num_classes)
    if mode == 'accuracy':
        return [list(row) for row in per_class_accuracy]
    if mode == 'rank':
        return rank_strength(per_class_accuracy, num_classes)
    raise ValueError(
        f"Unknown mate-score mode {mode!r}; expected 'count', 'accuracy' or 'rank'."
    )


def training_classes(certificate, num_classes, is_loner=False, rng=None):
    """Which classes an agent may actually be finetuned on this generation.

    Certified classes, plus one exploration slot for agents that did not pair
    off. Failing to find a mate buys the right to try something new -- and an
    agent whose certificate is empty is worth nothing to any mate, so it lands
    in the loner pool by construction and is never left with nothing to train
    on.

    The empty-certificate case is handled explicitly as well as via `is_loner`
    so that an agent can never be left with nothing to train on, however mate
    selection happens to pair it.
    """
    classes = set(certificate)

    if is_loner or not classes:
        uncertified = [c for c in range(num_classes) if c not in classes]
        if uncertified:
            if rng is None:
                import random as _random
                pick = _random.choice(uncertified)
            else:
                pick = int(rng.choice(uncertified))
            classes.add(pick)

    return sorted(classes)


def inherit(parent_certificates):
    """Certificate an offspring starts life with: the union of its parents'.

    Inherited for exactly one generation -- the offspring is evaluated and
    re-certified from its own measured performance at the start of the next
    generation, so an inherited-but-wrong certificate misdirects one round of
    training and then self-corrects. That is the bounded-drift property a name
    never had.
    """
    union = set()
    for cert in parent_certificates:
        union |= set(cert)
    return union


def coverage(certificates, num_classes):
    """Summary statistics for logging the state of the society."""
    if not certificates:
        return {}
    sizes = [len(c) for c in certificates]
    covered = set()
    for c in certificates:
        covered |= set(c)
    return {
        'cert_size_mean': sum(sizes) / len(sizes),
        'cert_size_min': min(sizes),
        'cert_size_max': max(sizes),
        'classes_covered': len(covered),
        'uncertified_agents': sum(1 for s in sizes if s == 0),
    }
