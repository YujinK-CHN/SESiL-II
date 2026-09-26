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


def training_classes(certificate, num_classes, is_loner=False, rng=None,
                     mode='random', per_class_accuracy=None):
    """Which classes an agent may actually be finetuned on this generation.

    An OFFSPRING trains on its certificate and nothing else. Its certificate is
    the union of its parents', so it keeps studying everything either parent
    was good at -- that is what stops the merge's knowledge from decaying.

    A LONER, and any agent whose certificate is empty, also gets ONE
    uncertified class. That slot is the only way knowledge from outside the
    population's current coverage gets in: mutation trains solely on certified
    classes, so a class that falls below the floor everywhere can never be
    relearned through the normal path.

    --mutation-mode chooses which uncertified class:

        random  uniformly at random. The original scheme, and unbiased.
        best    the one it already scores highest on. Exploitative: the class
                it is closest to earning a certificate in, so the slot is most
                likely to convert into coverage next generation.
        worse   the one it scores lowest on. Exploratory: the class the
                population is furthest from holding, so it targets exactly
                what has been lost -- at the cost of being the hardest to
                learn from one generation of training.

    'best' and 'worse' need measured accuracy; without it they fall back to
    random rather than failing, since an agent with no measurement yet is a
    real case at generation 0.
    """
    classes = set(certificate)

    if not (is_loner or not classes):
        return sorted(classes)

    uncertified = [c for c in range(num_classes) if c not in classes]
    if not uncertified:
        return sorted(classes)

    usable = (mode in ('best', 'worse')
              and per_class_accuracy is not None
              and len(per_class_accuracy) >= num_classes)

    if usable:
        chooser = max if mode == 'best' else min
        # Ties broken by class id so a run is repeatable.
        pick = chooser(sorted(uncertified), key=lambda c: per_class_accuracy[c])
    elif rng is None:
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
