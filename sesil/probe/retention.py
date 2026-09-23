"""How much of each parent's proficiency survives in a child.

Deliberately certificate-free. A certificate is a *licence to train*: a rank
within the population plus an absolute floor, which is the right object for
deciding what an agent may study and the wrong one for measuring an outcome.
It would make the same pair score differently depending on who else happens to
be in the population, discard magnitude, and drag --certify-floor into a
measurement. Proficiency is already directly observable as the per-class
accuracy vector, so that is what is used here.

The measure is a histogram intersection:

    R(parent, child) = sum_c min(a_parent[c], a_child[c]) / sum_c a_parent[c]

which is what "weight every class by how good the parent was, then cap the
child's credit at the parent's level" collapses to. Bounded [0, 1], free of
thresholds and hyperparameters, 1.0 exactly when the child matches or beats the
parent on every class. Classes the parent was near chance on contribute almost
nothing -- so the measure is self-regularising, with no division by small
numbers anywhere.

Two parents give two numbers, combined by harmonic mean. That choice is load
bearing: a child that keeps one parent whole and forgets the other entirely
scores 0.5 under an arithmetic mean -- indistinguishable from a genuinely
balanced merge -- and ~0 here. Collapse onto one parent is the characteristic
way crossover fails, so the aggregate has to be able to see it.
"""

import numpy as np


def _as_vector(a):
    return np.asarray(a, dtype=np.float64).ravel()


def retention(parent_acc, child_acc):
    """Fraction of `parent_acc`'s per-class proficiency present in `child_acc`.

    A parent that knows nothing (all-zero accuracy) has nothing to retain; that
    is reported as 1.0 rather than 0/0, since no proficiency was lost.
    """
    p, c = _as_vector(parent_acc), _as_vector(child_acc)
    if p.shape != c.shape:
        raise ValueError(f'shape mismatch: parent {p.shape} vs child {c.shape}')

    total = p.sum()
    if total <= 0:
        return 1.0
    return float(np.minimum(p, c).sum() / total)


def distinctive_retention(parent_acc, partner_acc, child_acc):
    """Retention restricted to what this parent knows and its partner does not.

    The plain measure counts classes both parents hold, so pairs that overlap
    heavily score well for a trivial reason -- the child could have got those
    classes from either side. Crossover is really being asked to preserve the
    *distinctive* part, which is where a merge actually fails.

    The weight a_parent[c] * (1 - a_partner[c]) is the threshold-free analogue
    of "certified by this parent, not by the other": full weight where this
    parent is strong and the partner is helpless, vanishing where the partner
    could have supplied the class anyway.
    """
    p, q, c = _as_vector(parent_acc), _as_vector(partner_acc), _as_vector(child_acc)
    if not (p.shape == q.shape == c.shape):
        raise ValueError('parent, partner and child vectors must be the same length')

    weight = p * (1.0 - q)
    total = weight.sum()
    if total <= 0:
        # The partner dominates this parent everywhere: nothing is distinctive,
        # so nothing distinctive can be lost.
        return 1.0

    # min(1, child/parent) with the 0/0 cell defined as fully retained; those
    # cells carry ~zero weight regardless.
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(p > 0, np.minimum(1.0, c / np.where(p > 0, p, 1.0)), 1.0)
    return float((weight * ratio).sum() / total)


def harmonic(r_a, r_b):
    """Aggregate of the two per-parent retentions.

    Zero when either side is zero, which is the entire reason it is not a mean.
    """
    if r_a <= 0 or r_b <= 0:
        return 0.0
    return float(2.0 * r_a * r_b / (r_a + r_b))


def pair_retention(acc_a, acc_b, acc_child):
    """Every retention number for one (parents, child) triple.

    `balanced` is the headline: the harmonic mean of the two distinctive
    retentions. Distinctive rather than total because total is inflated by
    whatever the parents already had in common, and harmonic because a child
    that collapses onto one parent has to score near zero.
    """
    r_a = retention(acc_a, acc_child)
    r_b = retention(acc_b, acc_child)
    d_a = distinctive_retention(acc_a, acc_b, acc_child)
    d_b = distinctive_retention(acc_b, acc_a, acc_child)

    return {
        'retention_a': r_a,
        'retention_b': r_b,
        'retention_total': harmonic(r_a, r_b),
        'distinctive_a': d_a,
        'distinctive_b': d_b,
        'balanced': harmonic(d_a, d_b),
    }
