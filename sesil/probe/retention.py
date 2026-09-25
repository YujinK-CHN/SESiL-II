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


def gain(parent_a, parent_b, child):
    """How much the child can do that the BETTER single parent could not.

    Retention has a blind spot: it is capped at what the parents already had,
    so it cannot tell a merge that combined two specialists from one that
    merely preserved a single parent. Two identical parents have nothing
    distinctive to lose, so nothing is lost, so retention scores them
    perfectly -- measured on this data, the one pair with identical
    certificates scored 0.939, the highest of any group, for a merge that
    gained nothing at all.

    This asks the complementary question. The reference is the element-wise
    max of the parents -- the best either could do on each class, which is what
    a merge would achieve if it kept everything from both. Scoring against that
    rather than against either parent alone is what makes mating a near-copy
    worthless: when the parents are the same, the reference is that same
    agent, and a child that reproduces it gains zero.

    Normalised by the headroom the pairing offered, so it measures how much of
    the AVAILABLE gain was captured, not how lucky the parents were. A pairing
    with no headroom -- identical parents -- has nothing to capture and returns
    0.0 rather than dividing by zero.
    """
    a, b, c = _as_vector(parent_a), _as_vector(parent_b), _as_vector(child)
    best_single = np.maximum(a, b)
    capped = np.minimum(c, best_single)     # credit is capped at what was available

    def captured(reference):
        """Share of the OTHER parent's exclusive skill this child picked up.

        Measured per class and only where the other parent was actually
        better, so it rewards breadth. Summing raw accuracy instead would be
        blind to how the child spreads its competence: 0.9 on three classes
        and 0.45 on six carry the same total, but only the second is a merge.
        """
        room = np.maximum(best_single - reference, 0.0)
        total = room.sum()
        if total <= 1e-12:
            return None                     # this parent already dominates
        return float(np.maximum(capped - reference, 0.0).sum() / total)

    from_a, from_b = captured(a), captured(b)
    if from_a is None or from_b is None:
        return 0.0                          # one parent dominates: nothing to gain

    # The weaker direction. A child that absorbed one parent whole and ignored
    # the other has not combined anything, so it must not score for the half
    # it did manage.
    return min(from_a, from_b)


def pair_retention(acc_a, acc_b, acc_child):
    """Every retention number for one (parents, child) triple.

    Two headline numbers, answering different questions:

    `balanced`  -- how much of each parent SURVIVED. Harmonic mean of the two
                   distinctive retentions, so a child that collapses onto one
                   parent scores near zero.
    `gain`      -- whether the merge was WORTH MAKING. Zero when the child is
                   no better than the better single parent, which is what
                   mating two near-identical agents produces.

    Neither subsumes the other. A merge can preserve both parents perfectly and
    gain nothing (identical parents), or gain a lot while losing one parent's
    rarer skills. Report both.
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
        'gain': gain(acc_a, acc_b, acc_child),
    }
