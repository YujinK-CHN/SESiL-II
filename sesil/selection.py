"""
Mate selection.

SESiL pairs agents by mutual, probabilistic choice over complementary skills.
The strategy consumes population_info -- a list of dicts, one per agent,
carrying 'Model Name', the per-class accuracy vector 'Per Class', the
'Certificate' granted this generation, and the 'Strength' weights resolved from
--mate-score -- and returns (pairs, loners).

What counts as a skill is the **certificate**, not a re-thresholding of the
accuracy vector. Certification already decided who is proficient in what, by
population-wide ranking plus an absolute floor (see sesil/certificate.py);
applying a second, different threshold here would let an agent be courted for a
class it is not licensed to train on.

An agent with an empty certificate is worth zero to every possible mate, since
mating_score sums the *mate's* skills. It therefore never gets reciprocated and
lands in the loner pool by construction -- where it earns an exploration slot.
"""

import random

import numpy as np


# --------------------------------------------------------------------------- #
# Mating score
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Per-class weights (--mate-score)
# --------------------------------------------------------------------------- #
# These decide what a certified class is WORTH to a chooser. That is a
# selection concern, not a certification one: certification decides WHICH
# classes an agent holds, and nothing here can change that set.

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
    raise ValueError(
        f"Unknown mate-score mode {mode!r}; expected 'count' or 'accuracy'."
    )


def mating_score(cert_a, cert_b, strength_b,
                 weight_extra=1.0, weight_common=0.1):
    """How much A values B as a mate.

    Classes B is certified in that A is not are worth weight_extra; classes both
    hold are worth weight_common.

    `strength_b` is B's per-class weight vector, supplied by --mate-score:

        count     all ones -- the pure certificate. Mate choice decides the
                  offspring's inherited certificate, which is the union of the
                  parents' certificate SETS, so scoring the set directly makes
                  the objective and the outcome the same object.
        accuracy  B's raw per-class accuracy. Prefers strong certificate
                  holders, but discriminates only inside an already-selected
                  band and flattens as the population saturates.

    Directional by construction: score(A, B) != score(B, A).
    """
    extra_skills = cert_b - cert_a
    common_skills = cert_a & cert_b

    score = weight_extra * sum(strength_b[i] for i in extra_skills)
    score += weight_common * sum(strength_b[i] for i in common_skills)
    return score


def globa_score_matrix(agent_ids, states, core, args):
    """Directional pair scores from the GLOBA decomposition.

    scores[a][b] is what B brings A: B's cells typed against A's occupancy,
    as a share of everything B brings. GLOBA works this way too -- model 1 is
    the base, model 2 the donor, and the six types describe the donor relative
    to the base.

    Asymmetric on purpose. SESiL pairs by mutual acceptance, so a score has to
    answer "what does this partner bring ME". A symmetric score makes every
    agent agree on who is best, which turns mutual choice into one global
    ranking: the top pair always matches, then the next, and nobody is ever
    left unpaired. Measured, that gives exactly 0 loners on an even population
    and exactly 1 on an odd one -- and loners are the only route by which a
    lost class re-enters the run.

    A task vector is `agent - core`, so this needs --pretrain-mode ssl.

    --globa-with picks which type is the score:

        D_minus   the donor moved structure the base also moved, in the
                  OPPOSITE direction -- what specialising differently looks
                  like. GLOBA calls it conflict because it is hard to merge;
                  over 980 measured merges it was the only GLOBA statistic to
                  beat random partner choice (22/35 seeds, p=0.032).

        E         the donor occupies cells the base does not, inside the base's
                  rows and columns -- GLOBA's "structural hole", the type its
                  theory rates highest. Never beat random in four measured
                  conditions; kept so the theory gets a fair test.

    Both directions come from one decomposition, so this costs the same as the
    symmetric version: one SVD per analysable layer per pair, recomputed every
    generation because the agents move.
    """
    from sesil.globa_stats import directional_pair_stats

    scores = {a: {} for a in agent_ids}
    for i, a in enumerate(agent_ids):
        for b in agent_ids[i + 1:]:
            a_sees_b, b_sees_a = directional_pair_stats(
                states[a], states[b], core,
                eta=args.probe_eta,
                svd_energy=args.probe_svd_energy,
                basis_energy=args.probe_basis_energy,
                head_prefix=args.head_prefix,
                weighting=args.probe_layer_weighting,
            )
            scores[a][b] = _globa_value(a_sees_b, args.globa_with)
            scores[b][a] = _globa_value(b_sees_a, args.globa_with)
    return scores


# D_plus is redundancy -- the donor moved structure the base already moved, the
# same way. Wanting LESS of it is the sensible direction, so it is reported as
# the share that is NOT redundant. Subtracting rather than negating keeps every
# score in [0, 1], which matters because a negative score would break any
# caller that treats scores as sampling weights.
_GLOBA_INVERTED = {'D_plus'}


def _globa_value(shares, kind):
    value = float(shares[kind])
    return 1.0 - value if kind in _GLOBA_INVERTED else value


def build_score_matrix(population_info, mode='certificate', **kwargs):
    """Directional mating scores: scores[a][b] = how much a wants b.

    'random' gives every candidate the same score, so probabilistic_choice
    samples uniformly. It is a real mode rather than a side effect of zeroing
    the weights: the control arm has to be something the code states, not
    something that falls out of a fallback branch.

    'certificate' scores each candidate by the classes it is certified on, with
    each class weighted by the vector already resolved for --cert-with.
    """
    if mode == 'random':
        return {a['Model Name']: {b['Model Name']: 1.0
                                  for b in population_info if b is not a}
                for a in population_info}

    # Hybrid scores on certificates too; what differs is how ties are settled.
    if mode == 'hybrid':
        mode = 'certificate'

    if mode != 'certificate':
        raise ValueError(
            f'Unknown mating mode {mode!r}; expected certificate, random, '
            f'globa or hybrid.')

    scores = {}
    for agent_a in population_info:
        cert_a = set(agent_a['Certificate'])
        scores[agent_a['Model Name']] = {}
        for agent_b in population_info:
            if agent_a is agent_b:
                continue
            scores[agent_a['Model Name']][agent_b['Model Name']] = mating_score(
                cert_a, set(agent_b['Certificate']), agent_b['Strength'], **kwargs
            )
    return scores


def _score_kwargs(args):
    """Scoring parameters for the configured mode.

    Random mode takes none -- passing weights it ignores would suggest they
    still do something.
    """
    if args.mating_mode == 'random':
        return {}
    if args.mating_mode == 'globa':
        return {}
    return dict(
        weight_extra=args.weight_extra,
        weight_common=args.weight_common,
    )


def best_choice(score_dict):
    """Pick the highest-scoring mate. Ties broken by name, so it is repeatable."""
    if not score_dict:
        return None
    return max(sorted(score_dict), key=lambda m: score_dict[m])


def lexicographic_choice(primary, secondary, tolerance=1e-9):
    """Highest primary score; ties settled by the secondary score.

    This exists because the certificate score has almost no resolution. It
    counts classes, so with 3 classes per agent every fully disjoint couple
    scores the same 3.0 -- measured on real populations, 12 or 13 of 28 pairs
    tie at the top and only 3 distinct values exist across the whole matrix.
    Taking its argmax is therefore an arbitrary pick among a dozen candidates,
    which is why complementarity correlated well with merge outcome (rho about
    +0.5) and still never reached the top quartile in six seeds.

    The secondary score is continuous and measures something else -- the two
    agree only about 24% of the time -- so it can order what the primary cannot
    separate. It never overrides the primary: a candidate outside the tied top
    can never win, whatever its secondary score.
    """
    if not primary:
        return None
    top = max(primary.values())
    tied = [m for m in sorted(primary) if primary[m] >= top - tolerance]
    if len(tied) == 1 or not secondary:
        return tied[0]
    return max(tied, key=lambda m: secondary.get(m, 0.0))


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
# Strategy
# --------------------------------------------------------------------------- #

def select_mates(population_info, args, scores=None, tiebreak=None):
    """Mutual mate choice, in rounds.

    One round: every agent still in the pool picks a partner from the agents
    still in the pool; picks that are RECIPROCATED become couples; both members
    leave the pool. Then the remaining agents try again. It ends when the pool
    is empty or --mating-rounds is reached, and whoever is left is a loner.

    Loners matter more than they look. They carry forward unchanged and earn
    one random UNCERTIFIED class to explore, which is the only route by which a
    class the population has lost can come back. So --mating-rounds is really
    an exploration dial: 1 means a loner is anyone whose first choice did not
    reciprocate, and a high value keeps re-matching until almost everyone is
    paired and nobody explores.

    How an agent picks depends on the mode. Certificate and random modes sample
    in proportion to score, so a fresh round can succeed where the last failed.
    GLOBA mode takes the argmax: it is a prediction of which partner merges
    best, and sampling around a prediction only blurs it. Because that choice
    is deterministic, a round that pairs nobody would repeat forever, so it
    stops early.

    `scores` may be supplied precomputed -- GLOBA mode does that, since its
    matrix needs the agents' weights and the phase-A backbone.

    Returns (pairs, loners).
    """
    if scores is None:
        scores = build_score_matrix(population_info, mode=args.mating_mode,
                                    **_score_kwargs(args))

    hybrid = args.mating_mode == 'hybrid' and tiebreak is not None
    deterministic = args.mating_mode == 'globa' or hybrid
    agents = list(scores.keys())
    n_agents = len(agents)

    pairs = []
    paired = set()

    for _ in range(max(args.mating_rounds, 1)):
        pool = [m for m in agents if m not in paired]
        if len(pool) < 2:
            break

        choices = {}
        for m in pool:
            options = {k: v for k, v in scores[m].items() if k not in paired}
            if hybrid:
                choices[m] = lexicographic_choice(
                    options, {k: v for k, v in tiebreak[m].items()
                              if k not in paired})
            elif deterministic:
                choices[m] = best_choice(options)
            else:
                choices[m] = probabilistic_choice(options)

        new_pairs = []
        for a, b in choices.items():
            if b is not None and choices.get(b) == a:
                if a not in paired and b not in paired:
                    new_pairs.append(tuple(sorted((a, b))))
                    paired.update([a, b])
        pairs.extend(new_pairs)

        # Deterministic choice cannot change its mind: if this round paired
        # nobody, every later round sees the same pool and the same rankings.
        if deterministic and not new_pairs:
            break

    loners = [m for m in agents if m not in paired]

    assert 2 * len(pairs) + len(loners) == n_agents,         f'Population size mismatch: {2 * len(pairs) + len(loners)} != {n_agents}'

    return pairs, loners
