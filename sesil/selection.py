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
        rank      B's population percentile on the class. Scale-free, so it
                  keeps separating agents late in a run when accuracies have
                  converged.

    Directional by construction: score(A, B) != score(B, A).
    """
    extra_skills = cert_b - cert_a
    common_skills = cert_a & cert_b

    score = weight_extra * sum(strength_b[i] for i in extra_skills)
    score += weight_common * sum(strength_b[i] for i in common_skills)
    return score


def build_score_matrix(population_info, **kwargs):
    """Directional mating scores: scores[a][b] = how much a wants b.

    Each record carries 'Strength', the per-class weight vector already
    resolved for the configured --mate-score mode, so all three modes share one
    code path and differ only in what that vector contains.
    """
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
    return dict(
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
# Strategy
# --------------------------------------------------------------------------- #

def select_mates(population_info, args):
    """Mutual, probabilistic mate choice.

    Each unpaired agent probabilistically picks a mate; only RECIPROCATED picks
    become couples. A couple yields one child per parent (see
    sesil.merge.extract_children), so each couple is recorded once and the
    population size is preserved. Agents nobody reciprocated become loners and
    carry forward unchanged, earning one random uncertified class to explore.

    Returns (pairs, loners).
    """
    scores = build_score_matrix(population_info, **_score_kwargs(args))

    agents = list(scores.keys())
    n_agents = len(agents)

    pairs = []
    paired = set()

    for _ in range(args.max_retries):
        choices = {m: probabilistic_choice(scores[m])
                   for m in agents if m not in paired}

        for a, b in choices.items():
            if b is not None and choices.get(b) == a:
                if a not in paired and b not in paired:
                    pairs.append(tuple(sorted((a, b))))
                    paired.update([a, b])

    loners = [m for m in agents if m not in paired]

    assert 2 * len(pairs) + len(loners) == n_agents, \
        f'Population size mismatch: {2 * len(pairs) + len(loners)} != {n_agents}'

    return pairs, loners
