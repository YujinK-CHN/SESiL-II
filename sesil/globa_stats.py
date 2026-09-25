"""GLOBA's decomposition, used as a screening instrument rather than a merger.

Given two agents and the shared backbone they were finetuned from, each agent
has a task vector tau = agent - core. GLOBA projects both task vectors into a
common basis built from their singular vectors, prunes the low-energy cells,
and partitions the surviving cells into six types by how the two parents occupy
them:

    A        one parent only, and the other touches neither that row nor that
             column -- complete orthogonality, nothing to negotiate
    B        one parent only, the other occupies the row
    C        one parent only, the other occupies the column
    E        one parent only, the other occupies both row and column
    D_plus   both parents, same sign -- agreement
    D_minus  both parents, opposite sign -- direct conflict

This module computes that partition and reports how the parents' update energy
is distributed across the six types. It does **not** merge anything: children in
this experiment are produced by SESiL's own merger. The question being asked is
whether the type structure of a pair predicts how good their offspring will be.

Pre-registered predictor: energy_frac['A'] - energy_frac['D_minus'], the one
combination with a mechanistic story (orthogonal updates compose freely,
opposite-sign updates cancel). Everything else this module reports is
exploratory and labelled as such by the caller.

Conv weights are 4-D. Following GLOBA, [out, in, k, k] is flattened to
[out, in*k*k]: output channels stay as rows, everything on the input side
collapses into the column axis. So row occupancy reads as "this parent uses an
output channel the other does not", which is a meaningful notion of
specialisation; the column side is a coarser mix of input channel and spatial
offset. 1-D tensors (all the BatchNorm parameters) have no matrix structure and
are skipped.

All linear algebra in float64.
"""

import numpy as np
import torch


TYPES = ('A', 'B', 'C', 'E', 'D_plus', 'D_minus')


def as_matrix(t):
    """[out, in, k, k] -> [out, in*k*k]; 2-D passes through unchanged."""
    return t.reshape(t.shape[0], -1)


def is_analysable(name, tensor, core, head_prefix):
    """Whether this tensor gets decomposed.

    Needs a counterpart in the core (no core, no task vector), must not be the
    classifier (agents get freshly random heads in phase B, so head differences
    are initialisation noise rather than learned structure), and must have
    matrix structure to decompose.
    """
    if name not in core:
        return False
    if head_prefix and name.startswith(head_prefix):
        return False
    return tensor.ndim in (2, 4)


def _truncate(U, S, Vh, energy):
    """Keep the singular triples whose cumulative squared energy stays within
    `energy`, always at least one."""
    if energy >= 1.0:
        return U, S, Vh
    total = (S ** 2).sum()
    if total <= 0:
        return U[:, :1], S[:1], Vh[:1]
    k = max(1, int(((torch.cumsum(S ** 2, 0) / total) <= energy).sum().item()))
    return U[:, :k], S[:k], Vh[:k]


def _lead_basis(M, basis_energy):
    """Orthonormal basis for the span of the concatenated singular vectors."""
    P, D, _ = torch.linalg.svd(M, full_matrices=False)
    if D.numel() == 0 or D[0] <= 0:
        return P[:, :1]
    if basis_energy >= 1.0:
        tol = D[0] * max(M.shape) * torch.finfo(M.dtype).eps
        k = int((D > tol).sum().item())
    else:
        cum = torch.cumsum(D ** 2, 0) / (D ** 2).sum()
        k = int(torch.searchsorted(cum, basis_energy).item()) + 1
    return P[:, :max(1, min(k, P.shape[1]))]


def prune(C, eta):
    """Zero all but the highest-energy cells holding `eta` of the total."""
    if eta >= 1.0:
        return C.clone()
    energy = (C * C).flatten()
    total = energy.sum()
    if total <= 0 or eta <= 0.0:
        return torch.zeros_like(C)
    vals, order = torch.sort(energy, descending=True, stable=True)
    cum = torch.cumsum(vals, 0)
    k = min(int(torch.searchsorted(cum, total * eta).item()) + 1, energy.numel())
    mask = torch.zeros_like(energy, dtype=torch.bool)
    mask[order[:k]] = True
    return torch.where(mask.view_as(C), C, torch.zeros_like(C))


def classify_directional(C_base, C_donor):
    """Type the DONOR's cells against the BASE's occupancy. GLOBA's own form.

    GLOBA is not symmetric: model 1 is taken whole and only model 2 is sorted
    into the six types, each defined by what model 1 does with that cell's row
    and column. Its type E, for instance, is

        (C1 == 0) & C1_occupied_rows & C1_occupied_cols

    -- cells the donor occupies that the base does not, inside the base's row
    and column span. Swap the two models and the masks change.

    That asymmetry is what SESiL's mating needs. Pairing is mutual acceptance,
    so a score has to mean "what does this partner bring ME", which differs by
    direction. The symmetric `classify` below unions both readings and makes
    score(a, b) == score(b, a), which collapses mutual choice into a single
    global ranking -- every agent then agrees on who is best, and pairing
    becomes deterministic greedy matching with no room for preference to
    differ.

    Only cells the donor occupies are typed; the base's own cells are its
    baseline, not something it can receive.
    """
    nz_base, nz_donor = C_base != 0, C_donor != 0
    row_base = nz_base.any(1, keepdim=True)
    col_base = nz_base.any(0, keepdim=True)

    both = nz_base & nz_donor
    same = torch.sign(C_base) == torch.sign(C_donor)
    only_donor = nz_donor & ~nz_base

    return {
        'A': only_donor & ~row_base & ~col_base,
        'B': only_donor & row_base & ~col_base,
        'C': only_donor & ~row_base & col_base,
        'E': only_donor & row_base & col_base,
        'D_plus': both & same,
        'D_minus': both & ~same,
    }


def classify(Cp, Cq):
    """Six boolean masks partitioning the cells either parent occupies.

    A cell held by exactly one parent is typed by what the OTHER parent does
    with that cell's row and column. Cells held by both are typed by sign
    agreement. Symmetric in (p, q) by construction.
    """
    nz_p, nz_q = Cp != 0, Cq != 0
    both = nz_p & nz_q
    same = torch.sign(Cp) == torch.sign(Cq)
    only_p, only_q = nz_p & ~nz_q, nz_q & ~nz_p

    row_p, col_p = nz_p.any(1, keepdim=True), nz_p.any(0, keepdim=True)
    row_q, col_q = nz_q.any(1, keepdim=True), nz_q.any(0, keepdim=True)

    def single(only, row_other, col_other):
        return {
            'A': only & ~row_other & ~col_other,
            'B': only & row_other & ~col_other,
            'C': only & ~row_other & col_other,
            'E': only & row_other & col_other,
        }

    sp = single(only_p, row_q, col_q)
    sq = single(only_q, row_p, col_p)

    masks = {t: sp[t] | sq[t] for t in ('A', 'B', 'C', 'E')}
    masks['D_plus'] = both & same
    masks['D_minus'] = both & ~same
    return masks


def layer_stats(tau_p, tau_q, eta=0.80, svd_energy=0.90, basis_energy=0.999):
    """Type-energy breakdown for one layer's pair of task vectors."""
    Up, Sp, Vhp = torch.linalg.svd(tau_p, full_matrices=False)
    Uq, Sq, Vhq = torch.linalg.svd(tau_q, full_matrices=False)

    Up_, Sp_, Vhp_ = _truncate(Up, Sp, Vhp, svd_energy)
    Uq_, Sq_, Vhq_ = _truncate(Uq, Sq, Vhq, svd_energy)

    Pu = _lead_basis(torch.cat((Up_, Uq_), 1), basis_energy)
    Pv = _lead_basis(torch.cat((Vhp_.T, Vhq_.T), 1), basis_energy)

    # Classify the structure the truncation kept, not the raw tensors.
    tp = Up_ @ torch.diag(Sp_) @ Vhp_ if svd_energy < 1.0 else tau_p
    tq = Uq_ @ torch.diag(Sq_) @ Vhq_ if svd_energy < 1.0 else tau_q

    Cp = prune(Pu.T @ tp @ Pv, eta)
    Cq = prune(Pu.T @ tq @ Pv, eta)
    masks = classify(Cp, Cq)

    # Energy is measured on the PARENTS' cells (Cp^2 + Cq^2), not on their
    # superposition (Cp + Cq)^2.
    #
    # GLOBA measures on the superposition because it is building a merged
    # tensor and wants the merged energy. That is exactly wrong for screening:
    # opposite-sign cells cancel there, so a D_minus cell contributes almost
    # nothing -- and two parents that cancel *perfectly* produce a total of
    # zero, making every fraction 0.0 and leaving the conflict detector blind
    # to total conflict. Measuring on the parents' own cells asks the question
    # we actually want: how much of what the parents learned sits in cells of
    # each type. Agreement is no longer inflated and conflict is no longer
    # suppressed.
    E = Cp * Cp + Cq * Cq
    total = float(E.sum())
    frac = {t: (float((E * masks[t]).sum()) / total if total > 0 else 0.0)
            for t in TYPES}

    # The superposition view kept alongside, since it is what GLOBA's own
    # merge would see, and the two disagreeing is itself informative.
    S = Cp + Cq
    total_s = float((S * S).sum())
    frac_merged = {t: (float(((S * masks[t]) ** 2).sum()) / total_s if total_s > 0 else 0.0)
                   for t in TYPES}

    norm_p, norm_q = float(tau_p.norm()), float(tau_q.norm())
    residual_p = float((tau_p - Pu @ Cp @ Pv.T).norm())
    residual_q = float((tau_q - Pu @ Cq @ Pv.T).norm())

    return {
        'energy_frac': frac,
        'energy_frac_merged': frac_merged,
        'cancellation': (1.0 - total_s / total) if total > 0 else 0.0,
        'basis_u': int(Pu.shape[1]),
        'basis_v': int(Pv.shape[1]),
        'nnz_p': int((Cp != 0).sum()),
        'nnz_q': int((Cq != 0).sum()),
        'typed_frac_p': 1.0 - (residual_p / norm_p) ** 2 if norm_p > 0 else 0.0,
        'typed_frac_q': 1.0 - (residual_q / norm_q) ** 2 if norm_q > 0 else 0.0,
        'norm_p': norm_p,
        'norm_q': norm_q,
        # Cosine of the two task vectors: the cheapest possible compatibility
        # statistic, carried so the fancy decomposition has to beat it.
        'cosine': (float((tau_p * tau_q).sum()) / (norm_p * norm_q)
                   if norm_p > 0 and norm_q > 0 else 0.0),
    }


def pair_stats(sd_p, sd_q, core, eta=0.80, svd_energy=0.90,
               basis_energy=0.999, head_prefix='linear.', weighting='energy'):
    """Whole-network type-energy breakdown for one candidate couple.

    Layers are collapsed into one set of numbers by a weighted mean. The
    default weights each layer by its share of the pair's total update energy,
    so a large late conv counts for more than a tiny early one; 'uniform'
    treats every layer alike. Per-layer values are returned alongside, since
    the signal may not live uniformly across depth.
    """
    per_layer = {}
    for name, tensor in sd_p.items():
        if name not in sd_q or not is_analysable(name, tensor, core, head_prefix):
            continue
        base = core[name].double()
        tau_p = as_matrix(tensor.double() - base)
        tau_q = as_matrix(sd_q[name].double() - base)
        per_layer[name] = layer_stats(tau_p, tau_q, eta, svd_energy, basis_energy)

    if not per_layer:
        raise ValueError(
            'No analysable layers. The agents and the core must come from the '
            'same architecture, and the core must be a real phase-A backbone.')

    if weighting == 'energy':
        weights = {n: s['norm_p'] ** 2 + s['norm_q'] ** 2 for n, s in per_layer.items()}
        if sum(weights.values()) <= 0:
            weights = {n: 1.0 for n in per_layer}
    elif weighting == 'uniform':
        weights = {n: 1.0 for n in per_layer}
    else:
        raise ValueError(f'Unknown layer weighting {weighting!r}')

    total_w = sum(weights.values())

    def collapse(get):
        return float(sum(weights[n] * get(s) for n, s in per_layer.items()) / total_w)

    energy_frac = {t: collapse(lambda s, t=t: s['energy_frac'][t]) for t in TYPES}
    energy_frac_merged = {t: collapse(lambda s, t=t: s['energy_frac_merged'][t])
                          for t in TYPES}

    summary = {
        'energy_frac': energy_frac,
        'energy_frac_merged': energy_frac_merged,
        'cancellation': collapse(lambda s: s['cancellation']),
        'typed_frac': collapse(lambda s: 0.5 * (s['typed_frac_p'] + s['typed_frac_q'])),
        'cosine': collapse(lambda s: s['cosine']),
        'norm_ratio': collapse(
            lambda s: min(s['norm_p'], s['norm_q']) / max(s['norm_p'], s['norm_q'])
            if max(s['norm_p'], s['norm_q']) > 0 else 0.0),
        'basis_u': collapse(lambda s: float(s['basis_u'])),
        'basis_v': collapse(lambda s: float(s['basis_v'])),
        'n_layers': len(per_layer),
    }

    # The pre-registered genetic test. Named separately so the analysis cannot
    # quietly promote whichever exploratory statistic happened to win.
    summary['globa_score'] = energy_frac['A'] - energy_frac['D_minus']

    return summary, per_layer


# --------------------------------------------------------------------------- #
# Directional scoring (mate selection)
# --------------------------------------------------------------------------- #

def directional_pair_stats(sd_a, sd_b, core, eta=0.80, svd_energy=0.90,
                           basis_energy=0.999, head_prefix='linear.',
                           weighting='energy'):
    """Energy fractions for BOTH directions of a pair, from one decomposition.

    Returns (a_sees_b, b_sees_a): each a dict of type -> share of what that
    donor brings, typed against that base. `a_sees_b` is what B offers A, which
    is the score A should use when choosing B.

    The SVD, the joint basis and the projected matrices are identical whichever
    parent is called the base -- only the typing differs -- so both directions
    come out of a single decomposition and this costs no more than the
    symmetric version.

    Energy is measured on the donor's own cells, normalised by the donor's
    total, so the number reads as "what fraction of what B brings is of this
    type, as far as A is concerned".
    """
    per_layer = {}
    for name, tensor in sd_a.items():
        if name not in sd_b or not is_analysable(name, tensor, core, head_prefix):
            continue
        base = core[name].double()
        tau_a = as_matrix(tensor.double() - base)
        tau_b = as_matrix(sd_b[name].double() - base)

        Ua, Sa, Vha = torch.linalg.svd(tau_a, full_matrices=False)
        Ub, Sb, Vhb = torch.linalg.svd(tau_b, full_matrices=False)
        Ua_, Sa_, Vha_ = _truncate(Ua, Sa, Vha, svd_energy)
        Ub_, Sb_, Vhb_ = _truncate(Ub, Sb, Vhb, svd_energy)

        Pu = _lead_basis(torch.cat((Ua_, Ub_), 1), basis_energy)
        Pv = _lead_basis(torch.cat((Vha_.T, Vhb_.T), 1), basis_energy)

        ta = Ua_ @ torch.diag(Sa_) @ Vha_ if svd_energy < 1.0 else tau_a
        tb = Ub_ @ torch.diag(Sb_) @ Vhb_ if svd_energy < 1.0 else tau_b
        Ca = prune(Pu.T @ ta @ Pv, eta)
        Cb = prune(Pu.T @ tb @ Pv, eta)

        def shares(C_base, C_donor):
            masks = classify_directional(C_base, C_donor)
            energy = C_donor * C_donor
            total = float(energy.sum())
            return {t: (float((energy * masks[t]).sum()) / total if total > 0 else 0.0)
                    for t in TYPES}

        per_layer[name] = {
            # a_sees_b: B is the donor, typed against A.
            'a_sees_b': shares(Ca, Cb),
            'b_sees_a': shares(Cb, Ca),
            'weight': float(tau_a.norm()) ** 2 + float(tau_b.norm()) ** 2,
        }

    if not per_layer:
        raise ValueError(
            'No analysable layers. Agents and core must share an architecture, '
            'and the core must be a real phase-A backbone.')

    if weighting == 'energy':
        w = {n: s['weight'] for n, s in per_layer.items()}
        if sum(w.values()) <= 0:
            w = {n: 1.0 for n in per_layer}
    elif weighting == 'uniform':
        w = {n: 1.0 for n in per_layer}
    else:
        raise ValueError(f'Unknown layer weighting {weighting!r}')
    total_w = sum(w.values())

    def collapse(key):
        return {t: float(sum(w[n] * s[key][t] for n, s in per_layer.items()) / total_w)
                for t in TYPES}

    return collapse('a_sees_b'), collapse('b_sees_a')
