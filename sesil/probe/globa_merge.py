"""GLOBA as a merge operator: two parents plus their shared core -> one child.

The probe already decomposes every candidate pair to score it. That
decomposition is most of a merge: having projected both task vectors into a
common basis and typed every cell, recombining them is one weighted sum away.
This module takes that last step, so the same run can ask two questions at
once -- whether GLOBA *predicts* a good pair, and whether GLOBA *makes* a
better child than SESiL's own merger on the same pair.

    tau_child = Pu @ (sum_t alpha_t * (Cp + Cq) * mask_t) @ Pv.T
                + rho * (Rp + Rq)
    child     = core + tau_child

`alpha_t` is a coefficient per cell type, `rho` one for the residual -- the
part of each task vector the typed cells do not carry. Keeping the residual is
what makes the identities hold: alpha = rho = 0.5 everywhere is exactly plain
weight averaging, and alpha = rho = 1 is exactly tau_a + tau_b. Nothing is
silently discarded, so a preset can only redistribute emphasis, never lose
information.

THE HEAD IS NOT AVERAGED BY DEFAULT, and this matters more than any preset.
Agents here are finetuned from one backbone with a freshly random classifier
each, on disjoint class subsets. Averaging two such classifiers destroys both:
each row is half a trained weight and half another agent's unrelated one.
SELBAL measured the same backbone merge at 0.29 with an averaged head against
0.91 with a label-aware one. So a class row is taken from whichever parent was
trained on that class, and only genuinely shared or genuinely unknown rows are
averaged.

Unlike ZipIt and permute, this needs no forward passes: no activation
statistics, no alignment pass over data. The child does need its BatchNorm
statistics recalibrated afterwards, which the caller does.
"""

import hashlib

import torch

from sesil.globa_stats import (
    TYPES,
    _lead_basis,
    _truncate,
    as_matrix,
    classify,
    is_analysable,
    prune,
)


# Each preset fixes the six type coefficients and the residual coefficient.
PRESETS = {
    # Plain weight averaging -- the reference point. Every cell, whatever its
    # type, contributes half. Useful precisely because it is boring: if a
    # fancier preset cannot beat it, the typing is not buying anything.
    'average': dict(alpha={t: 0.5 for t in TYPES}, rho=0.5),

    # Both task vectors added whole. Doubles the update magnitude where the
    # parents agree, which is either the point or a disaster depending on how
    # far they have drifted.
    'sum': dict(alpha={t: 1.0 for t in TYPES}, rho=1.0),

    # Cells only one parent touched, and which the other does not reach by row
    # or column, are free: nothing contests them, so take them whole.
    'orthogonal-full': dict(
        alpha={'A': 1.0, 'B': 0.5, 'C': 0.5, 'E': 0.5,
               'D_plus': 0.5, 'D_minus': 0.5},
        rho=0.5),

    # Anything held by exactly ONE parent is kept whole; only genuinely
    # contested cells are averaged. The most aggressive preset that still
    # cannot double-count, and the one that best matches the intuition that
    # disjoint specialists should both survive a merge.
    'single-full': dict(
        alpha={'A': 1.0, 'B': 1.0, 'C': 1.0, 'E': 1.0,
               'D_plus': 0.5, 'D_minus': 0.5},
        rho=0.5),
}


def preset_alpha(name):
    if name not in PRESETS:
        raise ValueError(
            f'Unknown GLOBA preset {name!r}. Choose one of {sorted(PRESETS)}.')
    p = PRESETS[name]
    return dict(p['alpha']), p['rho']


def decompose(tau_p, tau_q, eta, svd_energy, basis_energy):
    """Project both task vectors into a shared basis and type every cell.

    Returns (Pu, Pv, Cp, Cq, masks, Rp, Rq) with the exact identity
    tau = Pu @ C @ Pv.T + R for each parent, so the residuals carry whatever
    the typed cells do not.
    """
    Up, Sp, Vhp = torch.linalg.svd(tau_p, full_matrices=False)
    Uq, Sq, Vhq = torch.linalg.svd(tau_q, full_matrices=False)

    Up_, Sp_, Vhp_ = _truncate(Up, Sp, Vhp, svd_energy)
    Uq_, Sq_, Vhq_ = _truncate(Uq, Sq, Vhq, svd_energy)

    Pu = _lead_basis(torch.cat((Up_, Uq_), 1), basis_energy)
    Pv = _lead_basis(torch.cat((Vhp_.T, Vhq_.T), 1), basis_energy)

    tp = Up_ @ torch.diag(Sp_) @ Vhp_ if svd_energy < 1.0 else tau_p
    tq = Uq_ @ torch.diag(Sq_) @ Vhq_ if svd_energy < 1.0 else tau_q

    Cp = prune(Pu.T @ tp @ Pv, eta)
    Cq = prune(Pu.T @ tq @ Pv, eta)

    Rp = tau_p - Pu @ Cp @ Pv.T
    Rq = tau_q - Pu @ Cq @ Pv.T
    return Pu, Pv, Cp, Cq, classify(Cp, Cq), Rp, Rq


def merge_matrix(tau_p, tau_q, alpha, rho, eta, svd_energy, basis_energy):
    """Recombine one layer's two task vectors into the child's."""
    Pu, Pv, Cp, Cq, masks, Rp, Rq = decompose(
        tau_p, tau_q, eta, svd_energy, basis_energy)

    superposed = Cp + Cq
    combined = torch.zeros_like(superposed)
    for t in TYPES:                       # fixed order, so the sum is stable
        a = alpha[t]
        if a != 0.0:
            combined = combined + a * (superposed * masks[t])

    return Pu @ combined @ Pv.T + rho * (Rp + Rq)


def merge_head(tensor_p, tensor_q, classes_p, classes_q, mode):
    """Combine the classifier, row by row.

    Row i of a classifier weight (or bias) is the evidence for class i. When
    exactly one parent was trained on that class, only that parent's row means
    anything -- the other's is its untouched random initialisation, and
    averaging the two halves a trained row into noise. Rows both parents know,
    or neither does, are averaged as usual.
    """
    if mode == 'average' or classes_p is None or classes_q is None:
        return 0.5 * (tensor_p + tensor_q)

    known_p = set(int(c) for c in classes_p)
    known_q = set(int(c) for c in classes_q)

    out = 0.5 * (tensor_p + tensor_q)
    for row in range(tensor_p.shape[0]):
        if row in known_p and row not in known_q:
            out[row] = tensor_p[row]
        elif row in known_q and row not in known_p:
            out[row] = tensor_q[row]
    return out


def _fingerprint(state_dict):
    """Order-independent identity for a model, so merging is symmetric.

    Without a canonical order, merge(a, b) and merge(b, a) could differ through
    tie-breaking inside the basis construction. Sorting the two parents by a
    content hash makes the operator a function of the unordered pair.
    """
    h = hashlib.sha256()
    for key in sorted(state_dict):
        h.update(key.encode())
        h.update(state_dict[key].detach().cpu().contiguous().numpy().tobytes())
    return h.digest()


def merge(sd_a, sd_b, core, classes_a=None, classes_b=None,
          preset='single-full', head='label', eta=0.80,
          svd_energy=0.90, basis_energy=0.999, head_prefix='linear.'):
    """Two parents and their shared core -> one child state dict.

    Matrix-shaped tensors that exist in the core are decomposed and recombined.
    The classifier goes through merge_head. Everything else -- BatchNorm scales
    and shifts, running statistics -- has no matrix structure to decompose and
    is averaged, except integer buffers, which are copied.
    """
    if set(sd_a) != set(sd_b):
        raise ValueError('parents have different parameter sets')

    alpha, rho = preset_alpha(preset)

    # Canonical order: symmetry by construction. Class sets travel with their
    # parent so the label-aware head is not silently swapped.
    if _fingerprint(sd_b) < _fingerprint(sd_a):
        sd_a, sd_b = sd_b, sd_a
        classes_a, classes_b = classes_b, classes_a

    child = {}
    for name, tensor_a in sd_a.items():
        tensor_b = sd_b[name]

        if head_prefix and name.startswith(head_prefix):
            child[name] = merge_head(tensor_a, tensor_b,
                                     classes_a, classes_b, head)
            continue

        if not is_analysable(name, tensor_a, core, head_prefix):
            child[name] = (0.5 * (tensor_a + tensor_b)
                           if tensor_a.is_floating_point()
                           else tensor_a.clone())
            continue

        base = core[name].double()
        shape, dtype = tensor_a.shape, tensor_a.dtype
        tau_a = as_matrix(tensor_a.double() - base)
        tau_b = as_matrix(tensor_b.double() - base)

        tau_child = merge_matrix(tau_a, tau_b, alpha, rho,
                                 eta, svd_energy, basis_energy)
        child[name] = (base + tau_child.reshape(shape)).to(dtype)

    return child
