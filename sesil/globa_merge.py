"""GLOBA as a merge operator: a base parent, a donor parent, one child.

    child = base + (selected components of what the donor learned)

GLOBA is asymmetric and this module keeps it that way. Its own formula is

    C_merged = scale_base * C_base + sum_t scale_t * (C_donor * mask_t)
    child    = core + Pu @ C_merged @ Pv.T

The base's task vector is taken whole; only the DONOR's cells are sorted into
the six types, each defined by what the base does with that cell's row and
column. Swap the roles and you get a different child.

That is exactly what SESiL's crossover needs. A couple produces TWO children --
merge(a, b) and merge(b, a) -- so the population size is preserved and the
siblings genuinely differ: one is a plus what b brought, the other is b plus
what a brought. No interpolation weight has to be invented to tell them apart,
which is what --merge-bias existed to do for the ZipIt path.

GLOBA's published defaults take only two of the six types from the donor:

    scale_c1_identity = 1.0
    D_minus = 1.0    E = 1.0    A = B = C = D_plus = 0.0

Keep the base entire, add the cells where the donor disagrees with it and the
gaps the donor fills inside its span. Worth noting those are the same two types
that came out of the selection study on independent evidence.

THE HEAD IS NOT MERGED BY THE DECOMPOSITION. Agents here are finetuned from one
backbone with a freshly random classifier each, so averaging two classifiers
destroys both: each row becomes half a trained weight and half another agent's
untouched noise. A class row is taken from whichever parent is certified for
it, and only rows both or neither hold are averaged. Measured on this codebase,
that one choice was worth +0.13 to +0.25 in child quality -- more than every
other merge decision combined.

All linear algebra in float64; outputs cast back to the parents' dtype.
"""

import torch

from sesil.globa_stats import (
    TYPES,
    _lead_basis,
    _truncate,
    as_matrix,
    classify_directional,
    is_analysable,
    prune,
)


# How much of each DONOR type is carried into the child. The base is always
# taken whole (scale 1.0), as in GLOBA.
DONOR_PRESETS = {
    # GLOBA's published default: the donor contributes where it disagrees with
    # the base, and where it fills a gap inside the base's row/column span.
    'globa': {'D_minus': 1.0, 'E': 1.0,
              'A': 0.0, 'B': 0.0, 'C': 0.0, 'D_plus': 0.0},

    # Everything the donor has that the base does not touch at all, plus the
    # two GLOBA types. Adds the orthogonal cells GLOBA leaves out.
    'orthogonal': {'D_minus': 1.0, 'E': 1.0,
                   'A': 1.0, 'B': 1.0, 'C': 1.0, 'D_plus': 0.0},

    # Every typed cell the donor has. NOT the same as adding the two task
    # vectors whole: what pruning and truncation dropped from the donor is
    # still left out, because only typed cells can be assigned a scale. It
    # doubles the update wherever the two agree.
    'all': {t: 1.0 for t in TYPES},

    # Nothing from the donor -- the child is its base parent. Useful only as a
    # control: if a preset cannot beat this, the crossover added nothing.
    'none': {t: 0.0 for t in TYPES},
}


def donor_scales(name):
    if name not in DONOR_PRESETS:
        raise ValueError(
            f'Unknown GLOBA preset {name!r}. Choose one of {sorted(DONOR_PRESETS)}.')
    return dict(DONOR_PRESETS[name])


def merge_matrix(tau_base, tau_donor, scales, eta, svd_energy, basis_energy):
    """One layer: the base's task vector plus selected parts of the donor's."""
    Ub, Sb, Vhb = torch.linalg.svd(tau_base, full_matrices=False)
    Ud, Sd, Vhd = torch.linalg.svd(tau_donor, full_matrices=False)

    Ub_, Sb_, Vhb_ = _truncate(Ub, Sb, Vhb, svd_energy)
    Ud_, Sd_, Vhd_ = _truncate(Ud, Sd, Vhd, svd_energy)

    Pu = _lead_basis(torch.cat((Ub_, Ud_), 1), basis_energy)
    Pv = _lead_basis(torch.cat((Vhb_.T, Vhd_.T), 1), basis_energy)

    tb = Ub_ @ torch.diag(Sb_) @ Vhb_ if svd_energy < 1.0 else tau_base
    td = Ud_ @ torch.diag(Sd_) @ Vhd_ if svd_energy < 1.0 else tau_donor

    C_base = prune(Pu.T @ tb @ Pv, eta)
    C_donor = prune(Pu.T @ td @ Pv, eta)
    masks = classify_directional(C_base, C_donor)

    # The base whole, then the donor's typed cells at their own scales.
    C_merged = C_base.clone()
    for t in TYPES:                       # fixed order, so the sum is stable
        scale = scales[t]
        if scale != 0.0:
            C_merged = C_merged + scale * (C_donor * masks[t])

    # What pruning and truncation left out of the BASE only. The base is meant
    # to survive whole, so its residual is restored; the donor's is not, since
    # the point of the types is to choose what the donor contributes.
    residual = tau_base - Pu @ C_base @ Pv.T
    return Pu @ C_merged @ Pv.T + residual


def merge_head(tensor_base, tensor_donor, classes_base, classes_donor, mode):
    """Combine the classifier, row by row.

    Row i of a classifier weight (or bias) is the evidence for class i. A
    parent never trained on class i did not leave that row alone -- class i
    never appeared as a target, so training pushed the row DOWN. It is an
    anti-detector, and averaging it with the other parent's real detector
    cancels part of the signal and halves what survives.

    So a class exactly one parent holds is taken from that parent whole. Rows
    both hold, or neither does, are averaged: there is no better-informed
    choice available.
    """
    if mode == 'average' or classes_base is None or classes_donor is None:
        return 0.5 * (tensor_base + tensor_donor)

    known_base = set(int(c) for c in classes_base)
    known_donor = set(int(c) for c in classes_donor)

    out = 0.5 * (tensor_base + tensor_donor)
    for row in range(tensor_base.shape[0]):
        if row in known_base and row not in known_donor:
            out[row] = tensor_base[row]
        elif row in known_donor and row not in known_base:
            out[row] = tensor_donor[row]
    return out


def merge(sd_base, sd_donor, core, classes_base=None, classes_donor=None,
          preset='globa', head='label', eta=0.80, svd_energy=0.90,
          basis_energy=0.999, head_prefix='linear.'):
    """One child: `sd_base` plus selected components of `sd_donor`.

    Asymmetric by design -- call it twice with the parents swapped to get a
    couple's two children.

    Matrix-shaped tensors present in the core are decomposed and recombined.
    The classifier goes through merge_head. Everything else -- BatchNorm scales,
    shifts and running statistics -- has no matrix structure to decompose and is
    averaged, except integer buffers, which are copied from the base.
    """
    if set(sd_base) != set(sd_donor):
        raise ValueError('parents have different parameter sets')

    scales = donor_scales(preset)
    child = {}

    for name, tensor_base in sd_base.items():
        tensor_donor = sd_donor[name]

        if head_prefix and name.startswith(head_prefix):
            child[name] = merge_head(tensor_base, tensor_donor,
                                     classes_base, classes_donor, head)
            continue

        if not is_analysable(name, tensor_base, core, head_prefix):
            child[name] = (0.5 * (tensor_base + tensor_donor)
                           if tensor_base.is_floating_point()
                           else tensor_base.clone())
            continue

        base = core[name].double()
        shape, dtype = tensor_base.shape, tensor_base.dtype
        tau_base = as_matrix(tensor_base.double() - base)
        tau_donor = as_matrix(tensor_donor.double() - base)

        tau_child = merge_matrix(tau_base, tau_donor, scales,
                                 eta, svd_energy, basis_energy)
        child[name] = (base + tau_child.reshape(shape)).to(dtype)

    return child
