"""The learned mating function: --globa-with learned.

Every other --globa-with value names ONE cell type and scores a candidate by
that type's share alone. Which type to name was never settled by the evidence:
over 980 merges on 35 seeds no single type beat random partner choice after
correcting for the tests run, and which type looked best changed with the merge
configuration rather than being a property of the types. So the choice was a
design decision standing in for a measurement.

This replaces the choice with a fit. A small network reads the POPULATION's
context and emits one coefficient per cell type; a candidate's score is those
coefficients against its own type shares:

    w      = f(context)                     # six numbers, one rule
    score  = sum_t  w_t * share_t           # per candidate

The coefficients are constant within a population, which is what makes them a
rule that can be reported and argued about rather than an opaque per-pair
number. Reading the context at all is what lets one trained function serve
different merge configurations, which is the thing a fixed flag cannot do.

Two details that are easy to get wrong, both found by getting them wrong:

- The six shares SUM TO ONE. A uniform coefficient vector therefore scores
  every candidate identically, and under a ranking loss with no margin that
  costs nothing -- so the optimiser collapses to it and ranks arbitrarily.
  train() standardises the score within each population before the loss, which
  makes a constant score the worst solution rather than a free one.
- Normalising the coefficients onto the simplex would forbid a NEGATIVE weight,
  and "less of this type is better" is exactly what the hand-written rule said
  about D_plus. Normalising by sum|w| keeps the scale fixed while leaving the
  sign free.
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn

from sesil.globa_stats import TYPES

# Context is read at two different times -- while training in phase C, and
# again every generation at mating time -- so it has to be computable from
# things that exist in both places and cost nothing. Anything derived from a
# pair decomposition is excluded on purpose: needing it at mating time would
# mean decomposing every pair just to decide how to score pairs.
CONTEXT_FIELDS = (
    'mean_overall',        # population mean validation accuracy
    'std_overall',         # how spread out the population is
    'mean_cert_size',      # mean certificate size, in classes
    'cert_coverage',       # union of certificates / num_classes
    'merger_is_globa',     # merge configuration, 0/1
    'head_is_label',       # classifier handling, 0/1
)


def context_features(population_info, args):
    """The context vector, from quantities already measured this generation."""
    overalls = np.array([float(p.get('Joint', 0.0)) for p in population_info])
    certs = [set(p.get('Certificate', ())) for p in population_info]
    sizes = np.array([len(c) for c in certs], dtype=float)
    union = set().union(*certs) if certs else set()
    return np.array([
        float(overalls.mean()) if overalls.size else 0.0,
        float(overalls.std()) if overalls.size else 0.0,
        float(sizes.mean()) if sizes.size else 0.0,
        len(union) / max(int(args.num_classes), 1),
        1.0 if getattr(args, 'merger', '') == 'globa' else 0.0,
        1.0 if getattr(args, 'merge_head', '') == 'label' else 0.0,
    ], dtype=np.float64)


class MatingNet(nn.Module):
    """context -> one coefficient per cell type."""

    def __init__(self, n_ctx=len(CONTEXT_FIELDS), hidden=16, n_types=len(TYPES)):
        super().__init__()
        self.n_ctx = int(n_ctx)
        self.hidden = int(hidden)
        self.n_types = int(n_types)
        self.body = nn.Sequential(
            nn.Linear(self.n_ctx, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, self.n_types))

    def forward(self, ctx):
        w = self.body(ctx)
        # Fixed scale, free sign. See the module docstring.
        return w / w.abs().sum(-1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def coefficients(self, ctx):
        """The rule for one population, as a plain dict of type -> weight."""
        t = torch.as_tensor(np.asarray(ctx, dtype=np.float32))
        w = self(t).cpu().numpy().reshape(-1)
        return {name: float(v) for name, v in zip(TYPES, w)}


def score_from_coefficients(shares, coeffs):
    """One candidate's score: the coefficients against its own type shares.

    Shifted into [0, 1]. Scores are used as sampling weights by the certificate
    and random paths, and a negative weight would break them; mating also
    compares scores only against each other, so a monotone shift costs nothing.
    """
    raw = sum(float(coeffs[t]) * float(shares[t]) for t in TYPES)
    scale = sum(abs(float(coeffs[t])) for t in TYPES) or 1.0
    return 0.5 * (raw / scale + 1.0)


def _ranking_loss(score, outcome):
    """Logistic ranking loss over the ordered pairs within one population.

    Standardised first: with the shares summing to one, a constant score is
    otherwise a free optimum. Logistic rather than hinge so a mis-ordered pair
    keeps producing gradient instead of going flat once it clears a margin.
    """
    s = (score - score.mean()) / (score.std() + 1e-6)
    d_s = s[:, None] - s[None, :]
    better = (outcome[:, None] - outcome[None, :]) > 0
    if not better.any():
        return score.sum() * 0.0
    return -torch.nn.functional.logsigmoid(d_s[better]).mean()


def train(samples, hidden=16, epochs=400, lr=0.01, seed=0, verbose=True):
    """Fit a MatingNet.

    `samples` is a list of populations, each a dict with
        'context' : (n_ctx,)        the population's context vector
        'shares'  : (n_cand, 6)     type shares, one row per candidate
        'outcome' : (n_cand,)       measured merge quality, higher is better

    Trained on populations rather than on pooled candidates because the loss
    ranks candidates WITHIN a population; comparing a candidate from one
    population against one from another is not a comparison mating ever makes.
    """
    if not samples:
        raise ValueError('No samples to train the mating function on.')

    torch.manual_seed(seed)
    net = MatingNet(n_ctx=len(samples[0]['context']), hidden=hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    C = [torch.tensor(np.asarray(s['context'], dtype=np.float32)) for s in samples]
    S = [torch.tensor(np.asarray(s['shares'], dtype=np.float32)) for s in samples]
    Y = [torch.tensor(np.asarray(s['outcome'], dtype=np.float32)) for s in samples]

    history = []
    for epoch in range(int(epochs)):
        opt.zero_grad()
        loss = sum(_ranking_loss(S[i] @ net(C[i]), Y[i]) for i in range(len(samples)))
        loss = loss / len(samples)
        loss.backward()
        opt.step()
        if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1):
            history.append((epoch, float(loss)))
            print(f'[phase c] epoch {epoch:4d}  ranking loss {float(loss):.4f}')
    return net, history


def save(net, path, meta=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        'state_dict': net.state_dict(),
        'n_ctx': net.n_ctx,
        'hidden': net.hidden,
        'n_types': net.n_types,
        'context_fields': list(CONTEXT_FIELDS),
        'types': list(TYPES),
        'meta': meta or {},
    }, path)
    if meta is not None:
        with open(os.path.splitext(path)[0] + '.json', 'w') as f:
            json.dump(meta, f, indent=2, default=float)


def load(path, map_location='cpu'):
    blob = torch.load(path, map_location=map_location, weights_only=False)
    if list(blob.get('types', TYPES)) != list(TYPES):
        raise ValueError(
            f'Mating function at {path} was trained on cell types '
            f'{blob.get("types")}, but this build uses {list(TYPES)}. The '
            f'coefficients would be applied to the wrong shares.')
    if list(blob.get('context_fields', CONTEXT_FIELDS)) != list(CONTEXT_FIELDS):
        raise ValueError(
            f'Mating function at {path} was trained on context fields '
            f'{blob.get("context_fields")}, but this build builds '
            f'{list(CONTEXT_FIELDS)}. The network would read the wrong inputs.')
    net = MatingNet(n_ctx=blob['n_ctx'], hidden=blob['hidden'],
                    n_types=blob['n_types'])
    net.load_state_dict(blob['state_dict'])
    net.eval()
    return net


def default_path(args):
    """Where phase C writes the fitted function, and mating reads it."""
    return os.path.join(args.run_dir, 'mating_fn.pt')
