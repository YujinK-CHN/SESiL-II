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
            # .detach(): the graph is still attached at this point, and
            # float() on a grad-tracking tensor warns.
            value = float(loss.detach())
            history.append((epoch, value))
            print(f'[phase c] epoch {epoch:4d}  ranking loss {value:.4f}')
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


# --------------------------------------------------------------------------- #
# --mating-mode learn
# --------------------------------------------------------------------------- #
#
# A different question from --globa-with learned, and a different model.
#
#   --globa-with learned   reads the POPULATION and emits coefficients over the
#                          six types; a pair's score is those coefficients
#                          against its own shares. The rule is linear in the
#                          shares by construction.
#   --mating-mode learn    reads THE PAIR -- its genetic profile, the energy
#                          mapping over all six types -- and predicts how good
#                          that specific couple's offspring will be. Nothing
#                          constrains it to be linear, and the prediction is in
#                          the same units as the measured outcome.
#
# Why predict the QUALITY rather than a normalised rank: a rank normalised
# within a population always puts the best pair at 1.0, so an absolute floor
# could never say "every pair here is bad, nobody should mate". Predicting the
# outcome itself keeps the level, so the floor means something fixed.
#
# Why one shared network applied per pair, rather than one network emitting a
# vector over all pairs: a fixed-width output would tie the model to a single
# population size and to the arbitrary order pairs happen to be enumerated in.
# Per-pair with shared weights is permutation-equivariant and works for any
# population, which is what makes it scale.


class PairQualityNet(nn.Module):
    """genetic profile (six type shares) -> predicted offspring quality."""

    def __init__(self, n_types=len(TYPES), hidden=32):
        super().__init__()
        self.n_types = int(n_types)
        self.hidden = int(hidden)
        self.body = nn.Sequential(
            nn.Linear(self.n_types, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, 1))

    def forward(self, profile):
        # Sigmoid: the retention measures this is trained against are
        # fractions, so the prediction lives in the same bounded range and a
        # --mate-floor can be read as "offspring at least this good".
        return torch.sigmoid(self.body(profile)).squeeze(-1)

    @torch.no_grad()
    def predict(self, shares):
        """One pair, from the dict directional_pair_stats returns."""
        x = torch.tensor([[float(shares[t]) for t in TYPES]],
                         dtype=torch.float32)
        return float(self(x).item())


def _quality_loss(pred, target, rank_weight=1.0):
    """Regression on the measured quality, plus a ranking term.

    Both are needed and they do different jobs. Regression fixes the LEVEL,
    without which an absolute --mate-floor is meaningless. Ranking fixes the
    ORDER, which is what mating actually consumes, and a squared error alone
    will happily trade a correct ordering for a better average fit.
    """
    mse = torch.nn.functional.mse_loss(pred, target)
    d_p = pred[:, None] - pred[None, :]
    better = (target[:, None] - target[None, :]) > 0
    if better.any():
        rank = -torch.nn.functional.logsigmoid(d_p[better]).mean()
    else:
        rank = pred.sum() * 0.0
    return mse + rank_weight * rank, float(mse.detach()), float(rank.detach())


def train_quality(samples, hidden=32, epochs=600, lr=0.005, seed=0,
                  rank_weight=1.0, verbose=True):
    """Fit a PairQualityNet across several populations.

    `samples` is a list of populations, each a dict with
        'shares'  : (n_pairs, 6)   genetic profile per ORDERED pair
        'outcome' : (n_pairs,)     measured offspring quality, higher better

    Several populations rather than one on purpose: a rule fitted to a single
    society learns that society's accidents. The ranking term is computed
    WITHIN each population, because ranking a pair from one society against a
    pair from another is not a comparison mating ever makes; the regression
    term is pooled, since quality is on one scale everywhere.
    """
    if not samples:
        raise ValueError('No samples to train the pair-quality net on.')

    torch.manual_seed(seed)
    net = PairQualityNet(hidden=hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    S = [torch.tensor(np.asarray(s['shares'], dtype=np.float32)) for s in samples]
    Y = [torch.tensor(np.asarray(s['outcome'], dtype=np.float32)) for s in samples]

    history = []
    for epoch in range(int(epochs)):
        opt.zero_grad()
        total, mse_acc, rank_acc = 0.0, 0.0, 0.0
        for i in range(len(samples)):
            loss, mse, rank = _quality_loss(net(S[i]), Y[i], rank_weight)
            total = total + loss
            mse_acc += mse
            rank_acc += rank
        (total / len(samples)).backward()
        opt.step()
        if verbose and (epoch % max(1, epochs // 6) == 0 or epoch == epochs - 1):
            n = len(samples)
            history.append((epoch, mse_acc / n, rank_acc / n))
            print(f'[phase c] epoch {epoch:4d}  mse {mse_acc / n:.4f}  '
                  f'ranking {rank_acc / n:.4f}')
    return net, history


def save_quality(net, path, meta=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        'kind': 'pair_quality',
        'state_dict': net.state_dict(),
        'hidden': net.hidden,
        'n_types': net.n_types,
        'types': list(TYPES),
        'meta': meta or {},
    }, path)
    if meta is not None:
        with open(os.path.splitext(path)[0] + '.json', 'w') as f:
            json.dump(meta, f, indent=2, default=float)


def load_quality(path, map_location='cpu'):
    blob = torch.load(path, map_location=map_location, weights_only=False)
    if blob.get('kind') != 'pair_quality':
        raise ValueError(
            f'{path} holds a {blob.get("kind", "coefficient")} model, not a '
            f'pair-quality net. --mating-mode learn and --globa-with learned '
            f'fit different things and their files are not interchangeable.')
    if list(blob.get('types', TYPES)) != list(TYPES):
        raise ValueError(
            f'Pair-quality net at {path} was trained on cell types '
            f'{blob.get("types")}, but this build uses {list(TYPES)}.')
    net = PairQualityNet(n_types=blob['n_types'], hidden=blob['hidden'])
    net.load_state_dict(blob['state_dict'])
    net.eval()
    return net


def quality_path(args):
    """Where phase C writes the pair-quality net, and mating reads it."""
    return os.path.join(args.run_dir, 'pair_quality.pt')


def default_path(args):
    """Where phase C writes the fitted function, and mating reads it."""
    return os.path.join(args.run_dir, 'mating_fn.pt')
