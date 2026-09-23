"""
Mutation: finetune an agent for exactly its allotted budget.

Merging lands an offspring somewhere between its parents; a short finetune lets
it settle. This is the only place in a generation where gradients flow, so it
is where the per-agent training budget is spent.

The budget is denominated in sample-presentations rather than epochs. That
matters because agents cover different numbers of classes: with mutation
restricted to inherited labels, an agent holding 3 of 10 classes has a training
set a third the size. Charging it "one epoch" would silently give it a third of
the compute of a fully-covered agent. Instead every agent is trained on the
same number of presentations, cycling its subset as many times as needed.
"""

from utils import FractionalDataloader, train_logits


def mutate(model, train_loader, test_loader, sample_budget):
    """Finetune for exactly `sample_budget` sample-presentations.

    `sample_budget` may be smaller than the agent's training set (sub-epoch
    training) or larger (multiple passes); FractionalDataloader handles both,
    cycling the underlying loader when it runs out.

    Deliberately no `seed` argument. FractionalDataloader reseeds the global
    torch/numpy/random state on every __iter__ when given one, so passing a
    fixed seed here would hand every agent in every generation the identical
    data ordering -- erasing exactly the variation that makes two siblings
    diverge. The run is already reproducible from utils.set_seed() at startup;
    letting the global RNG advance is what keeps each mutation distinct.

    Returns (model, accuracy).
    """
    n_available = len(train_loader.dataset)
    fraction = sample_budget / max(n_available, 1)

    # Only wrap when the budget is not exactly one pass -- avoids the extra
    # iterator layer in the common whole-epoch case.
    if abs(fraction - 1.0) > 1e-9:
        loader = FractionalDataloader(train_loader, fraction)
    else:
        loader = train_loader

    model.train()
    # One optimisation pass over a loader sized to the budget, rather than N
    # epochs over the full subset: the loader length *is* the dose.
    model, final_acc = train_logits(
        model=model,
        train_loader=loader,
        test_loader=test_loader,
        epochs=1,
    )
    return model, final_acc
