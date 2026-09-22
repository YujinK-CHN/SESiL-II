"""
Mutation: finetune an offspring for a few epochs after merging.

Merging lands the offspring somewhere between its parents; a short finetune
lets it settle.  This is the only place in a generation where gradients flow,
so it is also where the per-generation training budget is spent.
"""

from utils import train_logits


def mutate(model, train_loader, test_loader, epochs=2):
    """Finetune a merged offspring in place.  Returns (model, accuracy)."""
    model.train()
    model, final_acc = train_logits(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
    )
    return model, final_acc
