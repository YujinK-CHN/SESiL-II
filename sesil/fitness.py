"""
Fitness evaluation.

Every agent is evaluated on the **whole label space**, not on a subset it
claims. That is what makes certification possible: proficiency is ranked across
the population per class, so each agent's accuracy on every class has to be
measured, including classes it was never trained on.

It is also cheaper than the old scheme, which masked the test set to the agent's
own labels and then walked it a second time just to count class frequencies.
One pass, no masking, no splits.
"""

import numpy as np
import torch
from torch.cuda.amp import autocast

from utils import get_device


def _predict(model, images):
    """Class predictions from either a plain model or a partially-zipped merge.

    With partial zipping (`--stop-node`), ModelMerge.forward returns one output
    per head rather than a single tensor. Combining them by softmax-then-max
    over heads gives a prediction over the full label space without telling the
    model which task the sample came from -- no task oracle.
    """
    outputs = model(images)

    if isinstance(outputs, list):
        stacked = torch.stack(outputs, dim=1)              # [B, heads, C]
        probs = stacked.softmax(dim=-1).to(stacked.dtype)
        outputs = probs.max(dim=-2)[0]                     # [B, C]

    return outputs.argmax(dim=-1)


def evaluate_all_classes(model, test_loader, num_classes):
    """Per-class accuracy over the entire label space.

    Returns:
        per_class: list of length num_classes, accuracy in [0, 1]
                   (0.0 for classes absent from the test set)
        overall:   accuracy across all samples
    """
    device = get_device(model)

    correct = torch.zeros(num_classes, dtype=torch.long)
    total = torch.zeros(num_classes, dtype=torch.long)

    model.eval()
    with torch.no_grad(), autocast():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = _predict(model, images)

            labels_cpu = labels.detach().cpu()
            hit = (preds == labels).detach().cpu()

            total += torch.bincount(labels_cpu, minlength=num_classes)
            correct += torch.bincount(labels_cpu[hit], minlength=num_classes)

    per_class = [
        (correct[c].item() / total[c].item()) if total[c] > 0 else 0.0
        for c in range(num_classes)
    ]
    overall = correct.sum().item() / max(total.sum().item(), 1)

    return per_class, overall


def summarise(per_class, overall, certificate=None):
    """Fitness record for one agent, ready for the csv and for mate selection.

    'Per Class' is the vector certification and mate selection both run on.
    'Certified Avg' is accuracy restricted to what the agent is certified in --
    how good it is at its own job, as opposed to 'Joint', which is how good it
    is overall.
    """
    record = {
        'Joint': overall,
        'Per Class': list(per_class),
        'Coverage': 0 if certificate is None else len(certificate),
    }

    if certificate:
        record['Certified Avg'] = float(np.mean([per_class[c] for c in sorted(certificate)]))
    else:
        record['Certified Avg'] = 0.0

    return record
