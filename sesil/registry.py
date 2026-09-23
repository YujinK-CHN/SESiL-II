"""
Merge-operator registry.

SESiL is one algorithm; the merge operator is an argument to it. Looking it up
here means adding a new operator is one line, not a copied training script.
"""

# --merger -> name of the function in matching_functions.py
MERGERS = {
    'zipit':   'match_tensors_zipit',
    'permute': 'match_tensors_permute',
    # Weight averaging is the identity alignment: no permutation or zipping is
    # applied, so get_merged_state_dict() ends up plainly averaging the models.
    'wavg':    'match_tensors_identity',
}


def get_merger_name(merger):
    """'permute' -> 'match_tensors_permute'."""
    if merger not in MERGERS:
        raise ValueError(
            f'Unknown merger {merger!r}. Choose one of {sorted(MERGERS)}.'
        )
    return MERGERS[merger]
