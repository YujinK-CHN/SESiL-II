"""
Method registry.

SESiL is one algorithm; the merge operator and the mate-selection strategy are
arguments to it.  Both are looked up here so that adding a new operator means
adding one line, not copying a 950-line training script.
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


def get_selection_fn(selection):
    """Return the mate-selection strategy named by --selection.

    Every strategy has the signature
        fn(population_info, args) -> (pairs, loners)
    where pairs is a list of (model_id_a, model_id_b) and loners is a list of
    model ids that carry forward unmerged.
    """
    from sesil import selection as selection_module

    table = {
        'bidirectional': selection_module.select_bidirectional,
        'breed':         selection_module.select_breed,
        'guided':        selection_module.select_guided,
        'hard':          selection_module.select_hard,
    }
    if selection not in table:
        raise ValueError(
            f'Unknown selection {selection!r}. Choose one of {sorted(table)}.'
        )
    return table[selection]
