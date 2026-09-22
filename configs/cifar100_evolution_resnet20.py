config = {
    'dataset': {
        'name': 'cifar50',
    },
    'model': {
        'name': 'resnet20x4',
        'dir': './checkpoints/cifar10_evolution/initial/',
        'bases': []
    },
    'merging_fn': 'match_tensors_permute',
    'eval_type': 'logits',
    'merging_metrics': ['covariance', 'mean'],
}