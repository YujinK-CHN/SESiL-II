config = {
    'dataset': {
        'name': 'cifar10',
    },
    'model': {
        'name': 'resnet20x4',
        'dir': './checkpoints/cifar10_evolution/initial/',
        'bases': []
    },
    'merging_fn': 'match_tensors_zipit',
    'eval_type': 'logits',
    'merging_metrics': ['covariance', 'mean'],
}