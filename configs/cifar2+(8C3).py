config = {
    'dataset': {
        'name': 'cifar100',
    },
    'model': {
        'name': 'resnet20x4',
        'dir': './checkpoints/cifar10_evolution/resnet20x4/initial/',
        'bases': []
    },
    'merging_fn': 'match_tensors_zipit',
    'eval_type': 'logits',
    'merging_metrics': ['covariance', 'mean'],
}