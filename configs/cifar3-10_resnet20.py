config = {
    'dataset': {
        'name': 'cifar10',
    },
    'model': {
        'name': 'resnet20x4',
        'dir': './checkpoints/cifar3-10_trainlogitsv2/resnet20x4/pairsplits/',
        'bases': []
    },
    'merging_fn': 'match_tensors_zipit',
    'eval_type': 'logits',
    'merging_metrics': ['covariance', 'mean'],
}