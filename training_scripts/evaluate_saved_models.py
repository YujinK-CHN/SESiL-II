import torch
import numpy as np
import torchvision
import torchvision.transforms as T
from models.resnets import resnet20
from utils import *

# -----------------
# Config
# -----------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = './data'
CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]
normalize = T.Normalize(np.array(CIFAR_MEAN)/255, np.array(CIFAR_STD)/255)

# -----------------
# Load CIFAR-10 test set
# -----------------
test_transform = T.Compose([T.ToTensor(), normalize])
test_set = torchvision.datasets.CIFAR100(root=DATA_DIR, train=False, download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)

# -----------------
# Load model checkpoint
# -----------------
def load_model(checkpoint_path, num_classes=100, w=4):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    # handle both formats
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model = resnet20(w=w, num_classes=num_classes).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# -----------------
# Evaluation helpers
# -----------------
def evaluate_model(model, test_loader):
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

def evaluate_per_class(model, test_loader, num_classes=100):
    correct = np.zeros(num_classes)
    total = np.zeros(num_classes)
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=1)
            for c in range(num_classes):
                mask = (y == c)
                correct[c] += (preds[mask] == c).sum().item()
                total[c] += mask.sum().item()
    return correct / np.maximum(total, 1)

import os
import hashlib
import json

# your encode_labels function without file saving per call
def encode_labels(labels):
    # normalize input
    if isinstance(labels, str):  # e.g. "0_9_8_9_2_6"
        labels = [int(x) for x in labels.split("_")]
    elif isinstance(labels, (list, tuple)):
        labels = list(map(int, labels))
    else:
        raise TypeError("labels must be str, list, or tuple")

    # create hash
    name = "_".join(map(str, labels))
    hash_id = hashlib.md5(name.encode()).hexdigest()[:8]
    return hash_id, labels

def rename_folders_and_generate_mapping(root_dir, mapping_file="mapping.json"):
    mapping = {}
    print(root_dir)
    print(os.listdir(root_dir))
    for folder in os.listdir(root_dir):
        print(folder)
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # encode folder name
        hash_id, labels = encode_labels(folder)

        # rename folder
        new_folder_path = os.path.join(root_dir, hash_id)
        os.rename(folder_path, new_folder_path)

        # update mapping
        mapping[hash_id] = labels
        print(f"{folder} -> {hash_id}")

    # save mapping.json once
    with open(os.path.join(root_dir, mapping_file), "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"\n✅ All folders renamed and mapping saved to {mapping_file}")
    return mapping

import os
# -----------------
# Example usage
# -----------------
if __name__ == "__main__":
    #history = np.load("./checkpoints/CIFAR-10_baseline/CIFAR-Acc.npy")
    #print(history)
    

    root_dir = "./checkpoints/cifar10_evolution/CIFAR-100C3-P20 ExMutate 9-25/initial"
    rename_folders_and_generate_mapping(root_dir)
    '''
    folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    for folder in folders:

        print(folder)
        ckpt = os.path.join(root_dir, folder, "resnet20x4_v0.pth.tar")
        model = load_model(ckpt, num_classes=100, w=4)

        # Overall accuracy
        acc = evaluate_model(model, test_loader)
        print(f"Overall CIFAR-100 accuracy: {acc:.2%}")

        # Per-class accuracy
        per_class_acc = evaluate_per_class(model, test_loader, num_classes=100)
        for i, a in enumerate(per_class_acc):
            print(f"Class {i}: {a:.2%}")
    '''

    
    

    '''
    CIFAR-10 baseline
    Overall CIFAR-10 accuracy: 93.66%
    Class 0: 94.60%
    Class 1: 96.60%
    Class 2: 90.60%
    Class 3: 85.00%
    Class 4: 96.90%
    Class 5: 91.90%
    Class 6: 95.00%
    Class 7: 94.90%
    Class 8: 96.10%
    Class 9: 95.00%
    Old (0–7) avg acc: 93.19%
    New (8–9) avg acc: 95.55%

    CIFAR-2 outlander
    Overall CIFAR-10 accuracy: 18.88%
    Class 0: 0.00%
    Class 1: 0.00%
    Class 2: 0.00%
    Class 3: 0.00%
    Class 4: 0.00%
    Class 5: 0.00%
    Class 6: 0.00%
    Class 7: 0.00%
    Class 8: 98.80%
    Class 9: 90.00%
    Old (0–7) avg acc: 0.00%
    New (8–9) avg acc: 94.40%
    '''
