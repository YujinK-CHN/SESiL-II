import os
import pdb
import clip
import torch
import random
from copy import deepcopy

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from model_merger import ModelMerge
from sklearn.model_selection import train_test_split
from models.resnets import resnet20
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm

CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]
normalize = T.Normalize(np.array(CIFAR_MEAN)/255, np.array(CIFAR_STD)/255)
denormalize = T.Normalize(-np.array(CIFAR_MEAN)/np.array(CIFAR_STD), 255/np.array(CIFAR_STD))

def load_clip_features(class_names, device):
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(device)
    model, preprocess = clip.load('ViT-B/32', device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)

    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #model_dir = '/home/student/Desktop/ZipIt/checkpoints/cifar50_trainlogitsv2'
    model_dir = '/home/student/Desktop/ZipIt/checkpoints/cifar10_trainlogitsv2'
    
    split_runs = 1
    models_per_run = 1
    # data_dir = '/nethome/gstoica3/research/pytorch-cifar100/data/cifar-100-python' #'/tmp'
    data_dir = './data/cifar-100-python/cifar-10-batches-py'
    wrapper = torchvision.datasets.CIFAR10
    num_classes = 10
    batch_size = 500
    model_width = 4
    epochs = 100
    
    model_dir = os.path.join(model_dir, f'resnet20x{model_width}', 'pairsplits')
    print(model_dir)
    os.makedirs(model_dir, exist_ok=True)
    
    train_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4), T.ToTensor(), normalize])
    test_transform = T.Compose([T.ToTensor(), normalize])
    train_dset = wrapper(root=data_dir, train=True, download=True, transform=train_transform)
    test_dset = wrapper(root=data_dir, train=False, download=True, transform=test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    if 'clip' in model_dir:
        clip_features = load_clip_features(test_dset.classes, device=device)
        out_dim = 512
    else:
        out_dim = num_classes# // 2

    for _ in range(split_runs):
    # REMOVE split logic
    # splits = train_test_split(...)
    # split_trainers = ...
    # split_testers = ...
    # split1, split2 = splits
    # label_remapping = ...

        for j in range(models_per_run):
            model = resnet20(w=model_width, num_classes=out_dim).cuda().train()

            if 'clip' in model_dir:
                model, final_acc = train_cliphead(
                    model=model, train_loader=train_loader, test_loader=test_loader,
                    class_vectors=clip_features,  # all classes
                    remap_class_idxs=None, epochs=epochs
                )
            else:
                model, final_acc = train_logits(
                    model=model, train_loader=train_loader,
                    test_loader=test_loader, epochs=epochs
                )

            print(f'Base model on all classes Acc: {final_acc}')
            print('Saving Base Model')
            save_dir = os.path.join(model_dir, 'all_classes')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'resnet20x{model_width}_v{len(os.listdir(save_dir))}.pth.tar')
            save_model(model, save_path)
                
    print('Done!')
