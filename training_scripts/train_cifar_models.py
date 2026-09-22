import os
import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from models.resnets import resnet20
from utils import *

# --------------------------
# Config
# --------------------------
DATA_DIR = './data'
MODEL_DIR = './checkpoints'
BATCH_SIZE = 500
EPOCHS = 218
MODEL_WIDTH = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# CIFAR stats
CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]
normalize = T.Normalize(np.array(CIFAR_MEAN) / 255, np.array(CIFAR_STD) / 255)

def save_model(model_or_state, path):
    """
    Save either a full nn.Module or just a state_dict to disk.
    """
    import os, torch

    # Case 1: nn.Module
    if hasattr(model_or_state, "state_dict"):
        sd = model_or_state.state_dict()

    # Case 2: already a state_dict (OrderedDict)
    elif isinstance(model_or_state, dict):
        sd = model_or_state

    else:
        raise TypeError(f"Unsupported type passed to save_model: {type(model_or_state)}")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(sd, path)
    print(f"✅ Model/state_dict saved to {path}")


# --------------------------
# Dataset helpers
# --------------------------
def get_cifar_dataset(classes=None):
    """
    If cifar8_classes is None → full CIFAR-10
    If cifar8_classes is list → subset with those class indices
    """
    transform_train = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4), T.ToTensor(), normalize])
    transform_test = T.Compose([T.ToTensor(), normalize])
    # change here for CIFAR-10
    base_train = torchvision.datasets.CIFAR100(root=DATA_DIR, train=True, download=True, transform=transform_train)
    base_test = torchvision.datasets.CIFAR100(root=DATA_DIR, train=False, download=True, transform=transform_test)

    if classes is not None:
        print(f"Using CIFAR subset with classes {classes}")
        train_set = SubsetWithRemap(base_train, classes)
        test_set = SubsetWithRemap(base_test, classes)
        num_classes = len(classes)
    else:
        train_set, test_set = base_train, base_test
        num_classes = 100  # change here for CIFAR-10

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    return train_loader, test_loader, num_classes



# --------------------------
# Training new model
# --------------------------
def train_logits(model, train_loader, test_loader, epochs=200, remap_class_idxs=None):
    collect_acc = []
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.00005)
    ne_iters = len(train_loader)

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1, end_factor=1e-7, total_iters=ne_iters
    )
    early_stopper = EarlyStopper(patience=epochs, min_delta=.0001)
    
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(reduction='mean')
    device = get_device(model)
    losses = []
    acc = 0.
    best_acc = 0.
    best_epoch = 0
    best_sd = None
    pbar = tqdm(range(epochs), desc=f'Training, prev acc: {acc}: ')
    
    for epoch in pbar:
        model = model.train()
        for i, (inputs, labels) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(inputs.to(device))
                if remap_class_idxs is not None:
                    remapped_labels = remap_class_idxs[labels].to(device)
                else:
                    remapped_labels = labels.to(device)
                    
                loss = loss_fn(logits, remapped_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(loss)
            losses.append(loss.item())

        # ✅ force evaluation with 10 classes
        acc, per_class_acc = evaluate_logits(
            model, test_loader, remap_class_idxs=remap_class_idxs, num_classes=10, return_confusion=True
        ) # change here for CIFAR-10
        print(sum(per_class_acc[8:]) / len(per_class_acc[8:]))
        collect_acc.append(sum(per_class_acc[8:]) / len(per_class_acc[8:]))
        print(per_class_acc)

        if acc > best_acc:
            best_sd = model.state_dict()
            best_acc = acc
            best_epoch = epoch
            
        if early_stopper.early_stop(acc):
            print(f'Stopping at Epoch: {epoch}. Best Accuracy {best_acc}, Achieved at Epoch {best_epoch}')
            break
        pbar.set_description(f'Training, prev acc: {sum(per_class_acc) / len(per_class_acc)}: ')

    # Load best model and evaluate again
    model.load_state_dict(best_sd)
    acc = evaluate_logits(
        model, test_loader, remap_class_idxs=remap_class_idxs, num_classes=10  # change here for CIFAR-10
    )
    print('Acc at Best Model: {}'.format(acc))
    save_dir = "./checkpoints/cifar8F2/baseline-f"
    os.makedirs(save_dir, exist_ok=True)  # create the folder if missing
    np.save(os.path.join(save_dir, "CIFAR-Acc.npy"), collect_acc)
    return model, best_acc

def evaluate_logits(
    model, 
    test_loader, 
    return_confusion=False, 
    use_flip_aug=False, 
    remap_class_idxs=None, 
    class_idxs=None, 
    eval_mask=None, 
    num_classes=10  # always force CIFAR-10 size
):
    model.eval()
    correct = 0
    total = 0
    totals = defaultdict(lambda: 0)
    corrects = defaultdict(lambda: 0)
    loss_fn = CrossEntropyLoss()
    device = next(iter(model.parameters())).device
    total_loss = 0
    total_iter = len(test_loader)

    with torch.no_grad(), autocast():
        for inputs, labels in tqdm(test_loader, 'Evaluating classification model'):
            inputs = inputs.to(device)
            outputs = model(inputs)

            if use_flip_aug:
                outputs += model(torch.flip(inputs, (3,)))
            
            if eval_mask is not None:
                outputs[:, eval_mask == 0] = -torch.inf
            
            pred = outputs.argmax(dim=-1)
            total += pred.shape[0]

            if remap_class_idxs is not None:
                remapped_labels = remap_class_idxs[labels].to(device)
            else:
                remapped_labels = labels.to(device)

            loss = loss_fn(outputs, remapped_labels)
            total_loss += loss

            for gt, p in zip(remapped_labels, pred):
                gt, p = gt.item(), p.item()
                totals[gt] += 1
                if gt == p:
                    correct += 1
                    corrects[gt] += 1

    # ✅ force CIFAR-10 size
    totals = [totals[i] for i in range(num_classes)]
    corrects = [corrects[i] for i in range(num_classes)]
    total_loss = total_loss / total_iter

    if return_confusion:
        # per-class accuracy (handle zero divisions safely)
        per_class_acc = [
            c / t if t > 0 else 0.0 for c, t in zip(corrects, totals)
        ]
        return correct / sum(totals), per_class_acc
    else:
        return correct / total

    
def train_new_model(num_epoch=None, classes=None, save_path=None, load_path=None, finetune=False):
    train_loader, test_loader, num_classes = get_cifar_dataset(classes)

    # Always build a fresh model with correct num_classes
    model = resnet20(w=MODEL_WIDTH, num_classes=num_classes).to(DEVICE)

    # Optionally load checkpoint
    if load_path is not None and os.path.exists(load_path):
        checkpoint = torch.load(load_path, map_location=DEVICE)
        if "state_dict" in checkpoint:   # case 1: checkpoint dict
            model.load_state_dict(checkpoint["state_dict"])
        else:                            # case 2: raw state_dict
            model.load_state_dict(checkpoint)
        print(f"Loaded model from {load_path}")
    else:
        print("No checkpoint loaded, training from scratch.")

    # If finetuning, you might want to freeze parts of the model
    if finetune:
        for name, param in model.named_parameters():
            if 'linear' not in name:  # keep backbone frozen
                param.requires_grad = False
        print("Finetuning mode: backbone frozen, classifier trainable.")

    # Train
    model, final_acc = train_logits(model, train_loader, test_loader, epochs=num_epoch)

    print(f"Final Accuracy: {final_acc:.2f}")

    # Save if requested
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_model(model, save_path)
        print(f"Saved model to {save_path}")

    return model




import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import torchvision

# --- small helper dataset: keep original label numbers (do NOT remap) ---
class SubsetKeepLabels(torch.utils.data.Dataset):
    def __init__(self, dataset, valid_classes):
        """
        Keep only samples whose label is in valid_classes,
        but keep original labels (e.g. 8 and 9).
        """
        self.dataset = dataset
        self.valid_classes = set(valid_classes)
        # works for torchvision CIFAR datasets which expose `targets`
        self.indices = [i for i, t in enumerate(dataset.targets) if t in self.valid_classes]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x, y = self.dataset[real_idx]
        return x, y

def finetune_model(
    pretrained_model_path,
    save_path,
    new_classes=[8, 9],
    base_classes=None,
    epochs=50
):
    """
    Finetune only the linear layer for new classes (8-9),
    with backbone frozen. Evaluate on all 10 CIFAR-10 classes.
    Works with checkpoints trained on 8 or 10 classes.
    """
    import torch
    import torch.nn as nn

    if base_classes is None:
        base_classes = list(range(8))
    num_old_classes = len(base_classes)
    num_classes = 10  # full CIFAR-10 output

    # ----------------------
    # 1. Dataset for new classes only (keep original labels)
    # ----------------------
    base_train = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True,
        transform=T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4), T.ToTensor(), normalize])
    )
    base_test = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True,
        transform=T.Compose([T.ToTensor(), normalize])
    )

    train_set = SubsetKeepLabels(base_train, new_classes)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(base_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # ----------------------
    # 2. Load pretrained state dict
    # ----------------------
    state_dict = torch.load(pretrained_model_path, map_location=DEVICE)
    if "state_dict" in state_dict:  # handle wrapped checkpoints
        state_dict = state_dict["state_dict"]

    # detect number of classes in checkpoint
    ckpt_out_features = state_dict["linear.weight"].shape[0]

    # ----------------------
    # 3. Build matching model and load weights
    # ----------------------
    model = resnet20(w=MODEL_WIDTH, num_classes=ckpt_out_features).to(DEVICE)
    model.load_state_dict(state_dict)
    print(f"Loaded pretrained model with {ckpt_out_features} output classes")

    # ----------------------
    # 4. If old model has < 10 classes, expand classifier
    # ----------------------
    if ckpt_out_features < num_classes:
        old_w = model.linear.weight.data.clone()
        old_b = model.linear.bias.data.clone()
        model.linear = nn.Linear(model.linear.in_features, num_classes).to(DEVICE)
        model.linear.weight.data[:ckpt_out_features] = old_w
        model.linear.bias.data[:ckpt_out_features] = old_b
        print(f"Expanded classifier from {ckpt_out_features} → {num_classes} classes")

    # ----------------------
    # 5. Freeze backbone, train only linear
    # ----------------------
    for name, param in model.named_parameters():
        param.requires_grad = ('linear' in name)

    # ----------------------
    # 6. Train
    # ----------------------
    finetuned_model, best_acc = train_logits(
        model,
        train_loader,
        test_loader,
        epochs=epochs,
        remap_class_idxs=None
    )

    # ----------------------
    # 7. Save model
    # ----------------------
    save_model(finetuned_model, save_path)
    print(f"✅ Finetuned model saved to {save_path}. Final Accuracy: {best_acc:.4f}")

    return finetuned_model








# --------------------------
# Finetune CIFAR-k → CIFAR-10 but train ONLY on new_classes
# --------------------------
def align_pretrained_model(pretrained_path,
                          new_classes,
                          save_path=None,
                          target_num_classes: int = 10,
                          freeze_backbone: bool = False,
                          train_epochs: int = 1):
    """
    pretrained_path: checkpoint that contains a CIFAR-k model (k = len(new_classes))
                     (handles raw state_dict or {"state_dict": ...})
    new_classes: list of target class indices in CIFAR-10 where the old k rows should be placed.
                 Example: for a 2-class model trained on CIFAR classes [8,9], use new_classes=[8,9].
    target_num_classes: final #classes (default 10).
    freeze_backbone: if True, freeze all parameters except classifier rows for new_classes.
    train_epochs: how many epochs to finetune (passed to train_logits).
    """
    # 1) load checkpoint (both possible formats)
    ckpt = torch.load(pretrained_path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        old_sd = ckpt["state_dict"]
        was_wrapped = True
    elif isinstance(ckpt, dict):
        old_sd = ckpt
        was_wrapped = False
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(ckpt)}")

    # old number of classes (from checkpoint linear layer)
    old_num_classes = old_sd["linear.weight"].shape[0]
    if len(new_classes) != old_num_classes:
        raise ValueError(f"Length of new_classes ({len(new_classes)}) must equal old model classes ({old_num_classes})")

    print(f"Loaded pretrained CIFAR-{old_num_classes} model from {pretrained_path}")

    # 2) create a fresh model with target_num_classes and copy any matching weights
    new_model = resnet20(w=MODEL_WIDTH, num_classes=target_num_classes).to(DEVICE)

    # copy all matching parameters from old_sd into new model's state_dict (except linear)
    new_sd = new_model.state_dict()
    for k, v in old_sd.items():
        # skip linear.* because shapes differ
        if k.startswith("linear."):
            continue
        if k in new_sd and new_sd[k].shape == v.shape:
            new_sd[k] = v.clone()

    # For the classifier: copy each old row i into the specified new_classes[i] row
    # (old_sd['linear.weight'] shape [k, in_features])
    old_lin_w = old_sd["linear.weight"].clone()
    old_lin_b = old_sd["linear.bias"].clone()

    # default init already present in new_sd for classifier; overwrite only the mapped rows
    for i, cls in enumerate(new_classes):
        if cls < 0 or cls >= target_num_classes:
            raise ValueError(f"new_classes contains invalid class index {cls} for target_num_classes={target_num_classes}")
        new_sd["linear.weight"][cls] = old_lin_w[i].clone()
        new_sd["linear.bias"][cls] = old_lin_b[i].clone()

    # load the assembled state dict
    new_model.load_state_dict(new_sd)

    print(f"Expanded classifier: placed old {old_num_classes} rows into indices {new_classes} of {target_num_classes} outputs.")

    # 3) make classifier rows OTHER THAN new_classes non-trainable (prevent gradient updates)
    # Create a mask: 1.0 for new_classes rows (allowed to train), 0.0 for frozen rows
    mask = torch.zeros(target_num_classes, device=DEVICE, dtype=torch.float32)
    mask[new_classes] = 1.0

    # register hooks to zero-out gradients for frozen rows
    def _weight_hook(grad):
        # grad shape: [out_features, in_features]
        return grad * mask.view(-1, 1)

    def _bias_hook(grad):
        # bias grad shape: [out_features]
        return grad * mask

    # IMPORTANT: register_hook must be attached to the Parameter object currently on device
    new_model.linear.weight.register_hook(_weight_hook)
    new_model.linear.bias.register_hook(_bias_hook)
    print("Registered gradient hooks: only rows %s will get grads in classifier." % (new_classes,))

    # 4) optionally freeze backbone parameters entirely (if requested)
    if freeze_backbone:
        for name, p in new_model.named_parameters():
            if not name.startswith("linear"):
                p.requires_grad = False
        print("Backbone frozen; only classifier rows for new_classes will be trainable.")
    else:
        print("Backbone left trainable (default).")

    # 5) prepare a train loader that contains ONLY samples with labels in new_classes (keeps original labels)
    #    use your existing dataset transforms / path via get_cifar_dataset if available; otherwise we recreate
    try:
        full_train_loader, full_test_loader, _ = get_cifar_dataset(cifar8_classes=None)  # your project helper
        full_train_ds = full_train_loader.dataset
        test_loader = full_test_loader
        batch_size = getattr(full_train_loader, "batch_size", 500)
        num_workers = getattr(full_train_loader, "num_workers", 8)
    except Exception:
        # Fallback: make datasets directly (assumes normalize/transforms are defined in your script)
        train_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4), T.ToTensor(), normalize])
        test_transform  = T.Compose([T.ToTensor(), normalize])
        full_train_ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        test_ds = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)
        batch_size = 500
        num_workers = 8

    train_subset = SubsetKeepLabels(full_train_ds, new_classes)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    print(f"Training set contains {len(train_subset)} samples from classes {new_classes} (only these will be used).")

    # 6) train using your existing training helper (it will only see samples with labels in new_classes)
    #    note: because we blocked grads for other classifier rows, those rows will not change even though the
    #    softmax includes them in computation.
    model_trained, final_acc = train_logits(new_model, train_loader, test_loader, epochs=train_epochs)

    print(f"Finetuned (only on new classes {new_classes}) CIFAR-{target_num_classes} accuracy: {final_acc:.4f}")

    # 7) save in same style as loaded checkpoint (raw sd or wrapped)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if was_wrapped:
            torch.save({"state_dict": model_trained.state_dict()}, save_path)
        else:
            torch.save(model_trained.state_dict(), save_path)
        print(f"Saved finetuned model to {save_path}")

    return model_trained



# --------------------------
# Dataset wrapper for subsets
# --------------------------
class SubsetWithRemap(torch.utils.data.Dataset):
    def __init__(self, dataset, valid_classes):
        """
        Wrap a dataset to only keep valid_classes and remap labels.
        Example: valid_classes=[8,9] → {8:0, 9:1}
        """
        self.dataset = dataset
        self.valid_classes = valid_classes
        self.class_map = {c: i for i, c in enumerate(valid_classes)}
        self.indices = [i for i, t in enumerate(dataset.targets) if t in valid_classes]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x, y = self.dataset[real_idx]
        return x, self.class_map[y]



# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":

    # 1. Train CIFAR-2 (e.g. classes 8-9)
    '''
    train_new_model(classes=[8, 9], save_path="./checkpoints/8_9/resnet20x4_v0.pth.tar")
    align_pretrained_model(
        pretrained_path="./checkpoints/cifar2_model_8_9/resnet20x4_v0.pth.tar",
        save_path="./checkpoints/cifar10_from_8_9/resnet20x4_v0.pth.tar",
        new_classes=[8, 9],
    )
    '''
    
    # 2. Train CIFAR-10
    '''
    train_new_model(num_epoch=755,
                    classes=None, 
                    #load_path="./checkpoints/CIFAR-100_baseline/resnet20x4_v0.pth.tar",
                    save_path="./checkpoints/CIFAR-100_baseline/resnet20x4_v0.pth.tar")
    
    '''
    # 3. Train CIFAR-8 (e.g. classes 0–7) Finetune CIFAR-8 → CIFAR-10 (add classes 8 and 9)
    '''
    train_new_model(num_epoch=40,
                    classes=list(range(8)), 
                    #load_path="./checkpoints/CIFAR-8_baseline/resnet20x4_v0.pth.tar",
                    save_path="./checkpoints/CIFAR-8_baseline/2/resnet20x4_v0.pth.tar")
    
    align_pretrained_model(
        pretrained_path="./checkpoints/cifar8F2/zipit/resnet20x4_v0.pth.tar",
        save_path="./checkpoints/cifar8F2/zipit/resnet20x4_v0.pth.tar",
        new_classes=[0,1,2,3,4,5,6,7],
    )
    '''
    finetune_model(
        pretrained_model_path="./checkpoints/CIFAR-8_baseline/2/resnet20x4_v0.pth.tar", 
        new_classes=[8,9],
        epochs=20,
        save_path="./checkpoints/cifar8F2/baseline-f/resnet20x4_v0.pth.tar")
    

    
