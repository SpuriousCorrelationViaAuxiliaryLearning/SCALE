import os
import math
import json
import random as rnd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
import pandas as  pd
import torchvision.utils as vision_utils
from PIL import Image
import torchvision
from torchvision import transforms
import pickle

class MCDOMINOES(Dataset):
    def __init__(self, x, y, p, target_resolution, isTrain):
        scale = 28.0 / 32.0
        self.x = x
        self.y_array = np.array(y)
        self.p_array = np.array(p)
        self.isTrain = isTrain
        self.confounder_array = np.array(p)
        self.n_classes = np.unique(self.y_array).size
        self.n_places = np.unique(self.confounder_array).size
        self.group_array = (self.y_array * self.n_places + self.confounder_array).astype('int')
        self.n_groups = self.n_classes * self.n_places
        self.group_counts = (
                torch.arange(self.n_groups).unsqueeze(1) == torch.from_numpy(self.group_array)).sum(1).float()
        self.y_counts = (
                torch.arange(self.n_classes).unsqueeze(1) == torch.from_numpy(self.y_array)).sum(1).float()
        self.p_counts = (
                torch.arange(self.n_places).unsqueeze(1) == torch.from_numpy(self.p_array)).sum(1).float()
        
        if isTrain:
            self.transform = transforms.Compose([
                transforms.RandomCrop(target_resolution, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
        else:
            self.transform = transforms.Compose([
                transforms.CenterCrop(target_resolution),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        p = self.confounder_array[idx]
        img = self.x[idx]
        img = self.transform(img)

        return img, y, g, p, idx

    def __getbatch__(self, idxs):
        x_batch, y_batch, g_batch, p_batch, idx_batch = [], [], [], [], []
        for idx in idxs:
            x, y, g, p, idx = self.__getitem__(idx)
            x_batch.append(x)
            y_batch.append(y)
            g_batch.append(g)
            p_batch.append(p)
            idx_batch.append(idx)
        return torch.stack(x_batch), torch.flatten(torch.Tensor(np.vstack(y_batch))), torch.flatten(
          torch.Tensor(np.vstack(g_batch))), \
               torch.flatten(torch.Tensor(np.vstack(p_batch))), torch.flatten(torch.Tensor(np.vstack(idx_batch)))



def format_mnist(imgs):
    imgs = np.stack([np.pad(imgs[i][0], 2, constant_values=0)[None,:] for i in range(len(imgs))])
    imgs = np.repeat(imgs, 3, axis=1)
    return torch.tensor(imgs)


def plot_samples(dataset, nrow=13, figsize=(10,7)):
    try:
        X, Y = dataset.tensors
    except:
        try:
            (X,) = dataset.tensors
        except:
            X = dataset
    fig = plt.figure(figsize=figsize, dpi=130)
    grid_img = vision_utils.make_grid(X[:nrow].cpu(), nrow=nrow, normalize=True, padding=1)
    _ = plt.imshow(grid_img.permute(1, 2, 0), interpolation='nearest')
    _ = plt.tick_params(axis=u'both', which=u'both',length=0)
    ax = plt.gca()
    _ = ax.xaxis.set_major_formatter(NullFormatter())
    _ = ax.yaxis.set_major_formatter(NullFormatter())
    plt.show()


def get_mcdominoes(target_resolution, VAL_SIZE, spurious_strength, data_dir, seed, indicies_val=None, indicies_target=None):
    save_dir = os.path.join(data_dir, f"mcdominoes_{spurious_strength}-{VAL_SIZE}-{seed}.pkl")
    if os.path.exists(save_dir):
        print("Loading Dataset")
        with open(save_dir, 'rb') as f:
            train_dataset, balanced_dataset, testset_dict = pickle.load(f)
        return train_dataset, balanced_dataset, testset_dict
    print("Generating Dataset")
    # Load mnist train
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    mnist_train_set = torchvision.datasets.MNIST(os.path.join(data_dir, "mnist"), train=True, download=True)
    mnist_train_input = mnist_train_set.data.view(-1, 1, 28, 28).float()/255.0
    mnist_train_target = mnist_train_set.targets
    rand_perm = torch.randperm(len(mnist_train_input))
    mnist_train_input = mnist_train_input[rand_perm]
    mnist_train_target = mnist_train_target[rand_perm]

    # Load mnist test
    mnist_test_set = torchvision.datasets.MNIST(os.path.join(data_dir, "mnist"), train=False, download=True)
    mnist_test_input = mnist_test_set.data.view(-1, 1, 28, 28).float()/255.0
    mnist_test_target = mnist_test_set.targets

    # Load cifar train
    cifar_train_set = torchvision.datasets.CIFAR10(os.path.join(data_dir, "cifar10"), train=True, download=True, transform=transform)
    cifar_train_input = []
    cifar_train_target = []
    for x, y in cifar_train_set:
        cifar_train_input.append(x)
        cifar_train_target.append(y)
    cifar_train_input = torch.stack(cifar_train_input)
    cifar_train_target = torch.Tensor(cifar_train_target)
    rand_perm = torch.randperm(len(cifar_train_input))
    cifar_train_input = cifar_train_input[rand_perm]
    cifar_train_target = cifar_train_target[rand_perm]

    # Load cifar test
    cifar_test_set = torchvision.datasets.CIFAR10(os.path.join(data_dir, "cifar10"), train=False, download=True, transform=transform)
    cifar_test_input = []
    cifar_test_target = []
    for x, y in cifar_test_set:
      cifar_test_input.append(x)
      cifar_test_target.append(y)
    cifar_test_input = torch.stack(cifar_test_input)
    cifar_test_target = torch.Tensor(cifar_test_target)

    mnist_train_input_new, mnist_train_target_new = [], []
    mnist_val_input, mnist_val_target = [], []
    mnist_test_input_new, mnist_test_target_new = [], []
    cifar_train_input_new, cifar_train_target_new = [], []
    cifar_val_input, cifar_val_target = [], []
    cifar_test_input_new, cifar_test_target_new = [], []

    # Train validation split spurious_strength
    for i in range(10):
        # For train and validation
        mnist_class_input = mnist_train_input[torch.where(mnist_train_target==i)]
        mnist_val_input.append(mnist_class_input[:int(VAL_SIZE/10)])
        mnist_val_target.extend([i]*int(VAL_SIZE/10))
        mnist_train_input_new.append(mnist_class_input[int(VAL_SIZE/10):5000])
        mnist_train_target_new.extend([i]*(5000-int(VAL_SIZE/10)))
        cifar_class_input = cifar_train_input[torch.where(cifar_train_target==i)]
        cifar_val_input.append(cifar_class_input[:int(VAL_SIZE/10)])
        cifar_val_target.extend([i]*int(VAL_SIZE/10))
        cifar_train_input_new.append(cifar_class_input[int(VAL_SIZE/10):5000])
        cifar_train_target_new.extend([i]*(5000-int(VAL_SIZE/10)))
        # For test
        mnist_class_input = mnist_test_input[torch.where(mnist_test_target==i)]
        mnist_test_input_new.append(mnist_class_input)
        mnist_test_target_new.extend([i]*1000)

        cifar_class_input = cifar_test_input[torch.where(cifar_test_target==i)]
        cifar_test_input_new.append(cifar_class_input)
        cifar_test_target_new.extend([i]*1000)

    mnist_train_input = format_mnist(torch.cat(mnist_train_input_new, dim=0))
    mnist_train_target = torch.tensor(mnist_train_target_new)
    mnist_val_input = format_mnist(torch.cat(mnist_val_input, dim=0))
    mnist_val_target = torch.tensor(mnist_val_target)
    mnist_test_input = format_mnist(torch.cat(mnist_test_input_new, dim=0))
    mnist_test_target = torch.tensor(mnist_test_target_new)

    cifar_train_input = torch.cat(cifar_train_input_new, dim=0)
    cifar_train_target = torch.tensor(cifar_train_target_new)
    cifar_val_input = torch.cat(cifar_val_input, dim=0)
    cifar_val_target = torch.tensor(cifar_val_target)
    cifar_test_input = torch.cat(cifar_test_input_new, dim=0)
    cifar_test_target = torch.tensor(cifar_test_target_new)
    # For train, shuffle fraction of dataset
    if spurious_strength != 1:
        fraction_permute = 1 - spurious_strength
        permute_indicies = rnd.sample(list(np.arange(len(cifar_train_input))), int(fraction_permute*len(cifar_train_input)))
        shuffled_indicies = rnd.sample(permute_indicies, len(permute_indicies))
        cifar_train_input[permute_indicies] = torch.clone(cifar_train_input[shuffled_indicies])
        cifar_train_target[permute_indicies] = torch.clone(cifar_train_target[shuffled_indicies])

    X_train = torch.cat((mnist_train_input, cifar_train_input), dim=2)

    # Then shuffle all
    P_train = mnist_train_target
    Y_train = cifar_train_target
    rand_perm = torch.randperm(len(X_train))
    X_train = X_train[rand_perm]
    P_train = P_train[rand_perm]
    Y_train = Y_train[rand_perm]

    # For validation and test, shuffle then concatenate
    rand_perm = torch.randperm(len(mnist_val_input))
    mnist_val_input = mnist_val_input[rand_perm]
    mnist_val_target = mnist_val_target[rand_perm]
    rand_perm = torch.randperm(len(cifar_val_input))
    cifar_val_input = cifar_val_input[rand_perm]
    cifar_val_target = cifar_val_target[rand_perm]
    X_val = torch.cat((mnist_val_input, cifar_val_input), dim=2)
    P_val = mnist_val_target
    Y_val = cifar_val_target

    X_target, P_target, Y_target = X_val[indicies_target], P_val[indicies_target], Y_val[indicies_target]
    X_val, P_val, Y_val = X_val[indicies_val], P_val[indicies_val], Y_val[indicies_val]

    rand_perm = torch.randperm(len(mnist_test_input))
    mnist_test_input = mnist_test_input[rand_perm]
    mnist_test_target = mnist_test_target[rand_perm]
    rand_perm = torch.randperm(len(cifar_test_input))
    cifar_test_input = cifar_test_input[rand_perm]
    cifar_test_target = cifar_test_target[rand_perm]

    X_test = torch.cat((mnist_test_input, cifar_test_input), dim=2)
    P_test = mnist_test_target
    Y_test = cifar_test_target

    train_dataset = MCDOMINOES(X_train, Y_train, P_train, target_resolution, True)
    val_dataset = MCDOMINOES(X_val, Y_val, P_val, target_resolution, False)
    test_dataset = MCDOMINOES(X_test, Y_test, P_test, target_resolution, False)
    testset_dict = {
        'Test': test_dataset,
        'Validation': val_dataset,
    }
    balanced_dataset = MCDOMINOES(X_target, Y_target, P_target, target_resolution, True)
    with open(save_dir, 'wb') as f:
        pickle.dump((train_dataset, balanced_dataset, testset_dict), f)
    return train_dataset, balanced_dataset, testset_dict
