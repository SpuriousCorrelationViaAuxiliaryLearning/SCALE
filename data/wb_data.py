import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler


def get_waterbirds(target_resolution, val_size, spurious_strength, data_dir, seed, indicies_val=None, indicies_target=None):
    save_dir = os.path.join(data_dir, f"waterbirds_{spurious_strength}-{val_size}-{seed}.pkl")
    if os.path.exists(save_dir):
        print("Loading Dataset")
        with open(save_dir, 'rb') as f:
            train_set, target_set, test_set_dict = pickle.load(f)
    else:
        train_transform = wb_transform(target_resolution=target_resolution, train=True, augment_data=True)
        test_transform = wb_transform(target_resolution=target_resolution, train=False, augment_data=False)
        train_set = WaterBirdsDataset(basedir=data_dir, split="train", transform=train_transform)
        target_set = WaterBirdsDataset(basedir=data_dir, split="val", transform=train_transform,
                                       indicies=indicies_target)
        test_set_dict = {
            'Test': WaterBirdsDataset(basedir=data_dir, split="test", transform=test_transform),
            'Validation': WaterBirdsDataset(basedir=data_dir, split="val", transform=test_transform,
                                            indicies=indicies_val),
        }
        if spurious_strength == 1:
            group_counts = train_set.group_counts
            minority_groups = np.argsort(group_counts.numpy())[:2]
            idx = np.where(np.logical_and.reduce([train_set.group_array != g for g in minority_groups], initial=True))[
                0]
            train_set.y_array = train_set.y_array[idx]
            train_set.group_array = train_set.group_array[idx]
            train_set.confounder_array = train_set.confounder_array[idx]
            train_set.filename_array = train_set.filename_array[idx]
            train_set.metadata_df = train_set.metadata_df.iloc[idx]
        with open(save_dir, 'wb') as f:
            pickle.dump((train_set, target_set, test_set_dict), f)
    return train_set, target_set, test_set_dict


class WaterBirdsDataset(Dataset):
    def __init__(self, basedir, split="train", transform=None, indicies=None):
        try:
            split_i = ["train", "val", "test"].index(split)
        except ValueError:
            raise(f"Unknown split {split}")
        metadata_df = pd.read_csv(os.path.join(basedir, "metadata.csv"))
        # print(len(metadata_df))
        self.metadata_df = metadata_df[metadata_df["split"] == split_i]
        # if the indicies is specified, we only obtain the datapoints corresponding to the indicies
        if indicies is not None:
            self.metadata_df = self.metadata_df.iloc[indicies]
        # print(len(self.metadata_df))
        self.basedir = basedir
        self.transform = transform
        self.y_array = self.metadata_df['y'].values
        self.p_array = self.metadata_df['place'].values


        # all the variables below are derived from metadata, y, and p
        self.n_classes = np.unique(self.y_array).size
        self.confounder_array = self.metadata_df['place'].values
        self.n_places = np.unique(self.confounder_array).size
        self.group_array = (self.y_array * self.n_places + self.confounder_array).astype('int')
        self.n_groups = self.n_classes * self.n_places
        self.group_counts = (
                torch.arange(self.n_groups).unsqueeze(1) == torch.from_numpy(self.group_array)).sum(1).float()
        self.y_counts = (
                torch.arange(self.n_classes).unsqueeze(1) == torch.from_numpy(self.y_array)).sum(1).float()
        self.p_counts = (
                torch.arange(self.n_places).unsqueeze(1) == torch.from_numpy(self.p_array)).sum(1).float()
        self.filename_array = self.metadata_df['img_filename'].values

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        p = self.confounder_array[idx]

        img_path = os.path.join(self.basedir, self.filename_array[idx])
        img = Image.open(img_path).convert('RGB')

        if self.transform:
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
        return torch.stack(x_batch), torch.flatten(torch.Tensor(np.vstack(y_batch))), torch.flatten(torch.Tensor(np.vstack(g_batch))), \
         torch.flatten(torch.Tensor(np.vstack(p_batch))), torch.flatten(torch.Tensor(np.vstack(idx_batch)))
    
        
def wb_transform(target_resolution, train, augment_data):
    scale = 256.0 / 224.0

    if (not train) or (not augment_data):
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform


def get_loader(data, train, reweight_groups, reweight_classes, reweight_places, **kwargs):
    if not train: # Validation or testing
        assert reweight_groups is None
        assert reweight_classes is None
        assert reweight_places is None
        shuffle = False
        sampler = None
    elif not (reweight_groups or reweight_classes or reweight_places): # Training but not reweighting
        shuffle = True
        sampler = None
    elif reweight_groups:
        # Training and reweighting groups
        # reweighting changes the loss function from the normal ERM (average loss over each training example)
        # to a reweighted ERM (weighted average where each (y,c) group has equal weight)
        group_weights = len(data) / data.group_counts
        weights = group_weights[data.group_array]

        # Replacement needs to be set to True, otherwise we'll run out of minority samples
        sampler = WeightedRandomSampler(weights, len(data), replacement=True)
        shuffle = False
    elif reweight_classes:  # Training and reweighting classes
        class_weights = len(data) / data.y_counts
        weights = class_weights[data.y_array]
        sampler = WeightedRandomSampler(weights, len(data), replacement=True)
        shuffle = False
    else: # Training and reweighting places
        place_weights = len(data) / data.p_counts
        weights = place_weights[data.p_array]
        sampler = WeightedRandomSampler(weights, len(data), replacement=True)
        shuffle = False

    loader = DataLoader(
        data,
        shuffle=shuffle,
        sampler=sampler,
        **kwargs)
    return loader


def log_data(logger, train_data, target_data, val_data, test_data, get_yp_func):
    logger.write(f'Training Data (total {len(train_data)})\n')
    for group_idx in range(train_data.n_groups):
        y_idx, p_idx = get_yp_func(group_idx)
        logger.write(f'    Group {group_idx} (y={y_idx}, p={p_idx}): n = {train_data.group_counts[group_idx]:.0f}\n')

    if target_data is None:
        logger.write(f'Target Data is None')
    else:
        logger.write(f'Target Data (total {len(target_data)})\n')
        for group_idx in range(target_data.n_groups):
            y_idx, p_idx = get_yp_func(group_idx)
            logger.write(f'    Group {group_idx} (y={y_idx}, p={p_idx}): n = {target_data.group_counts[group_idx]:.0f}\n')

    logger.write(f'Test Data (total {len(test_data)})\n')
    for group_idx in range(test_data.n_groups):
        y_idx, p_idx = get_yp_func(group_idx)
        logger.write(f'    Group {group_idx} (y={y_idx}, p={p_idx}): n = {test_data.group_counts[group_idx]:.0f}\n')

    logger.write(f'Validation Data (total {len(val_data)})\n')
    for group_idx in range(val_data.n_groups):
        y_idx, p_idx = get_yp_func(group_idx)
        logger.write(f'    Group {group_idx} (y={y_idx}, p={p_idx}): n = {val_data.group_counts[group_idx]:.0f}\n')
