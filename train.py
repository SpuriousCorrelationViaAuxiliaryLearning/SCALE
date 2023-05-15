import torch
import torchvision
import time
import numpy as np
import os
import tqdm
import argparse
import sys
import json
from functools import partial
from data.wb_data import WaterBirdsDataset, get_loader, wb_transform, log_data, get_waterbirds
from AuxiliaryOptimizer import AuxiliaryOptimizer
from utils import Logger, AverageMeter, set_seed, evaluate, get_y_p
from data.mcdominoes import get_mcdominoes
from torch.utils.data import Dataset, DataLoader
from data.celeba import get_celeba


import pickle


def parse_args():
    parser = argparse.ArgumentParser(description="Train model on dataset")

    # Data Args
    parser.add_argument("--dataset", type=str, default="cmnist",
                        help="Which dataset to use: [cmnist, mcdominoes]")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory where data is located")
    parser.add_argument("--val_target_size", type=int, default=1000, help="Size of validation+target dataset")
    parser.add_argument("--spurious_strength", type=float, default=1, help="Strength of spurious correlation")

    # Output Directory
    parser.add_argument(
        "--output_dir", type=str,
        default="logs/",
        help="Output directory")

    # Model Args
    parser.add_argument("--pretrained_model", action='store_true', help="Use pretrained model")

    # Training Args
    parser.add_argument("--scheduler", action='store_true', help="Learning rate scheduler")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum_decay", type=float, default=0.9)
    parser.add_argument("--init_lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)

    # Method Args
    parser.add_argument("--method", type=int, default=0, help="Which method to use")
    # Method 0: Normal ERM
    # Method 1: Only balanced dataset
    # Method 2: Balanced Optimizer

    # Additional Method 2 Args
    parser.add_argument("--group_size", type=int, default=64, help="Number kernels per group")
    parser.add_argument("--regularize_mode", type=int, default=0)

    args = parser.parse_args()
    return args


# parameters in config overwrites the parser arguments
def main(args):
    # --- Logger Start ---
    print('Preparing directory %s' % args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'command.sh'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        args_json = json.dumps(vars(args))
        f.write(args_json)
    set_seed(args.seed)
    logger = Logger(os.path.join(args.output_dir, 'log.txt'))
    # --- Logger End ---

    # --- Data Start ---
    indicies = np.arange(args.val_target_size)
    np.random.shuffle(indicies)
    # Explicitly split dataset into train-target-val-test split
    # First half for val
    indicies_val = indicies[:len(indicies) // 2]
    # Second half for target
    indicies_target = indicies[len(indicies) // 2:]

    # Obtain trainset, targetset, and testset_dict
    train_set, target_set, test_set_dict = None, None, None
    if args.dataset == "mcdominoes":
        target_resolution = (64, 32)
        train_set, target_set, test_set_dict = get_mcdominoes(target_resolution, args.val_target_size, args.spurious_strength,
                                                              args.data_dir, args.seed, indicies_val, indicies_target)
    elif args.dataset == "waterbirds":
        target_resolution = (224, 224)
        train_set, target_set, test_set_dict = get_waterbirds(target_resolution, args.val_target_size, args.spurious_strength,
                                                              args.data_dir, args.seed, indicies_val, indicies_target)
    elif args.dataset == "celeba":
        target_resolution = (224, 224)
        train_set, target_set, test_set_dict = get_celeba(target_resolution, args.val_target_size, args.spurious_strength,
                                                              args.data_dir, args.seed)

    num_classes, num_places = test_set_dict["Test"].n_classes, test_set_dict["Test"].n_places
    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 4, 'pin_memory': True}

    # For method 1, the training dataset is the balanced dataset
    if args.method == 1:
        train_loader = DataLoader(target_set, shuffle=True, **loader_kwargs)
    else:
        train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)

    test_loader_dict = {}
    for test_name, testset_v in test_set_dict.items():
        test_loader_dict[test_name] = DataLoader(testset_v, shuffle=False, **loader_kwargs)

    get_yp_func = partial(get_y_p, n_places=target_set.n_places)
    log_data(logger, train_set, target_set, test_set_dict['Validation'], test_set_dict['Test'], get_yp_func=get_yp_func)
    # --- Data End ---

    # --- Model Start ---
    if args.dataset == "mcdominoes":
        model = torchvision.models.resnet18(pretrained=args.pretrained_model)
    else:
        model = torchvision.models.resnet50(pretrained=args.pretrained_model)
    d = model.fc.in_features
    model.fc = torch.nn.Linear(d, num_classes)
    model.cuda()
    
    if args.method == 2:
        # Separate optimizers for shared parameters (FE) and non-shared parameters (head)
        model.fc = None
        optimizer = AuxiliaryOptimizer(model.parameters(), mode=args.regularize_mode, group_size=args.group_size,
                                       lr=args.init_lr, momentum=args.momentum_decay, weight_decay=args.weight_decay)
        # Create a separate linear layer for the main dataset
        fc_train = torch.nn.Linear(d, num_classes)
        fc_train.cuda()
        fc_target = torch.nn.Linear(d, num_classes)
        fc_target.cuda()
        fc_params = list(fc_train.parameters()) + list(fc_target.parameters())
        fc_optimizer = torch.optim.SGD(
            fc_params, lr=args.init_lr, momentum=args.momentum_decay, weight_decay=args.weight_decay)
        average_cosine_similarity = []
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.init_lr, momentum=args.momentum_decay, weight_decay=args.weight_decay)

    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs)
    else:
        scheduler = None

    criterion = torch.nn.CrossEntropyLoss()
    # --- Model End ---

    # --- Train Start ---
    best_worst_acc = 0

    for epoch in range(args.num_epochs):
        if args.method == 2:
            similarity_meter = AverageMeter()
        model.train()
        # Track metrics
        loss_meter = AverageMeter()
        method_loss_meter = AverageMeter()
        start = time.time()
        for batch in tqdm.tqdm(train_loader, disable=True):
            # Data
            x, y, g, p, idxs = batch
            x, y, p = x.cuda(), y.cuda(), p.cuda()
            optimizer.zero_grad()

            # --- Methods Start ---
            if args.method == 2:
                fc_optimizer.zero_grad()
                # Swap classification head before forward pass
                model.fc = fc_train

            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            # Normal ERM
            if args.method == 0 or args.method == 1:
                method_loss = 0

            # Methods 2
            else:
                # Store the gradients for the train set
                optimizer.store_gradients("train")
                optimizer.zero_grad()

                # Obtain batch of target data
                random_indices = np.random.choice(len(target_set), args.batch_size, replace=False)
                x_b, y_b, _, _, _ = target_set.__getbatch__(random_indices)
                x_b, y_b = x_b.cuda(), y_b.type(torch.LongTensor).cuda()

                # Swap classification head
                model.fc = fc_target
                # Forwards & backwards pass of target data
                logits_b = model(x_b)
                method_loss = criterion(logits_b, y_b)
                method_loss.backward()

                # Store balanced gradients
                optimizer.store_gradients("balanced")
                # Update classification head
                fc_optimizer.step()

            # Update main model
            opt_out = optimizer.step()
            if args.method == 2:
                similarity_meter.update(opt_out[1], x.size(0))
            method_loss_meter.update(method_loss, x.size(0))
            loss_meter.update(loss, x.size(0))
            # --- Methods Ends ---

        if args.scheduler:
            scheduler.step()

        # Save results

        # Evaluation
        # Iterating over datasets we test on
        for test_name, test_loader in test_loader_dict.items():
            minority_acc, majority_acc, avg_acc = evaluate(model, test_loader)
            logger.write(f"Minority {test_name} accuracy: {minority_acc:.3f}\t")
            logger.write(f"Average {test_name} accuracy: {avg_acc:.3f}\t")
            logger.write(f"Majority {test_name} accuracy: {majority_acc:.3f}\n")

        # Save best model based on worst group accuracy
        if minority_acc > best_worst_acc:
            torch.save(
                model.state_dict(), os.path.join(args.output_dir, 'best_checkpoint.pt'))
            best_worst_acc = minority_acc
        logger.write(f"Epoch {epoch}\t ERM Loss: {loss_meter.avg:.3f}\t Method Loss: {method_loss_meter.avg:.3f}\t Time Taken: {time.time()-start:.3f}\n")
        if args.method == 2:
            logger.write(f"Average cosine similarity: {similarity_meter.avg:.10f}")
        logger.write('\n')

    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_checkpoint.pt'))
    # --- Train End ---

    logger.write(f'Best validation worst-group accuracy: {best_worst_acc}')
    logger.write('\n')


if __name__ == "__main__":
    args = parse_args()
    main(args=args)
