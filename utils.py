import sys
import os
import torch
import torch.nn as nn
import numpy as np
import tqdm

import torch
import torch.nn as nn
import random as rnd


class Logger(object):
    def __init__(self, fpath=None, mode='w'):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rnd.seed(10)


def get_y_p(g, n_places):
    y = g // n_places
    p = g % n_places
    return y, p


def update_dict(acc_groups, y, g, logits):
    preds = torch.argmax(logits, axis=1)
    correct_batch = (preds == y)
    g = g.cpu()
    for g_val in np.unique(g):
        mask = g == g_val
        n = mask.sum().item()
        corr = correct_batch[mask].sum().item()
        acc_groups[g_val].update(corr / n, n)


def write_dict_to_tb(writer, dict, prefix, step):
    for key, value in dict.items():
        writer.add_scalar(f"{prefix}{key}", value, step)


def get_results(acc_groups, get_yp_func):
    groups = acc_groups.keys()
    results = {
            f"accuracy_{get_yp_func(g)[0]}_{get_yp_func(g)[1]}": acc_groups[g].avg
            for g in groups
    }
    # all_correct = sum([acc_groups[g].sum for g in groups])
    # all_total = sum([acc_groups[g].count for g in groups])
    # results.update({"mean_accuracy" : all_correct / all_total})
    # results.update({"worst_accuracy" : min(results.values())})
    return results


def evaluate(model, loader, silent=True):
    model.eval()
    minority_acc = AverageMeter()
    majority_acc = AverageMeter()
    avg_acc = AverageMeter()

    with torch.no_grad():
        for x, y, g, p, idxs in tqdm.tqdm(loader, disable=silent):
            x, y, p = x.cuda(), y.cuda(), p.cuda()
            logits = model(x)

            preds = torch.argmax(logits, axis=1)
            correct_batch = (preds == y)
            # Update average
            avg_acc.update(correct_batch.sum().item() / len(y), len(y))
            # Update minority
            mask = y != p
            n = mask.sum().item()
            if n != 0:
                corr = correct_batch[mask].sum().item()
                minority_acc.update(corr / n, n)

            # Update majority
            mask = y == p
            n = mask.sum().item()
            if n != 0:
                corr = correct_batch[mask].sum().item()
                majority_acc.update(corr / n, n)
    model.train()
    return minority_acc.avg, majority_acc.avg, avg_acc.avg


class MultiTaskHead(nn.Module):
    def __init__(self, n_features, n_classes_list):
        super(MultiTaskHead, self).__init__()
        self.fc_list = [
            nn.Linear(n_features, n_classes).cuda()
            for n_classes in n_classes_list
        ]

    def forward(self, x):
        outputs = []
        for head in self.fc_list:
            out = head(x)
            outputs.append(out)
        return outputs
    
def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def get_embed(m, x):
    x = m.conv1(x)
    x = m.bn1(x)
    x = m.relu(x)
    x = m.maxpool(x)
    x = m.layer1(x)
    x = m.layer2(x)
    x = m.layer3(x)
    x = m.layer4(x)
    x = m.avgpool(x)
    x = torch.flatten(x, 1)
    return x

def permute_dims(z):
    assert z.dim() == 2
    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1024, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)
    return torch.cat(perm_z, 1)

def feature_reg_loss_specific(model, x, y, p):
    features = get_embed(model, x)
    reg_loss = 0
    # Features for y=0
    reg_loss += torch.norm(features[torch.logical_and((y == 0), (p == 0))][:, :512].mean(dim=0) -
                           features[torch.logical_and((y == 0), (p == 1))][:, :512].mean(dim=0))
    # Features for y=1
    reg_loss += torch.norm(features[torch.logical_and((y == 1), (p == 0))][:, 400:800].mean(dim=0) -
                           features[torch.logical_and((y == 1), (p == 1))][:, 400:800].mean(dim=0))
    # Features for p=0
    reg_loss += torch.norm(features[torch.logical_and((y == 0), (p == 0))][:, 800:1200].mean(dim=0) -
                           features[torch.logical_and((y == 1), (p == 0))][:, 800:1200].mean(dim=0))
    # Features for p=1
    reg_loss += torch.norm(features[torch.logical_and((y == 1), (p == 1))][:, 1200:1600].mean(dim=0) -
                           features[torch.logical_and((y == 0), (p == 1))][:, 1200:1600].mean(dim=0))
    return reg_loss

def contrastive_loss(model, x, y, p, temperature, method):
    # Currently implemented basic contrastive loss
    features = get_embed(model, x)
    if method in [2, 3]:
        # Contrast tenth
        if method == 2:
            dividing_point = features.size()[1]//10 * 9
        # Contrast hundreth
        elif method == 3:
            dividing_point = features.size()[1]//100 * 99

        # Features to contrast for BG
        features_inverse = features[:, dividing_point:]
        features_inverse = torch.nn.functional.normalize(features_inverse, dim=1)
        features_inverse_transpose = features_inverse.T
        inverse_similarity_matrix = torch.exp(torch.div(torch.matmul(features_inverse, features_inverse_transpose), temperature))

        # Features to contrast for FG
        features = features[:, :dividing_point]

    features = torch.nn.functional.normalize(features, dim=1)
    features_transpose = features.T
    similarity_matrix = torch.exp(torch.div(torch.matmul(features, features_transpose), temperature))
    loss = 0
    count = 0
    # Anchor
    for anchor_idx in range(len(y)):
        anchor_label = y[anchor_idx]
        anchor_background = p[anchor_idx]

        # Positive
        positive_label = anchor_label
        positive_background = 1 - anchor_background
        positive_idxs = torch.where((y==positive_label) & (p==positive_background))[0]

        # Negative
        negative_label = 1 - anchor_label
        negative_background = anchor_background
        negative_idxs = torch.where((y==negative_label) & (p==negative_background))[0]

        # A positive pair has the same label but different background
        for positive_idx in positive_idxs:
            loss += -torch.log(similarity_matrix[anchor_idx][positive_idx]/(similarity_matrix[anchor_idx][positive_idx] + torch.sum(similarity_matrix[anchor_idx][negative_idxs])))
            count += 1
            
        if method in [2, 3]:
            # Positive
            positive_label = 1 - anchor_label
            positive_background = anchor_background
            positive_idxs = torch.where((y == positive_label) & (p == positive_background))[0]

            # Negative
            negative_label = anchor_label
            negative_background = 1 - anchor_background
            negative_idxs = torch.where((y == negative_label) & (p == negative_background))[0]

            # A positive pair has the same label but different background
            for positive_idx in positive_idxs:
                loss += -torch.log(inverse_similarity_matrix[anchor_idx][positive_idx] / (
                            inverse_similarity_matrix[anchor_idx][positive_idx] + torch.sum(
                        inverse_similarity_matrix[anchor_idx][negative_idxs])))
                count += 1
    if count == 0:
        return 0
    loss = loss / count
    # print(f"Loss: {loss}")
    return loss

    # reg_loss = 0
    # # Features for y
    # reg_loss += torch.norm(features[torch.logical_and((y == 0), (p == 0))][:, :1024].mean(dim=0) -
    #                        features[torch.logical_and((y == 0), (p == 1))][:, :1024].mean(dim=0))
    # reg_loss += torch.norm(features[torch.logical_and((y == 1), (p == 0))][:, :1024].mean(dim=0) -
    #                        features[torch.logical_and((y == 1), (p == 1))][:, :1024].mean(dim=0))
    # # Features for p
    # reg_loss += torch.norm(features[torch.logical_and((y == 0), (p == 0))][:, 1024:].mean(dim=0) -
    #                        features[torch.logical_and((y == 1), (p == 0))][:, 1024:].mean(dim=0))
    # reg_loss += torch.norm(features[torch.logical_and((y == 1), (p == 1))][:, 1024:].mean(dim=0) -
    #                        features[torch.logical_and((y == 0), (p == 1))][:, 1024:].mean(dim=0))
    # return reg_loss

def coral_loss(model, x_s, x_t, y_s, y_t, method):
    if method == 4:
        d_s = get_embed(model, x_s)
        c_s = torch.cov(d_s.T)
        d_t = get_embed(model, x_t)
        c_t = torch.cov(d_t.T)

        loss = torch.norm(torch.square(c_s-c_t), p='fro') / 10
    else:
        d_s = get_embed(model, x_s[torch.where(y_s==0)])
        c_s = torch.cov(d_s.T)
        d_t = get_embed(model, x_t[torch.where(y_t==0)])
        c_t = torch.cov(d_t.T)
        loss = torch.norm(torch.square(c_s-c_t), p='fro') / 10
        d_s = get_embed(model, x_s[torch.where(y_s==1)])
        c_s = torch.cov(d_s.T)
        d_t = get_embed(model, x_t[torch.where(y_t==1)])
        c_t = torch.cov(d_t.T)
        loss += torch.norm(torch.square(c_s-c_t), p='fro') / 10
        if torch.isnan(loss):
            return 0
    return loss


def retain_feature_loss(model, x, prev_features, coef):
    features = get_embed(model, x)
    retain_loss = torch.dot(torch.square(prev_features - features).mean(dim=0), coef)
    return retain_loss

def l2_norm(model, weight_decay):
    loss = 0
    norm = sum(p.pow(2.0).sum()
                for p in model.parameters())
    loss = loss + weight_decay * norm
    return loss

def l1_norm(model, weight_decay):
    loss = 0
    norm = sum(p.abs().sum()
                for p in model.parameters()) / 30
    loss = loss + weight_decay * norm
    return loss

def increasing_l1_norm(model, weight_decay):
    decays = np.repeat(np.linspace(args.weight_decay/10, args.weight_decay*10, num=54), repeats=3, axis=0)
    norm = sum(p.abs().sum() * decay
                    for p, decay in zip(model.parameters(), decays)) / 30
    loss = loss + norm
    return loss

def random_batch(dataset, bs):
    size = len(dataset)

# else:
#     if args.increasing_decay:

#     else:
#         norm = sum(p.abs().sum()
#                       for p in model.parameters()) / 30
#         loss = loss + args.weight_decay * norm


# Discriminator model to discriminate between predicted and independent distributions.
class Discriminator(nn.Module):
    def __init__(self, z_dim):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(z_dim, 1000),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Linear(1000, 1000),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Linear(1000, 1000),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Linear(1000, 1000),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Linear(1000, 1000),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Linear(1000, 2),
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                normal_init(m)

    def forward(self, z):
        return self.net(z).squeeze()

"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def correlation_loss(model, x):
    # features: bs, dim
    features = get_embed(model, x)
    # features: dim, bs
    features =  features.T
    # print(features.size())
    # correltaion_matrix: dim, dim
    # i-jth entry in the correlation matrix represents the correlation coefficient between the i-th and j-th features
    correlation_matrix = torch.corrcoef(features)
    # We want to split the feature dimension into two and penalize the correlations between the any pairwise correlations
    # between the first and second subsets
    # loss_mask: dim, dim
    # correlation_matrix = torch.nan_to_num(correlation_matrix)
    # print(correlation_matrix)
    # print(len(loss_mask)//2)
    with torch.no_grad():
        loss_mask = torch.zeros_like(correlation_matrix)
        mid_point = len(loss_mask) // 2
        loss_mask[:mid_point, mid_point:] = 1
        loss_mask[mid_point:, :mid_point] = 1
        loss_mask[torch.isnan(correlation_matrix)] = 0
    # print(loss_mask)
    # print(loss_mask * correlation_matrix)
    if len(correlation_matrix[loss_mask==1]) == 0:
        return 0
    correlation_loss = torch.sum(torch.abs(correlation_matrix[loss_mask==1])) / torch.sum(loss_mask)
    # print(torch.abs(loss_mask * correlation_matrix))
    # print(correlation_loss)
    return correlation_loss


def MTL_Loss():
    pass
