import torch.nn.functional as F
import torch
import numpy as np


def age2group(age, age_group):
    if isinstance(age, np.ndarray):
        groups = np.zeros_like(age)
    else:
        groups = torch.zeros_like(age).to(age.device)
    if age_group == 4:
        section = [30, 40, 50]
    elif age_group == 5:
        section = [20, 30, 40, 50]
    elif age_group == 7:
        section = [10, 20, 30, 40, 50, 60]
    else:
        raise NotImplementedError
    for i, thresh in enumerate(section, 1):
        groups[age > thresh] = i
    return groups


def age_criterion(age_logit, group_logit, gt_age, age_group):
    preds = F.softmax(age_logit, dim=1)
    pred_age = torch.sum(preds * torch.arange(preds.size(1)).to(preds.device), dim=1)
    reg_loss = F.mse_loss(pred_age, gt_age)
    class_loss = F.cross_entropy(group_logit, age2group(gt_age, age_group).long())
    return reg_loss + class_loss