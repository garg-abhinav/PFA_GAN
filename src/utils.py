import torch.nn.functional as F
import torch
import numpy as np
import os
import glob


def get_latest_checkpoint(net, log_dir, name, device):
    files = glob.glob(log_dir + '/*.pth')
    checkpoints = []
    for i in files:
        checkpoints.append(int(i.split('/')[-1].split('_')[-1].split('.')[0]))
    if len(checkpoints) == 0:
        return net, 0
    latest_cp = max(checkpoints)
    file = os.path.join(log_dir, f'{name}_step_{latest_cp}.pth')
    net.load_state_dict(torch.load(file, map_location=device))
    print(f'Model loaded from {file}')
    return net, latest_cp


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.zero_()


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
    class_loss = F.cross_entropy(group_logit, torch.tensor(age2group(gt_age, age_group),
                                                           dtype=torch.float32).to(gt_age.device))
    return reg_loss + class_loss