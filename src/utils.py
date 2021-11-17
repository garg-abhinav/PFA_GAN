import torch.nn.functional as F
import torch
import numpy as np
import os
import glob
import math


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


def save_checkoint(module, log_dir, name, step):
    file = os.path.join(log_dir, f'{name}_step_{step}.pth')
    torch.save(module.state_dict(), file)


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
    class_loss = F.cross_entropy(group_logit, age2group(gt_age, age_group).to(dtype=torch.long))
    return reg_loss + class_loss


def mse_loss(inputs, targets):
    return torch.mean((inputs - targets) ** 2)


def ssim_loss(img1, img2, window_size=10):
    channel = img1.shape[1]
    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return 1.0 - ssim_map.mean()


def create_window(window_size, channel):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * 1.5 ** 2)) for x in range(window_size)])
    window_1d = (gauss / gauss.sum()).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(channel, 1, window_size, window_size).contiguous()
    return window
