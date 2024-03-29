import logging
import os
import numpy as np
import pickle
import torch
from torch import optim
from tqdm import tqdm
import math
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from src import utils
from src.models import AgeEstimationNetwork
from src.dataset import AgeDataset
import config.config as exp_config


def train_net(net, device, global_step=0):

    # image_urls = pickle.load(open(os.path.join(exp_config.data_root, exp_config.image_urls), 'rb'))#[:200]
    # image_ages = pickle.load(open(os.path.join(exp_config.data_root, exp_config.image_ages), 'rb'))#[:200]

    train_data = AgeDataset(do_transforms=True)
    n_train = len(train_data)

    logging.info(f'Data loaded having {n_train} images')

    train_loader = DataLoader(train_data, batch_size=exp_config.age_batch_size,
                              shuffle=True, num_workers=exp_config.n_workers, pin_memory=True)
    print(train_loader.batch_sampler)
    writer = SummaryWriter(comment='AgeEstimator_LR_{}_BS_{}'.format(exp_config.age_lr, exp_config.age_batch_size))

    optimizer = optim.Adam(net.parameters(), lr=exp_config.age_lr, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=exp_config.age_lr_decay_steps,
                                          gamma=exp_config.age_lr_decay_rate)
    criterion = utils.age_criterion
    max_epochs = exp_config.age_max_epochs - global_step//math.ceil(n_train/exp_config.age_batch_size)
    logging.info(f'''Starting training from step {global_step}:
            Epochs:          {max_epochs}
            Batch size:      {exp_config.age_batch_size}
            Learning rate:   {exp_config.age_lr}
            Training size:   {n_train}
            Device:          {device.type}
        ''')
    net.train()
    for epoch in range(max_epochs):
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{max_epochs}', unit='img') as pbar:
            for batch in train_loader:
                gt_age = batch[1]
                imgs = batch[0]

                imgs = imgs.to(device=device, dtype=torch.float32)
                gt_age = gt_age.to(device=device, dtype=torch.float32)

                age_logit, group_logit = net(imgs)
                loss = criterion(age_logit, group_logit, gt_age, exp_config.age_group)
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss(batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

            for tag, value in net.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

        scheduler.step()

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
            logging.info('Created checkpoint directory')

        utils.save_checkoint(net, log_dir, 'age_estimator', global_step)

    writer.close()


if __name__ == '__main__':
    log_dir = os.path.join(exp_config.log_root, exp_config.age_dir)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = AgeEstimationNetwork(age_group=exp_config.age_group)
    logging.info(f'Age Estimation Network with {exp_config.age_group} age groups (classes)')

    global_step = 0
    if os.path.exists(log_dir):
        net, global_step = utils.get_latest_checkpoint(net, log_dir, 'age_estimator', device)
        logging.info(f'Age Estimator Model loaded from step {global_step}')

    net.to(device=device)

    train_net(net=net, device=device, global_step=global_step)