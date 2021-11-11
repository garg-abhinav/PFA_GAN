import logging
import os
import numpy as np
import torch
import torchvision
import collections
from torch import optim
from tqdm import tqdm
import math
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from src import utils
from src.models import AgeEstimationNetwork, Generator, Discriminator
import config.config as exp_config


class PFA_GAN:
    def __init__(self, device):
        self.log_dir = os.path.join(exp_config.log_root, exp_config.gan_dir)
        self.global_step = 0
        self.device = device

        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        logging.info(f'Using device {self.device}')

        # generator
        generator = Generator(age_group=exp_config.age_group)
        generator.apply(utils.weights_init)

        # discriminator
        discriminator = Discriminator(age_group=exp_config.age_group)

        # vgg face
        state_dict = torch.load(os.path.join(exp_config.log_root, exp_config.vgg_face_model))
        vgg_face = torchvision.models.vgg16(num_classes=2622)
        vgg_face.features = torch.nn.Sequential(collections.OrderedDict(
            zip(['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2',
                 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1',
                 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2',
                 'relu5_2', 'conv5_3', 'relu5_3', 'pool5'], vgg_face.features)))
        vgg_face.classifier = torch.nn.Sequential(collections.OrderedDict(
            zip(['fc6', 'relu6', 'drop6', 'fc7', 'relu7', 'drop7', 'fc8'], vgg_face.classifier)))
        new_state_dict = {l: torch.from_numpy(np.array(v)).view_as(p) for k, v in state_dict.items() for l, p in
                          vgg_face.named_parameters() if k in l}
        vgg_face.load_state_dict(new_state_dict)
        vgg_face = vgg_face.features[:23]
        vgg_face.eval()
        logging.info('VGG Face model loaded.')

        # age estimation network
        age_classifier = AgeEstimationNetwork(age_group=exp_config.age_group)
        age_classifier, step = utils.get_latest_checkpoint(age_classifier,
                                                           os.path.join(exp_config.log_root, exp_config.age_dir),
                                                           'age_estimator', device)
        age_classifier.eval()
        logging.info(f'Age Estimator Model loaded from step {step}')

        d_optim = torch.optim.Adam(discriminator.parameters(), exp_config.lr, betas=(exp_config.beta1, exp_config.beta2))
        g_optim = torch.optim.Adam(generator.parameters(), exp_config.lr, betas=(exp_config.beta1, exp_config.beta2))

        if os.path.exists(self.log_dir):
            generator, self.global_step = utils.get_latest_checkpoint(generator, self.log_dir, 'generator', device)
            discriminator, self.global_step = utils.get_latest_checkpoint(discriminator, self.log_dir, 'discriminator', device)
            d_optim, self.global_step = utils.get_latest_checkpoint(d_optim, self.log_dir, 'd_optim', device)
            g_optim, self.global_step = utils.get_latest_checkpoint(g_optim, self.log_dir, 'g_optim', device)
            logging.info(f'Generator, Discriminator and Optimizers loaded from step {self.global_step}')

        self.generator = generator.to(device=self.device)
        self.discriminator = discriminator.to(device=self.device)
        self.vgg_face = vgg_face.to(device=self.device)
        self.age_classifier = age_classifier.to(device=self.device)
        self.d_optim = d_optim
        self.g_optim = g_optim

        self.trainable_modules = [self.generator, self.discriminator, self.d_optim, self.g_optim]