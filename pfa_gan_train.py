import logging
import os
import numpy as np
import torch
import torchvision
import collections
from tqdm import tqdm
import math
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F
from src import utils
from src.models import AgeEstimationNetwork, Generator, Discriminator
from src.dataset import PFADataset
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
        # generator.apply(utils.weights_init)

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
        self.age_classifier.eval()
        self.vgg_face.eval()

        self.trainable_modules = {'generator': self.generator,
                                  'discriminator': self.discriminator,
                                  'd_optim': self.d_optim,
                                  'g_optim': self.g_optim}

    def fit(self):
        train_data = PFADataset(age_group=exp_config.age_group,
                                max_iter=exp_config.max_epochs,
                                batch_size=exp_config.batch_size,
                                source=exp_config.source,
                                do_transforms=True)
        n_train = len(train_data.image_list)

        logging.info(f'Data loaded having {n_train} images')

        train_loader = DataLoader(train_data, batch_size=exp_config.batch_size,
                                  shuffle=True, num_workers=exp_config.n_workers, pin_memory=True)

        writer = SummaryWriter(comment='GAN_LR_{}_BS_{}'.format(exp_config.lr, exp_config.batch_size))

        max_epochs = exp_config.max_epochs - self.global_step // math.ceil(n_train / exp_config.batch_size)

        logging.info(f'''Starting training from step {self.global_step}:
                    Epochs:          {max_epochs}
                    Batch size:      {exp_config.batch_size}
                    Learning rate:   {exp_config.lr}
                    Training size:   {n_train}
                    Device:          {self.device.type}
                ''')

        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
            logging.info('Created checkpoint directory')

        for epoch in range(max_epochs):
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{max_epochs}', unit='img') as pbar:
                for batch in train_loader:
                    source_img, true_img, source_label, target_label, true_label, true_age, mean_age = batch
                    true_img = true_img.to(device=self.device)
                    source_label = source_label.to(device=self.device)
                    target_label = target_label.to(device=self.device)
                    true_label = true_label.to(device=self.device)
                    true_age = true_age.to(device=self.device)
                    mean_age = mean_age.to(device=self.device)
                    self.generator.train()
                    self.discriminator.train()

                    # Train Discriminator
                    d_loss, g_source = self.train_discriminator(source_img, true_img, source_label, target_label, true_label, true_age, mean_age)

                    # Train Generator
                    g_loss, l1_loss, ssim_loss, id_loss, age_loss, total_loss = self.train_generator(source_img, true_img, source_label, target_label, true_label, true_age, mean_age, g_source)

                    pbar.set_postfix(**{"d_loss": d_loss,
                                        "g_loss": g_loss,
                                        "total_loss": total_loss})
                    pbar.update(batch[0].shape[0])  # assuming batch[0] is images

                    writer.add_scalar('d_loss/train', d_loss, self.global_step)
                    writer.add_scalar('g_loss/train', g_loss, self.global_step)
                    writer.add_scalar('l1_loss/train', l1_loss, self.global_step)
                    writer.add_scalar('ssim_loss/train', ssim_loss, self.global_step)
                    writer.add_scalar('id_loss/train', id_loss, self.global_step)
                    writer.add_scalar('age_loss/train', age_loss, self.global_step)
                    writer.add_scalar('total_loss/train', total_loss, self.global_step)
                    self.global_step += 1

                writer.add_scalar('learning_rate', self.g_optim.param_groups[0]['lr'], self.global_step)

            if ((epoch + 1) % exp_config.checkpoint == 0) or (epoch + 1 == max_epochs):
                for name, module in self.trainable_modules.items():
                    utils.save_checkoint(module, self.log_dir, name, self.global_step)
        writer.close()

    def train_discriminator(self, source_img, true_img, source_label, target_label, true_label, true_age, mean_age):
        # source_img, true_img, source_label, target_label, true_label, true_age, mean_age = batch
        g_source = self.generator(source_img, source_label, target_label)

        d1_logit = self.discriminator(true_img, true_label)
        d3_logit = self.discriminator(g_source.detach(), target_label)

        # Discriminator loss
        d_loss = (F.mse_loss(d1_logit, torch.ones_like(d1_logit, device=self.device)) +
                  F.mse_loss(d3_logit, torch.zeros_like(d3_logit, device=self.device))) / 2

        self.d_optim.zero_grad()
        d_loss.backward()
        self.d_optim.step()
        return d_loss, g_source

    def train_generator(self, source_img, true_img, source_label, target_label, true_label, true_age, mean_age, g_source):
        # source_img, true_img, source_label, target_label, true_label, true_age, mean_age = batch
        gan_logit = self.discriminator(g_source, target_label)

        # GAN loss
        g_loss = F.mse_loss(gan_logit, torch.ones_like(gan_logit, device=self.device)) / 2

        # AgeEstimator Loss
        age_logit, group_logit = self.age_classifier(g_source)
        age_loss = utils.age_criterion(age_logit, group_logit, mean_age, exp_config.age_group)

        # L1 loss
        l1_loss = F.l1_loss(g_source, source_img)

        # SSIM loss
        ssim_loss = utils.ssim_loss(g_source, source_img, 10)

        # ID loss
        id_loss = F.mse_loss(self.vgg_face_output(g_source), self.vgg_face_output(source_img))

        # pix_loss_weight = max(opt.pix_loss_weight,
        #                       exp_config.lambda_gan * (opt.decay_pix_factor ** (n_iter // opt.decay_pix_n)))

        total_loss = g_loss * exp_config.lambda_gan + \
                     (l1_loss * (1 - exp_config.alpha_ssim) + ssim_loss * exp_config.alpha_ssim +
                      id_loss * exp_config.alpha_id) * exp_config.lambda_id + \
                     age_loss * exp_config.lambda_age

        self.g_optim.zero_grad()
        total_loss.backward()
        self.g_optim.step()
        return g_loss, l1_loss, ssim_loss, id_loss, age_loss, total_loss

    def vgg_face_output(self, inputs):
        inputs = (F.hardtanh(inputs) * 0.5 + 0.5) * 255
        mean = torch.tensor([129.1863, 104.7624, 93.5940], device=self.device)
        std = torch.tensor([1.0, 1.0, 1.0], device=self.device)
        inputs = inputs.sub(mean[None, :, None, None]).div(std[None, :, None, None])
        return self.vgg_face(inputs)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    obj = PFA_GAN(device=device)
    obj.fit()
