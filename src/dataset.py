import config.config as opt
import numpy as np
import itertools
from src.utils import age2group
import random
import torch.utils.data as tordata
import pickle
import os
import matplotlib.image as mpimg
import torchvision

def load_source(urls, ages, train=True, age_group=opt.age_group):

    group = age2group(ages, age_group)

    return {'path': urls, 'age': ages, 'group': group}


class BaseDataset(tordata.Dataset):

    def __init__(self,
                 age_group=opt.age_group,
                 train=False,
                 max_iter=0,
                 batch_size=0,
                 transforms=False):
    
        self.age_group = age_group
        self.train = train
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.total_pairs = batch_size*max_iter
        self.transforms = transforms

        self.image_urls = np.array(pickle.load(open(os.path.join(opt.data_root, opt.image_urls), 'rb')))[:200]
        self.image_ages = np.array(pickle.load(open(os.path.join(opt.data_root, opt.image_ages), 'rb')))[:200]

        # self.image_urls = np.array(pickle.load(open('../GAN_Image_Dump.pkl', 'rb')))
        # self.image_ages = np.array(pickle.load(open('../GAN_Age_Dump.pkl', 'rb')))

        data = load_source(train=train, urls=self.image_urls, ages=self.image_ages)

        self.image_list, self.ages, self.groups = data['path'], data['age'], data['group']

        self.mean_ages = np.array([np.mean(self.ages[self.groups == i])
                                   for i in range(self.age_group)]).astype(np.float32)

#       Creating 2 arrays and appending
        self.label_group_images = []
        self.label_group_ages = []
        for i in range(self.age_group):
            self.label_group_images.append(
                self.image_list[self.groups == i].tolist())
            self.label_group_ages.append(
                self.ages[self.groups == i].astype(np.float32).tolist())

    def __len__(self):
        return self.total_pairs

    def read_image(self, image):
        # Reading the image from the directory
        img = mpimg.imread(os.path.join(os.path.join(opt.data_root, opt.cacd_data), image))
        # img = mpimg.imread(os.path.join('../CACD2000', image))
        return img

    def transform_image(self, image):

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(opt.image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])])

        return transforms(image)


class AgeDataset(BaseDataset):

    def __init__(self, transforms=True):

        super(AgeDataset, self).__init__(
            transforms=transforms)

        # self.image_urls = dataset
        # self.age = age

    def __getitem__(self, idx):
        url = self.image_urls[idx]
        img = self.read_image(url)

        if self.transform:
            img = self.transform_image(img)

        return img, self.age[idx]

    def __len__(self):
        return len(self.image_urls)


class PFADataset(BaseDataset):

    def __init__(self,
                 age_group,
                 max_iter,
                 batch_size,
                 source=opt.source,
                 transforms=None):

        super(PFADataset, self).__init__(
            age_group=age_group,
            batch_size=batch_size,
            max_iter=max_iter,
            transforms=transforms)

        np.random.seed(0)

        self.target_labels = np.random.randint(source + 1, self.age_group, self.total_pairs)

        pairs = np.array(list(itertools.combinations(range(age_group), 2)))
        p = [1, 1, 1, 0.5, 0.5, 0.5]
        p = np.array(p) / np.sum(p)
        pairs = pairs[np.random.choice(range(len(pairs)), self.total_pairs, p=p), :]
        source_labels, target_labels = pairs[:, 0], pairs[:, 1]
        self.source_labels = source_labels
        self.target_labels = target_labels

        self.true_labels = np.random.randint(0, self.age_group, self.total_pairs)

    def __getitem__(self, idx):

        source_label = self.source_labels[idx]
        target_label = self.target_labels[idx]
        true_label = self.true_labels[idx]

        source_img = self.read_image(random.choice(self.label_group_images[source_label]))

        index = random.randint(0, len(self.label_group_images[true_label]) - 1)

        true_img = self.read_image(self.label_group_images[true_label][index])

        true_age = self.label_group_ages[true_label][index]
        mean_age = self.mean_ages[target_label]

        if self.transforms:
            source_img = self.transform_image(source_img)
            true_img = self.transform_image(true_img)
        return source_img, true_img, source_label, target_label, true_label, true_age, mean_age



# x = PFADataset(age_group=opt.age_group, max_iter=2, batch_size=10, source=opt.source, transforms=True)
#
# print(x[2])

# print(x.source_labels)
#
# print(random.choice(x.label_group_images[3]))

# train_sampler = tordata.distributed.DistributedSampler(x, shuffle=False)
#
# train_loader = tordata.DataLoader(
#             dataset=x,
#             batch_size=10,
#             drop_last=True,
#             num_workers=16,
#             pin_memory=True,
#             sampler=train_sampler
#         )
#
# y = data_prefetcher(train_loader, [0, 1])
#
