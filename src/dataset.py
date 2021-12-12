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
                 do_transforms=False):
    
        self.age_group = age_group
        self.train = train
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.total_pairs = batch_size*max_iter
        self.do_transforms = do_transforms

        randomlist = random.sample(range(1, 160000), 20)

        self.image_urls = np.array(pickle.load(open(os.path.join(opt.data_root, opt.image_urls), 'rb')))[randomlist]
        self.image_ages = np.array(pickle.load(open(os.path.join(opt.data_root, opt.image_ages), 'rb')))[randomlist]

        # self.image_urls = np.array(pickle.load(open('../data/image_urls.pkl', 'rb')))
        # self.image_ages = np.array(pickle.load(open('../data/image_ages.pkl', 'rb')))

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
        # img = mpimg.imread(os.path.join('../data/CACD2000', image))
        return img

    def transform_image(self, image):
        print(image.shape)
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(opt.image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])])
        return transforms(image)


class AgeDataset(BaseDataset):

    def __init__(self, do_transforms=True):

        super(AgeDataset, self).__init__(
            do_transforms=do_transforms)

        # self.image_urls = dataset
        # self.age = age

    def __getitem__(self, idx):
        url = self.image_urls[idx]
        img = self.read_image(url)

        if self.do_transforms:
            img = self.transform_image(img)

        return img, self.image_ages[idx]

    def __len__(self):
        return len(self.image_urls)


class PFADataset(BaseDataset):

    def __init__(self,
                 age_group,
                 max_iter,
                 batch_size,
                 source=opt.source,
                 do_transforms=None):

        super(PFADataset, self).__init__(
            age_group=age_group,
            batch_size=batch_size,
            max_iter=max_iter,
            do_transforms=do_transforms)

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
        print('1')
        source_label = self.source_labels[idx]
        target_label = self.target_labels[idx]
        true_label = self.true_labels[idx]
        print('2')
        source_img = self.read_image(random.choice(self.label_group_images[source_label]))
        print('3')
        index = random.randint(0, len(self.label_group_images[true_label]) - 1)

        true_img = self.read_image(self.label_group_images[true_label][index])
        print('4')
        true_age = self.label_group_ages[true_label][index]
        mean_age = self.mean_ages[target_label]

        if self.do_transforms:
            print('s')
            source_img = self.transform_image(source_img)
            print('t')
            true_img = self.transform_image(true_img)
        print('5')
        return source_img, true_img, source_label, target_label, true_label, true_age, mean_age

###############Group Dataset

class GroupDataset(BaseDataset):

    def __init__(self,
                 do_transforms=None):
        super(GroupDataset, self).__init__(
            do_transforms=do_transforms)

        self.test_urls = pickle.load(open(os.path.join(opt.data_root, opt.test_image_urls), 'rb'))
        self.test_ages = pickle.load(open(os.path.join(opt.data_root, opt.test_image_ages), 'rb'))

    def __getitem__(self, idx):
        img = self.read_image(self.test_urls[idx])
        age = self.test_ages[idx]
        if self.do_transforms:
            img = self.transform_image(img)
        return img, age

    def __len__(self):
        return len(self.test_urls)

# x = PFADataset(age_group=opt.age_group, max_iter=2, batch_size=10, source=opt.source, do_transforms=True)
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
# test = GroupDataset(do_transforms='yes')
# print(len(test.test_urls))
# print(len(test.test_ages))