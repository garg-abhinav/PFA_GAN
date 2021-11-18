import config.config as opt
import numpy as np
import itertools
from torchvision.datasets.folder import pil_loader
from utils import age2group

def load_source(train=True, age_group = opt.age_group):

    #Importing data from pickle files available in directory
    image_urls = pickle.load(open(os.path.join(exp_config.data_root, exp_config.image_urls), 'rb'))
    image_ages = pickle.load(open(os.path.join(exp_config.data_root, exp_config.image_ages), 'rb'))

    #Getting age group from age2group
    group = age2group(age, image_ages)

    return {'path': image_urls, 'age': image_ages, 'group': group}


class BaseDataset():

    def __init__(self,
                 age_group = otp.age_group,
                 train=False,
                 max_iter=0,
                 batch_size=0,
                 transforms=None):
    
        self.age_group = age_group
        self.train = train
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.total_pairs = batch_size*max_iter
        self.transforms = transforms

        #Collecting individual inputs
        self.image_list, self.ages, self.groups = data['path'], data['age'], data['group']

        #Getting mean age at group level
        self.mean_ages = np.array([np.mean(self.ages[self.groups == i])
                                   for i in range(self.age_group)]).astype(np.float32)

        #Creating 2 arrays and appending
        self.label_group_images = []
        self.label_group_ages = []
        for i in range(self.age_group):
            self.label_group_images.append(
                self.image_list[self.groups == i].tolist())
            self.label_group_ages.append(
                self.ages[self.groups == i].astype(np.float32).tolist())

    def __len__(self):
        return self.total_pairs

#PFA Dataset class.
class PFADataset(BaseDataset):

    def __init__(self,
                 age_group,
                 max_iter,
                 batch_size,
                 source,
                 transforms=None,
                 **kwargs):
        super(PFADataset, self).__init__(
            age_group = age_group,
            batch_size = batch_size,
            max_iter = max_iter,
            transforms = transforms

        np.random.seed(0)

        #random integer(len-total pairs) between source+1 and age_group
        self.target_labels = np.random.randint(source + 1, self.age_group, self.total_pairs)

        #Array of combinations for the range of age_group
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

        #Check what this is doing
        source_img = pil_loader(random.choice(self.label_group_images[source_label]))

        index = random.randint(0, len(self.label_group_images[true_label]) - 1)
        #This
        true_img = pil_loader(self.label_group_images[true_label][index])

        true_age = self.label_group_ages[true_label][index]
        mean_age = self.mean_ages[target_label]

        if self.transforms is not None:
            source_img = self.transforms(source_img)
            true_img = self.transforms(true_img)
        return source_img, true_img, source_label, target_label, true_label, true_age, mean_age


