import torchvision
import config.config as opt

class AgeDataset():

    def __init__(self, dataset, age, transforms=None):

        self.dataset=dataset
        self.age=age

        if transforms is not None:

            self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(opt.image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])])

            self.dataset=self.transforms(self.dataset)

    def __getitem__(self, idx):

        img = self.dataset[idx]
        return img, self.age[idx]