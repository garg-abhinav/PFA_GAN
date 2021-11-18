import torchvision
from torch.utils.data import Dataset
import config.config as opt
import pickle
import os
import matplotlib.image as mpimg


class AgeDataset(Dataset):

    def __init__(self, dataset, age):
        self.image_urls = dataset
        self.age = age

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(opt.image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])])

    def read_image(self, image):
        # Reading the image from the directory
        img = mpimg.imread(os.path.join(os.path.join(opt.data_root, opt.cacd_data), image))
        return img

    def __getitem__(self, idx):
        url = self.image_urls[idx]
        img = self.read_image(url)
        img = self.transforms(img)

        return img, self.age[idx]

    def __len__(self):
        return len(self.image_urls)

with open('../GAN_Image_Dump.pkl','rb') as f:
    x = pickle.load(f)
    print(x[0])
    print(len(x))


with open('../GAN_Age_Dump.pkl','rb') as f:
    y = pickle.load(f)

test_case=AgeDataset(x, y)
print(test_case[0][0])

#print(test_case[0][0].shape)
