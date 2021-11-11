import scipy.io
import h5py
import pandas as pd
import mat73
from itertools import islice
import pprint
import matplotlib.image as mpimg
import numpy as np

class DataGAN:

    def __init__(self):

        self.actor_age = None
        self.actor_image_url = None
        self.actor_id = None
        self.image_array = None

    def load_data(self):

        data_dict = scipy.io.loadmat('celebrity2000_meta.mat')
        celebrity_data=data_dict['celebrityImageData']

        actor_age=[]
        actor_image_url=[]
        actor_id=[]

        for i in celebrity_data[0][0][0]:

            actor_age.append(i.tolist()[0])

        for i in celebrity_data[0][0][7]:

            actor_image_url.append(i.tolist()[0][0])

        for i in celebrity_data[0][0][1]:

            actor_id.append(i.tolist()[0])

        return actor_age, actor_image_url, actor_id

    def read_image(self, image):

        #Reading the image from the directory
        img1 = mpimg.imread('CACD2000/'+image)
        return img1

    def create_image_array(self, actor_image_url):

        image_array=[]

        #image_array = np.empty([163446,250,250,3])

        for i,url in enumerate(actor_image_url):

            #image_array[i]=read_image(url)

            image_array.append(self.read_image(url))

        return image_array


    def data_preprocess(self):

        self.actor_age, self.actor_image_url, self.actor_id = self.load_data()
        self.image_array= self.create_image_array(self.actor_image_url)

        return self.actor_age, self.actor_id, self.image_array


























