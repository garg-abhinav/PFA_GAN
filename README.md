# PFA-GAN Implementation
## CS543 Computer Vision | Fall 2021 | UIUC

Pytorch implementation of PFA-GAN: Progressive Face Aging with Generative
Adversarial Network ([paper](https://arxiv.org/pdf/2012.03459.pdf)). 

## Contributors
1. Abhinav Garg (garg19@illinois.edu)
2. Parth Gera (gera2@illinois.edu)

## Install dependencies
Here is an example of create environ **from scratch** with `anaconda`

```sh
# create conda env
conda create --name pfagan python=3.9
conda activate pfagan
# install dependencies
pip install -r requirements.txt
```

## Train

### 1. Prepare data

#### Pascal VOC2007

1. Download the dataset ([data](https://bcsiriuschen.github.io/CARC/)).
2. Create a directory named `data` and extract all the files there.
3. Create the metadata files (train and test) for training the network using `data_exploration.py`. Update the file as needed. 
4. The directory should have this basic structure

   ```Bash
   $data/CACD2000                           # jpg files goes here
   $data/celebrity2000_meta.mat             # base metadata file
   $data/image_ages.pkl                     # age of the person in the image for training
   $data/image_urls.pkl                     # training image paths
   $data/test_image_ages.pkl                # age of the person in the image for test
   $data/test_image_urls.pkl                # test image paths
   ```
5. Modify the paths accordingly in `config/config.py`.


### 2. Set Configuration
Update the parameters in `config/config.py` as per the experiment. 

### 3. Model Training

1. Train the age estimation network.
```bash
python age_estimation_network.py 
```
2. Download the DeepFace caffe [model](https://www.robots.ox.ac.uk/~vgg/software/vgg_face/) (the torch model on this website is buggy).

3. Use a caffe-to-pytorch converter to port the DeepFace model to pytorch and place it under the `models` directory. One such converter can be found [here](https://github.com/vadimkantorov/caffemodel2pytorch).

4. Train the GAN.
```bash
python pfa_gan_train.py 
```

5. Tensorboard file created under the `runs` directory can be used for monitoring the progress.

## Inference
To run inference on select test images, update the paths where trained models are located in `config/config.py`.
```bash
python test.py 
```

Use the Face++ API to evaluate the model.
