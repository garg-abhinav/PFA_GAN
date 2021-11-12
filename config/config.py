
# General
log_root = 'models/'
age_group = 4
vgg_face_model = 'vgg_face.pt'
image_size = 256
data_root = 'data/'
image_urls = 'image_urls.pkl'
image_ages = 'image_ages.pkl'

# AgeEstimationNetwork hyper parameters
age_lr = 0.0001
age_lr_decay_rate = 0.7
age_lr_decay_steps = 15
age_batch_size = 128
age_max_epochs = 50
age_dir = 'age_estimation_network/'

# PFAGAN hyper parameters
lr = 0.0001
beta1 = 0.5
beta2 = 0.99
batch_size = 12
max_iter = 200000
gan_dir = 'gan/'