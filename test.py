import torch
import torchvision
from src import utils
from config import config as exp_config
from src.models import Generator
from src.dataset import GroupDataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def generate_images(device, log_dir):
    # Load model
    generator = Generator(age_group=exp_config.age_group)
    generator, global_step = utils.get_latest_checkpoint(generator, log_dir, 'generator', device)
    generator.eval()

    test_data = GroupDataset(do_transforms=True)
    n_test = len(test_data.test_urls)

    print(f'Data loaded having {n_test} images')

    test_loader = DataLoader(test_data, batch_size=1, pin_memory=True)
    
    age_accuracy = [0] * exp_config.age_group
    confidence = [0] * exp_config.age_group
    counts = [0] * exp_config.age_group

    with tqdm(total=n_test, unit='img') as pbar:
        for i, batch in enumerate(test_loader):
            real_img, age = batch
            source = utils.age2group(age, exp_config.age_group)
            torchvision.utils.save_image(real_img * 0.5 + 0.5, 'real_img.jpg')
            bs, ch, w, h = real_img.size()
            fake_imgs = [real_img * 0.5 + 0.5, ]
            # generate fake images
            for target in range(source + 1, exp_config.age_group):
                output = generator(real_img, torch.ones(bs) * source, torch.ones(bs) * target)
                torchvision.utils.save_image(output * 0.5 + 0.5, 'fake_img.jpg')

                estimated_age = utils.get_estimated_age('fake_img.jpg', exp_config.key, exp_config.secret)
                estimated_group = utils.age2group(torch.tensor([estimated_age]), exp_config.age_group)
                verification_confidence = utils.get_verification_confidence('real_img.jpg', 'fake_img.jpg',
                                                                            exp_config.key, exp_config.secret)
    
                counts[target] += 1
                confidence[target] += verification_confidence
                if int(estimated_group) == int(target):
                    age_accuracy[target] += 1
    
                output = output * 0.5 + 0.5
                fake_imgs.append(output)

            pbar.update(1)
            
            fake_imgs = torch.stack(fake_imgs).transpose(1, 0).reshape((-1, ch, w, h))
            grid_img = torchvision.utils.make_grid(fake_imgs.clamp(0., 1.), nrow=exp_config.age_group - exp_config.source)
            torchvision.utils.save_image(grid_img, f'{exp_config.output_path}/output_img_{i}.jpg')

    print('Age Estimation Accuracy:', age_accuracy)
    print('Verification Confidence:', confidence)
    print('Counts:', counts)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = os.path.join(exp_config.log_root, exp_config.gan_dir)
    generate_images(device, log_dir)