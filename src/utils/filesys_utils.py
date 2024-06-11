import os
import imageio
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from torchvision.utils import make_grid, save_image

from utils import LOGGER, colorstr



def make_project_dir(config, is_rank_zero=False):
    """
    Make project folder.

    Args:
        config: yaml config.
        is_rank_zero (bool): make folder only at the zero-rank device.

    Returns:
        (path): project folder path.
    """
    prefix = colorstr('make project folder')
    project = config.project
    name = config.name

    save_dir = os.path.join(project, name)
    if os.path.exists(save_dir):
        if is_rank_zero:
            LOGGER.info(f'{prefix}: Project {save_dir} already exists. New folder will be created.')
        save_dir = os.path.join(project, name + str(len(os.listdir(project))+1))
    
    if is_rank_zero:
        os.makedirs(project, exist_ok=True)
        os.makedirs(save_dir)
    
    return Path(save_dir)


def yaml_save(file='data.yaml', data=None, header=''):
    """
    Save YAML data to a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        data (dict): Data to save in YAML format.
        header (str, optional): YAML header to add.

    Returns:
        (None): Data is saved to the specified file.
    """
    save_path = Path(file)
    print(data.dumps())
    with open(save_path, "w") as f:
        f.write(data.dumps(modified_color=None, quote_str=True))
        LOGGER.info(f"Config is saved at {save_path}")


def append_fake_images(fake_output, fake_list):
    fake_output = fake_output.detach().cpu()
    img = make_grid(fake_output, nrow=8).numpy()
    img = np.transpose(img, (1,2,0))
    fake_list.append(img)
    return fake_list


def gif_making(save_path, fake_list):
    gif_list = [(data/data.max()*255).astype(np.uint8) for data in fake_list]
    imageio.mimsave(save_path, gif_list)


def generated_img_per_epochs(save_path, fake_list):
    os.makedirs(save_path, exist_ok=True)
    
    for idx, img in enumerate(fake_list):
        if (idx+1) % 10 == 0 or idx == 0 or idx+1 == len(fake_list):
            i = '0'*(3-len(str(idx)))+str(idx+1)
            plt.figure(figsize=(10, 10))
            plt.imshow(img, interpolation='nearest')
            plt.savefig(os.path.join(save_path, f'{i}_epoch_img.jpg'))


def prepare_images(folder_dir, data):
    os.makedirs(folder_dir, exist_ok=True)
    
    for i, img in enumerate(data):
        save_image(img, os.path.join(folder_dir, f'img_{i}.png'))
    

def draw(real_data, fake_data, path):
    # Plot the real images
    plt.figure(figsize=(10, 5))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(make_grid(real_data[:64], padding=5, normalize=True).cpu(), (1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(make_grid(fake_data[:64], padding=5, normalize=True).cpu(), (1,2,0)))

    plt.savefig(path)