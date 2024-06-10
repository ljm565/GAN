import os
import imageio
import numpy as np
from pathlib import Path

from torchvision.utils import make_grid

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


def save_images(fake_output, fake_list):
    fake_output = fake_output.detach().cpu()
    img = make_grid(fake_output, nrow=8).numpy()
    img = np.transpose(img, (1,2,0))
    fake_list.append(img)
    return fake_list


def gif_making(base_path, fake_list, generation_gif_name):
    gif_list = [(data/data.max()*255).astype(np.uint8) for data in fake_list]
    imageio.mimsave(base_path+'result/'+generation_gif_name, gif_list)