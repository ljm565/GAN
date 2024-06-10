import torch
from torchvision.utils import make_grid, save_image
import os
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt











def image_path(base_path, score_cal_folder_name, data, real):
    if real:
        folder_path = base_path + 'test/' + score_cal_folder_name + '/real/'
        os.makedirs(folder_path, exist_ok=True)
    else:
        folder_path = base_path + 'test/' + score_cal_folder_name + '/fake/'
        os.makedirs(folder_path, exist_ok=True)

    for i in range(len(data)):
        img_path = folder_path + 'img_' + str(i) + '.png'
        save_image(data[i], img_path)

    return folder_path


def draw(real_data, fake_data, base_path, score_cal_folder_name):
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

    plt.savefig(base_path + 'test/' + score_cal_folder_name + '/RealandFake.png')