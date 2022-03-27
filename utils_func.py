import torch
from torchvision.utils import make_grid, save_image
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import imageio
import matplotlib.pyplot as plt


def save_checkpoint(file, model_dic, optimizer_dic):
    state = {'model': model_dic, 'optimizer': optimizer_dic}
    torch.save(state, file)


def make_img_data(path, trans):
    files = os.listdir(path)
    data = [trans(Image.open(path+file)) for file in tqdm(files) if not file.startswith('.')]
    return data


def save_images(fake_output, fake_list):
    fake_output = fake_output.detach().cpu()
    img = make_grid(fake_output, nrow=8).numpy()
    img = np.transpose(img, (1,2,0))
    fake_list.append(img)
    return fake_list


def gif_making(base_path, fake_list, generation_gif_name):
    gif_list = [(data/data.max()*255).astype(np.uint8) for data in fake_list]
    imageio.mimsave(base_path+'result/'+generation_gif_name, gif_list)


def generated_img_per_epochs(base_path, fake_list, generation_img_folder_name):
    folder_path = base_path+'result/'+generation_img_folder_name+'/'
    os.makedirs(folder_path, exist_ok=True)
    
    for idx, img in enumerate(fake_list):
        if (idx+1) % 10 == 0 or idx == 0 or idx+1 == len(fake_list):
            i = '0'*(3-len(str(idx)))+str(idx+1)
            plt.figure(figsize=(10, 10))
            plt.imshow(img, interpolation='nearest')
            plt.savefig(folder_path+i+'_epoch_img.jpg')


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