import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import time
import pickle
import math
import os 

from config import Config
from utils_func import *
from utils_data import DLoader
from model_GAN import Generator, Discriminator
from pytorch_fid.fid_score import calculate_fid_given_paths



class Trainer:
    def __init__(self, config:Config, device:torch.device, mode:str, continuous:int):
        self.config = config
        self.device = device
        self.mode = mode
        self.continuous = continuous
        self.dataloaders = {}

        # if continuous, load previous training info
        if self.continuous:
            with open(self.config.loss_data_path, 'rb') as f:
                self.loss_data = pickle.load(f)

        # path, data params
        self.base_path = self.config.base_path
        self.model_path = self.config.model_path
        self.color_channel = self.config.color_channel
        assert self.color_channel in [1, 3]
        self.convert2grayscale = True if self.color_channel==3 and self.config.convert2grayscale else False
        self.color_channel = 1 if self.convert2grayscale else self.color_channel

        # train params
        self.batch_size = self.config.batch_size
        self.epochs = self.config.epochs
        self.lr = self.config.lr
        self.noise_init_size = self.config.noise_init_size


        # split trainset to trainset and valset and make dataloaders
        if self.config.MNIST_train:
            # for reproducibility
            torch.manual_seed(999)

            # set to MNIST size
            self.config.width, self.config.height = 28, 28

            self.MNIST_valset_proportion = self.config.MNIST_valset_proportion
            self.trainset = dsets.MNIST(root=self.base_path, transform=transforms.ToTensor(), train=True, download=True)
            self.trainset, self.valset = random_split(self.trainset, [len(self.trainset)-int(len(self.trainset)*self.MNIST_valset_proportion), int(len(self.trainset)*self.MNIST_valset_proportion)])
            self.testset = dsets.MNIST(root=self.base_path, transform=transforms.ToTensor(), train=False, download=True)
        else:
            os.makedirs(self.base_path+'data', exist_ok=True)

            if os.path.isdir(self.base_path+'data/'+self.config.data_name):
                with open(self.base_path+'data/'+self.config.data_name+'/train.pkl', 'rb') as f:
                    self.trainset = pickle.load(f)
                with open(self.base_path+'data/'+self.config.data_name+'/val.pkl', 'rb') as f:
                    self.valset = pickle.load(f)
                with open(self.base_path+'data/'+self.config.data_name+'/test.pkl', 'rb') as f:
                    self.testset = pickle.load(f)
            else:
                os.makedirs(self.base_path+'data/'+self.config.data_name, exist_ok=True)
                self.trans = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                transforms.Resize((self.config.height, self.config.width)),
                                                transforms.ToTensor()]) if self.convert2grayscale else \
                            transforms.Compose([transforms.Resize((self.config.height, self.config.width)),
                                                transforms.ToTensor()]) 
                self.custom_data_proportion = self.config.custom_data_proportion
                assert math.isclose(sum(self.custom_data_proportion), 1)
                assert len(self.custom_data_proportion) <= 3
                
                if len(self.custom_data_proportion) == 3:
                    data = make_img_data(self.config.train_data_path, self.trans)
                    self.train_len, self.val_len = int(len(data)*self.custom_data_proportion[0]), int(len(data)*self.custom_data_proportion[1])
                    self.test_len = len(data) - self.train_len - self.val_len
                    self.trainset, self.valset, self.testset = random_split(data, [self.train_len, self.val_len, self.test_len], generator=torch.Generator().manual_seed(999))

                elif len(self.custom_data_proportion) == 2:
                    data1 = make_img_data(self.config.train_data_path, self.trans)
                    data2 = make_img_data(self.config.test_data_path, self.trans)
                    if self.config.two_folders == ['train', 'val']:
                        self.train_len = int(len(data1)*self.custom_data_proportion[0]) 
                        self.val_len = len(data1) - self.train_len
                        self.trainset, self.valset = random_split(data1, [self.train_len, self.val_len], generator=torch.Generator().manual_seed(999))
                        self.testset = data2
                    elif self.config.two_folders == ['val', 'test']:
                        self.trainset = data1
                        self.val_len = int(len(data2)*self.custom_data_proportion[0]) 
                        self.test_len = len(data2) - self.val_len
                        self.valset, self.testset = random_split(data2, [self.val_len, self.test_len], generator=torch.Generator().manual_seed(999))
                    else:
                        print("two folders must be ['train', 'val] or ['val', 'test']")
                        raise AssertionError

                elif len(self.custom_data_proportion) == 1:
                    self.trainset = make_img_data(self.config.train_data_path, self.trans)
                    self.valset = make_img_data(self.config.val_data_path, self.trans)
                    self.testset = make_img_data(self.config.test_data_path, self.trans)
                
                with open(self.base_path+'data/'+self.config.data_name+'/train.pkl', 'wb') as f:
                    pickle.dump(self.trainset, f)
                with open(self.base_path+'data/'+self.config.data_name+'/val.pkl', 'wb') as f:
                    pickle.dump(self.valset, f)
                with open(self.base_path+'data/'+self.config.data_name+'/test.pkl', 'wb') as f:
                    pickle.dump(self.testset, f)

            self.trainset, self.valset, self.testset = DLoader(self.trainset), DLoader(self.valset), DLoader(self.testset)
        
        self.dataloaders['train'] = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        self.dataloaders['val'] = DataLoader(self.valset, batch_size=self.batch_size, shuffle=False)
        if self.mode == 'test':
            self.dataloaders['test'] = DataLoader(self.testset, batch_size=self.batch_size, shuffle=True)


        self.fixed_test_noise = torch.randn(128, self.noise_init_size).to(device)
        self.G_model = Generator(self.config, self.color_channel).to(self.device)
        self.D_model = Discriminator(self.config, self.color_channel).to(self.device)
        self.criterion = nn.BCELoss()
        if self.mode == 'train':
            self.G_optimizer = optim.Adam(self.G_model.parameters(), lr=self.lr)
            self.D_optimizer = optim.Adam(self.D_model.parameters(), lr=self.lr)
            if self.continuous:
                self.check_point = torch.load(self.model_path, map_location=self.device)
                self.D_model.load_state_dict(self.check_point['model']['discriminator'])
                self.G_model.load_state_dict(self.check_point['model']['generator'])
                self.D_optimizer.load_state_dict(self.check_point['optimizer']['discriminator'])
                self.G_optimizer.load_state_dict(self.check_point['optimizer']['generator'])
                del self.check_point
                torch.cuda.empty_cache()
        elif self.mode == 'test':
            self.check_point = torch.load(self.model_path, map_location=self.device)
            self.D_model.load_state_dict(self.check_point['model']['discriminator'])
            self.G_model.load_state_dict(self.check_point['model']['generator'])
            self.D_model.eval(); self.G_model.eval()
            del self.check_point
            torch.cuda.empty_cache()

        
    def train(self):
        fake_list = []
        train_loss_history = {'discriminator': [], 'generator': []} if not self.continuous else self.loss_data['train_loss_history']
        val_loss_history = {'discriminator': [], 'generator': []} if not self.continuous else self.loss_data['val_loss_history']

        for epoch in range(self.epochs):
            start = time.time()
            print(epoch+1, '/', self.epochs)
            print('-'*10)
            
            for phase in ['train', 'val']:
                print('Phase: {}'.format(phase))
                if phase == 'train':
                    self.G_model.train()
                    self.D_model.train()
                else:
                    self.G_model.eval()
                    self.D_model.eval()

                G_total_loss, D_total_loss, Dx, D_G1, D_G2 = 0, 0, 0, 0, 0
                for i, (real_data, _) in enumerate(self.dataloaders[phase]):
                    batch_size = real_data.size(0)
                    self.G_optimizer.zero_grad()
                    self.D_optimizer.zero_grad()

                    with torch.set_grad_enabled(phase=='train'):

                        ###################################### Discriminator #########################################
                        # training discriminator for real data
                        real_data = real_data.to(self.device)
                        output_real = self.D_model(real_data)
                        target = torch.ones(batch_size, 1).to(self.device)
                        D_loss_real = self.criterion(output_real, target)
                        Dx += output_real.sum().item()

                        # training discriminator for fake data
                        fake_data = self.G_model(torch.randn(batch_size, self.noise_init_size).to(self.device))
                        output_fake = self.D_model(fake_data.detach())  # for ignoring backprop of the generator
                        target = torch.zeros(batch_size, 1).to(self.device)
                        D_loss_fake = self.criterion(output_fake, target)
                        D_loss = D_loss_real + D_loss_fake
                        D_G1 += output_fake.sum().item()

                        if phase == 'train':
                            D_loss.backward()
                            self.D_optimizer.step()
                        ##############################################################################################


                        ########################################## Generator #########################################
                        # training generator by interrupting discriminator
                        output_fake = self.D_model(fake_data)
                        target = torch.ones(batch_size, 1).to(self.device)
                        G_loss = self.criterion(output_fake, target)
                        D_G2 += output_fake.sum().item()

                        if phase == 'train':
                            G_loss.backward()
                            self.G_optimizer.step()
                        ##############################################################################################

                    D_total_loss += D_loss.item() * batch_size
                    G_total_loss += G_loss.item() * batch_size
                    if i % 100 == 0:
                        print('Epoch {}: {}/{} step - D loss: {}, G loss: {}'.format(epoch+1, i, len(self.dataloaders[phase]),\
                            D_loss.item(), G_loss.item()))
                
                D_epoch_loss = D_total_loss/len(self.dataloaders[phase].dataset)
                G_epoch_loss = G_total_loss/len(self.dataloaders[phase].dataset)
                Dx = Dx/len(self.dataloaders[phase].dataset)
                D_G1 = D_G1/len(self.dataloaders[phase].dataset)
                D_G2 = D_G2/len(self.dataloaders[phase].dataset)

                if phase == 'train':
                    print('{} - D loss: {:4f}, G loss: {:4f}, D(x): {}, D(G1): {}, D(G2): {}\n'.\
                        format(phase, D_epoch_loss, G_epoch_loss, Dx, D_G1, D_G2))
                    train_loss_history['discriminator'].append(D_epoch_loss)
                    train_loss_history['generator'].append(G_epoch_loss)

                if phase == 'val':
                    # for sanity check
                    assert D_G1 == D_G2
                    print('{} - D loss: {:4f}, G loss: {:4f}, D(x): {}, D(G): {}\n'.\
                        format(phase, D_epoch_loss, G_epoch_loss, Dx, D_G1))
                    val_loss_history['discriminator'].append(D_epoch_loss)
                    val_loss_history['generator'].append(G_epoch_loss)

                    print('model pt file is being saved\n')
                    model_dic = {'discriminator': self.D_model.state_dict(), 'generator': self.G_model.state_dict()}
                    optimizer_dic = {'discriminator': self.D_optimizer.state_dict(), 'generator': self.G_optimizer.state_dict()}
                    save_checkpoint(self.model_path, model_dic, optimizer_dic)
            
            fake_imgs = self.G_model(self.fixed_test_noise)
            fake_list = save_images(fake_imgs, fake_list)
            print("time: {} s\n".format(time.time() - start))

        print('train finished\n')
        
        if self.config.training_visualization:
            print('Saving the generated images related data...\n')
            gif_making(self.config.base_path, fake_list, self.config.generation_gif_name)
            generated_img_per_epochs(self.config.base_path, fake_list, self.config.generation_img_folder_name)
        self.loss_data = {'train_loss_history': train_loss_history, 'val_loss_history': val_loss_history}
        return {'discriminator': self.D_model, 'generator': self.G_model}, self.loss_data


    def test(self):
        base_path, score_cal_folder_name = self.config.base_path, self.config.score_cal_folder_name
        
        real_data = next(iter(self.dataloaders['test']))[0]
        fake_data = self.G_model(self.fixed_test_noise)

        real_path = image_path(base_path, score_cal_folder_name, real_data, True)
        fake_path = image_path(base_path, score_cal_folder_name, fake_data, False)

        # calculate_fid_given_paths (paths, batch_size, device, dims)
        fid_value = calculate_fid_given_paths([real_path, fake_path], 50, self.device, 2048)

        # draw real and fake images
        draw(real_data, fake_data, base_path, score_cal_folder_name)

        return fid_value