import gc
import sys
import time
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributed as dist
from torchvision.utils import make_grid

from tools import TrainingLogger
from trainer.build import get_model, get_data_loader
from utils import TQDM, RANK, LOGGER, colorstr, init_seeds
from utils.filesys_utils import *
from utils.training_utils import *




class Trainer:
    def __init__(
            self, 
            config,
            mode: str,
            device,
            is_ddp=False,
            resume_path=None,
        ):
        init_seeds(config.seed + 1 + RANK, config.deterministic)

        # init
        self.mode = mode
        self.is_training_mode = self.mode in ['train', 'resume']
        self.device = torch.device(device)
        self.is_ddp = is_ddp
        self.is_rank_zero = True if not self.is_ddp or (self.is_ddp and device == 0) else False
        self.config = config
        self.world_size = len(self.config.device) if self.is_ddp else 1
        self.dataloaders = {}
        if self.is_training_mode:
            self.save_dir = make_project_dir(self.config, self.is_rank_zero)
            self.wdir = self.save_dir / 'weights'

        # path, data params
        self.config.is_rank_zero = self.is_rank_zero
        self.resume_path = resume_path

        # color channel init
        self.convert2grayscale = True if self.config.color_channel==3 and self.config.convert2grayscale else False
        self.color_channel = 1 if self.convert2grayscale else self.config.color_channel
        self.config.color_channel = self.color_channel
        
        # sanity check
        assert self.config.color_channel in [1, 3], colorstr('red', 'image channel must be 1 or 3, check your config..')

        # save the yaml config
        if self.is_rank_zero and self.is_training_mode:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.config.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / 'args.yaml', self.config)  # save run args

        # init model, dataset, dataloader, etc.
        self.modes = ['train', 'validation'] if self.is_training_mode else ['test']
        self.generator, self.discriminator = self._init_model(self.config, self.mode)
        self.dataloaders = get_data_loader(self.config, self.modes, self.is_ddp)
        self.training_logger = TrainingLogger(self.config, self.is_training_mode)
        
        # init criterion, optimizer, etc.
        self.epochs = self.config.epochs
        self.noise_init_size = self.config.noise_init_size
        self.fixed_test_noise = torch.randn(128, self.noise_init_size).to(self.device)
        self.criterion = nn.BCELoss()
        if self.is_training_mode:
            self.g_optimizer = optim.Adam(self.generator.parameters(), lr=self.config.lr)
            self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.config.lr)


    def _init_model(self, config, mode):
        def _resume_model(resume_path, device, is_rank_zero):
            checkpoints = torch.load(resume_path, map_location=device)
            generator.load_state_dict(checkpoints['model']['generator'])
            discriminator.load_state_dict(checkpoints['model']['discriminator'])
            del checkpoints
            torch.cuda.empty_cache()
            gc.collect()
            if is_rank_zero:
                LOGGER.info(f'Resumed model: {colorstr(resume_path)}')
            return generator, discriminator

        # init model and tokenizer
        resume_success = False
        do_resume = mode == 'resume' or (mode == 'validation' and self.resume_path)
        generator, discriminator = get_model(config, self.device)

        # resume model or resume model after applying peft
        if do_resume and not resume_success:
            generator, discriminator = _resume_model(self.resume_path, self.device, config.is_rank_zero)

        # init ddp
        if self.is_ddp:
            torch.nn.parallel.DistributedDataParallel(generator, device_ids=[self.device])
            torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[self.device])
        
        return generator, discriminator


    def do_train(self):
        self.train_cur_step = -1
        self.train_time_start = time.time()
        
        if self.is_rank_zero:
            LOGGER.info(f'\nUsing {self.dataloaders["train"].num_workers * (self.world_size or 1)} dataloader workers\n'
                        f"Logging results to {colorstr('bold', self.save_dir)}\n"
                        f'Starting training for {self.epochs} epochs...\n')
        
        if self.is_ddp:
            dist.barrier()

        for epoch in range(self.epochs):
            start = time.time()
            self.epoch = epoch

            if self.is_rank_zero:
                LOGGER.info('-'*100)

            for phase in self.modes:
                if self.is_rank_zero:
                    LOGGER.info('Phase: {}'.format(phase))

                if phase == 'train':
                    self.epoch_train(phase, epoch)
                    if self.is_ddp:
                        dist.barrier()
                else:
                    self.epoch_validate(phase, epoch)
                    if self.is_ddp:
                        dist.barrier()
            
            # save generated fake images
            fake_imgs = self.generator(self.fixed_test_noise)
            fake_list = save_images(fake_imgs, fake_list)

            # clears GPU vRAM at end of epoch, can help with out of memory errors
            torch.cuda.empty_cache()
            gc.collect()

            if self.is_rank_zero:
                LOGGER.info(f"\nepoch {epoch+1} time: {time.time() - start} s\n\n\n")

        if RANK in (-1, 0) and self.is_rank_zero:
            LOGGER.info(f'\n{epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            

    def epoch_train(
            self,
            phase: str,
            epoch: int
        ):
        self.generator.train()
        self.discriminator.train()
        train_loader = self.dataloaders[phase]
        nb = len(train_loader)
        d_x, d_g1, d_g2 = 0, 0, 0

        if self.is_ddp:
            train_loader.sampler.set_epoch(epoch)

        # init progress bar
        if RANK in (-1, 0):
            logging_header = ['D-loss', 'G-loss', 'd_x', 'd_g1', 'd_g2']
            pbar = init_progress_bar(train_loader, self.is_rank_zero, logging_header, nb)

        for i, (real_data, _) in pbar:
            self.train_cur_step += 1
            batch_size = real_data.size(0)
            self.g_optimizer.zero_grad()
            self.d_optimizer.zero_grad()

            ###################################### Discriminator #########################################
            # training discriminator for real data
            real_data = real_data.to(self.device)
            output_real = self.discriminator(real_data)
            target = torch.ones(batch_size, 1).to(self.device)
            d_loss_real = self.criterion(output_real, target)
            d_x += output_real.mean()

            # training discriminator for fake data
            fake_data = self.generator(torch.randn(batch_size, self.noise_init_size).to(self.device))
            output_fake = self.discriminator(fake_data.detach())  # for ignoring backprop of the generator
            target = torch.zeros(batch_size, 1).to(self.device)
            d_loss_fake = self.criterion(output_fake, target)
            d_loss = d_loss_real + d_loss_fake
            d_g1 += output_fake.mean()

            d_loss.backward()
            self.d_optimizer.step()
            ##############################################################################################


            ########################################## Generator #########################################
            # training generator by interrupting discriminator
            output_fake = self.discriminator(fake_data)
            target = torch.ones(batch_size, 1).to(self.device)
            g_loss = self.criterion(output_fake, target)
            d_g2 += output_fake.mean()

            g_loss.backward()
            self.g_optimizer.step()
            ##############################################################################################

            if self.is_rank_zero:
                self.training_logger.update(
                    phase, 
                    epoch + 1,
                    self.train_cur_step,
                    batch_size, 
                    **{'validation_loss_d': d_loss.item(), 'validation_loss_g': g_loss.item()},
                    **{'d_x': d_x.item(), 'd_g1': d_g1.item(), 'd_g2': d_g2.item()}
                )
                loss_log = [d_loss.item(), g_loss.item(), d_x.item(), d_g1.item(), d_g2.item()]
                msg = tuple([f'{epoch + 1}/{self.epochs}'] + loss_log)
                pbar.set_description(('%15s' * 1 + '%15.4g' * len(loss_log)) % msg)
            
        # upadate logs
        if self.is_rank_zero:
            self.training_logger.update_phase_end(phase, printing=True)
        
        
    def epoch_validate(
            self,
            phase: str,
            epoch: int,
            is_training_now=True
        ):
        
        with torch.no_grad():
            if self.is_rank_zero:
                val_loader = self.dataloaders[phase]
                nb = len(val_loader)
                logging_header = ['D-loss', 'G-loss', 'd_x', 'd_g1', 'd_g2']
                pbar = init_progress_bar(val_loader, self.is_rank_zero, logging_header, nb)
                d_x, d_g1, d_g2 = 0, 0, 0

                self.generator.eval()
                self.discriminator.eval()

                for i, (real_data, _) in pbar:
                    batch_size = real_data.size(0)

                    ###################################### Discriminator #########################################
                    # training discriminator for real data
                    real_data = real_data.to(self.device)
                    output_real = self.discriminator(real_data)
                    target = torch.ones(batch_size, 1).to(self.device)
                    d_loss_real = self.criterion(output_real, target)
                    d_x += output_real.mean()

                    # training discriminator for fake data
                    fake_data = self.generator(torch.randn(batch_size, self.noise_init_size).to(self.device))
                    output_fake = self.discriminator(fake_data.detach())  # for ignoring backprop of the generator
                    target = torch.zeros(batch_size, 1).to(self.device)
                    d_loss_fake = self.criterion(output_fake, target)
                    d_loss = d_loss_real + d_loss_fake
                    d_g1 += output_fake.mean()
                    ##############################################################################################


                    ########################################## Generator #########################################
                    # training generator by interrupting discriminator
                    output_fake = self.discriminator(fake_data)
                    target = torch.ones(batch_size, 1).to(self.device)
                    g_loss = self.criterion(output_fake, target)
                    d_g2 += output_fake.mean()
                    ##############################################################################################

                    assert d_g1 == d_g2     # for sanity check
                    self.training_logger.update(
                        phase, 
                        epoch, 
                        self.train_cur_step if is_training_now else 0, 
                        batch_size, 
                        **{'train_loss_d': d_loss.item(), 'train_loss_g': g_loss.item()},
                        **{'d_x': d_x.item(), 'd_g1': d_g1.item(), 'd_g2': d_g2.item()}
                    )

                    loss_log = [d_loss.item(), g_loss.item(), d_x.item(), d_g1.item(), d_g2.item()]
                    msg = tuple([f'{epoch + 1}/{self.epochs}'] + loss_log)
                    pbar.set_description(('%15s' * 1 + '%15.4g' * len(loss_log)) % msg)
                
                # upadate logs and save model
                self.training_logger.update_phase_end(phase, printing=True)
                if is_training_now:
                    self.training_logger.save_model(
                        self.wdir, 
                        {'generator': self.generator, 'discriminator': self.discriminator}
                    )
                    self.training_logger.save_logs(self.save_dir)
        

    def test(self, phase, result_num):
        if result_num > len(self.dataloaders[phase].dataset):
            LOGGER.info(colorstr('red', 'The number of results that you want to see are larger than total test set'))
            sys.exit()

        # makd directory
        vis_save_dir = os.path.join(self.config.save_dir, 'vis_outputs') 
        os.makedirs(vis_save_dir, exist_ok=True)
        
        # concatenate all testset for t-sne and results
        with torch.no_grad():
            total_x, total_output, total_mu, total_log_var, total_y = [], [], [], [], []
            test_loss = 0
            self.model.eval()
            for x, y in self.dataloaders[phase]:
                x = x.to(self.device)
                output, mu, log_var = self.model(x)
                test_loss +=  vae_loss(x, output, mu, log_var, self.decoder_loss).item()
                
                total_x.append(x.detach().cpu())
                total_output.append(output.detach().cpu())
                total_mu.append(mu.detach().cpu())
                total_log_var.append(log_var.detach().cpu())
                total_y.append(y.detach().cpu())

            total_x = torch.cat(tuple(total_x), dim=0)
            total_output = torch.cat(tuple(total_output), dim=0)
            total_mu = torch.cat(tuple(total_mu), dim=0)
            total_log_var = torch.cat(tuple(total_log_var), dim=0)
            total_y = torch.cat(tuple(total_y), dim=0)

        total_output = total_output.view(total_output.size(0), -1, self.config.height, self.config.width)
        total_mu = total_mu.view(total_mu.size(0), -1)
        total_log_var = total_log_var.view(total_log_var.size(0), -1)
        total_y = total_y.numpy()
        z = make_z(total_mu.numpy(), total_log_var.numpy())
        LOGGER.info(colorstr('green', f'testset loss: {test_loss/len(self.dataloaders[phase].dataset)}'))

        # select random index of the data
        ids = set()
        while len(ids) != result_num:
            ids.add(random.randrange(len(total_output)))
        ids = list(ids)

        # save the result img 
        LOGGER.info('start result drawing')
        k = 0
        plt.figure(figsize=(7, 3*result_num))
        for id in ids:
            orig = total_x[id].squeeze(0) if self.color_channel == 1 else total_x[id].permute(1, 2, 0)
            out = total_output[id].squeeze(0) if self.color_channel == 1 else total_output[id].permute(1, 2, 0)
            plt.subplot(result_num, 2, 1+k)
            plt.imshow(orig, cmap='gray')
            plt.subplot(result_num, 2, 2+k)
            plt.imshow(out, cmap='gray')
            k += 2
        plt.savefig(os.path.join(vis_save_dir, 'vae_result.png'))

        # t-sne visualization
        if not self.config.MNIST_train:
            LOGGER.info(colorstr('red', 'Now visualization is possible only for MNIST dataset. You can revise the code for your own dataset and its label..'))
            sys.exit()

        # latent variable visualization
        LOGGER.info('start visualizing the latent space')
        np.random.seed(42)
        tsne = TSNE()

        x_test_2D = tsne.fit_transform(z)
        x_test_2D = (x_test_2D - x_test_2D.min())/(x_test_2D.max() - x_test_2D.min())

        plt.figure(figsize=(10, 10))
        plt.scatter(x_test_2D[:, 0], x_test_2D[:, 1], s=10, cmap='tab10', c=total_y)
        cmap = plt.cm.tab10
        image_positions = np.array([[1., 1.]])
        for index, position in enumerate(x_test_2D):
            dist = np.sum((position - image_positions) ** 2, axis=1)
            if np.min(dist) > 0.02: # if far enough from other images
                image_positions = np.r_[image_positions, [position]]
                imagebox = mpl.offsetbox.AnnotationBbox(
                    mpl.offsetbox.OffsetImage(torch.squeeze(total_x).cpu().numpy()[index], cmap='binary'),
                    position, bboxprops={'edgecolor': cmap(total_y[index]), 'lw': 2})
                plt.gca().add_artist(imagebox)
        plt.axis('off')
        plt.savefig(os.path.join(vis_save_dir, 'tsne_result.png'))
    
        # latent space chaning visualization
        z_mean, interp = [], []
        numbers = self.config.numbers

        for i in range(10):
            loc = np.where(total_y == i)[0]
            z_mean.append(np.mean(z[loc], axis=0))

        z1, z2 = z_mean[numbers[0]], z_mean[numbers[1]]

        with torch.no_grad():
            for coef in (np.linspace(0, 1, 10)):
                interp.append((1 - coef)*z1 + coef*z2)
            z = torch.from_numpy(np.array(interp)).to(self.device)
            output = self.model.decoder(z)

        output = output.view(10, 1, 28, 28).detach().cpu()
        img = make_grid(output, nrow=10).numpy()
        plt.figure(figsize=(30, 7))
        plt.imshow(np.transpose(img, (1, 2, 0)), interpolation='nearest')
        plt.savefig(os.path.join(vis_save_dir, 'latent_space_changing.png'))

