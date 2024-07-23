import gc
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributed as dist

from tools import TrainingLogger, calculate_fid_given_paths
from trainer.build import get_model, get_data_loader
from utils import RANK, LOGGER, colorstr, init_seeds
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

        # init model, dataset, dataloader, etc.
        self.modes = ['train', 'validation'] if self.is_training_mode else ['train', 'validation', 'test']
        self.generator, self.discriminator = self._init_model(self.config, self.mode)
        self.dataloaders = get_data_loader(self.config, self.modes, self.is_ddp)
        self.training_logger = TrainingLogger(self.config, self.is_training_mode)

        # save the yaml config
        if self.is_rank_zero and self.is_training_mode:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.config.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / 'args.yaml', self.config)  # save run args
        
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
            try:
                checkpoints = torch.load(resume_path, map_location=device)
            except RuntimeError:
                LOGGER.warning(colorstr('yellow', 'cannot be loaded to MPS, loaded to CPU'))
                checkpoints = torch.load(resume_path, map_location=torch.device('cpu'))
            generator.load_state_dict(checkpoints['model']['generator'])
            discriminator.load_state_dict(checkpoints['model']['discriminator'])
            del checkpoints
            torch.cuda.empty_cache()
            gc.collect()
            if is_rank_zero:
                LOGGER.info(f'Resumed model: {colorstr(resume_path)}')
            return generator, discriminator

        # init model and tokenizer
        do_resume = mode == 'resume' or (mode == 'validation' and self.resume_path)
        generator, discriminator = get_model(config, self.device)

        # resume model
        if do_resume:
            generator, discriminator = _resume_model(self.resume_path, self.device, config.is_rank_zero)

        # init ddp
        if self.is_ddp:
            torch.nn.parallel.DistributedDataParallel(generator, device_ids=[self.device])
            torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[self.device])
        
        return generator, discriminator


    def do_train(self):
        fake_list = []
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
            
            # take generated fake images
            fake_imgs = self.generator(self.fixed_test_noise)
            fake_list = append_fake_images(fake_imgs, fake_list)

            # clears GPU vRAM at end of epoch, can help with out of memory errors
            torch.cuda.empty_cache()
            gc.collect()

            if self.is_rank_zero:
                LOGGER.info(f"\nepoch {epoch+1} time: {time.time() - start} s\n\n\n")

        if RANK in (-1, 0) and self.is_rank_zero:
            LOGGER.info(colorstr('Saving the generated images related data...\n'))
            visoutput_path = self.save_dir / 'vis_outputs'
            os.makedirs(visoutput_path, exist_ok=True)
            gif_making(visoutput_path / 'training.gif', fake_list)
            generated_img_per_epochs(visoutput_path / 'generated_images', fake_list)

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
                    **{'train_loss_d': d_loss.item(), 'train_loss_g': g_loss.item()},
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
                        **{'validation_loss_d': d_loss.item(), 'validation_loss_g': g_loss.item()},
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
        

    def cal_fid(self, phase):
        visoutput_path = os.path.join(self.config.save_dir, 'vis_outputs')

        real_data = next(iter(self.dataloaders[phase]))[0]
        fake_data = self.generator(self.fixed_test_noise)

        real_path = os.path.join(visoutput_path, 'fid/real_images')
        fake_path = os.path.join(visoutput_path, 'fid/fake_images')
        
        prepare_images(real_path, real_data)
        prepare_images(fake_path, fake_data)

        # calculate_fid_given_paths (paths, batch_size, device, dims)
        fid_value = calculate_fid_given_paths([real_path, fake_path], 50, self.device, 2048)

        # draw real and fake images
        draw(real_data, fake_data, os.path.join(visoutput_path, 'fid/RealAndFake.png'))

        return fid_value