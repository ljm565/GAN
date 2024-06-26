# Training GAN
Here, we provide guides for training a GAN model.

### 1. Configuration Preparation
To train a GAN model, you need to create a configuration.
Detailed descriptions and examples of the configuration options are as follows.

```yaml
# base
seed: 0
deterministic: True

# environment config
device: cpu                 # You can DDP training with multiple gpus. e.g. gpu: [0], [0,1], [1,2,3], cpu: cpu, mac: mps

# project config
project: outputs/GAN         # Project directory
name: MNIST                 # Trained model-related data are saved at ${project}/${name} folde

# image setting config
height: 28                  # Image size for preprocessing
width: 28                   # Image size for preprocessing
color_channel: 1            # [1, 3], you can choose one
convert2grayscale: False    # if True and color_channel is 3, you can train color image with grayscaled image

# data config
workers: 0                  # Don't worry to set worker. The number of workers will be set automatically according to the batch size.
MNIST_train: True           # if True, MNIST will be loaded automatically.
MNIST:
    path: data/
    MNIST_valset_proportion: 0.2      # MNIST has only train and test data. Thus, part of the training data is used as a validation set.
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null

# train config
batch_size: 128
epochs: 10
lr: 0.001
hidden_dim: 256
noise_init_size: 128

# logging config
common: ['train_loss_d', 'train_loss_g', 'validation_loss_d', 'validation_loss_g', 'd_x', 'd_g1', 'd_g2']
```


### 2. Training
#### 2.1 Arguments
There are several arguments for running `src/run/train.py`:
* [`-c`, `--config`]: Path to the config file for training.
* [`-m`, `--mode`]: Choose one of [`train`, `resume`].
* [`-r`, `--resume_model_dir`]: Path to the model directory when the mode is resume. Provide the path up to `${project}/${name}`, and it will automatically select the model from `${project}/${name}/weights/` to resume.
* [`-l`, `--load_model_type`]: Choose one of [`loss`, `last`].
    * `loss` (default): Resume the model with the minimum validation loss.
    * `last`: Resume the model saved at the last epoch.
* [`-p`, `--port`]: (default: `10001`) NCCL port for DDP training.


#### 2.2 Command
`src/run/train.py` file is used to train the model with the following command:
```bash
# training from scratch
python3 src/run/train.py --config configs/config.yaml --mode train

# training from resumed model
python3 src/run/train.py --config config/config.yaml --mode resume --resume_model_dir ${project}/${name}
```

When the model training is complete, the checkpoint is saved in `${project}/${name}/weights` and the training config is saved at `${project}/${name}/args.yaml`.