# FID Calculation
Here, we provide guides for calculating FID score between real and fake data from the trained GAN model.

### 1. FID
#### 1.1 Arguments
There are several arguments for running `src/run/cal_fid.py`:
* [-r, --resume_model_dir]: Directory to the model to calculate FID score. Provide the path up to `{$project}/{$name}`, and it will automatically select the model from `{$project}/{$name}/weights/` to calculate FID score.
* [-l, --load_model_type]: Choose one of [`loss`, `last`].
    * `loss` (default): Resume the model with the minimum validation loss.
    * `last`: Resume the model saved at the last epoch.
* [-d, --dataset_type]: (default: `test`) Choose one of [`train`, `validation`, `test`].


#### 1.2 Command
`src/run/cal_fid.py` file is used to calculate FID scores of the model with the following command:
```bash
python3 src/run/cal_fid.py --resume_model_dir {$project}/{$name}
```