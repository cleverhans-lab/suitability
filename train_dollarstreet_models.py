#!/usr/bin/env python
import time
import os

if not os.path.isdir('./job_files'):
    os.mkdir('./job_files')

job_file = f'./job_files/train_model_dollarstreet.slrm'
with open(job_file, 'w+') as fh:
    fh.writelines("#!/bin/bash\n")
    fh.writelines("#SBATCH -A nic\n")
    fh.writelines("#SBATCH -q nic\n")
    fh.writelines("#SBATCH -p nic\n")
    fh.writelines("#SBATCH --gres=gpu:1\n")
    fh.writelines("#SBATCH -c 4\n")
    fh.writelines("#SBATCH --mem=40GB\n")
    fh.writelines("#SBATCH -w caballus\n")
    fh.writelines("#SBATCH --output=./out_err_slurm/%j.out\n")
    fh.writelines("#SBATCH --error=./out_err_slurm/%j.err\n")
    fh.writelines(f"python dollarstreet_model.py --root_dir /mfsnic/u/apouget/data/dollarstreet/dataset_dollarstreet/ --log_dir /mfsnic/u/apouget/experiments/dollarstreet_fewer_epochs --use_wandb --wandb_api_key_path /h/321/apouget/wandb_key.txt --wandb_kwargs project=DollarStreet name=initial --batch_size 32 --epochs 30 --lr 0.1 --momentum 0.9 --weight_decay 1e-4 --lr_step_size 30 --lr_gamma 0.1 \n")

os.system("sbatch %s" %job_file)
time.sleep(0.3)