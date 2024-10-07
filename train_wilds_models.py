#!/usr/bin/env python
import time
import os

datasets = ["camelyon17"]

if not os.path.isdir('./job_files'):
    os.mkdir('./job_files')

for data in datasets:
    job_file = f'./job_files/wilds_exps_{data}.slrm'
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
        fh.writelines(f"python wilds/examples/run_expt.py --dataset {data} --algorithm ERM --download --root_dir /mfsnic/u/apouget/data/ --log_dir /mfsnic/u/apouget/experiments/{data} --use_wandb --wandb_api_key_path /h/321/apouget/wandb_key.txt --wandb_kwargs project=WILDS name={data} \n")
    os.system("sbatch %s" %job_file)
    time.sleep(0.3)