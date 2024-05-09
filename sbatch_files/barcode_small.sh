#!/bin/bash

#SBATCH -A snigdha
#SBATCH --nodelist=gnode085
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH -c 9
#SBATCH --time=96:00:00
#SBATCH --mail-user=snigdha.a@research.iiit.ac.in
#SBATCH --mail-type=ALL

cd ~
cd MGM_project/semanticGAN

python bar_s_train_gan.py  --img_dataset '/ssd_scratch/cvit/vanshg/barcode_data' --seg_dataset '/ssd_scratch/cvit/vanshg/barcode_data' --inception '/ssd_scratch/cvit/vanshg/barcode_output.pkl' --seg_name celeba-mask