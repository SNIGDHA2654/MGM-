#!/bin/bash

#SBATCH -A snigdha
#SBATCH --nodelist=gnode076
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH -c 9
#SBATCH --time=96:00:00
#SBATCH --mail-user=snigdha.a@research.iiit.ac.in
#SBATCH --mail-type=ALL

cd ~
cd MGM_project/semanticGAN

python /home2/snigdha/MGM_project/semanticGAN/train_seg_gan.py  --img_dataset '/home2/snigdha/MGM_project/barcode_data' --seg_dataset '/home2/snigdha/MGM_project/barcode_data' --inception '/home2/snigdha/MGM_project/bar_output.pkl' --seg_name celeba-mask