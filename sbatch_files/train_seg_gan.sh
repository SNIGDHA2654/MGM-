#!/bin/bash

#SBATCH -A snigdha
#SBATCH --nodelist=gnode075
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH -c 9
#SBATCH --time=96:00:00
#SBATCH --mail-user=snigdha.a@research.iiit.ac.in
#SBATCH --mail-type=ALL

cd ~
cd MGM_project/semanticGAN

python /home2/snigdha/MGM_project/semanticGAN/gan_train.py  --img_dataset '/home2/snigdha/MGM_project/CelebAMask-HQ' --seg_dataset '/home2/snigdha/MGM_project/CelebAMask-HQ' --inception '/home2/snigdha/MGM_project/output.pkl' --seg_name celeba-mask --ckpt /home2/snigdha/MGM_project/006000.pt
# python /home2/snigdha/MGM_project/semanticGAN/gan_train.py  --img_dataset '/home2/snigdha/MGM_project/CelebAMask-HQ' --seg_dataset '/home2/snigdha/MGM_project/CelebAMask-HQ' --inception '/home2/snigdha/MGM_project/output.pkl' --seg_name celeba-mask --ckpt /home2/snigdha/MGM_project/006000.pt