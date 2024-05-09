"""
This module computes Inception scores and other metrics for datasets using an Inception v3 model.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle
from loader_mod import load_inception_net, calculate_inception_score
from celeb import CelebAMaskDataset
#import inception_utils
#from dataloader_1.dataset import CelebAMaskDataset
from torch.utils.data import ConcatDataset

@torch.no_grad()
def extract_features(loader, inception, device):
    """Extract features from the data loader using the inception model."""
    pools, logits = [], []
    #print("Entering Extract Feats!")
    for data in loader:
        img = data['image']
        img = img.permute(0,3,1,2).float()
        img = img.to(device)
#         img = data['image'].to(device)
        #print("<<<<<<<<<<<<<IMG SHAPE>>>>>>>>>>>>>>", img.shape)
        if img.shape[1] != 3:
            img = img.expand(-1, 3, -1, -1)  # Ensure 3 color channels
        print("########## TYPE OF IMG ###########", len(img))
        pool_val, logits_val = inception(img)
        #del inception
#         logits_val= inception(img)
        pools.append(pool_val.detach().cpu().numpy())
#         logits.append(torch.nn.functional.softmax(logits_val, dim=1).detach().cpu().numpy())
        logits.append(torch.nn.functional.softmax(logits_val, dim=1).detach().cpu().numpy())
        
    return np.concatenate(pools), np.concatenate(logits)

def get_dataset(path, dataset_name):
    """Load the labeled and unlabeled dataset based on the input arguments."""
    if dataset_name == 'celeba-mask':
        # Assuming that the labeled and unlabeled data are stored in the same root path but in different subdirectories
        labeled_dataset = CelebAMaskDataset( path, is_label=True, phase='train-val')
        unlabeled_dataset = CelebAMaskDataset( path, is_label=False)

#         # Optionally, you can load only labeled or only unlabeled depending on another argument or condition
#         if args.use_both_types:
            # Concatenate both datasets into a single dataset
        dataset = ConcatDataset([labeled_dataset, unlabeled_dataset])
#         else:
#             # Decide based on some condition or configuration
#             dataset = labeled_dataset if args.prefer_labeled else unlabeled_dataset
    else:
        raise Exception('No such dataset loader defined.')
    
    return dataset

def main():
    """Main function to compute Inception features and scores."""
    # Configuration parameters
    size = 1024
    batch = 50
    #n_sample = 50000
    output = '/ssd_scratch/cvit/snigdha/idd/idd_output.pkl'
    image_mode = 'RGB'
    dataset_name = 'celeba-mask'
    path = '/ssd_scratch/cvit/snigdha/idd'
    print('entered')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    inception = load_inception_net()
    inception = inception.to(device)
    print(type(inception))
    
    dataset = get_dataset(path, dataset_name)
    #rint("Dataset Class Initiated")
    loader = DataLoader(dataset, batch_size=batch, num_workers=0)
    #rint("Dataloader Created ...")
#     logits = extract_features(loader, inception, device)
#     print(f'Extracted {logits.shape[0]} features')

#     IS_mean, IS_std = calculate_inception_score(logits)
#     print(f'Training data from dataloader has an Inception Score of {IS_mean:.5f} +/- {IS_std:.5f}')
#     print('Calculating means and covariances...')
#     mean, cov = np.mean(logits, axis=0), np.cov(logits, rowvar=False)

#     with open(output, 'wb') as f:
#         pickle.dump({'mean': mean, 'cov': cov, 'size': size, 'path': path}, f)
        
    pools, logits = extract_features(loader, inception, device)
    print(f'Extracted {pools.shape[0]} features')

    IS_mean, IS_std = calculate_inception_score(logits)
    print(f'Training data from dataloader has an Inception Score of {IS_mean:.5f} +/- {IS_std:.5f}')
    print('Calculating means and covariances...')
    mean, cov = np.mean(pools, axis=0), np.cov(pools, rowvar=False)

    with open(output, 'wb') as f:
        pickle.dump({'mean': mean, 'cov': cov, 'size': size, 'path': path}, f)


if __name__ == '__main__':
    main()
