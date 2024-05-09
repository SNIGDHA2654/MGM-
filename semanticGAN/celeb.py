from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image

class CelebAMaskDataset(Dataset):
    def __init__(self, dataroot, is_label=True, phase='train'):
        """
        Initializes the CelebAMaskDataset.

        Parameters:
            dataroot (str): The root directory where the data is stored.
            is_label (bool): Whether the dataset should load labeled data (True) or unlabeled (False).
            phase (str): The phase of the dataset to use ('train', 'val', 'test', or 'train-val').
        """
        self.dataroot = dataroot
        self.is_label = is_label
        self.phase = phase
        self.idx_list = self._load_index_list()

    def _load_index_list(self):
        """Load index list based on the phase, label requirement, and data root."""
        # Define the subdirectory based on whether data is labeled or not
#         data_root = os.path.join(self.dataroot, 'label_data' if self.is_label else 'unlabel_data')
        data_root = self.dataroot
        
        # Set the filename based on the dataset phase or the type of data
        if self.is_label:
            if self.phase == 'train':
                filename = 'train_full_list.txt'
            elif self.phase == 'val':
                filename = 'val_full_list.txt'
            elif self.phase == 'test':
                filename = 'test_list.txt'
            elif self.phase == 'train-val':
                self.tmp_data_path = data_root+ '/label_data'
                # Combine train and validation lists for labeled data
                train_list = self._load_file(self.tmp_data_path+'/train_full_list.txt')
                val_list = self._load_file(self.tmp_data_path+'/val_full_list.txt')
                return np.concatenate((train_list, val_list))
            else:
                raise ValueError("Invalid phase specified for labeled data.")
    
            # Handle the unlabeled data
            #filename = 'unlabel_list.txt'  # Default for all phases if unlabeled
        
        self.tmp_data_path = data_root+ '/unlabel_data'
        file_path = self.tmp_data_path + "/unlabel_list.txt"
        return self._load_file(file_path)

    def _load_file(self, file_path):
        """Helper function to load a file and raise an error if not found."""
#         print(file_path)
#         print(os.path.exists(file_path))
#         print(type(file_path))
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Expected file at {file_path} not found.")
        return np.loadtxt(file_path, dtype=str)

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        """Retrieve an item by its index."""
        img_path = self.tmp_data_path +'/images/' + self.idx_list[idx]
#         print("THE PATH OF IMAGE IS", img_path)
        img = Image.open(img_path).convert('RGB')
        return {'image': np.array(img)}
