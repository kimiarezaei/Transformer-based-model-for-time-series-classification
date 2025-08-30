"""Customized dataset and related functions for preprocessing the data"""
from torch.utils.data import random_split
import os
import time
import torch
from torch.utils.data.dataloader import DataLoader, Dataset



# split train data into train and validation set
def split_dataset(dataset, params):
    # Calculate the number of training samples 
    train_size = int(len(dataset) * params.train_portion)
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size])


class MyDataset_train(Dataset):
    def __init__(self, root_dir):
        self.root = root_dir

        self.file_path = [os.path.join(root_dir, file) for file in os.listdir(root_dir) if file.endswith('.pt')]

        self.length = len(self.file_path)  # Store the length of the dataset
        self.signal_all = []
        self.label_all = []

        # getting the matrix and its label
        print('loading train data...')
        start_time = time.time()
        for path in self.file_path_total:
            sample_data =  torch.load(path)
            self.signal_all.append(sample_data['signal'])
            self.label_all.append(sample_data['class'])

        self.signalsT = torch.stack(self.signal_all, dim=0)
        self.labelsT = torch.tensor(self.label_all)    
        end_time = time.time()
        total_time = end_time - start_time
        print('train data loaded in:', total_time, 'seconds')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        signal, label = self.signalsT[idx] , self.labelsT[idx]
        return signal, label   


# seperating data into batches
def MyDataLoader(train_ds, val_ds, test_ds, params):   
    train_dl = DataLoader(train_ds, batch_size=params.batch_size, shuffle=True)
    validation_dl = DataLoader(val_ds, batch_size=params.batch_size)
    test_dl = DataLoader(test_ds, batch_size=params.batch_size)
    print('number of train samples:', len(train_ds), 'number of validation samples:', len(val_ds), 'number of test samples:', len(test_ds))
    return train_dl, validation_dl, test_dl
