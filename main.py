
import numpy as np
import random
from datetime import date
import torch
import wandb
import argparse
import sys
import os

from dataset_builder import MyDataset, split_dataset, MyDataLoader 
from train import train, wandbinitialization
from test import test
from utils import save_models, Params, folder_creator, DeviceDataLoader
from model import MyTransformer, init_weights
today = date.today()

# paths
PYTHON = sys.executable
parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', 
                    default=r"your directory", 
                    help='Directory containing data')

parser.add_argument('--info_dir', 
                    default=r'parameters/my_params.json',
                    help='Directory containing json file for hyperparameters')


parser.add_argument('--save_dir', 
                    default=rf'saved files\{today}',
                    help='Directory for saving the result')

args = parser.parse_args()

# Make a folder to save files
folder_creator(args.save_dir)

project_name = f'{today}_transformer'

# predefined parameters
params  = Params(args.info_dir)

# Set random seeds
np.random.seed(params.random_seed)
random.seed(params.random_seed)
torch.manual_seed(params.random_seed)
torch.cuda.manual_seed(params.random_seed)
torch.cuda.manual_seed_all(params.random_seed)

# Read files and define training and test dataset
# the directory of files
train_dir = os.path.join(args.data_dir, 'train_folder')
test_dir = os.path.join(args.data_dir, 'test_folder')

# Create the custom dataset
# train_total = MyDataset_train(train_dir1, train_dir2, train_dir3)
train_total = MyDataset(train_dir)
test_ds = MyDataset(test_dir)
print(' dataset is initialized')

train_ds, val_ds = split_dataset(train_total, params)
print('split is done:',len(train_ds), len(val_ds))

# seperating data into batches
train_dl, validation_dl, test_dl = MyDataLoader(train_ds, val_ds, test_ds, params)

model = MyTransformer(params)
model.apply(init_weights)

# move data to GPU
device = 'cuda' 
train_loader = DeviceDataLoader(train_dl,device)
validation_loader = DeviceDataLoader(validation_dl,device)
test_loader = DeviceDataLoader(test_dl,device)
model = model.to(device)

# train and validate the model
if params.use_wandb:
    wandb.login()
    wandbinitialization(project_name, params)

# train the model
model, best_epoch, best_model = train(model, params, train_loader, validation_loader, device)

# save the model
save_models(model, best_model, args.save_dir)

# test model on test set
test(model, test_loader, device, args.save_dir)


if not params.apply_early_stop and best_epoch != params.epochs-1:
    test(best_model, test_loader, device, args.save_dir)
