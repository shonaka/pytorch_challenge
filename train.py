"""
Main file for PyTorch Challenge Final Project
"""
from __future__ import print_function, division
import os
import pdb
import zipfile
import urllib.request
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchsummary import summary
from utils.logfun import set_logger, timer
from pathlib import Path

def download_data(data_dir, data_name, zip_name, url):
    # Function to unzip
    def unzip(data_dir):
        print("Unzipping the .zip file...")
        with zipfile.ZipFile(str(data_dir / zip_name), 'r') as zip_ref:
            zip_ref.extractall(str(data_dir))

    # First check if the file exists already
    if os.path.exists(str(data_dir / zip_name)) or os.path.exists(str(data_dir / data_name)):
        print("Zip file already exists. Checking if it needs unzipping.")
        if os.path.exists(str(data_dir / data_name)):
            print("You are good to go.")
        else:
            # Only need to unzip
            unzip(data_dir)
    else: # Needs to download and then unzip
        print("Downloading the .zip file... it may take a while...")
        urllib.request.urlretrieve(url, str(data_dir / zip_name))
        # Now unzip
        unzip(data_dir)

def check_dir_and_create(dir_path):
    if os.path.exists(dir_path):
        print("The directory already exists.")
    else:
        print("The directory is missing, creating one")
        os.mkdir(dir_path)

def create_dataloader(normalization, directories, batch_size):
    # Defining transformations
    train_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(normalization['mean'],
                                                                normalization['std'])])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(normalization['mean'],
                                                                normalization['std'])])
    data = {
        'train': datasets.ImageFolder(root=str(directories['train']), transform=train_transforms),
        'valid': datasets.ImageFolder(root=str(directories['valid']), transform=valid_transforms)
    }
    # Defining dataloaders. Making sure only the training has random shuffle on
    dataloaders = {
        'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
        'valid': DataLoader(data['valid'], batch_size=batch_size)
    }
    return dataloaders


if __name__ == '__main__':
    # Specifying some paths
    DATA_DIR = Path("data")
    RESULTS_DIR = Path("results")
    # Just checking if the directory exists, if not creating
    check_dir_and_create(str(DATA_DIR))
    check_dir_and_create(str(RESULTS_DIR))

    # Data URL
    URL = "https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip"
    DATA_NAME = "flower_data"
    ZIP_NAME = "flower_data.zip"

    # Custom function for logging
    log = set_logger(str(RESULTS_DIR), "pytorch_challenge.py.log")

    # Using a custom function to download the data
    download_data(data_dir=DATA_DIR, data_name=DATA_NAME, zip_name=ZIP_NAME, url=URL)

    # Use GPU if available
    torch_gpu = torch.cuda.is_available()

    # Directories to training and validation
    directories = {
        'train': DATA_DIR / DATA_NAME / "train",
        'valid': DATA_DIR / DATA_NAME / "valid"
    }
    normalization = {
        'mean': [0.5, 0.5, 0.5],
        'std':  [0.5, 0.5, 0.5]
    }
    dataloaders = create_dataloader(normalization, directories, batch_size=32)

    # Logging some information
    log.info("Normalization used for mean: {}".format(str(normalization['mean'])))
    log.info("Normalization used for std: {}".format(str(normalization['std'])))
    log.info("Batch size: {}".format(batch_size))
    log.info("Number of training samples: {}".format(len(dataloaders['train'].dataset.samples)))
    log.info("Number of classes: {}".format(len(dataloaders['train'].dataset.classes)))
    log.info("Dimensions of an image: {}".format(str(next(iter(dataloaders['train']))[0].shape)))

    pdb.set_trace()