"""
Main file for PyTorch Challenge Final Project
"""
from __future__ import print_function, division
import os
import pdb
import json
import time
import zipfile
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchsummary import summary
from utils.logfun import set_logger, timer
from pathlib import Path
from tqdm import tqdm

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
    data_transforms = {
        'train': transforms.Compose([transforms.Resize(256),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     normalization]),
        'valid': transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     normalization])
    }
    image_datasets = {
        'train': datasets.ImageFolder(root=str(directories['train']),
                                      transform=data_transforms['train']),
        'valid': datasets.ImageFolder(root=str(directories['valid']),
                                      transform=data_transforms['valid'])
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    # Defining dataloaders. Making sure only the training has random shuffle on
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
        'valid': DataLoader(image_datasets['valid'], batch_size=batch_size)
    }
    return dataset_sizes, dataloaders

class SimpleCNN(nn.Module):
    """
    A class for simple CNN classifier.
    """
    def __init__(self, params):
        """
        Define a constructor and initialize the network.
        """
        super(SimpleCNN, self).__init__()
        self.params = params
        # Defining your own block
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(128 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, self.params['nc'])
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Actually defining the structure.
        :param x: input batch data. dimension: [batch, rgb=3, height, width]
        :return x: output after forward passing
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # flatten layer
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_and_eval(model, datasizes, dataloaders, torch_gpu, log, num_epochs):
    # For training
    def train(model, optimizer, criterion, datasizes, dataloaders, torch_gpu):
        # Make sure the model in training mode
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for i, (data, target) in tqdm(enumerate(dataloaders['train'])):
            # If using gpu, make sure you put the data into cuda
            if torch_gpu:
                data, target = data.cuda(), target.cuda()
            # Clear previous gradients
            optimizer.zero_grad()
            # Make prediction
            out = model(data)
            _, preds = torch.max(out.data, 1)
            # Compute gradients of all variables wrt loss
            loss = criterion(out, target)
            loss.backward()
            # Update
            optimizer.step()
            # Track the acc and loss for visualization
            running_loss += loss.item() * data.size(0)
            running_corrects += torch.sum(preds == target.data)

        # calculate the loss and acc per epoch
        train_loss = running_loss / datasizes['train']
        train_acc = running_corrects.double() / datasizes['train']
        return train_loss, train_acc

    # For Validation
    def valid(model, criterion, datasizes, dataloaders, torch_gpu):
        # making sure it's not in training mode anymore
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        # No need to track gradients when evaluating
        with torch.no_grad():
            # validation for loop
            for i, (data, target) in tqdm(enumerate(dataloaders['valid'])):
                # If using gpu, make sure you put the data into cuda
                if torch_gpu:
                    data, target = data.cuda(), target.cuda()
                # prediction
                out = model(data)
                _, preds = torch.max(out.data, 1)
                # calculate the loss
                loss = criterion(out, target)
                # keeping track of acc and loss
                running_loss += loss.item() * data.size(0)
                running_corrects += torch.sum(preds == target.data)

        # calculate the loss and acc per epoch
        valid_loss = running_loss / datasizes['valid']
        valid_acc = running_corrects.double() / datasizes['valid']
        return valid_loss, valid_acc

    # Define optimizers and loss function
    optimizer = torch.optim.Adam(model.parameters(), amsgrad=True)
    criterion = nn.CrossEntropyLoss()

    # Define empty lists for keeping track of results
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []

    # Iterate over number of epochs
    for e in range(num_epochs):
        train_loss, train_acc = train(model, optimizer, criterion, datasizes, dataloaders, torch_gpu)
        valid_loss, valid_acc = valid(model, criterion, datasizes, dataloaders, torch_gpu)
        # Logging
        log.info("Epoch: {}".format(e+1))
        log.info("  Train Loss: {:.2f}, Train Acc: {:.2f}".format(train_loss, train_acc))
        log.info("  Valid Loss: {:.2f}, Valid Acc: {:.2f}".format(valid_loss, valid_acc))
        # for later visualization
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)

    return train_loss_list, train_acc_list, valid_loss_list, valid_acc_list


if __name__ == '__main__':

    # Specify parameters. Later move to config.yaml
    batch_size = 32

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
    normalization = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
    d_size, d_loaders = create_dataloader(normalization, directories, batch_size=batch_size)

    # Get some other parameters
    num_classes = len(d_loaders['train'].dataset.classes)

    # Logging some information
    log.info("Batch size: {}".format(batch_size))
    log.info("Using GPU: {}".format(str(torch_gpu)))
    log.info("Number of training samples: {}".format(len(d_loaders['train'].dataset.samples)))
    log.info("Number of classes: {}".format(num_classes))
    log.info("Dimensions of an image: {}".format(str(next(iter(d_loaders['train']))[0].shape)))

    # Loading labels provided by Udacity
    # https://github.com/udacity/pytorch_challenge
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    log.info("Categories: {}".format(str(cat_to_name)))

    # Building a model
    params = {
        'nc': num_classes
    }

    # Define your model
    model = SimpleCNN(params).cuda() if torch_gpu else SimpleCNN(params).cuda()

    # Good for checking the architecture
    summary(model, input_size=(3, 224, 224), batch_size=batch_size)

    # A function to perform training and validation
    log.info("Start Training")
    start = time.time()
    t_loss, t_acc, v_loss, v_acc = train_and_eval(model, d_size, d_loaders, torch_gpu, log, num_epochs=20)
    end = time.time()
    log.info("Finsihed Training")
    hours, mins, seconds = timer(start, end)
    log.info("Training and testing took: {:0>2} Hours {:0>2} minutes {:05.2f} seconds".format(int(hours), int(mins), seconds))
