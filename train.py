"""
Main file for PyTorch Challenge Final Project
"""
from __future__ import print_function, division
import json
import time
import torch
from torchvision import transforms
from pathlib import Path
from torchsummary import summary
from torchvision import models
# Custom functions and classes
from trainer.trainer import train_and_eval
from utils.util import download_data, check_dir_and_create
from utils.logfun import set_logger, timer
from utils.visualization import fig_loss_acc
from data_loader.data_loaders import create_dataloader
from model.model import SimpleCNN, Pretrained
# Just for debugging purpose. You could delete this later.
import pdb


if __name__ == '__main__':
    # ==== Later move this part to config.yaml or argparse
    batch_size = 32
    num_epochs = 25
    model_type = 'resnet34'
    model_name = model_type + '.pth.tar' # when saving model

    # Specifying some paths
    DATA_DIR = Path("data")
    RESULTS_DIR = Path("results") / model_type
    LOG_DIR = RESULTS_DIR / "logs"
    FIG_DIR = RESULTS_DIR / "figures"
    # Just checking if the directory exists, if not creating
    check_dir_and_create(str(DATA_DIR))
    check_dir_and_create(str(RESULTS_DIR))
    check_dir_and_create(str(LOG_DIR))
    check_dir_and_create(str(FIG_DIR))

    # Data URL
    URL = "https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip"
    DATA_NAME = "flower_data"
    ZIP_NAME = "flower_data.zip"

    # Custom function for logging
    log = set_logger(str(LOG_DIR), "pytorch_challenge.py.log")

    # Using a custom function to download the data
    download_data(data_dir=DATA_DIR, data_name=DATA_NAME, zip_name=ZIP_NAME, url=URL)

    # Use GPU if available
    torch_gpu = torch.cuda.is_available()

    # Directories to training and validation
    directories = {x: DATA_DIR / DATA_NAME / x for x in ['train', 'valid']}
    # If you were to use transfer learning on pre-trained network that was trained on
    # ImageNet, you need to specifically use the following normalization parameters
    # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    if model_type == 'simplecnn':
        normalization = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    else:
        normalization = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
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
        'model_type': model_type
    }

    # Define the model
    if model_type == 'simplecnn':
        model = SimpleCNN(params)
    else:
        model = Pretrained(params)

    # Make sure to put the model into GPU
    model.cuda() if torch_gpu else model.cpu()

    # Good for checking the architecture
    summary(model, input_size=(3, 224, 224), batch_size=batch_size)

    # A function to perform training and validation
    log.info("Start Training")
    start = time.time()
    t_loss, t_acc, v_loss, v_acc = train_and_eval(model,
                                                  d_size,
                                                  d_loaders,
                                                  torch_gpu,
                                                  log,
                                                  num_epochs=num_epochs)
    end = time.time()
    log.info("Finsihed Training")
    hours, mins, seconds = timer(start, end)
    log.info("Training and testing took: {:0>2} Hours {:0>2} minutes {:05.2f} seconds".format(int(hours), int(mins), seconds))

    # Save the model
    torch.save(model.state_dict(), str(RESULTS_DIR / model_name))

    # Log the results and save the figures
    fig_loss_acc(t_loss, v_loss, "loss", FIG_DIR)
    fig_loss_acc(t_acc, v_acc, "acc", FIG_DIR)