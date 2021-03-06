"""
Main file for PyTorch Challenge Final Project
"""
from __future__ import print_function, division
import json
import yaml
import time
import torch
import optuna
from torchvision import transforms
from pathlib import Path
from torchsummary import summary
from torchvision import models
# Custom functions and classes
from trainer.trainer import train_and_eval, objective
from utils.arg_yaml import get_args, get_parser
from utils.util import download_data, check_dir_and_create
from utils.logfun import set_logger, timer
from utils.visualization import fig_loss_acc
from data_loader.data_loaders import create_dataloader
from model.model import SimpleCNN, Pretrained
# Just for debugging purpose. You could delete this later.
import pdb


if __name__ == '__main__':
    # Get the default and specified arguments
    args = get_args(get_parser("config.yaml"))

    # Defining a name for the model to be saved
    header = args.trial + '_' + args.model_type
    model_name = header + '.pth.tar'

    # Specifying some paths
    DATA_DIR = Path("data")
    RESULTS_DIR = Path("results") / header
    LOG_DIR = RESULTS_DIR / "logs"
    FIG_DIR = RESULTS_DIR / "figures"
    # Just checking if the directory exists, if not creating
    check_dir_and_create(str(DATA_DIR))
    check_dir_and_create(str(RESULTS_DIR))
    check_dir_and_create(str(LOG_DIR))
    check_dir_and_create(str(FIG_DIR))

    # Custom function for logging
    log = set_logger(str(LOG_DIR), args.log_name)

    # Using a custom function to download the data
    download_data(data_dir=DATA_DIR, data_name=args.file_name, zip_name=args.zip_name, url=args.url)

    # Use GPU if available
    torch_gpu = torch.cuda.is_available()

    # Directories to training and validation
    directories = {x: DATA_DIR / args.file_name / x for x in ['train', 'valid']}
    # If you were to use transfer learning on pre-trained network that was trained on
    # ImageNet, you need to specifically use the following normalization parameters
    # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    if args.model_type == 'simplecnn':
        normalization = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    else:
        normalization = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    d_size, d_loaders = create_dataloader(normalization, directories, batch_size=args.batch_size)

    # Get some other parameters
    num_classes = len(d_loaders['train'].dataset.classes)

    # Logging some information
    log.info("PyTorch version: {}".format(torch.__version__))
    log.info("Model using: {}".format(args.model_type))
    log.info("Hyperparameter optimization: {}".format(args.optuna_tune))
    log.info("Batch size: {}".format(args.batch_size))
    log.info("Using GPU: {}".format(str(torch_gpu)))
    log.info("Number of training samples: {}".format(len(d_loaders['train'].dataset.samples)))
    log.info("Number of classes: {}".format(num_classes))
    log.info("Dimensions of an image: {}".format(str(next(iter(d_loaders['train']))[0].shape)))

    # Loading labels provided by Udacity
    # https://github.com/udacity/pytorch_challenge
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # Building a model
    params = {
        'nc': num_classes,
    }

    # Define the model
    if args.model_type == 'simplecnn':
        model = SimpleCNN(args, params)
    else:
        model = Pretrained(args, params)
    # Make sure to put the model into GPU
    model.cuda() if torch_gpu else model.cpu()
    # Using multiple GPUs
    model = torch.nn.DataParallel(model)

    # Good for checking the architecture
    summary(model, input_size=(3, 224, 224), batch_size=args.batch_size)

    # A function to perform training and validation
    log.info("Start Training")
    start = time.time()
    if args.optuna_tune:
        # Running hyperparameter optimization using optuna
        # n_warmup_steps = 10 to at least run 10 epochs before deciding to prune
        study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
        # use lambda if you want to pass more agruments other than the "trial"
        study.optimize(lambda trial: objective(trial, params, d_size, d_loaders, torch_gpu, args),
                       n_trials=args.optuna_trials)
        # after the optimization, this is how you get the best parameters
        best_params = study.best_params
        # The best error_rate
        best_error = study.best_value
        log.info("Best params are: {}".format(str(best_params)))
        log.info("Best error_rate is: {:.4f}".format(best_error))
        # now running with the best parameters for saving results purposes
        # TODO: optimize this part a little, I shouldn't have to run twice
        # TODO: make parsing of the best parameters automatic as well
        args.optim_lr = float(best_params['lr'])
        args.optim_type = str(best_params['optimizer'])
        args.optim_amsgrad = bool(best_params['amsgrad'])
        args.optim_weight_decay = float(best_params['weight_decay'])
        log.info("Final testing with the best parameters")
        t_loss, t_acc, v_loss, v_acc = train_and_eval(model,
                                                  d_size,
                                                  d_loaders,
                                                  torch_gpu,
                                                  log,
                                                  args)
    else:
        t_loss, t_acc, v_loss, v_acc = train_and_eval(model,
                                                  d_size,
                                                  d_loaders,
                                                  torch_gpu,
                                                  log,
                                                  args)
    end = time.time()
    log.info("Finsihed Training")
    hours, mins, seconds = timer(start, end)
    log.info("Training and testing took: {:0>2} Hours {:0>2} minutes {:05.2f} seconds".format(int(hours), int(mins), seconds))

    # Save the model
    torch.save(model.state_dict(), str(RESULTS_DIR / model_name))

    # Log the results and save the figures
    fig_loss_acc(t_loss, v_loss, "loss", FIG_DIR)
    fig_loss_acc(t_acc, v_acc, "acc", FIG_DIR)

    # Log the parameters and results
    dict_params = vars(args)
    dict_params['final_train_loss'] = round(t_loss[-1], 4)
    dict_params['final_train_acc'] = round(t_acc[-1], 4)
    dict_params['final_valid_loss'] = round(v_loss[-1], 4)
    dict_params['final_valid_acc'] = round(v_acc[-1], 4)
    print(type(dict_params))
    print(dict_params)
    with open(str(RESULTS_DIR / "results.yaml"), 'w') as output_file:
        yaml.dump(dict_params, output_file, default_flow_style=False)
    with open(str(RESULTS_DIR / "results.json"), 'w') as output_file:
        json.dump(dict_params, output_file)
