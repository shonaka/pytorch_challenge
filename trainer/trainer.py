import torch
import torch.nn as nn
import optuna
from tqdm import tqdm
from model.model import SimpleCNN, Pretrained


def get_optimizer(model, args):
    """
    A function to get optimizer with specified parameters.
    :param model: A model you defined as PyTorch class.
    :param args: configuration and parameters
    :return optimizer: optimizer with specified parameters.
    """
    # Define optimizers and loss function
    # If you are using PyTorch 0.4.0 you need this weird filter
    # https://github.com/pytorch/pytorch/issues/679
    # no longer needed in 1.0.0
    if args.optim_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=args.optim_lr,
                                    amsgrad=args.optim_amsgrad)
    elif args.optim_type == 'Momentum':
        optimizer = optim.SGD(model.parameters(),
                            lr=args.optim_lr,
                            momentum=args.optim_momentum,
                            weight_decay=args.optim_weight_decay)
    else:
        raise("The optimizer type not defined. Double check the configuration file.")

    return optimizer


def train_and_eval(model, datasizes, dataloaders, torch_gpu, log, args):
    """
    A function that wraps training and validation.
    :param model: Your defined model as a class
    :param datasizes: number of samples inside your dataset
    :param dataloaders: Your defined dataloader as a DataLoader class
    :param torch_gpu: whether to use GPU or not (True or False)
    :param log: wrapper for logging
    :param args: configurations and parameters
    """
    # Define optimizer
    optimizer = get_optimizer(model, args)
    # Define loss criterion
    criterion = nn.CrossEntropyLoss()

    # Define empty lists for keeping track of results
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []

    # Iterate over number of epochs
    for e in range(args.num_epochs):
        dict_loss = {}
        dict_acc = {}
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            # For keeping track of progress
            running_loss = 0.0
            running_corrects = 0

            for i, (data, target) in tqdm(enumerate(dataloaders[phase])):
                # If using gpu, make sure you put the data into cuda
                if torch_gpu:
                    data, target = data.cuda(), target.cuda()
                # Clear previous gradients
                optimizer.zero_grad()
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Make prediction
                    out = model(data)
                    _, preds = torch.max(out.data, 1)
                    # Calculate the loss
                    loss = criterion(out, target)
                    if phase == 'train':
                        # Backpropagate loss
                        loss.backward()
                        # Update
                        optimizer.step()
                # Track the acc and loss for visualization
                running_loss += loss.item() * data.size(0)
                running_corrects += torch.sum(preds == target.data).item()

            # calculate the loss and acc per epoch
            dict_loss[phase] = running_loss / datasizes[phase]
            dict_acc[phase] = running_corrects / datasizes[phase]

        # Logging
        log.info("Epoch: {}".format(e+1))
        log.info("  Train Loss: {:.3f}, Train Acc: {:.3f}".format(dict_loss['train'], dict_acc['train']))
        log.info("  Valid Loss: {:.3f}, Valid Acc: {:.3f}".format(dict_loss['valid'], dict_acc['valid']))
        # for later visualization
        train_loss_list.append(dict_loss['train'])
        train_acc_list.append(dict_acc['train'])
        valid_loss_list.append(dict_loss['valid'])
        valid_acc_list.append(dict_acc['valid'])

    return train_loss_list, train_acc_list, valid_loss_list, valid_acc_list


def optuna_optimizer(trial, model, args):
    """
    Using hyperparameter optimization optuna to tune optimizer
    :param trial: trial used to optimize hyperparameters from optuna
    :param model: A model you defined as PyTorch class.
    :param args: configuration and parameters
    :return optimizer: optimizer with specified parameters.
    """
    # Define optimizers and loss function
    optimizer_names = ['Adam', 'Momentum']
    optimizer_name = trial.suggest_categorical('optimizer', optimizer_names)
    amsgrad = trial.suggest_categorical('amsgrad', [True, False])
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-8, 1e-3)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr,
                                     weight_decay=weight_decay,
                                     amsgrad=amsgrad)
    elif optimizer_name == 'Momentum':
        optimizer = torch.optim.SGD(model.parameters(),
                                         lr=lr,
                                         momentum=args.optim_momentum,
                                         weight_decay=weight_decay)
    else:
        raise("The optimizer type not defined. Double check the configuration file.")

    return optimizer


def objective(trial, params, datasizes, dataloaders, torch_gpu, args):
    """
    Objective function to be minimized
    :param trial: trial used to optimize hyperparameters from optuna
    :param datasizes: number of samples inside your dataset
    :param dataloaders: Your defined dataloader as a DataLoader class
    :param torch_gpu: whether to use GPU or not (True or False)
    :param log: wrapper for logging
    :param args: configurations and parameters
    """
    # Define model
    if args.model_type == 'simplecnn':
        model = SimpleCNN(args, params)
    else:
        model = Pretrained(args, params)
    # Make sure to put the model into GPU
    model.cuda() if torch_gpu else model.cpu()
    # Using multiple GPUs
    model = torch.nn.DataParallel(model)

    # Define optimizer
    optimizer = optuna_optimizer(trial, model, args)
    # Define loss criterion
    criterion = nn.CrossEntropyLoss()

    # Iterate over number of epochs
    for e in range(args.num_epochs):
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            # For keeping track of progress
            running_corrects = 0

            for i, (data, target) in tqdm(enumerate(dataloaders[phase])):
                # If using gpu, make sure you put the data into cuda
                if torch_gpu:
                    data, target = data.cuda(), target.cuda()
                # Clear previous gradients
                optimizer.zero_grad()
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Make prediction
                    out = model(data)
                    _, preds = torch.max(out.data, 1)
                    # Calculate the loss
                    loss = criterion(out, target)
                    if phase == 'train':
                        # Backpropagate loss
                        loss.backward()
                        # Update
                        optimizer.step()
                # Track the acc and loss for visualization
                running_corrects += torch.sum(preds == target.data).item()

            # calculate the validation acc per epoch
            if phase == 'valid':
                acc_valid = running_corrects / datasizes[phase]
                # calculate the error_rate
                error_rate = 1 - acc_valid
                # report the error_rate
                trial.report(error_rate, e)
                # if terminating in the middle because of the performance
                if trial.should_prune(e):
                    raise optuna.structs.TrialPruned()

    return error_rate

