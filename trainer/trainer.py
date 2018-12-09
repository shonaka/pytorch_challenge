import torch
import torch.nn as nn
from tqdm import tqdm

def train_and_eval(model, datasizes, dataloaders, torch_gpu, log, num_epochs):
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
                running_corrects += torch.sum(preds == target.data)

            # calculate the loss and acc per epoch
            dict_loss[phase] = running_loss / datasizes[phase]
            dict_acc[phase] = running_corrects.double() / datasizes[phase]

        # Logging
        log.info("Epoch: {}".format(e+1))
        log.info("  Train Loss: {:.2f}, Train Acc: {:.2f}".format(dict_loss['train'], dict_acc['train']))
        log.info("  Valid Loss: {:.2f}, Valid Acc: {:.2f}".format(dict_loss['valid'], dict_acc['valid']))
        # for later visualization
        train_loss_list.append(dict_loss['train'])
        train_acc_list.append(dict_acc['train'])
        valid_loss_list.append(dict_loss['valid'])
        valid_acc_list.append(dict_acc['valid'])

    return train_loss_list, train_acc_list, valid_loss_list, valid_acc_list

