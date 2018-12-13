import torch
import torch.nn as nn
import torchvision.models as models

class SimpleCNN(nn.Module):
    """
    A class for simple CNN classifier.
    """
    def __init__(self, args, params):
        """
        Define a constructor and initialize the network.
        """
        super(SimpleCNN, self).__init__()
        self.args = args
        self.params = params
        # Defining your own block
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block5 = nn.Sequential(
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
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        # flatten layer
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def Pretrained(args, params):
    # Load the pretrained model from torchvision
    if args.model_type == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif args.model_type == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif args.model_type == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif args.model_type == 'resnet101':
        model = models.resnet101(pretrained=True)
    elif args.model_type == 'resnet152':
        model = models.resnet152(pretrained=True)
    else:
        raise "There is no model you specified."
    # Freeze the parameters in the conv layers
    for param in model.parameters():
        param.requires_grad = False
    # Replace the last fully-connected layer to our needs
    # extract the number of input features to the last fc layer
    num_features = model.fc.in_features
    # construct a new fc layer with our number of output classes
    model.fc = nn.Linear(num_features, params['nc'])

    return model


