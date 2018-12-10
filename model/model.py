import torch
import torch.nn as nn
import torchvision.models as models

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
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def Pretrained(params):
    # Load the pretrained model from torchvision
    resnet18 = models.resnet18(pretrained=True)
    # Freeze the parameters in the conv layers
    for param in resnet18.parameters():
        param.requires_grad = False
    # Replace the last fully-connected layer to our needs
    # extract the number of input features to the last fc layer
    num_features = resnet18.fc.in_features
    # construct a new fc layer with our number of output classes
    resnet18.fc = nn.Linear(num_features, params['nc'])

    return resnet18


