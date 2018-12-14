from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def create_dataloader(normalization, directories, batch_size):
    """
    A function to create dataloader based on specified parameters and path
    """
    options = ['train', 'valid']
    # Defining transformations
    data_transforms = {
        options[0]: transforms.Compose([transforms.Resize(256),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     normalization]),
        options[1]: transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     normalization])
    }
    image_datasets = {x: datasets.ImageFolder(root=str(directories[x]),
                        transform=data_transforms[x]) for x in options
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in options}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
                    for x in options
    }

    return dataset_sizes, dataloaders

