import os 
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

print(NUM_WORKERS)

def create_dateloaders(
    train_dir: str,
    test_dir: str, 
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS
):
    
    # Creates training and testing Data Loaders.

    # Takes in a training directory and testing directory path and turns
    # them into Pytorch datasets and then into pytorch DataLoaders.
    
    #   Args:
    #     train_dir: Path to training directory.
    #     test_dir: Path to testing directory.
    #     transform: torchvision transforms to perform on training and testing data.
    #     batch_size: Number of samples per batch in each of the DataLoaders.
    #     num_workers: An integer for number of workers per DataLoader.

    #   Returns:
    #     A tuple of (train_dataloader, test_dataloader, class_names).
    #     Where class_names is a list of the target classes.
    #     Example usage:
    #       train_dataloader, test_dataloader, class_names = \
    #         = create_dataloaders(train_dir=path/to/train_dir,
    #                              test_dir=path/to/test_dir,
    #                              transform=some_transform,
    #                              batch_size=32,
    #                              num_workers=4)


    # using ImageFolder to create datasets
    train_data = datasets.ImageFolder(train_dir, transform=transforms)
    test_data = datasets.ImageFolder(test_dir, transform=transforms)

    #get class names
    class_names = train_data.classes 

    #turn images into data loaders 
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle= True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle= True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )



    return train_dataloader, test_dataloader, class_names