
import torch
from torch import nn
import os 
from pathlib import Path
import zipfile
import matplotlib.pyplot as plt

def set_seeds(seed: int=42):
    #sets random sets for torch operations 

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def download_data(source: str,
                  destination: str,
                  remove_source: bool = True) -> Path:
    #Downloads a zupped dataset from source and unips to destination.

    data_path = Path("data/")
    image_path = data_path / destination

    if image_path.is_dir():
        print(f"{image_path} already exists, skipping download")
    else:
        print(f"did not find {image_path}, creating one")
        image_path.mkdir(parents=True, exist_ok=True)

        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f: 
            request = request.get(source)
            print(f"downloading {target_file} from {source}")
            f.write(request.content)

        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"unzipping {target_file}")
            zip_ref.extractall(image_path)

        if remove_source:
            os.remove(data_path/target_file)

    return image_path


def plot_loss_curves(results):
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()



def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    
    #create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    #create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt")
    model_save_path = target_dir_path / model_name
    
    print(f"saving model to {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
