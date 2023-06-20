from engine import train
import torch
from models import ViT
import data_setup

vit = ViT(num_classes=3)

train_dataloader, test_dataloader = data_setup.get_dataloaders()






optimizer = torch.optim.Adam(params=vit.parameters(),
                             lr=3e-3,
                             betas=(0.9,0.999),
                             weight_decay=0.3)

loss_fn = torch.nn.CrossEntropyLoss()

results = train(model=vit,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=20,
                device="cuda")