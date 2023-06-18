import os 
import torch 
import data_setup, engine, utils 
from torchvision import transforms 


class PatchEmbedding(torch.nn.Module):
    #Turns a 2D input image into a 1D sequence learnable embedding vector
    
    def __init__(self,
                 in_channels:int=3,
                 patch_size:int=16,
                 embedding_dim:int=768):
        super().__init__()

    # Create a layer to turn an image into patches 
        self.patch_size = patch_size

        self.patcher = torch.nn.Conv2d(in_channels=in_channels, 
                                out_channels= embedding_dim,
                                kernel_size=patch_size,
                                stride=patch_size,
                                padding= 0)
    
    # Create a layer to flatten the patch feature maps into a single dimension 

        self.flatten = torch.nn.Flatten(start_dim=2, end_dim=3)

    
    #define the forward method 

    def forward(self, x):
        #create assertion to check that inputs are the correct shape 
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0 

        #perform forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        return x_flattened.permute(0,2,1)
    



train_dataloader = data_setup.get_dataloaders()

image_batch, label_batch = next(iter(train_dataloader))

image = image_batch[0]

patchify = PatchEmbedding(in_channels=3, patch_size=16, embedding_dim=768)

processed_image = patchify(image.unsqueeze(0))
print(processed_image.shape)