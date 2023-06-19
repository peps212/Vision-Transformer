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



class MultiheadSelfAttentionBlock(torch.nn.Module):
    #Creates a multihead self attention block

    def __init__(self,
                 embedding_dim=768,
                 num_heads=12,
                 attn_dropout=0
                 ):
        super().__init__()

        #create norm layer

        self.layer_norm = torch.nn.LayerNorm(normalized_shape=embedding_dim)

        #create the multihead attention layer 
        self.multihead_attn = torch.nn.MultiheadAttention(embed_dim=embedding_dim,
                                                          num_heads=num_heads,
                                                          dropout=attn_dropout,
                                                          batch_firts=True)
        

    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x, key=x, value=x, need_weights=False)

        return attn_output
    




class MLPBlock(torch.nn.Module):
    #creates a layer normalized multilayer perceptron block
        def __init__(self,
                 embedding_dim:int=768, # Hidden Size D from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 dropout:float=0.1):
            super().__init__()

            self.layer_norm = torch.nn.LayerNorm(normalized_shape=embedding_dim)

            self.mlp = torch.nn.Sequential(
                 torch.nn.Linear(in_features=embedding_dim, out_features=mlp_size),
                 torch.nn.GELU(),
                 torch.nn.Dropout(p=dropout),
                 torch.nn.Linear(in_features=mlp_size, # needs to take same in_features as out_features of layer above
                      out_features=embedding_dim),
                 torch.nn.Dropout(p=dropout)
            )

        def forward(self, x):
            x = self.layer_norm(x)
            x = self.mlp(x)
            return x
        

class TransformerEncoderBlock(torch.nn.Module):
    """Creates a Transformer Encoder block."""
    # 2. Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 embedding_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 mlp_dropout:float=0.1, # Amount of dropout for dense layers from Table 3 for ViT-Base
                 attn_dropout:float=0): # Amount of dropout for attention layers
        super().__init__()

        # 3. Create MSA block (equation 2)
        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                     num_heads=num_heads,
                                                     attn_dropout=attn_dropout)
        
        # 4. Create MLP block (equation 3)
        self.mlp_block =  MLPBlock(embedding_dim=embedding_dim,
                                   mlp_size=mlp_size,
                                   dropout=mlp_dropout)
        
    # 5. Create a forward() method  
    def forward(self, x):
        
        # 6. Create residual connection for MSA block (add the input to the output)
        x =  self.msa_block(x) + x 
        
        # 7. Create residual connection for MLP block (add the input to the output)
        x = self.mlp_block(x) + x 
        
        return x


class ViT(torch.nn.Module):
    #Creates a vision transformer architecture with ViT-Base Hyperparameters by default
    def __init__(self,
                 img_size:int=224, # Training resolution from Table 3 in ViT paper
                 in_channels:int=3, # Number of channels in input image
                 patch_size:int=16, # Patch size
                 num_transformer_layers:int=12, # Layers from Table 1 for ViT-Base
                 embedding_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 attn_dropout:float=0, # Dropout for attention projection
                 mlp_dropout:float=0.1, # Dropout for dense/MLP layers 
                 embedding_dropout:float=0.1, # Dropout for patch and position embeddings
                 num_classes:int=1000): # Default for ImageNet but can customize this
        super().__init__() # don't forget the super().__init__()!

        assert img_size % patch_size == 0
        self.num_patches = (img_size * img_size) // patch_size**2

        #create learnable class embedding
        self.class_embedding = torch.nn.Parameters(data=torch.randn(1,1, embedding_dim), requires_grad=True)

        #Create learnable position embedding
        self.position_embedding = torch.nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim),
                                               requires_grad=True)
        
        #Create embedding dropout value
        self.embedding_dropout = torch.nn.Dropout(p=embedding_dropout)

        # 8. Create patch embedding layer
        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)
        
        # 9. Create Transformer Encoder blocks (we can stack Transformer Encoder blocks using nn.Sequential()) 
        # Note: The "*" means "all"
        self.transformer_encoder = torch.nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                            num_heads=num_heads,
                                                                            mlp_size=mlp_size,
                                                                            mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)])
        # 10. Create classifier head
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(normalized_shape=embedding_dim),
            torch.nn.Linear(in_features=embedding_dim, 
                      out_features=num_classes)
        )
    
    def forward(self, x):
        
        # 12. Get batch size
        batch_size = x.shape[0]
        
        # 13. Create class token embedding and expand it to match the batch size (equation 1)
        class_token = self.class_embedding.expand(batch_size, -1, -1) # "-1" means to infer the dimension (try this line on its own)

        # 14. Create patch embedding (equation 1)
        x = self.patch_embedding(x)

        # 15. Concat class embedding and patch embedding (equation 1)
        x = torch.cat((class_token, x), dim=1)

        # 16. Add position embedding to patch embedding (equation 1) 
        x = self.position_embedding + x

        # 17. Run embedding dropout (Appendix B.1)
        x = self.embedding_dropout(x)

        # 18. Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3)
        x = self.transformer_encoder(x)

        # 19. Put 0 index logit through classifier (equation 4)
        x = self.classifier(x[:, 0]) # run on each sample in a batch at 0 index

        return x