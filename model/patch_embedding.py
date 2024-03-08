import torch
import torch.nn as nn
from ..utils import to_2tuple

############################
# Image to patch embedding #
############################
class PatchEmbedding(nn.Module):
    """
    To use Embedding,
    embedding = *.embedding()
    """
    def __init__(
            self, 
            img_size : int = 256, 
            patch_size : int = 8, 
            in_channels : int = 3, 
            embedding_dimension : int = 512
            ):
        
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        n_patch = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patch = n_patch
        self.proj = nn.Conv2d(in_channels, embedding_dimension, kernel_size=patch_size, stride=patch_size)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x) -> torch.Tensor:
        x = self.proj(x)
        return x