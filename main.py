import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


#######################################
# Style Transforms Transformer Module #
#######################################

class StyTrans(nn.Module):
    def __init__(self, cnn, decoder, embedding, transformer, args):
        super().__init__()
        cnn_layers = list(cnn.children())
        self.cnn_1 = nn.Sequential(*cnn_layers[:4])
        self.cnn_2 = nn.Sequential(*cnn_layers[4:11])
        self.cnn_3 = nn.Sequential(*cnn_layers[11:18])
        self.cnn_4 = nn.Sequential(*cnn_layers[18:31])
        self.cnn_5 = nn.Sequential(*cnn_layers[31:44])

        for name in ['cnn_1','cnn_2','cnn_3','cnn_4','cnn_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.mse_loss = nn.MSELoss()
        self.tranformer = transformer
        self.decoder = decoder
        self.embedding = embedding

        
