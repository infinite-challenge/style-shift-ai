import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .utils import *
import pdb
#######################################
# Style Transforms Transformer Module #
#######################################
class StyTrans(nn.Module):
    """
    To use style transforms transformer,
    ex) network = *.StyTrans(...)
    """
    def __init__(self, cnn, decoder, embedding, transformer):
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
        self.transformer = transformer
        self.decoder = decoder
        self.embedding = embedding

    def checkpt_of_ecnoder(self, input):
        """ 
        set checkpoint for each blocks of cnn (cnn_* )
        """
        results = [input]
        for i in range(1, 6):
            func = getattr(self, 'cnn_{:d}'.format(i))
            results.append(func(results[-1]))
        return results[1:]
    
    def compute_content_loss(self, input, target):
        """ 
        compute mse_loss between input(decoder output) and target(original content or style)
        """
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)
    
    def compute_style_loss(self, input, target):
        """ 
        compute mse_loss between input(decoder output) and target(original style)
        """
        assert input.size() == target.size()
        assert (target.requires_grad is False)
        input_mean, input_std = compute_mean_std(input)
        target_mean, target_std = compute_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, sample_c : torch.tensor, sample_s : torch.tensor):
        """
        return stylized image and computed losses
        """
        content_input = sample_c
        style_input = sample_s

        content_features = self.checkpt_of_ecnoder(sample_c)
        style_features = self.checkpt_of_ecnoder(sample_s)

        style = self.embedding(sample_s)
        content = self.embedding(sample_c)

        embed_position_c = None
        embed_position_s = None
        mask = None
        hs = self.transformer(style, content, mask, embed_position_c, embed_position_s)
        Ics = self.decoder(hs)
        
        Ics_features = self.checkpt_of_ecnoder(Ics)
        loss_c = 0
        for i in [-1, -2]:
            loss_c += self.compute_content_loss(normalize(Ics_features[i]), 
                                                normalize(content_features[i]))
        loss_s = 0
        for i in range(5):
            loss_s += self.compute_style_loss(Ics_features[i], 
                                              style_features[i])
        
        Icc = self.decoder(self.transformer(content, content, mask, embed_position_c, embed_position_c))
        Iss = self.decoder(self.transformer(style, style, mask, embed_position_s, embed_position_s))


        loss_l1 = 0
        for I, sample in zip([Icc, Iss],[content_input, style_input]):
            loss_l1 += self.compute_content_loss(I, sample)

        Icc_features = self.checkpt_of_ecnoder(Icc)
        Iss_features = self.checkpt_of_ecnoder(Iss)

        loss_l2 = 0
        for i in range(5):
            loss_l2 += self.compute_content_loss(Icc_features[i], 
                                                 content_features[i])
            loss_l2 += self.compute_content_loss(Iss_features[i], 
                                                 style_features[i])

        return Ics, loss_c, loss_s, loss_l1, loss_l2