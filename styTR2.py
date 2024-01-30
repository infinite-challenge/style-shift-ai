import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import collections.abc as container_abcs
from itertools import repeat


def compute_mean_std(feature, eps=1e-6):
    """
    compute features' mean and standard deviation
    """
    size = feature.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feature_var = feature.view(N, C, -1).var(dim=2) + eps
    feature_std = feature_var.sqrt().view(N,C,1,1)
    feature_mean = feature.view(N, C, -1).mean(dim=2).view(N,C,1,1)
    return feature_mean, feature_std

def normalize(feature, eps=1e-5):
    """
    normalized feature by using mean std
    """
    feature_mean, feature_std = compute_mean_std(feature, eps)
    normalized = (feature-feature_mean)/feature_std
    return normalized

def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

############################
# Image to patch embedding #
############################
class embedding(nn.Moduel):
    """
    To use Embedding,
    embedding = *.embedding()
    """
    def __init__(self, img_size = 256, patch_size=8, in_channels=3, embedding_dimension=512):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        n_patch = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patch = n_patch
        self.proj = nn.Conv2d(in_channels, embedding_dimension, kernel_size=patch_size, stride=patch_size)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        return x
    
########################################
# Convolution Module : vgg and decoder #
########################################
class Convolutions(nn.Module):
    """
    To Use vgg and decoder from this 'Convolutions' class,
    convolution_module = *.Covolutions()
    decoder = convolution_module.decoder
    vgg = convolution_module.vgg
    """
    def __init__(self):
        super(Convolutions, self).__init__()

        self.decoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 256, (3, 3)),
        )

        self.vgg = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU()
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU()
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU()
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU()
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU()
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU()
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU()
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.decoder(x), self.vgg(x)


#######################################
# Style Transforms Transformer Module #
#######################################
class StyTrans(nn.Module):
    """
    To use style transforms transformer,
    ex) network = *.StyTrans(...)
    """
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
        assert (input.size()) == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = compute_mean_std(input)
        target_mean, target_std = compute_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, sample_c : torch.tensor, sample_s : torch.tensor):
        """
        return stylized image and computed losses
        """
        content_features = self.checkpt_of_ecnoder(sample_c)
        style_features = self.checkpt_of_ecnoder(sample_s)

        style = self.embedding(sample_c)
        content = self.embedding(sample_s)

        embed_position_c = None
        embed_position_s = None
        mask = None
        Ics = self.decoder(self.transformer(style, mask, content, embed_position_c, embed_position_s))
        Ics_features = self.checkpt_of_ecnoder(Ics)

        loss_c = 0
        for i in [-1, -2]:
            loss_c += self.compute_content_loss(normalize(Ics_features[i]), 
                                                normalize(content_features[i]))
        loss_s = 0
        for i in range(5):
            loss_s += self.compute_style_loss(Ics_features[i], 
                                              style_features[i])
        
        Icc = self.decoder(self.transformer(content, mask, content, embed_position_c, embed_position_c))
        Iss = self.decoder(self.transformer(style, mask, style, embed_position_s, embed_position_s))

        Icc_features = self.checkpt_of_ecnoder(Icc)
        Iss_features = self.checkpt_of_ecnoder(Iss)

        loss_l1 = 0
        for I, sample in zip([Icc, Iss],[sample_c, sample_s]):
            loss_l1 += self.compute_content_loss(I, sample)

        loss_l2 = 0
        for i in range(5):
            loss_l2 += self.compute_content_loss(Icc_features[i], 
                                                 content_features[i])
            loss_l2 += self.compute_content_loss(Iss_features[i], 
                                                 style_features[i])

        return Ics, loss_c, loss_s, loss_l1, loss_l2