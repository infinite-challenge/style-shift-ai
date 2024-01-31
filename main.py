import argparse
from pathlib import Path
import os
import torch
import torch.nn as nn
import numpy as np

from os.path import basename, splitext

from torchvision import transforms
from torchvision.utils import save_image
import model.decoder as decoder
import model.vgg as vgg
import model.patch_embedding as embedding
import transformer
import styTR2

from collections import OrderedDict
import lightning as pl


parser = argparse.ArgumentParser()

parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--style', type=str,
                    help='File path to the style image or multiple style \
                    images seperated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output images')

parser.add_argument('--mode', type=str, default='test',
                    help='The mode of the model (train/test)')

parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')
parser.add_argument('--decoder_path', type=str, default='experiments/decoder_iter_160000.pth')
parser.add_argument('--Trans_path', type=str, default='experiments/transformer_iter_160000.pth')
parser.add_argument('--embedding_path', type=str, default='experiments/embedding_iter_160000.pth')

parser.add_argument('--style_interpolation_weight', type=str, default="")
parser.add_argument('--a', type=float, default=1.0)
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine','learned'),
                    help='type of positional embedding to use on top of the image features')
parser.add_argument('--hidden_dim', default=512, type=int,
                    help='size of the embeddings (dimensions of the transformer)')
args = parser.parse_args()

content_size = (512, 512)
style_size = (512, 512)
crop_size = 256
save_extension = '.jpg'
output_path = args.output
preserve_color = 'store_true'
alpha = args.a

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LightningStyleShift(pl.LightningModule):
    def __init__(self, 
                 cnn, 
                 decoder, 
                 embedding, 
                 transformer
                 ):
        super().__init__()
        self.model = styTR2.StyTrans(cnn, decoder, embedding, transformer)
        self.criterion = nn.MSELoss()

    def foward(self, content, style):
        return self.model(content, style)
    
    def training_step(self, batch, batch_idx):
        content, style = batch
        Ics, loss_c, loss_s, loss_l1, loss_l2 = self.model(content, style)

        
        


        return loss
    

def load_dict(module, file_path : str):
    new_file_dict = OrderedDict()
    file_dict = torch.load(file_path)
    for key, value in file_dict.items():
        new_file_dict[key] = value
    return module.load_state_dict(new_file_dict)

h, w = content_size

with torch.no_grad():
    y_hat = network(content, style)

y_hat.cpu()
output = transforms.Compose([
    transforms.Resize((h, w), antialias=True)])(y_hat)

output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(output_path, splitext(basename(content_path))[0],
                                                   splitext(basename(style_path))[0], save_extension)

save_image(output, output_name)