import argparse
from pathlib import Path
import os
from typing import Any, Optional
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from os.path import basename, splitext

from torch.utils import data
from torchvision import transforms
from torchvision.utils import save_image

import model.decoder as decoder
import model.vgg as vgg
from model.patch_embedding import PatchEmbedding
from model.transformer import Transformer
from utils import data_transform, content_transform, img_tensor_2_pil

import styTR2

from collections import OrderedDict
import argparse
import lightning as pl
from collections import OrderedDict

from dataloader import ImageDataModule, LitCIFAR10DataModule, CustomImageDataset
from model.sampler import InfiniteSampleWrapper

from tqdm import tqdm
import pdb

parser = argparse.ArgumentParser()

parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--style', type=str,
                    help='File path to the style image or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output images')

parser.add_argument('--mode', type=str, default='test',
                    help='The mode of the model (train/test)')

parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')
parser.add_argument('--decoder_path', type=str, default='experiments/decoder_iter_160000.pth')
parser.add_argument('--trans_path', type=str, default='experiments/transformer_iter_160000.pth')
parser.add_argument('--embedding_path', type=str, default='experiments/embedding_iter_160000.pth')

parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_decay', type=float, default=1e-5)
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
NUM_WORKERS=int(os.cpu_count()/2)

random_seed = 42
torch.manual_seed(random_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LightningStyleShift(pl.LightningModule):
    def __init__(self, 
                cnn, 
                decoder, 
                embedding, 
                transformer,
                content_weight : float = 7.0,
                style_weight : float = 10.0,
                l_identity1_weight : float = 70.0,
                l_identity2_weight : float = 1.0,
                lr : float = 5e-4,
                lr_decay : float = 1e-5,
                output_path : str = './output/test',
                 ):
        super().__init__()
        self.model = styTR2.StyTrans(cnn, decoder, embedding, transformer)
        self.criterion = nn.MSELoss()
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.l_identity1_weight = l_identity1_weight
        self.l_identity2_weight = l_identity2_weight
        self.lr = lr
        self.lr_decay = lr_decay
        self.training_step_outputs = []
        self.output_path = output_path

    def foward(self, content, style):
        return self.model(content, style)

    def evaluate_custom_lr_lambda(self, epoch):
        if epoch < 1e4:
            return 0.1 * (1.0 + 3e-4 * epoch)
        else:
            return 2e-4 / self.lr / (1.0 + self.lr_decay * (epoch - 1e4)) 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.model.decoder.parameters()},
            {'params': self.model.transformer.parameters()},
            {'params': self.model.embedding.parameters()},
        ], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.evaluate_custom_lr_lambda)
        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, metric: Any | None) -> None:
        scheduler.step()

    def training_step(self, batch, batch_idx):
        
        content, style = batch
        content = content[0]
        style = style[0]
        # content, style = batch

        Ics, loss_c, loss_s, loss_l1, loss_l2 = self.model(content, style)

        loss_c = self.content_weight * loss_c
        loss_s = self.style_weight * loss_s
        loss_l1 = self.l_identity1_weight * loss_l1
        loss_l2 = self.l_identity2_weight * loss_l2

        loss = loss_c + loss_s + loss_l1 + loss_l2

        # training output save
        out = torch.cat((content, Ics), dim=0)
        out = torch.cat((style, out), dim=0)

        self.training_step_outputs.append(out.to('cpu'))

        self.log_dict({
            'loss': loss,
            'loss_c': loss_c,
            'loss_s': loss_s,
            'loss_l1': loss_l1,
            'loss_l2': loss_l2,
        }, on_step=True, prog_bar=True, on_epoch=True, logger=True)

        return loss
    
    def forward(self, content, style):
        return self.model(content, style)
    
    def on_train_epoch_start(self):
        print(f'epoch: {self.current_epoch}')

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> int | None:
        print(f'batch_idx: {batch_idx}')
        print(f'epoch: {self.current_epoch}')

    def on_train_epoch_end(self):
        # every 100 epochs, save the output
        if self.current_epoch % 2 == 0:
            output_file = f'{self.output_path}/epoch_{self.current_epoch}'
            for i in range(len(self.training_step_outputs)):
                save_image(self.training_step_outputs[i], output_file+f'_{i}.jpg')

        # every 1000 epochs, save the state dict
        if self.current_epoch % 1000 == 0 and self.current_epoch != 0:
            transformer_dict = self.model.transformer.state_dict()
            for key in transformer_dict.keys():
                transformer_dict[key] = transformer_dict[key].to('cpu')
            torch.save(transformer_dict, f'{self.output_path}/transformer_iter_{self.current_epoch}.pth')
            embedding_dict = self.model.embedding.state_dict()
            for key in embedding_dict.keys():
                embedding_dict[key] = embedding_dict[key].to('cpu')
            torch.save(embedding_dict, f'{self.output_path}/embedding_iter_{self.current_epoch}.pth')
            decoder_dict = self.model.decoder.state_dict()
            for key in decoder_dict.keys():
                decoder_dict[key] = decoder_dict[key].to('cpu')
            torch.save(decoder_dict, f'{self.output_path}/decoder_iter_{self.current_epoch}.pth')

        self.training_step_outputs.clear()

def data_transform():
    transfrom_list = [
        transforms.Resize(size = (512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transfrom_list)

def imgtensor2pil(img_tensor):
    pil_img = transforms.ToPILImage()(img_tensor)
    return pil_img

mode = args.mode
output_path = args.output

content_weight : float = 7.0
style_weight : float = 10.0
l_identity1_weight : float = 70.0
l_identity2_weight : float = 1.0

if not os.path.exists(output_path):
    os.makedirs(output_path)

if __name__ == '__main__':

    if mode == 'train':    
        network = styTR2.StyTrans(vgg.vgg, decoder.decoder, PatchEmbedding(), Transformer())

        optimizer = torch.optim.Adam([
            {'params': network.decoder.parameters()},
            {'params': network.transformer.parameters()},
            {'params': network.embedding.parameters()},
        ], lr=args.lr)    
        # load the data
        for epoch in tqdm(range(10)):
            if epoch < 1e4:
                lr = args.lr * 0.1 * (1.0 + 3e-4 * epoch)
            else:
                lr = 2e-4 / (1.0 + args.lr_decay * (epoch- 1e4))
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            data_tf = data_transform()
            content_dataset = CustomImageDataset("./content", data_tf)
            style_dataset = CustomImageDataset("./style", data_tf)

            content = data.DataLoader(
                content_dataset, batch_size = 8,
                # sampler=InfiniteSampleWrapper(content_dataset), 
                num_workers=0)
            style = data.DataLoader(
                style_dataset, batch_size = 8,
                # sampler=InfiniteSampleWrapper(style_dataset), 
                num_workers=0)
                        
            # image_batch = zip(content, style)
            # pdb.set_trace()            
            # content, style = data_module.train_dataloader()

            Ics, loss_c, loss_s, loss_l1, loss_l2 = network(content, style)

            if epoch % 5 == 0:
                output_name = '{:s}/test/{:s}{:s}'.format(
                                args.save_dir, str(epoch),".jpg"
                            )
                out = torch.cat((content,Ics),0)
                out = torch.cat((style,out),0)
                save_image(out, output_name)

            loss_c = content_weight * loss_c
            loss_s = style_weight * loss_s
            loss_l1 = l_identity1_weight * loss_l1
            loss_l2 = l_identity2_weight * loss_l2
            loss = loss_c + loss_s + loss_l1 + loss_l2

            print(loss.sum().cpu().detach().numpy(),"-content:",loss_c.sum().cpu().detach().numpy(),"-style:",loss_s.sum().cpu().detach().numpy()
              ,"-l1:",loss_l1.sum().cpu().detach().numpy(),"-l2:",loss_l2.sum().cpu().detach().numpy()
              )
            
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
        # content_dataset = ImageDataModule(args.content, data_tf)
        # style_dataset = ImageDataModule(args.style, data_tf)

        # content = data.DataLoader(
        #     content_dataset, batch_size = 4,
        #     sampler=InfiniteSampleWrapper(content_dataset), 
        #     num_workers=8)
        # style = data.DataLoader(
        #     style_dataset, batch_size = 4,
        #     sampler=InfiniteSampleWrapper(style_dataset), 
        #     num_workers=8)

        # image_batch = zip(content, style)

        # load the model

    elif mode == 'test':
        # load the state dict
        vgg = vgg.vgg
        decoder = decoder.decoder
        transformer = Transformer()
        embedding = PatchEmbedding()
        
        vgg.load_state_dict(torch.load(args.vgg))
        vgg = nn.Sequential(*list(vgg.children())[:44])

        decoder.eval()
        transformer.eval()
        vgg.eval()

        new_decoder_state_dict = OrderedDict()
        decoder_state_dict = torch.load(args.decoder_path)
        for k, v in decoder_state_dict.items():
            name = k
            new_decoder_state_dict[name] = v
        
        decoder.decoder.load_state_dict(new_decoder_state_dict)

        new_embedding_state_dict = OrderedDict()
        embedding_state_dict = torch.load(args.embedding_path)

        for k, v in embedding_state_dict.items():
            name = k
            new_embedding_state_dict[name] = v

        embedding.load_state_dict(new_embedding_state_dict)

        new_transformer_state_dict = OrderedDict()
        transformer_state_dict = torch.load(args.trans_path)

        for k, v in transformer_state_dict.items():
            name = k
            new_transformer_state_dict[name] = v

        transformer.load_state_dict(new_transformer_state_dict)

        network = styTR2.StyTrans(vgg, decoder, embedding, transformer)
        network.eval()
        network.to(device) # Edit

        content = Image.open(args.content).convert('RGB')
        style = Image.open(args.style).convert('RGB')

        content_size = content.size

        content_tf = data_transform(content_size)
        style_tf = data_transform(style_size)

        resize_tf = content_transform(content_size)

        content = content_tf(content)
        style = style_tf(style)

        content = content.to(device).unsqueeze(0)
        style = style.to(device).unsqueeze(0)

        with torch.no_grad():
            output_tensor = network(content, style)

        output_tensor.cpu()

        output = resize_tf(output_tensor)

        output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(output_path, splitext(basename(args.content))[0],
                                                        splitext(basename(args.style))[0], save_extension)

        save_image(output, output_name)