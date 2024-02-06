import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np
from copy import deepcopy

import pdb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformer Model for Style Shift (StyTR2)
# reference : https://github.com/diyiiyiii/StyTR-2/blob/main/models/transformer.py
class Transformer(nn.Module):
    
    def __init__(self, 
                d_model : int = 512, 
                nhead : int = 8, 
                num_encoder_layers : int = 3, 
                num_decoder_layers : int = 3, 
                dim_feedforward : int = 2048, 
                dropout : float = 0.1, 
                activation : str = "relu", 
                normalize_before : bool = False, 
                return_intermediate_dec : bool = False
                ):
    
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None

        self.encoder_c = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.encoder_s = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)

        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate_dec)

        # reset parameters
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.average_pooling = nn.AdaptiveAvgPool2d(18)
        self.new_ps = nn.Conv2d(512, 512, (1, 1))

    # reset parameters
    # description : reset parameters for transformer
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # forward function for transformer
    def forward(self, 
                style : torch.Tensor,
                content : torch.Tensor,
                mask : Optional[torch.Tensor] = None,
                pos_embed_content : Optional[torch.Tensor] = None,
                pos_embed_style : Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        
        # positional encoding for content
        content_pool = self.average_pooling(content)
        pos_content = self.new_ps(content_pool)
        pos_embed_content = F.interpolate(pos_content, mode='bilinear', size=style.shape[-2:])

        # flattent the style and content
        # N x C x H x W -> (H X W) x N x C
        content = content.flatten(2).permute(2, 0, 1)
        if pos_embed_content is not None:
            pos_embed_content = pos_embed_content.flatten(2).permute(2, 0, 1)

        style = style.flatten(2).permute(2, 0, 1)
        if pos_embed_style is not None:
            pos_embed_style = pos_embed_style.flatten(2).permute(2, 0, 1)

        # transformer encoder and decoder
        style = self.encoder_s(style, src_key_padding_mask = mask, pos = pos_embed_style)
        content = self.encoder_c(content, src_key_padding_mask = mask, pos = pos_embed_content)
        
        # transformer decoder
        hidden_state = self.decoder(tgt = content, memory = style, memory_key_padding_mask = mask, 
                                    pos = pos_embed_style, query_pos = pos_embed_content)[0]

        # (H X W) x N x C -> N x C x H x W
        N, B, C = hidden_state.shape
        W = H = int(np.sqrt(N))
        hidden_state = hidden_state.permute(1, 2, 0)
        hidden_state = hidden_state.view(B, C, H, W)
        
        return hidden_state 

# Transformer Encoder and Decoder

# Transformer Encoder
# description : transformer encoder
class TransformerEncoder(nn.Module):
    
    def __init__(self, 
                 encode_layer : nn.Module, 
                 num_layers : int, 
                 norm : nn.Module = None
                 ):

        super().__init__()

        # layer list is consist of num_layers of encode_layer
        self.layers = nn.ModuleList([deepcopy(encode_layer) for _ in range(num_layers)]) 
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, 
                x : torch.Tensor,
                mask : Optional[torch.Tensor] = None,
                src_key_padding_mask : Optional[torch.Tensor] = None,
                pos : Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        
        # forward function for each layer
        for layer in self.layers:
            x = layer(x, src_mask = mask, src_key_padding_mask = src_key_padding_mask, pos = pos)

        # normalization
        if self.norm is not None:
            x = self.norm(x)

        return x 

# Transformer Decoder
# description : transformer decoder
class TransformerDecoder(nn.Module):

    def __init__(
            self, 
            decode_layer : nn.Module, 
            num_layers : int, 
            norm : nn.Module = None, 
            return_intermediate : bool = False
            ) -> torch.Tensor:
        
        super().__init__()

        # layer list is consist of num_layers of decode_layer
        self.layers = nn.ModuleList([deepcopy(decode_layer)  for _ in range(num_layers)]) 
        self.num_layers = num_layers
        self.norm = norm

        # return intermediate
        # description : intermediate output for each layer
        self.return_intermediate = return_intermediate

    def forward(self, 
                tgt : torch.Tensor,
                memory : torch.Tensor,
                tgt_mask : Optional[torch.Tensor] = None,
                memory_mask : Optional[torch.Tensor] = None,
                tgt_key_padding_mask : Optional[torch.Tensor] = None,
                memory_key_padding_mask : Optional[torch.Tensor] = None,
                pos : Optional[torch.Tensor] = None,
                query_pos : Optional[torch.Tensor] = None
                ) -> torch.Tensor:

        """
        parameters
        tgt : content image
        memory : style image
        tgt_mask : mask for content image
        memory_mask : mask for style image
        tgt_key_padding_mask : key padding mask for content image
        memory_key_padding_mask : key padding mask for style image
        pos : positional encoding for style image
        query_pos : positional encoding for content image
        """

        # intermediate list (if return_intermediate is True)
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask = tgt_mask, memory_mask = memory_mask, tgt_key_padding_mask = tgt_key_padding_mask, 
                        memory_key_padding_mask = memory_key_padding_mask, pos = pos, query_pos = query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            tgt = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)
    
# Transformer Encoder and Decoder Layer

# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    
    def __init__(self, 
                 d_model : int, 
                 nhead : int, 
                 dim_feedforward : int = 2048, 
                 dropout : float = 0.1, 
                 activation : str = "relu", 
                 normalize_before : bool = False
                 ):
        
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Feed Forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    # positional encoding
    def positional_encoding(self, x : torch.Tensor, pos : Optional[torch.Tensor] = None) -> torch.Tensor:
        return x if pos is None else x + pos

    def forward(self, 
                src : torch.Tensor, 
                src_mask : Optional[torch.Tensor] = None, 
                src_key_padding_mask : Optional[torch.Tensor] = None, 
                pos : Optional[torch.Tensor] = None) -> torch.Tensor:
        
        if self.normalize_before:
            return self.forward_pre_norm(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post_norm(src, src_mask, src_key_padding_mask, pos)

    def forward_pre_norm(self, 
                        src : torch.Tensor, 
                        src_mask : Optional[torch.Tensor] = None, 
                        src_key_padding_mask : Optional[torch.Tensor] = None, 
                        pos : Optional[torch.Tensor] = None) -> torch.Tensor:

        src2 = self.norm1(src)
        q = k = self.positional_encoding(src2, pos)
        src2 = self.self_attn(q, k, value = src2, attn_mask = src_mask, 
                              key_padding_mask = src_key_padding_mask)[0]    
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)

        return src

    def forward_post_norm(self,
                        src : torch.Tensor, 
                        src_mask : Optional[torch.Tensor] = None, 
                        src_key_padding_mask : Optional[torch.Tensor] = None, 
                        pos : Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # query, key
        q = k = self.positional_encoding(src, pos)

        src2 = self.self_attn(q, k, value = src, attn_mask = src_mask, 
                              key_padding_mask = src_key_padding_mask)[0]

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
    
        return src

# Transformer Decoder Layer
class TransformerDecoderLayer(nn.Module):
        
    def __init__(self, 
                d_model : int, 
                nhead : int, 
                dim_feedforward : int = 2048, 
                dropout : float = 0.1, 
                activation : str = "relu", 
                normalize_before : bool = False
                ):
        
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Feed Forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    # positional encoding
    def positional_encoding(self, x : torch.Tensor, pos : Optional[torch.Tensor] = None) -> torch.Tensor:
        return x if pos is None else x + pos

    # forward function for decoder layer
    def forward(self,
                tgt : torch.Tensor, 
                memory : torch.Tensor, 
                tgt_mask : Optional[torch.Tensor] = None, 
                memory_mask : Optional[torch.Tensor] = None, 
                tgt_key_padding_mask : Optional[torch.Tensor] = None, 
                memory_key_padding_mask : Optional[torch.Tensor] = None, 
                pos : Optional[torch.Tensor] = None, 
                query_pos : Optional[torch.Tensor] = None) -> torch.Tensor:
        
        if self.normalize_before:
            return self.forward_pre_norm(tgt, memory, tgt_mask, memory_mask, 
                                         tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post_norm(tgt, memory, tgt_mask, memory_mask, 
                                      tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

    def forward_pre_norm(self, 
                        tgt : torch.Tensor, 
                        memory : torch.Tensor, 
                        tgt_mask : Optional[torch.Tensor] = None, 
                        memory_mask : Optional[torch.Tensor] = None, 
                        tgt_key_padding_mask : Optional[torch.Tensor] = None, 
                        memory_key_padding_mask : Optional[torch.Tensor] = None, 
                        pos : Optional[torch.Tensor] = None, 
                        query_pos : Optional[torch.Tensor] = None) -> torch.Tensor:

        tgt2 = self.norm1(tgt)
        q = k = self.positional_encoding(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value = tgt2, attn_mask = tgt_mask, key_padding_mask = tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query = self.positional_encoding(tgt2, query_pos), 
                                   key = self.positional_encoding(memory, pos), 
                                   value = memory, attn_mask = memory_mask, 
                                   key_padding_mask = memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt
    
    def forward_post_norm(self,
                        tgt : torch.Tensor, 
                        memory : torch.Tensor, 
                        tgt_mask : Optional[torch.Tensor] = None, 
                        memory_mask : Optional[torch.Tensor] = None, 
                        tgt_key_padding_mask : Optional[torch.Tensor] = None, 
                        memory_key_padding_mask : Optional[torch.Tensor] = None, 
                        pos : Optional[torch.Tensor] = None, 
                        query_pos : Optional[torch.Tensor] = None) -> torch.Tensor:
        
        q = self.positional_encoding(tgt, query_pos)
        k = self.positional_encoding(memory, pos)
        v = memory

        tgt2 = self.self_attn(
            q, k, v, 
            attn_mask = tgt_mask, 
            key_padding_mask = tgt_key_padding_mask
            )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query = self.positional_encoding(tgt, query_pos), 
            key = self.positional_encoding(memory, pos), 
            value = memory, attn_mask = memory_mask, 
            key_padding_mask = memory_key_padding_mask
            )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


# activation mapping function
# description : activation name to activation function
def _get_activation_fn(activation : str) -> Optional[nn.Module]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")