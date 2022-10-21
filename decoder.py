"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
import numpy as np

from myProj.encoder import Posenc, PositionalEncoding
from myProj.neural import MultiHeadedAttention, PositionwiseFeedForward, DecoderState

MAX_SIZE = 5000


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, ff, dropout,sep_dec=False) -> None:
        super(DecoderLayer, self).__init__()

        self.attention = MultiHeadedAttention(
            heads, d_model, dropout=dropout, sep_dec=sep_dec)

        # self.contextAttn=MultiHeadedAttention(heads,)
        self.feedin = PositionwiseFeedForward(
            d_model=d_model, dropout=dropout, d_ff=ff)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        mask = torch.from_numpy(
            np.triu(np.ones((1, MAX_SIZE, MAX_SIZE), k=1).astype('uint8')))
        self.register_buffer('mask', mask)

    def forward(self, inp, mem_bank, sr_pad, tg_pad, prev_inp=None, lay_cah=None, tgt=None):
        dmask = None
        norm_inp = self.layer_norm(inp)
        store_inp = norm_inp

        if prev_inp is not None:
            store_inp = torch.cat((prev_inp, norm_inp), dim=1)
        else:

            dmask = torch.gt(tg_pad +
                             self.mask[:, :tg_pad.size(1),
                                       :tg_pad.size(1)], 0)
        q = self.attention(store_inp, store_inp, norm_inp,
                           mask=dmask, lay_cah=lay_cah, type="self", tgt=tgt)
        q = self.drop(q)+inp
        q_new = self.layer_norm(q)
        output = self.feedin(self.drop(self.attention(mem_bank, mem_bank, q_new,
                                                      mask=sr_pad,
                                                      layer_cache=lay_cah,
                                                      type="context", tgt_segs=tgt)))
        return output,store_inp

class TransDecoderState(DecoderState):
    def __init__(self,args) -> None:
        super(TransDecoderState,self).__init__()
        self.args=args
        self.prev_layer=None

# defines entire decoder
class Decoder(nn.Module):
    def __init__(self,layers,d_model,heads,ff,dropout,embed,sep_dec=False) -> None:
        super(Decoder,self).__init__()
        self.sep_model=d_model
        self.layers=layers
        self.embedding=embed
        self.pos_emb=Posenc(dropout,embed.embedding_dim)
        self.main_layer=nn.ModuleList([DecoderLayer(self.sep_model,heads,ff,dropout,sep_dec)] for i in range(layers))

        self.layer_norm=nn.LayerNorm(d_model,eps=1e-6)
    