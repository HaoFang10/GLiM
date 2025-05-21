import torch.nn as nn
from typing import Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph import Graph as G, apply, add_Graph, Graph_like    

import math



class EncLayer(nn.Module):
    def __init__(self, dim_in, dim_qk, dim_v, dim_ff, n_heads, dropout=0., drop_mu=0., return_attn=False):
        super().__init__()
        self.return_attn = return_attn
        self.add = Add()
        self.ln = Apply(nn.LayerNorm(dim_in))
        self.attn = SelfAttn(n_heads=n_heads, d_in=dim_in, d_out=dim_in, d_qk=dim_qk, d_v=dim_v)  # <-
        self.ffn = Apply(nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(dim_ff, dim_in)
        ))

    def forward(self, G):
        h = self.ln(G)
        attn_score, h = self.attn(h)
        G = self.add(G, h)
        h = self.ffn(G)
        return (attn_score, self.add(G, h)) if self.return_attn else self.add(G, h)


class tokenGT(nn.Module):
    def __init__(self, n_layers: int, dim_in, dim_out, dim_hidden, dim_qk, dim_v, dim_ff, n_heads, drop_input=0.,
                 dropout=0., drop_mu=0., last_layer_n_heads=16):
        super().__init__()
        # assert last_layer_n_heads >= 16
        self.input = Apply(
            nn.Sequential(
                nn.Linear(dim_in, dim_hidden),
                nn.Dropout(drop_input, inplace=True)
            )
        )
        layers = []
        for i in range(n_layers):
            layers.append(EncLayer(dim_hidden, dim_qk, dim_v, dim_ff, n_heads, dropout, drop_mu, return_attn=False))
        layers.append(
            EncLayer(dim_hidden, dim_qk, dim_v, dim_ff, last_layer_n_heads, dropout, drop_mu, return_attn=True))
        self.layers = nn.Sequential(*layers)

        self.output = Apply(
            nn.Sequential(
                nn.LayerNorm(dim_hidden),
                nn.Linear(dim_hidden, dim_out)
            )
        )

    def forward(self, G):  # G.values: [ max(n+e), dim_hidden]
        G = self.input(G)  # G.values: [ max(n+e), dim_hidden]
        attn_score, G = self.layers(G)  # attn_score: [last_layer_n_heads, |E|, |E|]
        # G.values: [bsize, max(n+e), dim_hidden]
        return attn_score, self.output(G)  # attn_score : [last_layer_n_heads, |E|, |E|]   # self.output(G).values: [ max(n+e), dim_out]





class Apply(nn.Module):
    def __init__(self, f: Callable[[torch.Tensor], torch.Tensor], skip_masking=False):
        super().__init__()
        self.f = f
        self.skip_masking = skip_masking

    def forward(self, G: Union[torch.Tensor, G]) -> Union[torch.Tensor, G]:
        return apply(G, self.f, self.skip_masking)


class Add(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(G1: Union[torch.Tensor, G], G2: Union[torch.Tensor, G]) -> Union[torch.Tensor, G]:
        return add_Graph(G1, G2)





class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor):
        # Q, K: (nheads, |E|, d_qk)
        # V: (n_heads, |E|, d_v)
        # mask: (1, 1, |E|)

        dim_qk = Q.size(-1)
        attn_score = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(dim_qk)  # (nheads, |E|, |E|)
        attn_score = attn_score.masked_fill(~mask, -1e9)  # (nheads, |E|, |E|)
        attn_score = F.softmax(attn_score, dim=-1)  # (nheads, |E|, |E|)
        output = torch.matmul(attn_score, V)  # (nheads, |E|, d_v)
        return attn_score, output  # attn_score: (nheads, |E|, |E|), output: (nheads, |E|, d_v)


class SelfAttn(nn.Module):
    def __init__(self, n_heads=15, d_in=64, d_out=64, d_qk=512, d_v=512):
        super().__init__()
        self.n_heads = n_heads
        self.d_in = d_in
        self.d_out = d_out
        self.d_qk = d_qk
        self.d_v = d_v
        self.scaled_dot_attention = ScaledDotProductAttention()
        self.fc1 = nn.Linear(d_in, 2 * n_heads * d_qk)
        self.fc_v = nn.Linear(d_in, n_heads * d_v)
        self.fc_out = nn.Linear(n_heads * d_v, d_out)

    def forward(self, G):  # G.values: [ |E|, d_in)
        e, _ = G.values.shape
        h = self.fc1(G.values)  # (|E|, 2*n_heads*d_qk)
        Q = h[..., :self.n_heads * self.d_qk].view(e, self.n_heads, self.d_qk)  # (|E|, n_heads, d_qk)
        K = h[..., self.n_heads * self.d_qk:].view(e, self.n_heads, self.d_qk)  # (|E|, n_heads, d_qk)

        V = self.fc_v(G.values)  # (|E|, n_heads*d_v)
        V = V.masked_fill(~G.mask.unsqueeze(-1), 0)
        V = V.view(e, self.n_heads, self.d_v)  # (|E|, n_heads, d_v)

        Q = Q.transpose(0, 1)  # (n_heads, |E|, d_qk)
        K = K.transpose(0, 1)  # (n_heads, |E|, d_qk)
        V = V.transpose(0, 1)  # (n_heads, |E|, d_v)

        G_mask = G.mask.unsqueeze(0).unsqueeze(0)  # (1, 1, |E|)
        attn_score, prod_attn = self.scaled_dot_attention(Q, K, V, mask=G_mask)  # prod_attn: (n_heads, |E|, d_v); attn_score: (nheads, |E|, |E|)

        prod_attn = prod_attn.transpose(0, 1).contiguous()  # (|E|, n_heads, d_v)
        prod_attn = prod_attn.view(e, -1)  # (|E|, n_heads * d_v)

        output = self.fc_out(prod_attn)  # (|E|, d_out)
        return attn_score, Graph_like(G, output, skip_masking=False)