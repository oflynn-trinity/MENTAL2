import torch
import torch.nn as nn
import torch.nn.functional as F

import math

#assumed input dimensions for k, v (batch_size, m1, d_model) for q (batch_size, m2, d_model)
#output dimensions for .forward() are (batch_size, m2, d_model)
class CrossAttention(nn.Module):

    def __init__(self, d_model: int, dropout: float):
        super().__init__()

        self.d_model = d_model

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        key = self.W_k(k)
        query = self.W_q(q)
        value = self.W_v(v)

        key_t = key.transpose(1,2)

        attention1 = torch.matmul(query / math.sqrt(self.d_model), key_t)
        final_attention = self.dropout(F.softmax(attention1,-1))
        unweighted = torch.matmul(final_attention, value)
        result = (self.W_o(unweighted))

        return result
    
class Fuser(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()

        self.d_model = d_model
        self.dropout = dropout

        self.fusion1 = CrossAttention(self.d_model, self.dropout)
        self.fusion2 = CrossAttention(self.d_model, self.dropout)
        self.fusion3 = CrossAttention(self.d_model, self.dropout)

    def forward(self, primary1, primary2, primary3, secondary):
        primary1 = self.fusion1(secondary, primary1, primary1)
        primary2 = self.fusion2(secondary, primary2, primary2)
        primary3 = self.fusion3(secondary, primary3, primary3)

        return primary1, primary2, primary3
        


        