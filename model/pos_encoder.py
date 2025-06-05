import torch
import torch.nn as nn
import math

#assumed input shape is (batch_sz, input_sz, d_model)
#adds positional encoding where position is determined by the second dimension
#output shape is same as input

class PositionalEncoder(nn.Module):
    def __init__(self, input_sz: int, d_model: int):
        super().__init__()

        self.input_sz = input_sz
        self.d_model = d_model

        #initialize positional encoding matrix
        pe = torch.zeros(input_sz, d_model)

        position = torch.arange(input_sz).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, input):
        return input + self.pe
    
class AllPosEncoder(nn.Module):
    def __init__(self, neoDem_sz: int, n_segments: int, n_channels: int, n_bands: int, d_model: int):
        super().__init__()

        self.neoDem_sz = neoDem_sz
        self.segments = n_segments
        self.channels = n_channels
        self.bands = n_bands
        self.d_model = d_model

        self.neoDemPos = PositionalEncoder(self.neoDem_sz, self.d_model)
        self.temporalPos = PositionalEncoder(self.segments, self.d_model)
        self.spatialPos = PositionalEncoder(self.channels, self.d_model)
        self.spectralPos = PositionalEncoder(self.bands, self.d_model)

    def forward(self, neoDem, temporal, spatial, spectral):
        return self.neoDemPos(neoDem), self.temporalPos(temporal), self.spatialPos(spatial), self.spectralPos(spectral)
