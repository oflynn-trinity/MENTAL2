import torch
import torch.nn as nn

# the TransformerEnsemble component consists of three transformer encoders
# this component takes three input tensors, passes each into a separate encoder, and
# returns a single concatenated result tensor

class TransformerEnsemble(nn.Module):

    def __init__(self, d_model: int, d_ff: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout

        self.encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)

        self.spatial_transformer = nn.TransformerEncoder(self.encoder_layer, n_layers)
        self.spectral_transformer = nn.TransformerEncoder(self.encoder_layer, n_layers)
        self.temporal_transformer = nn.TransformerEncoder(self.encoder_layer, n_layers)

    def forward(self, in_spatial, in_spectral, in_temporal):
        out_spatial = self.spatial_transformer(in_spatial)
        out_spectral = self.spectral_transformer(in_spectral)
        out_temporal = self.temporal_transformer(in_temporal)

        return torch.cat((out_spatial,out_spectral,out_temporal), 2)   #concats along final dimension, assumes tensors are of shape (batch_size, channels/bands/windows, d_model)
        