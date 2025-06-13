import torch
import torch.nn as nn

#takes a tensor of shape (batch_sz, n_segments, n_channels, n_bands)
#and creates a feature-wise embedding of the selected dimension
#return shape is (batch_sz, size of selected input dimension, d_model)

class PSDEmbedding(nn.Module):
    def __init__(self, n_segments:int , n_channels: int, n_bands: int, d_model: int, dim: int):
        super().__init__()

        self.segments = n_segments
        self.channels = n_channels
        self.bands = n_bands
        self.d_model = d_model
        self.dim = dim

        assert 0 < dim < 4 #check for valid dimension selection

        if self.dim == 1:
            self.dim_sz = self.segments
            self.intermediate_sz = self.channels * self.bands
        elif self.dim == 2:
            self.dim_sz = self.channels
            self.intermediate_sz = self.segments * self.bands
        elif self.dim == 3:
            self.dim_sz = self.bands
            self.intermediate_sz = self.segments * self.channels

        self.linear = nn.Linear(self.intermediate_sz, self.d_model)

    def forward(self, psd):
        #make sure combined dimensions are last to preserve structure when reshaping
        if self.dim == 2:
            restructured = psd.transpose(1,2)
        elif self.dim == 3:
            restructured = psd.transpose(1,3)
        else:
            restructured = psd

        shaped = restructured.reshape(psd.size()[0], self.dim_sz, self.intermediate_sz)
        return self.linear(shaped)

class AllPSDEmbedding(nn.Module):
    def __init__(self, n_segments:int , n_channels: int, n_bands: int, d_model: int):
        super().__init__()

        self.segments = n_segments
        self.channels = n_channels
        self.bands = n_bands
        self.d_model = d_model

        self.temporalEmbed = PSDEmbedding(self.segments, self.channels, self.bands, self.d_model, 1)
        self.spatialEmbed = PSDEmbedding(self.segments, self.channels, self.bands, self.d_model, 2)
        self.spectralEmbed = PSDEmbedding(self.segments, self.channels, self.bands, self.d_model, 3)

    def forward(self, psd):
        return self.temporalEmbed(psd), self.spatialEmbed(psd), self.spectralEmbed(psd)