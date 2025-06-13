import torch
import torch.nn as nn

from .components.psd_embedding import AllPSDEmbedding
from .components.pos_encoder import PositionalEncoder
from .components.transformers import TransformerEnsemble
from .components.classifier import Classifier

class EEGOnlyModel(nn.Module):
    def __init__(self, d_model, n_segments, n_channels, n_bands, d_ff, output_sz, dropout):
        super().__init__()

        self.d_model = d_model
        self.segments = n_segments
        self.channels = n_channels
        self.bands = n_bands
        self.d_ff = d_ff
        self.output_sz = output_sz
        self.dropout = dropout

        self.psdEmbed = AllPSDEmbedding(self.segments, self.channels, self.bands, d_model)

        self.temporalPosEncoder = PositionalEncoder(self.segments, self.d_model)
        self.spatialPosEncoder = PositionalEncoder(self.channels, self.d_model)
        self.spectralPosEncoder = PositionalEncoder(self.bands, self.d_model)

        self.temporalLinear = nn.Linear(self.segments, 63)
        self.spatialLinear = nn.Linear(self.channels, 63)
        self.spectralLinear = nn.Linear(self.bands, 63)

        self.transformers = TransformerEnsemble(63, self.d_model, self.d_ff, self.dropout)

        self.classifier = Classifier(3 * self.d_model, self.d_model, self.output_sz, self.dropout)

    def forward(self, psd):
        temporal, spatial, spectral = self.psdEmbed(psd)

        temporal = self.temporalPosEncoder(temporal)
        spatial = self.spatialPosEncoder(spatial)
        spectral = self.spectralPosEncoder(spectral)

        temporal = self.temporalLinear(temporal.transpose(1,2)).transpose(1,2)
        spatial = self.spatialLinear(spatial.transpose(1,2)).transpose(1,2)
        spectral = self.spectralLinear(spectral.transpose(1,2)).transpose(1,2)

        summaryVector = self.transformers(temporal,spatial,spectral)

        return self.classifier(summaryVector)


