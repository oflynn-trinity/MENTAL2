import torch
import torch.nn as nn

from .components.psd_embedding import AllPSDEmbedding
from .components.pos_encoder import AllPosEncoder
from .components.cross_attention import Fuser
from .components.transformers import TransformerEnsemble
from .components.classifier import Classifier

#input shape of psd data is (batch_sz, n_segments, n_channels, n_bands)
#input shape of NEO-FFI data is (batch_sz, 60)
#input shape of demographic data is (batch_sz, 3)

NEO_SZ = 60
DEM_SZ = 3

class MENTAL2(nn.Module):
    def __init__(self, d_model, n_segments, n_channels, n_bands, d_ff, output_sz, dropout):
        super().__init__()

        self.d_model = d_model
        self.segments = n_segments
        self.channels = n_channels
        self.bands = n_bands
        self.d_ff = d_ff
        self.output_sz = output_sz
        self.dropout = dropout

        self.neoEmbed = nn.Linear(1,self.d_model)
        self.demEmbed = nn.Linear(1,self.d_model)
        self.psdEmbed = AllPSDEmbedding(self.segments, self.channels, self.bands, d_model)

        self.posEncoder = AllPosEncoder(NEO_SZ + DEM_SZ, self.segments, self.channels, self.bands, self.d_model)

        self.fuser = Fuser(self.d_model, self.dropout)
        
        self.transformers = TransformerEnsemble(63, self.d_model, self.d_ff, self.dropout)

        self.classifier = Classifier(3 * self.d_model, self.d_model, self.output_sz, self.dropout)

    def forward(self, psd, neo, dem):
        #embed data
        embeddedNeo = self.neoEmbed(neo.unsqueeze(2)) #shape (batch_sz, 60, d_model)
        embeddedDem = self.demEmbed(dem.unsqueeze(2)) #shape (batch_sz, 3, d_model)
        emTemporal, emSpatial, emSpectral = self.psdEmbed(psd) #shape (batch_sz, # of segments/channels/bands, d_model)

        #concatenate embeddings along feature dimension
        neoDem = torch.cat((embeddedNeo, embeddedDem), 1) #shape (batch_sz, 63, d_model)

        #add positional encodings before fusion
        posNeoDem, posTemporal, posSpatial, posSpectral = self.posEncoder(neoDem, emTemporal, emSpatial, emSpectral) #shape of posX same as emX

        #separately fuse embedded PSD with embedded NEO-FFI/demographic
        fusedTemporal, fusedSpatial, fusedSpectral = self.fuser(posTemporal, posSpatial, posSpectral, posNeoDem) #shapes (batch_sz, 63, d_model)

        #passed fused data through transformers
        summary = self.transformers(fusedTemporal, fusedSpatial, fusedSpectral) #shape (batch_sz, 3 * d_model)

        #MLP classifies results
        return self.classifier(summary) #shape (batch_sz, output_sz)
