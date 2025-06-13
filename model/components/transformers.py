import torch
import torch.nn as nn

# the TransformerEnsemble component consists of three transformer encoders
# This component takes three input tensors, passes each into a separate encoder, and
# returns a result summary of the three transformers

# input shape expected is (batch_sz, input_sz, d_model), output shape will be (batch_sz, d_model * 3)

class TransformerEnsemble(nn.Module):

    def __init__(self, input_sz: int, d_model: int, d_ff: int, dropout: float, n_heads = 8, n_layers = 4):
        super().__init__()

        self.input_sz = input_sz
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout

        self.encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first = True)

        self.transformer1 = nn.TransformerEncoder(self.encoder_layer, n_layers)
        self.transformer2 = nn.TransformerEncoder(self.encoder_layer, n_layers)
        self.transformer3 = nn.TransformerEncoder(self.encoder_layer, n_layers)

        self.linear = nn.Linear(self.input_sz, 1)

    def forward(self, in1, in2, in3):
        out1 = self.transformer1(in1)
        out2 = self.transformer2(in2)
        out3 = self.transformer3(in3)
        #output shape is same as input shape, (batch_sz, input_sz, d_model)
        combined = torch.cat((out1,out2,out3), 2) #concats along final dimension
        collapsed = self.linear(combined.transpose(1,2)) #collapses input_sz dimension

        return torch.squeeze(collapsed)