import torch
import torch.nn as nn

#assumed input shape is (batch_sz, input_sz)
#MLP produces an output shape of (batch_sz, output_shape)
#output matrix undergoes sigmoid activation if output_sz is 1, softmax otherwise

class Classifier(nn.Module):

    def __init__(self, input_sz: int, d_model: int, output_sz: int, dropout: int):
        super().__init__()

        self.input_sz = input_sz
        self.d_model = d_model
        self.output_sz = output_sz

        assert self.d_model % 4 == 0

        self.mlp = nn.Sequential(
            nn.Linear(input_sz, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(d_model // 4, output_sz)
        )

        if output_sz == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, input):
        return self.activation(self.mlp(input))