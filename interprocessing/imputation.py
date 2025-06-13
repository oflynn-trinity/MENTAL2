import os

import numpy as np
import torch
from torch.utils import data

from dataset import ImputingDataset
from dataset import ImputingMissingDataset
from vae import VAE

##################################################################################################
##################################################################################################
##################################################################################################

INPUT_DIM = 15663
Z_DIM = 512
DROPOUT = 0.1

# Create Dataset and Dataset Loader
complete_data = np.load('/Users/owenflynn/research/MENTAL2/TD-BRAIN-sample/normalized_unimputed/sample_combined_complete.npy')
complete_dataset = ImputingDataset('sample_combined_complete.npy', '/Users/owenflynn/research/MENTAL2/TD-BRAIN-sample/normalized_unimputed')
data_loader = data.DataLoader(complete_dataset, batch_size=3, shuffle=True)

# Create an instance of the encoder
encoder = VAE(INPUT_DIM, Z_DIM, DROPOUT)

optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

epochs = 5

for epoch in range(epochs):

    for vals in data_loader:
        output, mu, var = encoder.forward(vals)

        recon_loss = torch.nn.MSELoss()
        loss = recon_loss(output, vals)

        kl_loss = - torch.sum(1 + torch.log(var.pow(2))-mu.pow(2)-var.pow(2))

        loss = loss + kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("-----------------------")
    print("Epoch: " + str(epoch))
    print(" Loss: " + str(loss))
    print("-----------------------")

missing_data = np.load('/Users/owenflynn/research/MENTAL2/TD-BRAIN-sample/normalized_unimputed/sample_combined_missing.npy')
missing_dataset = ImputingMissingDataset('sample_combined_missing.npy', '/Users/owenflynn/research/MENTAL2/TD-BRAIN-sample/normalized_unimputed')
missing_data_loader = data.DataLoader(missing_dataset, batch_size=1, shuffle=True)
imputed = []

for (ind, mask, missing) in missing_data_loader:
    print(ind.shape,mask.shape,missing.shape)

    masked = ind*mask
    masked = masked.type(torch.float32)

    out = encoder.forward(masked[0])
    imputed_ind = torch.mul(missing[0], out[0])
    print(imputed_ind.shape)
    
    filled = masked[0]+imputed_ind
    filled = filled.detach().numpy()
    filled = np.array(filled)

    imputed.append(filled)

labels = np.expand_dims(missing_data[:,0],1)
imputed = np.concatenate((labels,np.array(imputed).astype(str)),axis=1)
print("Shape of imputed data ", imputed.shape)

all_data = np.concatenate((imputed, complete_data))
print("Final data shape ", all_data.shape)

np.save(os.path.join('/Users/owenflynn/research/MENTAL2/TD-BRAIN-sample/imputed_features','complete_samples.npy'), all_data)


# For EO
'''
missing_dataset = ImputingMissingDataset('small_missing_samples_EO_top5.npy', '/data/zhanglab/ggreiner/MENTAL/TDBRAIN')
missing_data_loader = data.DataLoader(missing_dataset, batch_size=1, shuffle=True)
imputed = []

for (ind, mask, missing) in missing_data_loader:
    masked = ind*mask
    masked = masked.type(torch.float32)
    test = masked.size()

    out = encoder.forward(masked[0][1:])
    imputed_ind = torch.mul(missing[0][1:], out[0])
    
    filled = masked[0][1:]+imputed_ind
    filled = filled.detach().numpy()
    test = [ind[0][0].detach().numpy()]
    test = np.array(test)
    filled = np.array(filled)
    res = np.concatenate([test, filled])

    imputed.append(res)

imputed = np.array(imputed)
print(imputed.shape)

np.save(os.path.join('/data/zhanglab/ggreiner/MENTAL/TDBRAIN','small_imputed_samples_EO_top5.npy'), imputed)
'''