import numpy as np
import os

directory = '/Users/owenflynn/research/MENTAL2/TD-BRAIN-sample/imputed_features'
individuals_file = 'complete_samples.npy'
individuals = np.load(os.path.join(directory, individuals_file))

ind = individuals[0]
print(ind)
print(ind.shape)

data = ind[1:].astype(np.float32)
print(data.min().item(), data.max().item())
print(np.where(np.isnan(data))[0])

