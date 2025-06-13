import os
import numpy as np
from scipy import stats

INPUT_DIR = '/Users/owenflynn/research/MENTAL2/TD-BRAIN-sample/all_unimputed_features'
OUTPUT_DIR = '/Users/owenflynn/research/MENTAL2/TD-BRAIN-sample/normalized_unimputed'

def normalize_all(in_file, out_file):
    labeled_data = np.load(in_file, allow_pickle=True)
    indications = np.expand_dims(labeled_data[:,0],1)
    data = labeled_data[:,1:].astype(np.float32)

    for i in range(data.shape[1]):
        col = data[:, i]
        if np.std(col) == 0:
            data[:, i] = 0
        else:
            data[:, i] = stats.zscore(col, axis=None)
    
    data = data.astype(str)

    print(indications.shape, data.shape)
    final = np.concatenate((indications, data), axis=1)
    print(final.shape)

    np.save(out_file, final)

def normalize_nonNEO(in_file, out_file):
    labeled_data = np.load(in_file, allow_pickle=True)
    indications = np.expand_dims(labeled_data[:,0],1)
    dem = labeled_data[:,1:4].astype(np.float32)
    neo = labeled_data[:,4:64]
    psd = labeled_data[:,64:].astype(np.float32)

    for i in range(dem.shape[1]):
        dem[:,i] = stats.zscore(dem[:,i], axis=None).astype(str)

    for i in range(psd.shape[1]):
        psd[:,i] = stats.zscore(psd[:,i], axis=None).astype(str)

    res = np.concatenate((indications, dem), axis=1)
    res = np.concatenate((res, neo), axis=1)
    res = np.concatenate((res, psd), axis=1)

    np.save(out_file, res)

files = os.listdir(INPUT_DIR)
for f in files:
    if f.__contains__('complete'):
        normalize_all(os.path.join(INPUT_DIR,f),os.path.join(OUTPUT_DIR,f))
    else:
        normalize_all(os.path.join(INPUT_DIR,f),os.path.join(OUTPUT_DIR,f))




