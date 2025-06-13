import os

import numpy as np

# path of TDBRAIN
participants_path = 'TD-BRAIN-sample'

# out path
out_path = 'TD-BRAIN-sample'

def generate_NEO_Samples(ptc, out):
    survey = np.loadtxt(os.path.join(ptc, "participants.csv"), delimiter=",", dtype=str, usecols = range(65))

    complete_samples = []
    seen = []

    for ind in survey[1:]:
        
        # Only consider individuals that we have survey data for
        # This means excluding the individuals marked for replication
        id = ind[0]
        if((not id[0] == '1') and (not seen.__contains__(id)) and (not ind[5] == 'n/a')):
            seen.append(id)

            neo = ind[5:]
            neo = [int(n) for n in neo]
            neo = np.array(neo)
            
            indication = ind[2]
            if(indication != 'n/a'):
                indication = np.array([indication])
                print(indication.shape, neo.shape)
                res = np.concatenate((indication, neo))
                complete_samples.append(res)
            
            
    all_complete_samples = np.array(complete_samples)
    np.save(os.path.join(out,'missing_NEO_samples'), all_complete_samples)

    print("   Total samples: " + str(survey.shape[0]))
    print("Complete samples: " + str(all_complete_samples.shape[0]))

    return all_complete_samples

def generate_missing_samples(ptc, out):
    survey = np.loadtxt(os.path.join(ptc, "participants.csv"), delimiter=",", dtype=str, usecols = range(65))

    missing_samples = []
    seen = []

    for ind in survey[1:]:
        
        # Only consider individuals that we have survey data for
        # This means excluding the individuals marked for replication
        id = ind[0]
        if(ind[5] == 'n/a'):
            seen.append(id)

            neo = ind[5:]
            neo = [-1 for n in neo]
            neo = np.array(neo)
            
            indication = ind[2]
            if(indication != 'n/a'):
                indication = np.array([indication])
                print(indication.shape, neo.shape)
                res = np.concatenate((indication, neo))
                missing_samples.append(res)
            
            
    missing_samples = np.array(missing_samples)
    np.save(os.path.join(out,'missing_NEO_samples'), missing_samples)

    print("  Total samples: " + str(survey.shape[0]))
    print("Missing samples: " + str(missing_samples.shape[0]))

    return missing_samples

generate_missing_samples(participants_path, out_path)