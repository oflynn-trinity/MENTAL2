import os
import numpy as np

# path of directory where we will save the PSD features
psds_path = 'TD-BRAIN-sample/PSD_full'

# path of TDBRAIN
participants_path = 'TD-BRAIN-sample'

# out path
out_path = 'TD-BRAIN-sample/all_unimputed_features'

EC_OUT_NAME = 'sample_EC'
EO_OUT_NAME = 'sample_EO'
COMBINED_OUT_NAME = 'sample_combined'
PARTICIPANT_FILE = 'participants.csv'
CSV_COLS = list(range(65)) + [81]

#type should be 'EC' or 'EO'
def generate_Single_Type_Samples(type, ptc, psd, out):
    survey = np.loadtxt(os.path.join(ptc, PARTICIPANT_FILE), delimiter=",", dtype=str, usecols = CSV_COLS)

    seen = []
    complete_samples = []
    missing_samples = []

    for ind in survey[1:]:
        
        # Only consider individuals that we have survey data for
        # This means excluding the individuals marked for replication
        id = ind[0]
        #if(not seen.__contains__(id)):
        if((not id[0] == '1') and (not ind[5] == 'n/a')):
            seen.append(id)
            # Navigate to the directory with the psd information

            psds = loadPSD(ind,psd,type)

            neo = loadNEO(ind)
            
            indication = ind[2]
            if(indication != 'n/a'):
                ind_data = compileSamples(ind, psds, neo)
                complete_samples.append(ind_data)

        elif(not id[0] == '1'):
            seen.append(id)

            psds = loadPSD(ind,psd,type)

            neo = np.full(60, -1)

            indication = ind[2]
            if(indication != 'n/a'):
                ind_data = compileSamples(ind, psds, neo)
                missing_samples.append(ind_data)

    complete_samples = np.array(complete_samples)
    missing_samples = np.array(missing_samples)

    np.save(out+'_complete', complete_samples)
    np.save(out+'_missing', missing_samples)

    print(type+" samples")
    print("   Total samples: " + str(survey.shape[0]))
    print("Complete samples: " + str(complete_samples.shape[0]))
    print("   Total samples: " + str(survey.shape[0]))
    print(" Missing samples: " + str(missing_samples.shape[0]))

def generate_EC_EO_Samples(ptc, psd, out):
    survey = np.loadtxt(os.path.join(ptc, PARTICIPANT_FILE), delimiter=",", dtype=str, usecols = CSV_COLS)

    seen = []
    complete_samples = []
    missing_samples = []

    for ind in survey[1:]:
        
        # Only consider individuals that we have survey data for
        # This means excluding the individuals marked for replication
        id = ind[0]
        #if(not seen.__contains__(id)):
        if((not id[0] == '1') and (not ind[5] == 'n/a')):
            seen.append(id)
            # Navigate to the directory with the psd information

            ecPSD = loadPSD(ind,psd,'EC')
            eoPSD = loadPSD(ind,psd,'EO')
            psds = np.concatenate((ecPSD, eoPSD))

            neo = loadNEO(ind)
            
            indication = ind[2]
            if(indication != 'n/a'):
                ind_data = compileSamples(ind, psds, neo)
                complete_samples.append(ind_data)

        elif(not id[0] == '1'):
            seen.append(id)

            ecPSD = loadPSD(ind,psd,'EC')
            eoPSD = loadPSD(ind,psd,'EO')
            psds = np.concatenate((ecPSD, eoPSD))

            neo = np.full(60, -1)

            indication = ind[2]
            if(indication != 'n/a'):
                ind_data = compileSamples(ind, psds, neo)
                missing_samples.append(ind_data)

    complete_samples = np.array(complete_samples)
    missing_samples = np.array(missing_samples)

    np.save(out+'_complete', complete_samples)
    np.save(out+'_missing', missing_samples)

    print("EC and EO samples")
    print("   Total samples: " + str(survey.shape[0]))
    print("Complete samples: " + str(complete_samples.shape[0]))
    print("   Total samples: " + str(survey.shape[0]))
    print(" Missing samples: " + str(missing_samples.shape[0]))

def loadPSD(ind, psd, type):
    loc = os.path.join(psd, "sub-"+ind[0])
    files = os.listdir(loc)

    sn = ind[1]
    psds = None
    for f in files:
        if(f.__contains__(type) and f.__contains__(sn)):
            found = True
            # Load the PSD values from the files
            pth = os.path.join(loc,f)
            psds = np.load(pth, allow_pickle=True)
            psds = np.squeeze(psds)
            psds = psds.flatten()
    return psds

def loadNEO(ind):
    neo = ind[5:65]
    neo = [int(n) for n in neo]
    return np.array(neo)

def compileSamples(ind, psds, neo):
    indication = np.array([ind[2]])
    age = np.array([ind[3]])
    gender = np.array([ind[4]])
    ed = np.array([ind[65]])
    res = np.concatenate((indication, age))
    res = np.concatenate((res, gender))
    res = np.concatenate((res, ed))
    res = np.concatenate((res, neo))
    res = np.concatenate((res, psds))
    return res

generate_Single_Type_Samples('EC',participants_path, psds_path, os.path.join(out_path,EC_OUT_NAME)) 
generate_Single_Type_Samples('EO',participants_path, psds_path, os.path.join(out_path,EO_OUT_NAME)) 
generate_EC_EO_Samples(participants_path, psds_path,  os.path.join(out_path,COMBINED_OUT_NAME)) 