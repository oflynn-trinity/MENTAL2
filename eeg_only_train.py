import numpy as np
import torch
from torch.utils import data

import matplotlib.pyplot as plt

from interprocessing.dataset import EEGBothDataset
from interprocessing.dataset import SplitDataset
from model.eeg_only_model import EEGOnlyModel


##################################################################################################
##################################################################################################
##################################################################################################

SPLITS = [15,3]
EPS = 0.0001

def run_train(learn_rate, wd, batch_sz, epochs, output_sz, outfile):
    diagnoses = ['-1', 'HEALTHY', 'MDD', 'ADHD', 'SMC', 'OCD', 'TINNITUS', 'INSOMNIA', 'PARKINSON', 'DYSLEXIA',
                'ANXIETY', 'PAIN', 'CHRONICPAIN', 'PDD NOS', 'BURNOUT', 'BIPOLAR', 'ASPERGER', 
                'DEPERSONALIZATION', 'ASD', 'WHIPLASH', 'MIGRAINE', 'EPILEPSY', 'GTS', 'PANIC', 
                'STROKE', 'TBI', 'ANOREXIA', 'CONVERSION DX', 'DPS', 'DYSPRAXIA', 'LYME', 'MSA-C', 
                'PTSD', 'TRAUMA', 'TUMOR', 'DYSCALCULIA']


    main_dataset = EEGBothDataset('sample_combined.npy', '/Users/owenflynn/research/MENTAL2/TD-BRAIN-sample/PSD_features')

    res = data.random_split(main_dataset, SPLITS)

    train_loader = data.DataLoader(res[0], batch_size=batch_sz, shuffle=True)
    test_loader  = data.DataLoader(res[1], batch_size=batch_sz, shuffle=True)

    model = EEGOnlyModel(512, 120, 26, 5, 512, 1, 0.1)


    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=wd)

    accs = []
    sens = []
    spec = []

    for epoch in range(epochs):

        for (p_entry, label) in train_loader:

            label_reshaped = np.reshape(label, (batch_sz, output_sz))

            output = model.forward(p_entry)

            loss = torch.nn.MSELoss()
            res = loss(output, label_reshaped)

            optimizer.zero_grad()
            res.backward()
            optimizer.step()        
        
        correct = 0
        vals = []
        cvals = []
        fvals = []
        for (p_entry, label) in test_loader:

            label_reshaped = np.reshape(label, (batch_sz,output_sz))

            output = model.forward(p_entry)

            preds = []
            for i in range(0, batch_sz):
                if(output[i] >= 0.5):
                    preds.append(1)
                else:
                    preds.append(0)
                vals.append(float(output[i].detach()))

            label = label.squeeze_(1)
            conds = []

            for i in range(0, len(label)):
                conds.append(label[i].item())

            print(conds)
            # Variables for calculating specificity and sensitivity
            N = 0 + EPS
            P = 0 + EPS
            TP = 0
            TN = 0
            for i in range(0, len(conds)):
                lb = conds[i]
                pd = preds[i]
                if(lb == 1):
                    P+=1
                    if(lb==pd): 
                        correct += 1
                        TP+=1
                        cvals.append(vals[i])
                    else:
                        fvals.append(vals[i])
                if(lb == 0):
                    N+=1
                    if(lb==pd):
                        correct += 1
                        TN+=1
                        cvals.append(vals[i])
                    else:
                        fvals.append(vals[i])

            

        total = (test_loader.__len__())*batch_sz
        acc = correct/total
        accs.append(acc)
        print("Epoch "+str(epoch)+" test accuracy = "+str(acc))

        sensitivity = TP/P
        sens.append(sensitivity)

        specificity = TN/N
        spec.append(specificity)

        """
        plt.figure(figsize=(15,10))
        plt.hist(vals, bins=np.arange(0, 1.01, 0.05))
        plt.xticks(np.arange(0, 1.01, 0.05))
        plt.yticks(np.arange(0,151,5))
        plt.title("Histogram of Output Values for epoch " + str(epoch))
        plt.ylabel("Count")
        plt.xlabel("Output Value")
        plt.savefig("epoch"+str(epoch)+"_b15_w6_l3_values", pad_inches=0.1)

        print(cvals.__len__()+fvals.__len__())
        print(cvals, fvals)

        plt.figure(figsize=(15,10))
        plt.hist([cvals, fvals], bins=np.arange(0, 1.01, 0.05), label=['Correct', 'Incorrect'], color=['g', 'r'])
        plt.xticks(np.arange(0, 1.01, 0.05))
        plt.yticks(np.arange(0,151,5))
        plt.title("Histogram of Correct and Incorrect Values for epoch " + str(epoch))
        plt.ylabel("Count")
        plt.xlabel("Output Value")
        plt.savefig("epoch"+str(epoch)+"_b15_w6_l3_accvalues", pad_inches=0.1)

        plt.close('all')
        """

            
    accs = np.array(accs)
    sens = np.array(sens)
    spec = np.array(spec)

    labels = np.arange(0, epochs, 1)

    plt.figure(figsize=(15,10))
    plt.plot(labels, accs)
    plt.title("Accuracy of Model for " + str(epoch) + " epochs, batch size " + str(batch_sz))
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig(outfile+"accuracy")
    plt.clf()

    plt.figure(figsize=(15,10))
    plt.plot(labels, sens)
    plt.title("Sensitivity of Model for " + str(epoch) + " epochs, batch size " + str(batch_sz))
    plt.ylabel("Sensitivity")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig(outfile+"sensitivity")
    plt.clf()

    plt.figure(figsize=(15,10))
    plt.plot(labels, spec)
    plt.title("Specificity of Model for " + str(epoch) + " epochs, batch size " + str(batch_sz))
    plt.ylabel("Specificity")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig(outfile+"specificity")
    plt.clf()



# running code


epoch = [1000]
batches = [15]

learn = 1e-3
weight_decay = 1e-6

run_train(learn_rate= learn, wd=weight_decay, batch_sz=3, epochs= 50, output_sz=1, outfile="results/test1")

        