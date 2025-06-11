from end2end_alphaPowerandiAPF import end2end_alphaPowerandiAPF
import os

varargs = {}
main_dir = '/Users/owenflynn/research/MENTAL2/TD-BRAIN-sample/'

varargs['sourcepath'] = main_dir + 'derivatives'
varargs['participantspath'] = main_dir
varargs['preprocpath'] = main_dir + 'preprocessed'
varargs['resultspath'] = main_dir + 'preproc_results'
varargs['chans'] = 'Pz'

if not os.path.exists(varargs['preprocpath']):
    os.mkdir(varargs['preprocpath'])
if not os.path.exists(varargs['resultspath']):
    os.mkdir(varargs['resultspath'])

print('Reading data from '+varargs['sourcepath']+" and writing to "+varargs['participantspath']+', '+varargs['preprocpath']+', and '+varargs['resultspath'])

end2end_alphaPowerandiAPF(varargs)
