import pickle
import numpy as np
from state_boundary_detection import StateSegment

#load the datafile and remove voxels with low inter-subject synchrony
file = open('group_data_roi1_radius6.p', 'rb')
res = pickle.load(file)
ISS = res['ISS']
ISSm=np.nanmean(ISS,0)
indices=np.squeeze(np.argwhere(ISSm>=0.35))
group_data = res['group_data'][:,:,indices]

#run the state deteciton algorithm for a maximum of 150 states and without cross validation
traindata = np.squeeze(np.mean(group_data[:, :, :], axis=0))
states = StateSegment(traindata, traindata, maxK=150)
states.train(CV=False, outextra=False, type='GS')

#relevant outputs
#optimum as defined by the t-distance
optimum_fold = states.optimum
#optimum as defined by wac
optimum_simple_fold = states.optimum_wac
#the state boundary timecourse for the optimal number of states
fin_bounds_folds = states.fin_bounds
#the t-distance estimates for all values of k
tdist = states.tdist
#the wac estimates for all values of k
wac = states.wac