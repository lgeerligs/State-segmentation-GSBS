import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from scipy.stats import  ttest_ind, zscore
from scipy.optimize import linear_sum_assignment
from statesegmentation import GSBS
from brainiak.eventseg.event import EventSegment as HMM
from joblib import Parallel, delayed

def deltas_states(deltas: np.ndarray) -> np.ndarray:
    deltas.astype(int)
    states = np.zeros(deltas.shape[0], int)
    for i, delta in enumerate(deltas[1:]):
        states[i + 1] = states[i] + 1 if delta else states[i]

    return states

def fit_metrics_simulation(real_bounds, recovered_bounds):
    recovered_bounds[recovered_bounds>0]=1
    real_bounds[real_bounds>0]=1
    real_bounds.astype(int)
    recovered_bounds.astype(int)

    real_states = deltas_states(real_bounds)
    recovered_states = deltas_states(recovered_bounds)
    simm, simz = correct_fit_metric(real_states, recovered_states)

    real_locations = np.where(real_bounds)[0]
    recovered_locations = np.where(recovered_bounds)[0]
    dist = np.zeros(np.max(recovered_states)+1)
    for count, i in enumerate(recovered_locations):
        dist[count] = (np.abs(real_locations - i)).min()
        loc=(np.abs(real_locations - i)).argmin()
        if i<real_locations[loc]:
            dist[count]=-dist[count]

    return simm, simz, dist

#function to state detection with HMM and compute the relevant fit metrics
def compute_fits_hmm(data:np.ndarray, k:int, mindist:int, type='HMM', y=None, t1=None, ind1=None, zs=False):
    if type == 'HMM':
        hmm = HMM(k)
    elif type == 'HMMsplit':
        hmm = HMM(k, split_merge = True)

    if zs == True:
        data = zscore(data, axis=0, ddof=1)
    hmm.fit(data)

    if y is None:
        tdata=data
    else:
        if zs == True:
            y = zscore(y, axis=0, ddof=1)
        tdata=y

    _, LL_HMM = hmm.find_events(tdata)

    hmm_bounds = np.insert(np.diff(np.argmax(hmm.segments_[0], axis=1)), 0, 0).astype(int)

    if t1 is None and ind1 is None:
        ind = np.triu(np.ones(tdata.shape[0], bool), mindist)
        z = GSBS._zscore(tdata)
        t = np.cov(z)[ind]
    else:
        ind=ind1
        t=t1

    stateseq = deltas_states(deltas=hmm_bounds)[:, None]
    diff, same, alldiff = (lambda c: (c == 1, c == 0, c > 0))(cdist(stateseq, stateseq, "cityblock")[ind])
    WAC_HMM = np.mean(t[same]) - np.mean(t[alldiff])
    tdist_HMM = 0 if sum(same) < 2 else ttest_ind(t[same], t[diff], equal_var=False)[0]

    return LL_HMM, WAC_HMM, tdist_HMM, hmm_bounds, t, ind

# subfunction to compute reliability
def compute_reliability(data, pflag=True):
    indlist = np.arange(0, data.shape[0])

    states = np.zeros([len(indlist), data.shape[1]])

    for i in indlist:
        states[i]=deltas_states(data[i, :].astype(int))

    if pflag == True:
        reliability_sim, reliability_simz = zip(*Parallel(n_jobs=50)(delayed(compute_reliability1)(np.mean(data[np.setdiff1d(indlist, i), :], axis=0), states[i]) for i in indlist))

    if pflag == False:
        reliability_sim = np.zeros(len(indlist))
        reliability_simz = np.zeros(len(indlist))
        for i in indlist:
            #correlate each subject with the rest of the group
             avgdata = np.mean(data[np.setdiff1d(indlist, i), :], axis=0)

            #get the k most observed boundaries and compute accuracy on group level with fixed k
             k=np.int(np.max(states[i]))
             group_deltas_loc = np.argsort(-avgdata)[0:k]
             group_deltas = np.zeros(avgdata.shape)
             group_deltas[group_deltas_loc]=1

             states_group = deltas_states(group_deltas.astype(int))
             reliability_sim[i], reliability_simz[i]=correct_fit_metric(states_group, states[i],pflag=True)

    return reliability_sim,reliability_simz


def compute_reliability1(avgdata, states):
        k = np.int(np.max(states))
        group_deltas_loc = np.argsort(-avgdata)[0:k]
        group_deltas = np.zeros(avgdata.shape)
        group_deltas[group_deltas_loc] = 1

        states_group = deltas_states(group_deltas.astype(int))
        reliability_sim, reliability_simz = correct_fit_metric(states_group, states, pflag=False)

        return reliability_sim, reliability_simz

def compute_reliability_pcor(data):
    indlist = np.arange(0, data.shape[0])
    reliability_pcor = np.zeros(len(indlist))

    for i in indlist:
        #correlate each subject with the rest of the group
        avgdata = np.mean(data[np.setdiff1d(indlist, i), :], axis=0)
        reliability_pcor[i]=np.corrcoef(avgdata, data[i,:])[0,1]

    return reliability_pcor

def correct_fit_metric(c1, c2, pflag=False):
    sim = get_accuracy(c1, c2)
    nc1 = len(np.unique(c1))
    nc2 = len(np.unique(c2))
    nt = len(c1)
    if pflag==False:
        simr = np.zeros((1000, 1))
        for i in range(0, 1000):
            simr[i]=randomize_fit(nt, nc1, nc2,i)

    elif pflag==True:
        simr = np.array(Parallel(n_jobs=50)(delayed(randomize_fit)(nt, nc1, nc2,i) for i in range(0,1000)))

    simz = (sim - np.mean(simr))/np.std(simr)
    simm = (sim - np.mean(simr))/(1-np.mean(simr))
    return simm, simz


def randomize_fit(nt, nc1, nc2, rep):
    boundloc1 = np.random.choice(nt, [nc1 - 1, 1], replace=False)
    bounds1 = np.zeros((nt, 1)).astype(int)
    bounds1[boundloc1] = 1
    states1 = deltas_states(bounds1)
    boundloc2 = np.random.choice(nt, [nc2 - 1, 1], replace=False)
    bounds2 = np.zeros((nt, 1)).astype(int)
    bounds2[boundloc2] = 1
    states2 = deltas_states(bounds2)
    simr = get_accuracy(states1, states2)
    return simr

def get_accuracy(c1,c2):
    c = confusion_matrix(c1,c2)
    row_ind, col_ind = linear_sum_assignment(-c)
    accuracy=c[row_ind,col_ind].sum()/len(c1)
    return accuracy

