import numpy as np
from scipy.spatial import distance
from brainiak.eventseg.event import EventSegment as HMM
from scipy.stats import ttest_ind

xp = np

class StateSegment:
    def __init__(self, X: xp.ndarray, Y: xp.ndarray, maxK:int, mindist:int = 1) -> None:

        #X = time * voxels
        #Y = time * voxels
        #Note that data is not z-scored in time, this will reduce the dominance of voxels that have high ISS
        #For the HMM analyses, the data has to be z-scored, this is done within the code that runs the HMM

        self.X = X
        self.Y = Y
        self.K = maxK
        self.minK = 2
        self.mindist = mindist

        self.all_bounds = None
        self.fin_bounds = None
        self.deltas = None
        self.hmm_bounds = None
        self.optimum = None
        self.optimum_wac = None
        self.tdist = None
        self.wac = None
        self.fin_bounds_cdist = None

        self.I = xp.triu(xp.ones(self.X.shape[0], int), self.mindist) == 1  # P x P logical index matrix
        self.YCZ = np.squeeze(StateSegment._zscore(StateSegment._corr(self.Y)[self.I][None]))  # N x N test correlation matrix - z-scored
        self.XCZ = np.squeeze(StateSegment._zscore(StateSegment._corr(self.X)[self.I][None]))  # N x N test correlation matrix - z-scored
        self.XC = np.squeeze(StateSegment._corr(self.X)[self.I][None])  # N x N test correlation matrix


    @staticmethod
    def _corr(X: xp.ndarray) -> xp.ndarray:
        return xp.cov(StateSegment._zscore(X))

    @staticmethod
    def _zscore(X: xp.ndarray) -> xp.ndarray:
        return (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)

    def train_HMM(self, fixedK:bool=False):
        hmm_bounds = xp.full([self.K+1, self.X.shape[0]], 0).astype(int)

        if fixedK is True:
            ev = HMM(self.K)
            ev.fit(self.X)
            hmm_bounds = np.insert(np.diff(np.argmax(ev.segments_[0], axis=1)), 0, 0)
            return hmm_bounds
        else:
            for k in range(self.minK, self.K + 1):
                ev = HMM(k)
                ev.fit(self.X)
                hmm_bounds[k, :] = np.insert(np.diff(np.argmax(ev.segments_[0], axis=1)), 0, 0)
            self.hmm_bounds = hmm_bounds

    def train(self, CV:bool=True, outextra:bool=False, type:str='GS') -> None:
        bounds = xp.full(self.X.shape[0], 0).astype(int)
        deltas = xp.full(self.X.shape[0], False).astype(bool)
        states = xp.full(self.X.shape[0], 0).astype(int)
        WBdist = xp.full(self.X.shape[0], 0).astype(float)
        Wdist = xp.full(self.X.shape[0], 0).astype(float)
        tdist = xp.full([self.K+1], 0).astype(float)
        wac = xp.full([self.K+1], 0).astype(float)

        if type == 'HMM':
            self.train_HMM()

        if outextra is True:
            self.all_m_W = np.copy(tdist)
            self.all_m_Bcon = np.copy(tdist)
            self.all_m_Ball = np.copy(tdist)
            self.all_sd_W = np.copy(tdist)
            self.all_sd_Bcon = np.copy(tdist)
            self.all_sd_Ball = np.copy(tdist)

        for k in range(self.minK, self.K + 1):
            states.fill(0)
            WBdist.fill(0)
            Wdist.fill(0)

            if type is 'GS':

                for i, delta in enumerate(deltas[1:]):
                    states[i + 1] = states[i] + 1 if delta else states[i]

                Z = StateSegment._zscore(self.X)
                X = xp.full(self.X.shape, 0).astype(float)
                for state in np.unique(states):
                    index = state == states
                    X[index] = self.X[index].mean(0)


                for n in range(1, self.X.shape[0]):
                    if np.sum(deltas[n]) < 1:

                        # using the average correlations of within-state timepoints does not work as well as averaging the signal first (as below),
                        # also, using the WBdist to define boundaries does not work at all!

                        # tstates=np.copy(states)
                        # tstates[n:]=tstates[n:]+1
                        #
                        # same_state = np.multiply(distance.cdist(np.expand_dims(tstates, axis=1), np.expand_dims(tstates, axis=1),
                        #                              'cityblock')==0, self.I)
                        #
                        # W = self.XC[same_state[self.I]]
                        # Wdist[n]=np.mean(W)

                        state = xp.nonzero(states == states[n])[0]

                        x = X[state]  # instead of x = X[n] ಠ_ಠ ?!1
                        x = np.array(x)

                        X[state[0]:             n] = self.X[state[0]:             n].mean(0)
                        X[n: state[-1] + 1] = self.X[n: state[-1] + 1].mean(0)

                        Wdist[n] = X.shape[1] * (StateSegment._zscore(X) * Z).mean() / (X.shape[1] - 1)

                        X[state] = x

                n = Wdist.argmax()
                bounds[n] = k
                deltas[n] = True

                #if we use CV to find the optimum
                if CV is True:
                    ty = self.YCZ
                else:
                    ty = self.XCZ

                tstates = np.copy(states)
                tstates[n:] = tstates[n:] + 1

            elif type is 'HMM':
                deltas = self.hmm_bounds[k,:]

                for i, delta in enumerate(deltas[1:]):
                    states[i + 1] = states[i] + 1 if delta else states[i]

                tstates=np.copy(states)

            same_state = np.multiply(
                distance.cdist(np.expand_dims(tstates, axis=1), np.expand_dims(tstates, axis=1),
                               'cityblock') == 0, self.I)
            diff_state = np.multiply(
                distance.cdist(np.expand_dims(tstates, axis=1), np.expand_dims(tstates, axis=1),
                               'cityblock') == 1, self.I)

            all_diff_state = np.multiply(
                distance.cdist(np.expand_dims(tstates, axis=1), np.expand_dims(tstates, axis=1),
                               'cityblock') > 0, self.I)

            W = ty[same_state[self.I]]
            Bcon = ty[diff_state[self.I]]
            Ball = ty[all_diff_state[self.I]]

            [tdist[k], p] = ttest_ind(W, Bcon, equal_var=False)
            wac[k]=np.mean(W)-np.mean(Ball)

            if np.sum(same_state[self.I]) < 2:
                wac[k] = 0
                tdist[k] = 0

            if outextra is True:
                self.all_m_W[k] = np.mean(W)
                self.all_m_Bcon[k] = np.mean(Bcon)
                self.all_m_Ball[k] = np.mean(Ball)
                self.all_sd_W[k] = np.std(W)
                self.all_sd_Bcon[k] = np.std(Bcon)
                self.all_sd_Ball[k] = np.std(Ball)

        self.deltas = deltas
        self.all_bounds = bounds
        self.tdist = tdist
        self.wac = wac
        self.optimum = tdist.argmax()
        self.optimum_wac = wac.argmax()

        # get the final set of boundaries
        fin_bounds = np.copy(bounds)
        fin_bounds[fin_bounds > self.optimum] = 0
        fin_bounds[fin_bounds > 0] = 1
        self.fin_bounds = fin_bounds

        # get the boundary strength
        states = xp.full(self.X.shape[0], 0).astype(int)
        for i, delta in enumerate(fin_bounds[1:]):
            states[i + 1] = states[i] + 1 if delta else states[i]

        E = xp.full((len(np.unique(states)), self.X.shape[1]), 0).astype(float)
        cdist = xp.full((len(np.unique(states)) - 1), 0).astype(float)
        for i in np.unique(states):
            E[i] = self.X[states == i].mean(0)
            if i > 0:
                co = 1 - np.corrcoef(E[i], E[i - 1])
                cdist[i - 1] = co[0, 1]

        fin_bounds_cdist = np.double(fin_bounds)
        fin_bounds_cdist[fin_bounds == 1] = cdist
        self.fin_bounds_cdist = fin_bounds_cdist


