from numpy import concatenate, cov, ndarray, nonzero, ones, triu, unique, zeros, logical_and, arange, max,where,mean, std,sum, median, int, copy, tile
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, ttest_ind, zscore
from typing import Optional, Tuple

class GSBS:
    def __init__(self, kmax: int, x: ndarray, dmin: int = 1, y: Optional[ndarray] = None, blocksize: Optional[int] = 50, outextra: Optional[bool] = False, finetune:Optional[int] = 1) -> None:
        """Given an ROI timeseries, this class uses a greedy search algorithm (GSBS) to
        segment the timeseries into neural states with stable activity patterns.
        GSBS identifies the timepoints of neural state transitions, while t-distance is used
        to determine the optimal number of neural states.

        You can find more information about the method here:
        Geerligs L., van Gerven M., Güçlü U (2020) Detecting neural state transitions underlying event segmentation
        biorXiv. https://doi.org/10.1101/2020.04.30.069989

        Arguments:
            kmax {int} -- the maximal number of neural states that should be estimated in the greedy search
            x {ndarray} -- a multivoxel ROI timeseries - timepoint by voxels array

        Keyword Arguments:
            dmin {Optional[int]} -- the number of TRs around the diagonal of the time by time correlation matrix that are not taken
                            into account in the computation of the t-distance metric. (default: {1})
            y {Optional[ndarray]} -- a multivoxel ROI timeseries - timepoint by voxels array
                                      if y is given, the t-distance will be based on cross-validation,
                                      such that the state boundaries are identified using the data in x and the
                                      optimal number of states is identified using the data in y. If y is not given
                                      the state boundaries and optimal number of states are both based on x. (default: {None})
             blocksize {Optional[int]} -- to speed up the computation when the number of timepoints is large, the algorithm
                                        can first detect local optima for boundary locations within a block of one or multiple states before obtaining
                                        the optimum across all states. Blocksize indicates the minimal number of timepoints that
                                        constitute a block. (default: {50})
             finetune  {Optional[int]} -- the number of TRs around each state boundary in which the algorithm searches for the optimal boundary
                                        during the finetuning step. If finetune is 0, no finetuning of state boundaries is performed. If finetune is <0
                                        all TRs are included during the finetuning step. Note that the latter option will be computationally intensive
                                        (default: {1})

        """
        self.x = x
        self.y = y
        self.kmax = kmax
        self.dmin = dmin
        self.finetune = finetune
        self.blocksize = blocksize
        self._argmax = None
        self.all_bounds = zeros((self.kmax + 1, self.x.shape[0]), int)
        self._bounds = zeros(self.x.shape[0], int)
        self._deltas = zeros(self.x.shape[0], bool)
        self._tdists = zeros(self.kmax + 1, float)
        self.outextra = outextra

        if outextra is True:
            self.all_m_W = zeros(self.kmax + 1, float)
            self.all_m_Bcon = zeros(self.kmax + 1, float)
            self.all_m_Ball = zeros(self.kmax + 1, float)
            self.all_sd_W = zeros(self.kmax + 1, float)
            self.all_sd_Bcon = zeros(self.kmax + 1, float)
            self.all_sd_Ball = zeros(self.kmax + 1, float)
            self.WAC = zeros(self.kmax + 1, float)
            self.mdist = zeros(self.kmax + 1, float)
            self.meddist = zeros(self.kmax + 1, float)


    def get_bounds(self, k:int = None) -> ndarray:
        """
        Keyword Arguments:
            k {Optional[int]} -- number of states
                By default the function returns the boundaries for the optimal number of states (k=nstates).
                When k is given, the boundaries for k states are returned.
        Returns:
            ndarray -- array with length == number of timepoints, where a zero indicates no state transition
            at a particular timepoint and a higher number indicates a state transition. State transitions
            are numbered in the order in which they are detected in GSBS (stronger boundaries tend
            to be detected first).
        """
        assert self._argmax is not None
        if k is None:
            k=self._argmax
        if self.finetune!=0:
            return self.all_bounds[k]
        else:
            return self._bounds * self.get_deltas(k)

    @ property
    def bounds(self) -> ndarray:
        """
        Returns:
            ndarray -- array with length == number of timepoints, where a zero indicates no state transition
            at a particular timepoint and a higher number indicates a state transition. State transitions
            are numbered in the order in which they are detected in GSBS (stronger boundaries tend
            to be detected first).
            The number of states is equal to the optimal number of states (nstates).
        """
        return self.get_bounds(k = None)

    def get_deltas(self, k:int = None) -> ndarray:
        """
        Keyword Arguments:
            k {Optional[int]} -- number of states
                By default the function returns the deltas for the optimal number of states (k=nstates).
                When k is given, the deltas for k states are returned.
        Returns:
            ndarray -- array with length == number of timepoints, where a zero indicates no state transition
            at a particular timepoint and a one indicates a state transition.
        """
        assert self._argmax is not None

        if k is None:
            k = self._argmax

        if self.finetune!=0:
            deltas = logical_and(self.all_bounds[k] <= k, self.all_bounds[k] > 0)
        else:
            deltas = logical_and(self._bounds <= k, self._bounds > 0)
        deltas = deltas * 1

        return deltas

    @ property
    def deltas(self) -> ndarray:
        """
        Returns:
            ndarray -- array with length == number of timepoints, where a zero indicates no state transition
            at a particular timepoint and a one indicates a state transition.
            The number of states is equal to the optimal number of states (nstates).
        """
        return self.get_deltas(k = None)

    @property
    def tdists(self) -> ndarray:
        """
        Returns:
            ndarray -- array with length == kmax
            contains the t-distance estimates for each value of k (number of states)
        """
        assert self._argmax is not None
        return self._tdists

    def get_states(self, k:int = None) -> ndarray:
        """
        Keyword Arguments:
            k {Optional[int]} -- number of states
                By default the function returns the states for the optimal number of states (k=nstates).
                When k is given, k states are returned.

        Returns:
            ndarray -- array with length == number of timepoints,
            where each timepoint is numbered according to the neural state it is in.
        """
        assert self._argmax is not None
        if k is None:
            k=self._argmax
        states = self._states(self.get_deltas(k), self.x)+1
        return states

    @ property
    def states(self) -> ndarray:
        """
        Returns:
            ndarray -- array with length == number of timepoints, where each timepoint is numbered according to
            the neural state it is in. The number of states is equal to the optimal number of states (nstates).
        """
        return self.get_states(k = None)

    @property
    def nstates(self) -> ndarray:
        """
        Returns:
            integer -- optimal number of states as determined by t-distance
        """
        assert self._argmax is not None
        return self._argmax

    def get_state_patterns(self, k:int = None) -> ndarray:
        """
        Keyword Arguments:
            k {Optional[int]} -- number of states
                By default the function returns the state patterns for the optimal number of states (k=nstates).
                When k is given, the state patterns for k states are returned.

        Returns:
            ndarray -- timepoint by nstates array
            Contains the average voxel activity patterns for each of the estimates neural states
        """
        assert self._argmax is not None
        if k is None:
            k=self._argmax
        deltas = self.get_deltas(k)
        states = self._states(deltas, self.x)
        states_unique = unique(states)
        xmeans = zeros((len(states_unique), self.x.shape[1]), float)

        for state in states_unique:
            xmeans[state] = self.x[state == states].mean(0)

        return xmeans

    @ property
    def state_patterns(self) -> ndarray:
        """
        Returns:
            ndarray -- timepoint by nstates array
            Contains the average voxel activity patterns for each of the estimates neural states.
            The numer of states is equal to the optimal number of states (nstates).
        """
        return self.get_state_patterns(k = None)

    def get_strengths(self, k:int = None) -> ndarray:
        """
        Keyword Arguments:
            k {Optional[int]} -- number of states
                By default the function returns the transition strengths for the optimal number of states (k=nstates).
                When k is given, the transition strengths for k states are returned.

        Returns:
            ndarray -- array with length == number of timepoints, where a zero indicates no state transition
            at a particular timepoint and another value indicates a state transition. The numbers indicate
            the strength of a state transition, as indicated by the correlation-distance between neural
            activity patterns in consecutive states.
        """
        assert self._argmax is not None
        if k is None:
            k=self._argmax
        deltas = self.get_deltas(k)
        states = self._states(deltas, self.x)
        states_unique = unique(states)
        pcorrs = zeros(len(states_unique) - 1, float)
        xmeans = zeros((len(states_unique), self.x.shape[1]), float)

        for state in states_unique:
            xmeans[state] = self.x[state == states].mean(0)
            if state > 0:
                pcorrs[state - 1] = pearsonr(xmeans[state], xmeans[state - 1])[0]

        strengths = zeros(deltas.shape, float)
        strengths[deltas == 1] = 1 - pcorrs

        return strengths


    @ property
    def strengths(self) -> ndarray:
        """
         Returns:
            ndarray -- array with length == number of timepoints, where a zero indicates no state transition
            at a particular timepoint and another value indicates a state transition. The numbers indicate
            the strength of a state transition, as indicated by the correlation-distance between neural
            activity patterns in consecutive states. The number of states is equal to the optimal number of states (nstates).
        """
        return self.get_strengths(k = None)

    def fit(self) -> None:
        """This function performs the GSBS and t-distance computations to determine
        the location of state boundaries and the optimal number of states.
        """
        ind = triu(ones(self.x.shape[0], bool), self.dmin)
        z = GSBS._zscore(self.x)
        if self.y is None:
            t = cov(z)[ind]
        else:
            t = cov(GSBS._zscore(self.y))[ind]

        for k in range(2, self.kmax + 1):
            states = self._states(self._deltas, self.x)
            wdists = self._wdists_blocks(self._deltas, states, self.x, z, self.blocksize)
            argmax = wdists.argmax()

            self._bounds[argmax] = k
            self._deltas[argmax] = True

            if self.finetune != 0 and k > 2:
                self._bounds = self._finetune(self._bounds, self.x, z, self.finetune)
                self._deltas=self._bounds>0
                self.all_bounds[k]=self._bounds

            states = self._states(self._deltas, self.x)[:,None]
            diff, same, alldiff = (lambda c: (c == 1, c == 0, c > 0))(cdist(states, states, "cityblock")[ind])
            self._tdists[k] = 0 if sum(same) < 2 else ttest_ind(t[same], t[diff], equal_var=False)[0]

            if self.outextra is True:
                self.all_m_W[k] = mean(t[same])
                self.all_m_Bcon[k] = mean(t[diff])
                self.all_m_Ball[k] = mean(t[alldiff])
                self.all_sd_W[k] = std(t[same])
                self.all_sd_Bcon[k] = std(t[diff])
                self.all_sd_Ball[k] = std(t[alldiff])
                self.WAC[k] = mean(t[same])-mean(t[alldiff])
                self.mdist[k] = mean(t[same]) - mean(t[diff])
                self.meddist[k] = median(t[same]) - median(t[diff])

        self._argmax = self._tdists.argmax()

        if self.outextra is True:
            self.nstates_WAC = self.WAC.argmax()
            self.nstates_mdist = self.mdist.argmax()
            self.nstates_meddist = self.meddist.argmax()


    @staticmethod
    def _finetune(bounds: ndarray, x: ndarray, z: ndarray, finetune: int):
        finebounds = copy(bounds.astype(int))
        for kk in unique(bounds[bounds > 0]):
            ind = (finebounds == kk).nonzero()[0][0]
            finebounds[ind] = 0
            deltas = finebounds > 0
            states = GSBS._states(deltas, x)
            if finetune < 0:
                boundopt = arange(1,states.shape[0],1)
            else:
                boundopt = arange(max((1, ind-finetune)), min((states.shape[0],ind+finetune+1)), 1)
            wdists = GSBS._wdists(deltas, states, x, z, boundopt)
            argmax = wdists.argmax()
            finebounds[argmax] = kk

        return finebounds


    @staticmethod
    def _states(deltas: ndarray, x: ndarray) -> ndarray:
        states = zeros(x.shape[0], int)
        for i, delta in enumerate(deltas[1:]):
            states[i + 1] = states[i] + 1 if delta else states[i]

        return states

    @staticmethod
    def _wdists(deltas: ndarray, states: ndarray, x: ndarray, z: ndarray , boundopt: ndarray = None) -> ndarray:
        xmeans = zeros(x.shape, float)
        wdists = -ones(x.shape[0], float)
        ns = zeros(x.shape[0], float)
        if boundopt is None:
            boundopt = arange(1, x.shape[0])

        # original method
        for state in map(lambda s: s == states, unique(states)):
            xmeans[state] = x[state].mean(0)

        for i in boundopt:
            if deltas[i] == 0:
                state = nonzero(states[i] == states)[0]
                xmean = copy(xmeans[state])  # ಠ_ಠ
                xmeans[state[0]: i] = x[state[0]: i].mean(0)
                xmeans[i: state[-1] + 1] = x[i: state[-1] + 1].mean(0)
                wdists[i] = xmeans.shape[1] * (GSBS._zscore(xmeans) * z).mean() / (xmeans.shape[1] - 1)
                xmeans[state] = xmean

        return wdists

    @staticmethod
    def _wdists_blocks(deltas: ndarray, states: ndarray, x: ndarray, z: ndarray, blocksize: int) -> ndarray:

        if len(unique(states)) > 1:
            boundopt = zeros(max(states)+1)
            prevstate=-1
            for s in unique(states):
                state = where((states > prevstate) & (states <= s))[0]
                numt = state.shape[0]
                if numt > blocksize or s == max(states):
                    xt = x[state]
                    zt = z[state]
                    wdists = GSBS._wdists(deltas=deltas[state], states=states[state], x=xt, z=zt)
                    boundopt[s] = wdists.argmax() + state[0]
                    prevstate = s

            boundopt = boundopt[boundopt>0]
            boundopt = boundopt.astype(int)

        else:
            boundopt = None

        wdists = GSBS._wdists(deltas=deltas, states=states, x=x, z=z, boundopt=boundopt)

        return wdists

    @staticmethod
    def _zscore(x: ndarray) -> ndarray:
        return (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True, ddof=1)
