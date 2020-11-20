import numpy as np
from statesegmentation import GSBS
import gsbs_extra
from typing import Tuple
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, ttest_ind, mannwhitneyu
from sklearn.model_selection import KFold
from hrf_estimation import hrf
import timeit
from help_functions import fit_metrics_simulation, compute_fits_hmm, deltas_states
from brainiak.eventseg.event import EventSegment as HMM
from importlib import reload

savedir = '/home/lingee/wrkgrp/Cambridge_data/Movie_HMM/simulations/'

class Simulations:
    def __init__(self, nvox, ntime, nstates, nsub, sub_std, group_std, TR, sub_evprob, length_std,maxK, peak_delay:float=6, peak_disp:float=1, extime:int=2):
        self.nvox=nvox
        self.ntime=ntime
        self.nstates=nstates
        self.nsub=nsub
        self.group_std=group_std
        self.sub_std=sub_std
        self.TR=TR
        self.sub_evprob=sub_evprob
        self.length_std=length_std
        self.peak_delay=peak_delay
        self.peak_disp=peak_disp
        self.extime = extime
        self.maxK=maxK


    @staticmethod
    def sample(n: int, p: float, r: float) -> Tuple[np.ndarray, np.ndarray, float, float]:
        # n = number of timepoints
        # p = number of states
        # r = variability of state lengths
        x_0 = np.linspace(0, n, p + 1).astype(int)[1: -1]
        x_r = []
        q = r * (n / p) - 0.5

        for i in range(p - 1):
            while True:
                x = x_0[i] + np.random.randint(-q, q + 1)
                if (i > 0 and x_r[i - 1] >= x) or (i == 0 and x < 1) or (x > n-(p-i)-1):
                    continue
                else:
                    break

            x_r.append(x)

        x_r = np.array(x_r)
        # x_d = np.concatenate(([x_r[0]], x_r[1:] - x_r[:-1], [n - x_r[-1]]))
        bounds = np.zeros(n).astype(int)
        bounds[x_r]=1
        states = deltas_states((bounds))
        print(max(states))
        # x_d = np.concatenate(([x_r[0]], x_r[1:] - x_r[:-1], [n - x_r[-1]]))

        return bounds, states #x_d, min(x_d), max(x_d)

    def generate_simulated_data_HRF(self, nstates=None, group_std=None, sub_std=None, sub_evprob=None, length_std=None, peak_delay=None, peak_disp=None, extime=None, TR=None, TRfactor=1, nsub=None, rep=500):

        nstates = nstates or self.nstates
        group_std = group_std or self.group_std
        sub_std = sub_std or self.sub_std
        length_std = length_std or self.length_std
        sub_evprob = sub_evprob or self.sub_evprob
        peak_delay = peak_delay or self.peak_delay
        peak_disp = peak_disp or self.peak_disp
        extime = extime or np.int(self.extime/TRfactor)
        TR = TR or self.TR*TRfactor
        nsub = nsub or self.nsub

        np.random.seed(rep)
        state_means = np.random.randn(self.nvox, nstates)

        bounds,state_labels = Simulations.sample(self.ntime, nstates, length_std)
        nb = np.array(np.where(bounds == 0)[0])
        b = np.array(np.where(bounds == 1)[0])

        evData = np.zeros([self.ntime+extime, self.nvox])
        for t in range(0,self.ntime):
            evData[t,:] = state_means[:, state_labels[t]]

        #extend the final state to make sure it is present in the final signal
        for te in range(self.ntime, self.ntime+extime):
            evData[te,:]=evData[-1,:]

        spmhrf = hrf.spm_hrf_compat(np.arange(0,30,TR), peak_delay=peak_delay, peak_disp=peak_disp)
        BOLDevData = self.convolve_with_hrf(evData, spmhrf, extime)

        groupSig = group_std * np.random.randn(self.ntime+extime, self.nvox)
        BOLDgroupSig = self.convolve_with_hrf(groupSig, spmhrf, extime)

        subData = np.zeros([nsub,  self.ntime, self.nvox])
        subbounds = np.zeros([nsub, self.ntime])
        if sub_evprob == 0:
            for s in range(0,nsub):
                subSig = sub_std * np.random.randn(self.ntime+extime, self.nvox)
                BOLDsubSig = self.convolve_with_hrf(subSig, spmhrf, extime)
                subData[s,:,:] = BOLDsubSig + BOLDgroupSig + BOLDevData

        else:
            for s in range(0, nsub):
                p_b = b[np.nonzero(np.random.binomial(1,sub_evprob,nstates-1))[0]]
                p_nb = nb[np.nonzero(np.random.binomial(1,sub_evprob/(np.shape(nb)[0]/(nstates-1)),np.shape(nb)[0]))[0]]
                samp = np.concatenate((p_b, p_nb))

                sub_state_labels=np.copy(state_labels)
                for t in samp:
                    # a state boundary disappears
                    if bounds[t] == 1:
                        ev=np.argwhere(sub_state_labels == sub_state_labels[t])
                        sub_state_labels[ev] = sub_state_labels[t-1]

                    # a state boundary appears
                    elif bounds[t] == 0:
                        times=np.arange(t,self.ntime,1)
                        ev = np.argwhere(sub_state_labels[times] == sub_state_labels[t])
                        sub_state_labels[times[ev]] = np.amax(sub_state_labels) + 1

                if np.amax(sub_state_labels)>(nstates-1):
                    substate_means = np.random.randn(self.nvox, np.amax(sub_state_labels)-(nstates-1))

                subbounds[s,1:] = np.diff(sub_state_labels)

                subevData = np.zeros([self.ntime+extime, self.nvox])
                for t in range(0, self.ntime):
                    if sub_state_labels[t] < nstates:
                        subevData[t,:] = state_means[:, sub_state_labels[t]]
                    else:
                        subevData[t,:] = substate_means[:, sub_state_labels[t]-nstates]

                # extend the final state to make sure it is present in the final signal
                for te in range(self.ntime, self.ntime + extime):
                    subevData[te,:] = subevData[self.ntime - 1,:]

                BOLDsubevData = self.convolve_with_hrf(subevData, spmhrf, extime)

                subSig = sub_std * np.random.randn(self.ntime + extime, self.nvox)
                BOLDsubSig = self.convolve_with_hrf(subSig, spmhrf, extime)

                subData[s,:,:] = BOLDsubSig + BOLDgroupSig + BOLDsubevData

        return bounds, subData, subbounds

    def convolve_with_hrf(self, signal, hrf, extime):

        BOLDsignal = np.zeros([self.ntime + extime + len(hrf) - 1, self.nvox])
        for n in range(0, self.nvox):
            BOLDsignal[:,n] = np.convolve(signal[:,n], hrf)
        BOLDsignal = BOLDsignal[extime:extime + self.ntime,:]
        return BOLDsignal


    # simulation 1, vary state length and estimate how accurately we can recover state boundaries
    def run_simulation_evlength(self,length_std, nstates_list,run_HMM, rep, TRfactor=1, finetune=1):

        res=dict()
        list2=['dists_GS','dists_HMM', 'dists_HMMsplit']
        for key in list2:
            res[key] = np.zeros([np.shape(length_std)[0], np.shape(nstates_list)[0], nstates_list[-1]])

        list = ['sim_GS', 'sim_HMM','sim_HMMsplit', 'simz_GS', 'simz_HMM', 'simz_HMMsplit']
        for key in list:
            res[key] = np.zeros([np.shape(length_std)[0], np.shape(nstates_list)[0]])
        res['statesreal']=np.zeros([np.shape(length_std)[0], np.shape(nstates_list)[0],self.ntime])
        res['bounds'] = np.zeros([np.shape(length_std)[0], np.shape(nstates_list)[0], self.ntime])
        res['bounds_HMMsplit'] = np.zeros([np.shape(length_std)[0], np.shape(nstates_list)[0], self.ntime])

        for idxl, l in enumerate(length_std):
            for idxn, n in enumerate(nstates_list):
                print(rep, l)
                bounds, subData,_ = self.generate_simulated_data_HRF(length_std=l, nstates=n, TRfactor=TRfactor, rep=rep)
                res['statesreal'][idxl,idxn,:]=deltas_states(bounds)
                states = gsbs_extra.GSBS(kmax=n, x=subData[0,:,:], finetune=finetune)
                states.fit()
                res['sim_GS'][idxl,idxn], res['simz_GS'][idxl, idxn],res['dists_GS'][idxl,idxn,0:n] = fit_metrics_simulation(bounds, np.double(states.get_bounds(k=n)>0))
                res['bounds'][idxl,idxn,:]=states.bounds

                if run_HMM is True:
                    ev = HMM(n, split_merge=False)
                    ev.fit(subData[0,:,:])
                    hmm_bounds = np.insert(np.diff(np.argmax(ev.segments_[0], axis=1)), 0, 0).astype(int)
                    ev = HMM(n, split_merge=True)
                    ev.fit(subData[0, :, :])
                    hmm_bounds_split = np.insert(np.diff(np.argmax(ev.segments_[0], axis=1)), 0, 0).astype(int)
                    res['sim_HMM'][idxl, idxn], res['simz_HMM'][idxl, idxn], res['dists_HMM'][idxl, idxn, 0:n] = fit_metrics_simulation(bounds, hmm_bounds)
                    res['sim_HMMsplit'][idxl, idxn],  res['simz_HMMsplit'][idxl, idxn], res['dists_HMMsplit'][idxl, idxn, 0:n] = fit_metrics_simulation(bounds, hmm_bounds_split)
                    res['bounds_HMMsplit'][idxl, idxn, :] = hmm_bounds_split

        return res


    #simulation 2, how do the different fit measures compare, depending on how many states there are (more states should cause more similarity between distinct states)
    def run_simulation_compare_nstates(self, nstates_list, mindist, run_HMM, finetune, zs, rep):

        res2 = dict()
        list = ['optimum_tdist','optimum_wac','optimum_mdist','optimum_meddist','optimum_mwu','optimum_LL_HMM','optimum_WAC_HMM',
                'optimum_mdist_HMM','optimum_meddist_HMM','optimum_mwu_HMM','optimum_tdist_HMM',
                'sim_GS_tdist', 'sim_GS_WAC', 'simz_GS_tdist', 'simz_GS_WAC',
                'sim_HMM_LL','simz_HMM_LL','sim_HMMsplit_LL','simz_HMMsplit_LL','sim_HMM_WAC','simz_HMM_WAC',
                'sim_HMMsplit_WAC','simz_HMMsplit_WAC','sim_HMM_tdist','simz_HMM_tdist','sim_HMMsplit_tdist','simz_HMMsplit_tdist']
        for i in list:
            res2[i]= np.zeros([np.shape(nstates_list)[0]])
        list2 = ['tdist', 'wac', 'mdist', 'meddist',  'LL_HMM', 'WAC_HMM', 'tdist_HMM', 'fit_W_mean', 'fit_W_std', 'fit_Ball_mean', 'fit_Ball_std', 'fit_Bcon_mean', 'fit_Bcon_std']
        for i in list2:
            res2[i] = np.zeros([np.shape(nstates_list)[0], self.maxK+1])

        for idxl, l in enumerate(nstates_list):
                print(rep, l)
                bounds, subData,_ = self.generate_simulated_data_HRF(nstates=l, rep=rep)
                states = gsbs_extra.GSBS(x=subData[0,:,:], kmax=self.maxK, outextra=True, dmin=mindist, finetune=finetune)
                states.fit()
                res2['sim_GS_tdist'][idxl],  res2['simz_GS_tdist'][idxl], dist = fit_metrics_simulation(bounds, states.deltas)
                res2['sim_GS_WAC'][idxl], res2['simz_GS_WAC'][idxl], dist = fit_metrics_simulation(bounds, states.get_deltas(k=states.nstates_WAC))

                if run_HMM is True:
                    t=None
                    ind=None

                    for i in range(2,self.maxK):
                        res2['LL_HMM'][idxl, i], res2['WAC_HMM'][idxl, i],res2['tdist_HMM'][idxl, i], \
                        hmm_bounds, t, ind = compute_fits_hmm(subData[0, :, :], i, mindist, type='HMM', y=None, t1=t, ind1=ind, zs=zs)

                    res2['optimum_LL_HMM'][idxl] = np.argmax(res2['LL_HMM'][idxl][2:90])+2
                    res2['optimum_WAC_HMM'][idxl] = np.argmax(res2['WAC_HMM'][idxl])
                    res2['optimum_tdist_HMM'][idxl] = np.argmax(res2['tdist_HMM'][idxl])

                    i = int(res2['optimum_LL_HMM'][idxl])
                    _, _, _, hmm_bounds, t, ind = compute_fits_hmm(data=subData[0, :, :], k=i, mindist=1, type='HMM', y=None, t1=t, ind1=ind)
                    res2['sim_HMM_LL'][idxl],  res2['simz_HMM_LL'][idxl], dist = fit_metrics_simulation(bounds, hmm_bounds)
                    _, _, _, hmm_bounds, t, ind = compute_fits_hmm(data=subData[0, :, :], k=i, mindist=1, type='HMMsplit', y=None, t1=t, ind1=ind)
                    res2['sim_HMMsplit_LL'][idxl],  res2['simz_HMMsplit_LL'][idxl], dist = fit_metrics_simulation(bounds, hmm_bounds)

                    i = int(res2['optimum_WAC_HMM'][idxl])
                    _, _, _, hmm_bounds, t, ind = compute_fits_hmm(data=subData[0, :, :], k=i, mindist=1, type='HMM', y=None, t1=t, ind1=ind)
                    res2['sim_HMM_WAC'][idxl],  res2['simz_HMM_WAC'][idxl], dist = fit_metrics_simulation(bounds, hmm_bounds)
                    _, _, _, hmm_bounds, t, ind = compute_fits_hmm(data=subData[0, :, :], k=i, mindist=1, type='HMMsplit', y=None, t1=t, ind1=ind)
                    res2['sim_HMMsplit_WAC'][idxl],  res2['simz_HMMsplit_WAC'][idxl], dist = fit_metrics_simulation(bounds, hmm_bounds)

                    i = int(res2['optimum_tdist_HMM'][idxl])
                    _, _, _, hmm_bounds, t, ind = compute_fits_hmm(data=subData[0, :, :], k=i, mindist=1, type='HMM', y=None, t1=t, ind1=ind)
                    res2['sim_HMM_tdist'][idxl],  res2['simz_HMM_tdist'][idxl], dist = fit_metrics_simulation(bounds, hmm_bounds)
                    _, _, _, hmm_bounds, t, ind = compute_fits_hmm(data=subData[0, :, :], k=i, mindist=1, type='HMMsplit', y=None, t1=t, ind1=ind)
                    res2['sim_HMMsplit_tdist'][idxl],  res2['simz_HMMsplit_tdist'][idxl], dist = fit_metrics_simulation(bounds, hmm_bounds)

                res2['optimum_tdist'][idxl]=states.nstates
                res2['optimum_wac'][idxl]=states.nstates_WAC
                res2['optimum_meddist'][idxl] = states.nstates_meddist
                res2['optimum_mdist'][idxl] = states.nstates_mdist

                res2['fit_W_mean'][idxl, :] = states.all_m_W
                res2['fit_W_std'][idxl, :] = states.all_sd_W
                res2['fit_Ball_mean'][idxl, :] = states.all_m_Ball
                res2['fit_Ball_std'][idxl, :] = states.all_sd_Ball
                res2['fit_Bcon_mean'][idxl, :] = states.all_m_Bcon
                res2['fit_Bcon_std'][idxl, :] = states.all_sd_Bcon

                res2['tdist'][idxl,:]=states.tdists
                res2['wac'][idxl, :] = states.WAC
                res2['mdist'][idxl,:]=states.mdist
                res2['meddist'][idxl,:]=states.meddist

        return res2


    #simulation 3, can we correctly estimate the number of states in the group when there is ideosyncracy in state boundaries between participants?
    def run_simulation_sub_noise(self, CV_list, sub_std_list, kfold_list, nsub, rep):

        res3=dict()
        list=['optimum', 'sim_GS','sim_GS_fixK', 'simz_GS', 'simz_GS_fixK']
        for key in list:
            res3[key] = np.zeros([np.shape(CV_list)[0], np.shape(sub_std_list)[0], np.shape(kfold_list)[0]])
        res3['tdist'] = np.zeros([np.shape(CV_list)[0], np.shape(sub_std_list)[0], np.shape(kfold_list)[0], self.maxK + 1])

        list = ['optimum_subopt', 'sim_GS_subopt', 'simz_GS_subopt']
        for key in list:
            res3[key] = np.zeros([np.shape(sub_std_list)[0], nsub])

        for idxs, s in enumerate(sub_std_list):
            bounds, subData,_ = self.generate_simulated_data_HRF(sub_std=s, nsub=nsub, rep=rep)

            for idxi, i in enumerate(kfold_list):
                print(rep, s, i)
                if i>1:
                    kf = KFold(n_splits=i, shuffle=True)
                    for idxl, l in enumerate(CV_list):

                        tdist_temp = np.zeros([i,self.maxK+1]);  optimum_temp = np.zeros(i); GS_sim_temp = np.zeros(i)
                        GS_sim_temp_fixK = np.zeros(i); simz_temp = np.zeros(i); simz_temp_fixK = np.zeros(i)

                        count=-1
                        for train_index, test_index in kf.split(np.arange(0,np.max(kfold_list))):
                            count=count+1
                            print(count)
                            if l is False:
                                states = gsbs_extra.GSBS(x=np.mean(subData[test_index, :, :], axis=0), kmax=self.maxK)
                            elif l is True:
                                states = gsbs_extra.GSBS(x=np.mean(subData[train_index, :, :], axis=0), y=np.mean(subData[test_index, :, :], axis=0), kmax=self.maxK)
                            states.fit()

                            optimum_temp[count] = states.nstates
                            tdist_temp[count, :] = states.tdists
                            GS_sim_temp[count], simz_temp[count], dist = fit_metrics_simulation(bounds, states.deltas)
                            GS_sim_temp_fixK[count] , simz_temp_fixK[count], dist = fit_metrics_simulation(bounds, states.get_deltas(k=self.nstates))

                        res3['optimum'][idxl, idxs, idxi] = np.mean(optimum_temp)
                        res3['sim_GS'][idxl, idxs, idxi] = np.mean(GS_sim_temp)
                        res3['sim_GS_fixK'][idxl, idxs, idxi] = np.mean(GS_sim_temp_fixK)
                        res3['simz_GS'][idxl, idxs, idxi] = np.mean(simz_temp)
                        res3['simz_GS_fixK'][idxl, idxs, idxi] = np.mean(simz_temp_fixK)
                        res3['tdist'][idxl, idxs, idxi, :] = tdist_temp.mean(0)

                else:
                    states = gsbs_extra.GSBS(x=np.mean(subData[:, :, :], axis=0), kmax=self.maxK)
                    states.fit()

                    res3['optimum'][:, idxs, idxi] = states.nstates
                    res3['sim_GS'][:, idxs, idxi], res3['simz_GS'][:, idxs, idxi],dists = fit_metrics_simulation(bounds, states.deltas)
                    res3['sim_GS_fixK'][:, idxs, idxi],res3['simz_GS_fixK'][:, idxs, idxi],dists = fit_metrics_simulation(bounds, states.get_deltas(k=self.nstates))
                    res3['tdist'][:, idxs, idxi, :] = states.tdists

                    # subbounds = states.fitsubject(subData)
                    # for isub in range(nsub):
                    #     res3['optimum_subopt'][idxs, isub] = np.shape(subbounds[isub][subbounds[isub]>0])[0]
                    #     res3['sim_GS_subopt'][idxs, isub], res3['simz_GS_subopt'][idxs, isub], dists = fit_metrics_simulation(bounds, subbounds[isub])
        return res3


    #simulation 4, can we correctly estimate the number of states in the group when there is ideosyncracy in state boundaries between participants?
    def run_simulation_sub_specific_states(self, CV_list, sub_evprob_list, kfold_list, sub_std, nsub, rep):

        res4=dict()
        list=['optimum', 'sim_GS','sim_GS_fixK',  'simz_GS', 'simz_GS_fixK']
        for key in list:
            res4[key] = np.zeros([np.shape(CV_list)[0], np.shape(sub_evprob_list)[0], np.shape(kfold_list)[0]])
        res4['tdist'] = np.zeros([np.shape(CV_list)[0], np.shape(sub_evprob_list)[0], np.shape(kfold_list)[0], self.maxK + 1])
        list = ['optimum_subopt', 'sim_GS_subopt', 'simz_GS_subopt']
        for key in list:
            res4[key] = np.zeros([np.shape(sub_evprob_list)[0], nsub])

        for idxs, s in enumerate(sub_evprob_list):
            bounds, subData, subbounds = self.generate_simulated_data_HRF(sub_evprob=s, nsub=nsub, sub_std=sub_std, rep=rep)

            for idxi, i in enumerate(kfold_list):
                print(rep, s, i)
                if i > 1:
                    kf = KFold(n_splits=i, shuffle=True)

                    for idxl, l in enumerate(CV_list):

                        tdist_temp = np.zeros([i,self.maxK+1]);  optimum_temp = np.zeros(i); GS_sim_temp = np.zeros(i)
                        GS_sim_temp_fixK = np.zeros(i); simz_temp = np.zeros(i); simz_temp_fixK = np.zeros(i)

                        count = -1
                        for train_index, test_index in kf.split(np.arange(0, np.max(kfold_list))):
                            count = count + 1
                            if l is False:
                                states = gsbs_extra.GSBS(x=np.mean(subData[test_index, :, :], axis=0), kmax=self.maxK)
                            elif l is True:
                                states = gsbs_extra.GSBS(x=np.mean(subData[train_index, :, :], axis=0),
                                              y=np.mean(subData[test_index, :, :], axis=0), kmax=self.maxK)
                            states.fit()

                            optimum_temp[count] = states.nstates
                            tdist_temp[count, :] = states.tdists
                            GS_sim_temp[count],  simz_temp[count], dist = fit_metrics_simulation(
                                bounds, states.bounds)
                            GS_sim_temp_fixK[count], simz_temp_fixK[
                                count], dist = fit_metrics_simulation(bounds, states.get_bounds(k=self.nstates))

                        res4['optimum'][idxl, idxs, idxi] = np.mean(optimum_temp)
                        res4['sim_GS'][idxl, idxs, idxi] = np.mean(GS_sim_temp)
                        res4['sim_GS_fixK'][idxl, idxs, idxi] = np.mean(GS_sim_temp_fixK)
                        res4['simz_GS'][idxl, idxs, idxi] = np.mean(simz_temp)
                        res4['simz_GS_fixK'][idxl, idxs, idxi] = np.mean(simz_temp_fixK)
                        res4['tdist'][idxl, idxs, idxi, :] = tdist_temp.mean(0)

                else:
                    states = gsbs_extra.GSBS(x=np.mean(subData[:, :, :], axis=0), kmax=self.maxK)
                    states.fit()

                    res4['optimum'][:, idxs, idxi] = states.nstates
                    res4['sim_GS'][:, idxs, idxi], res4['simz_GS'][:, idxs, idxi],dists = fit_metrics_simulation(bounds, states.bounds)
                    res4['sim_GS_fixK'][:, idxs, idxi], res4['simz_GS_fixK'][:, idxs, idxi],dists = fit_metrics_simulation(bounds, states.get_bounds(k=self.nstates))
                    res4['tdist'][:, idxs, idxi, :] = states.tdists

                    # subbounds = states.fitsubject(subData)
                    # for isub in range(nsub):
                    #     res4['optimum_subopt'][idxs, isub] = np.max(subbounds[isub])
                    #     res4['sim_GS_subopt'][idxs, isub], res4['simz_GS_subopt'][idxs, isub], dists = fit_metrics_simulation(bounds,subbounds[isub])

        return res4

    # simulation 5, vary the peak and dispersion of the HRF
    def run_simulation_hrf_shape(self, nstates_list, peak_delay_list, peak_disp_list, rep):
        print(rep)
        res5=dict()
        list=['optimum']
        for key in list:
            res5[key] = np.zeros([np.shape(nstates_list)[0], np.shape(peak_delay_list)[0], np.shape(peak_disp_list)[0]])

        for idxe,e in enumerate(nstates_list):
            for idxde, de in enumerate(peak_delay_list):
                for idxdp, dp in enumerate(peak_disp_list):

                    bounds, subData,_ = self.generate_simulated_data_HRF(nstates=e, peak_delay=de, peak_disp=dp, rep=rep)
                    states = gsbs_extra.GSBS(x=subData[0,:,:],kmax=self.maxK)
                    states.fit()
                    res5['optimum'][idxe, idxde, idxdp] = states.nstates

        return res5


    def run_simulation_computation_time(self, nstates, rep):
        bounds, subData,_ = self.generate_simulated_data_HRF(rep=rep)
        res6 = dict()
        res6['duration_GSBS'] = np.zeros([nstates])
        res6['duration_HMM_fixK'] = np.zeros([nstates])
        res6['duration_HMMsm_fixK'] = np.zeros([nstates])

        for i in range(2,nstates):
            print(rep, i)
            states = gsbs_extra.GSBS(x=subData[0, :, :], kmax=i)
            tic = timeit.default_timer()
            states.fit()
            res6['duration_GSBS'][i] = timeit.default_timer()-tic

            tic = timeit.default_timer()
            ev = HMM(i, split_merge=False)
            ev.fit(subData[0, :, :])
            res6['duration_HMM_fixK'][i] = timeit.default_timer() - tic

            tic = timeit.default_timer()
            ev = HMM(i, split_merge=True)
            ev.fit(subData[0, :, :])
            res6['duration_HMMsm_fixK'][i] = timeit.default_timer() - tic

        res6['duration_HMM_estK'] = np.cumsum(res6['duration_HMM_fixK'])
        res6['duration_HMMsm_estK'] = np.cumsum(res6['duration_HMMsm_fixK'])

        return res6