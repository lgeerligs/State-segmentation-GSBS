import numpy as np
from state_boundary_detection import StateSegment
from sklearn.model_selection import KFold
from hrf_estimation import hrf

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

    def generate_simulated_data_HRF(self, nstates=None, group_std=None, sub_std=None, sub_evprob=None, length_std=None, peak_delay=None, peak_disp=None, extime=None, TR=None, nsub=None):

        nstates = nstates or self.nstates
        group_std = group_std or self.group_std
        sub_std = sub_std or self.sub_std
        length_std = length_std or self.length_std
        sub_evprob = sub_evprob or self.sub_evprob
        peak_delay = peak_delay or self.peak_delay
        peak_disp = peak_disp or self.peak_disp
        extime = extime or self.extime
        TR = TR or self.TR
        nsub = nsub or self.nsub

        state_means = np.random.randn(self.nvox, nstates)

        state_labels = np.zeros(self.ntime, dtype=int)
        start_TR = 0
        for e in range(nstates - 1):
            length = round(((self.ntime - start_TR) / (nstates - e)) * (1 + length_std * np.random.randn()))
            length = min(max(length, 1), self.ntime - start_TR - (nstates - e))
            state_labels[start_TR:(start_TR + length)] = e
            start_TR = start_TR + length
        state_labels[start_TR:] = nstates - 1

        bounds = np.insert(np.diff(state_labels),0,0)
        nb = np.array(np.where(bounds == 0)[0])
        b = np.array(np.where(bounds == 1)[0])

        evData = np.zeros([self.nvox, self.ntime+extime])
        for t in range(0,self.ntime):
            evData[:, t] = state_means[:, state_labels[t]]

        #extend the final state to make sure it is present in the final signal
        for te in range(self.ntime, self.ntime+extime):
            evData[:, te]=evData[:, self.ntime-1]

        spmhrf = hrf.spm_hrf_compat(np.arange(0,30,TR), peak_delay=peak_delay, peak_disp=peak_disp)
        BOLDevData = self.convolve_with_hrf(evData, spmhrf)

        groupSig = group_std * np.random.randn(self.nvox,self.ntime+extime)
        BOLDgroupSig = self.convolve_with_hrf(groupSig, spmhrf)

        subData = np.zeros([nsub, self.nvox, self.ntime])
        if sub_evprob == 0:
            for s in range(0,nsub):
                subSig = sub_std * np.random.randn(self.nvox, self.ntime+extime)
                BOLDsubSig = self.convolve_with_hrf(subSig, spmhrf)
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

                subevData = np.zeros([self.nvox, self.ntime+self.extime])
                for t in range(0, self.ntime):
                    if sub_state_labels[t] < nstates:
                        subevData[:, t] = state_means[:, sub_state_labels[t]]
                    else:
                        subevData[:, t] = substate_means[:, sub_state_labels[t]-nstates]

                # extend the final state to make sure it is present in the final signal
                for te in range(self.ntime, self.ntime + self.extime):
                    subevData[:, te] = subevData[:, self.ntime - 1]

                BOLDsubevData = self.convolve_with_hrf(subevData, spmhrf)

                subSig = sub_std * np.random.randn(self.nvox, self.ntime + self.extime)
                BOLDsubSig = self.convolve_with_hrf(subSig, spmhrf)

                subData[s,:,:] = BOLDsubSig + BOLDgroupSig + BOLDsubevData

        return bounds, subData

    def convolve_with_hrf(self, signal, hrf):

        BOLDsignal = np.zeros([self.nvox, self.ntime + self.extime + len(hrf) - 1])
        for n in range(0, self.nvox):
            BOLDsignal[n, :] = np.convolve(signal[n, :], hrf)
        BOLDsignal = BOLDsignal[:, self.extime:self.extime + self.ntime]
        return BOLDsignal


    # simulation 1, vary state length and estimate how accurately we can recover state boundaries
    def run_simulation_evlength(self,length_std, rep):

        GS_sim = np.zeros([np.shape(length_std)[0]])
        HMM_sim = np.zeros([np.shape(length_std)[0]])
        gs_bounds = np.zeros([np.shape(length_std)[0], self.ntime])
        hmm_bounds = np.zeros([np.shape(length_std)[0], self.ntime])
        real_bounds = np.zeros([np.shape(length_std)[0], self.ntime])

        for idxl, l in enumerate(length_std):
            print(rep, l)
            bounds, subData = self.generate_simulated_data_HRF(length_std=l)
            states = StateSegment(X=subData[0,:,:].T, Y=subData[0,:,:].T,maxK=self.nstates, mindist=1)
            states.train(False)
            hmm_bounds[idxl,:] = states.train_HMM(fixedK=True)
            gs = states.all_bounds
            gs[gs > self.nstates] = 0
            gs[gs > 0] = 1
            gs_bounds[idxl] = gs
            GS_sim[idxl] = np.corrcoef(bounds, gs)[0,1]
            HMM_sim[idxl] = np.corrcoef(bounds, hmm_bounds[idxl,:])[0,1]
            real_bounds[idxl] = bounds

        return [GS_sim, HMM_sim, gs_bounds, hmm_bounds, real_bounds]


    #simulation 2, how do the different fit measures compare, depending on how many states there are (more states should cause more similarity between distinct states)
    def run_simulation_compare_fit(self, nstates_list, group_std_list, mindist, rep):

        optimum = np.zeros([np.shape(nstates_list)[0],np.shape(group_std_list)[0]])
        optimum_wac = np.zeros([np.shape(nstates_list)[0],np.shape(group_std_list)[0]])
        tdist = np.zeros([np.shape(nstates_list)[0], np.shape(group_std_list)[0], self.maxK+1])
        wac = np.zeros([np.shape(nstates_list)[0], np.shape(group_std_list)[0], self.maxK+1])
        fit_W_mean = np.zeros([np.shape(nstates_list)[0], np.shape(group_std_list)[0], self.maxK+1])
        fit_W_std = np.zeros([np.shape(nstates_list)[0], np.shape(group_std_list)[0], self.maxK+1])
        fit_Ball_mean = np.zeros([np.shape(nstates_list)[0], np.shape(group_std_list)[0], self.maxK+1])
        fit_Ball_std = np.zeros([np.shape(nstates_list)[0], np.shape(group_std_list)[0], self.maxK+1])
        fit_Bcon_mean = np.zeros([np.shape(nstates_list)[0], np.shape(group_std_list)[0], self.maxK+1])
        fit_Bcon_std = np.zeros([np.shape(nstates_list)[0], np.shape(group_std_list)[0], self.maxK+1])

        for idxg, g in enumerate(group_std_list):
            for idxl, l in enumerate(nstates_list):
                print(rep, l, g)
                bounds, subData = self.generate_simulated_data_HRF(nstates=l, group_std=g)
                states = StateSegment(X=subData[0,:,:].T, Y=subData[0,:,:].T, maxK=self.maxK, mindist=mindist)
                states.train(False, outextra=True)
                optimum[idxl,idxg]=states.optimum
                optimum_wac[idxl,idxg]=states.optimum_wac

                fit_W_mean[idxl, idxg, :] = states.all_m_W
                fit_W_std[idxl, idxg, :] = states.all_sd_W
                fit_Ball_mean[idxl, idxg, :] = states.all_m_Ball
                fit_Ball_std[idxl, idxg, :] = states.all_sd_Ball
                fit_Bcon_mean[idxl, idxg, :] = states.all_m_Bcon
                fit_Bcon_std[idxl, idxg, :] = states.all_sd_Bcon

                tdist[idxl,idxg,:]=states.tdist
                wac[idxl, idxg, :] = states.wac

        return [optimum,optimum_wac,tdist,wac, fit_W_mean, fit_W_std, fit_Ball_mean, fit_Ball_std, fit_Bcon_mean, fit_Bcon_std]



    #simulation 3, can we correctly estimate the number of states in the group when there is ideosyncracy in state boundaries between participants?
    def run_simulation_sub_noise(self, CV_list, sub_std_list, kfold_list, rep):

        optimum = np.zeros([np.shape(CV_list)[0],np.shape(sub_std_list)[0], np.shape(kfold_list)[0]])
        optimum_wac = np.zeros([np.shape(CV_list)[0],np.shape(sub_std_list)[0],np.shape(kfold_list)[0]])
        GS_sim = np.zeros([np.shape(CV_list)[0],np.shape(sub_std_list)[0],np.shape(kfold_list)[0]])
        GS_sim_fixK = np.zeros([np.shape(CV_list)[0],np.shape(sub_std_list)[0],np.shape(kfold_list)[0]])
        tdist = np.zeros([np.shape(CV_list)[0], np.shape(sub_std_list)[0], np.shape(kfold_list)[0],self.maxK+1])
        wac = np.zeros([np.shape(CV_list)[0], np.shape(sub_std_list)[0], np.shape(kfold_list)[0], self.maxK + 1])


        for idxs, s in enumerate(sub_std_list):

            bounds, subData = self.generate_simulated_data_HRF(sub_std=s, nsub=np.max(kfold_list))


            for idxi, i in enumerate(kfold_list):
                print(rep, s, i)
                if i>1:
                    kf = KFold(n_splits=i, shuffle=True)
                    for idxl, l in enumerate(CV_list):

                        WBdist = np.zeros([i,self.maxK+1])
                        WBdist_simple = np.zeros([i, self.maxK + 1])

                        optimum_temp = np.zeros(i)
                        optimum_wac_temp = np.zeros(i)
                        GS_sim_temp = np.zeros(i)
                        GS_sim_temp_fixK = np.zeros(i)

                        count=-1
                        for train_index, test_index in kf.split(np.arange(0,np.max(kfold_list))):
                            count=count+1
                            if l is False:
                                states = StateSegment(np.mean(subData[test_index,:,:],axis=0).T, np.mean(subData[test_index,:,:], axis=0).T,maxK=self.maxK)
                                states.train(False)
                            elif l is True:
                                states = StateSegment(np.mean(subData[train_index, :, :], axis=0).T,
                                                            np.mean(subData[test_index, :, :], axis=0).T, maxK=self.maxK)
                                states.train(True)
                            optimum_temp[count] = states.optimum
                            optimum_wac_temp[count] = states.optimum_wac
                            WBdist[count,:]=states.tdist
                            WBdist_simple[count,:]=states.wac
                            GS_sim_temp[count] = np.corrcoef(states.fin_bounds, bounds)[0,1]

                            gs_bounds = np.copy(states.all_bounds)
                            gs_bounds[gs_bounds > self.nstates] = 0
                            gs_bounds[gs_bounds > 0] = 1
                            GS_sim_temp_fixK[count] = np.corrcoef(bounds, gs_bounds)[0, 1]
                        optimum_wac[idxl, idxs, idxi] = np.mean(optimum_wac_temp)
                        optimum[idxl, idxs, idxi] = np.mean(optimum_temp)
                        GS_sim[idxl, idxs, idxi] = np.mean(GS_sim_temp)
                        GS_sim_fixK[idxl, idxs, idxi] = np.mean(GS_sim_temp_fixK)
                        tdist[idxl, idxs, idxi, :] = WBdist.mean(0)
                        wac[idxl, idxs, idxi, :] = WBdist_simple.mean(0)

                else:
                    states = StateSegment(np.mean(subData[:, :, :], axis=0).T,
                                                np.mean(subData[:, :, :], axis=0).T, maxK=self.maxK)
                    states.train(False)
                    gs_bounds = np.copy(states.all_bounds)
                    gs_bounds[gs_bounds > self.nstates] = 0
                    gs_bounds[gs_bounds > 0] = 1

                    optimum_wac[:, idxs, idxi] = states.optimum_wac
                    optimum[:, idxs, idxi] = states.optimum
                    GS_sim[:, idxs, idxi] = np.corrcoef(states.fin_bounds, bounds)[0, 1]
                    GS_sim_fixK[:, idxs, idxi] = np.corrcoef(bounds, gs_bounds)[0, 1]
                    tdist[:, idxs, idxi, :] = states.tdist
                    wac[:, idxs, idxi, :] = states.wac


        return [optimum_wac, optimum, GS_sim, GS_sim_fixK, tdist, wac]



    #simulation 4, can we correctly estimate the number of states in the group when there is ideosyncracy in state boundaries between participants?
    def run_simulation_sub_specific_states(self, CV_list, sub_evprob_list, kfold_list, rep):

        optimum = np.zeros([np.shape(CV_list)[0],np.shape(sub_evprob_list)[0], np.shape(kfold_list)[0]])
        optimum_wac = np.zeros([np.shape(CV_list)[0],np.shape(sub_evprob_list)[0],np.shape(kfold_list)[0]])
        GS_sim = np.zeros([np.shape(CV_list)[0],np.shape(sub_evprob_list)[0],np.shape(kfold_list)[0]])
        GS_sim_fixK = np.zeros([np.shape(CV_list)[0],np.shape(sub_evprob_list)[0],np.shape(kfold_list)[0]])
        tdist = np.zeros([np.shape(CV_list)[0], np.shape(sub_evprob_list)[0], np.shape(kfold_list)[0],self.maxK+1])
        wac = np.zeros([np.shape(CV_list)[0], np.shape(sub_evprob_list)[0], np.shape(kfold_list)[0], self.maxK + 1])


        for idxs, s in enumerate(sub_evprob_list):

            bounds, subData = self.generate_simulated_data_HRF(sub_evprob=s, nsub=np.max(kfold_list))

            for idxi, i in enumerate(kfold_list):
                print(rep, s, i)
                if i > 1:
                    kf = KFold(n_splits=i, shuffle=True)

                    for idxl, l in enumerate(CV_list):

                        WBdist = np.zeros([i,self.maxK+1])
                        WBdist_simple = np.zeros([i, self.maxK + 1])

                        optimum_temp = np.zeros(i)
                        optimum_wac_temp = np.zeros(i)
                        GS_sim_temp = np.zeros(i)
                        GS_sim_temp_fixK = np.zeros(i)

                        count=-1
                        for train_index, test_index in kf.split(np.arange(0,np.max(kfold_list))):
                            count=count+1
                            if l is False:
                                states = StateSegment(np.mean(subData[test_index, :, :], axis=0).T,
                                                            np.mean(subData[test_index, :, :], axis=0).T, maxK=self.maxK)
                                states.train(False)
                            elif l is True:
                                states = StateSegment(np.mean(subData[train_index, :, :], axis=0).T,
                                                            np.mean(subData[test_index, :, :], axis=0).T, maxK=self.maxK)
                                states.train(True)
                            optimum_temp[count] = states.optimum
                            optimum_wac_temp[count] = states.optimum_wac
                            WBdist[count,:]=states.tdist
                            WBdist_simple[count,:]=states.wac
                            GS_sim_temp[count] = np.corrcoef(states.fin_bounds, bounds)[0,1]

                            gs_bounds = np.copy(states.all_bounds)
                            gs_bounds[gs_bounds > self.nstates] = 0
                            gs_bounds[gs_bounds > 0] = 1
                            GS_sim_temp_fixK[count] = np.corrcoef(bounds, gs_bounds)[0, 1]

                        optimum_wac[idxl,idxs, idxi] = np.mean(optimum_wac_temp)
                        optimum[idxl,idxs, idxi] = np.mean(optimum_temp)
                        GS_sim[idxl,idxs, idxi] = np.mean(GS_sim_temp)
                        GS_sim_fixK[idxl, idxs, idxi] = np.mean(GS_sim_temp_fixK)
                        tdist[idxl,idxs, idxi,:] = WBdist.mean(0)
                        wac[idxl, idxs, idxi, :] = WBdist_simple.mean(0)

                else:
                    states = StateSegment(np.mean(subData[:, :, :], axis=0).T,
                                                np.mean(subData[:, :, :], axis=0).T, maxK=self.maxK)
                    states.train(False)
                    gs_bounds = np.copy(states.all_bounds)
                    gs_bounds[gs_bounds > self.nstates] = 0
                    gs_bounds[gs_bounds > 0] = 1

                    optimum_wac[:, idxs, idxi] = states.optimum_wac
                    optimum[:, idxs, idxi] = states.optimum
                    GS_sim[:, idxs, idxi] = np.corrcoef(states.fin_bounds, bounds)[0, 1]
                    GS_sim_fixK[:, idxs, idxi] = np.corrcoef(bounds, gs_bounds)[0, 1]
                    tdist[:, idxs, idxi, :] = states.tdist
                    wac[:, idxs, idxi, :] = states.wac

        return [optimum_wac, optimum, GS_sim, GS_sim_fixK, tdist, wac]



    # simulation 5, vary the peak and dispersion of the HRF
    def run_simulation_hrf_shape(self, nstates_list, peak_delay_list, peak_disp_list, rep):
        print(rep)
        optimum = np.zeros([np.shape(nstates_list)[0], np.shape(peak_delay_list)[0], np.shape(peak_disp_list)[0]])
        GS_sim = np.zeros([np.shape(nstates_list)[0], np.shape(peak_delay_list)[0], np.shape(peak_disp_list)[0]])
        GS_sim_fixk = np.zeros([np.shape(nstates_list)[0], np.shape(peak_delay_list)[0], np.shape(peak_disp_list)[0]])
        HMM_sim = np.zeros([np.shape(nstates_list)[0], np.shape(peak_delay_list)[0], np.shape(peak_disp_list)[0]])

        for idxe,e in enumerate(nstates_list):
            for idxde, de in enumerate(peak_delay_list):
                for idxdp, dp in enumerate(peak_disp_list):

                    bounds, subData = self.generate_simulated_data_HRF(nstates=e, peak_delay=de, peak_disp=dp)
                    states = StateSegment(subData[0, :, :].T, subData[0, :, :].T, maxK=e, mindist=1)
                    hmm_bounds = states.train_HMM(fixedK=True)
                    HMM_sim[idxe, idxde, idxdp] = np.corrcoef(bounds, hmm_bounds)[0, 1]

                    states = StateSegment(subData[0,:,:].T, subData[0,:,:].T,maxK=100, mindist=1)
                    states.train(False)
                    gs_bounds_fixk = np.copy(states.all_bounds)
                    gs_bounds_fixk[gs_bounds_fixk > e] = 0
                    gs_bounds_fixk[gs_bounds_fixk > 0] = 1
                    optimum[idxe, idxde, idxdp] = states.optimum
                    GS_sim[idxe, idxde, idxdp] = np.corrcoef(bounds, states.fin_bounds)[0,1]
                    GS_sim_fixk[idxe, idxde, idxdp] = np.corrcoef(bounds, gs_bounds_fixk)[0,1]

        return [GS_sim, GS_sim_fixk, HMM_sim, optimum]



