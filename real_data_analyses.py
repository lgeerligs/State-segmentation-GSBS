import numpy as np
import os
import pickle
from statesegmentation import GSBS
from sklearn.model_selection import KFold
import gsbs_extra
from scipy.io import loadmat
#this is a nilearn function that I adapted to return the signals from all voxels in the searchlight
import nifti_spheres_masker_allvox
from brainiak.eventseg.event import EventSegment as HMM
from help_functions import compute_fits_hmm, compute_reliability, correct_fit_metric, deltas_states, compute_reliability_pcor
from sklearn.metrics import adjusted_mutual_info_score



# get the data for each of the rois
def get_data(radiuslist, coords, CBUIDs, savedir, datadir, default_radius):
    # create dct, similar to spm_dctmtx
    N = 192
    TR = 2.47
    n = np.arange(0, N)
    C = np.zeros([np.shape(n)[0], N])
    C[:, 0] = np.ones([np.shape(n)[0]]) / np.sqrt(N)
    for k in np.arange(1, N):
        C[:, k] = np.sqrt(2 / N) * np.cos(np.pi * (2 * n + 1) * (k - 1) / (2 * N))
    nHP = np.int(np.floor(2 * (N * TR) / (1 / 0.008)) + 1)
    filter = C[:, 0:nHP + 1]

    for radius in radiuslist:
        for roi in range(0, len(coords)):
            for hyp in range(0, 2):
                if hyp == 1:
                    savename = savedir + 'group_data_roi' + str(roi) + '_radius' + str(radius) + '.p'
                    dataloc = datadir + 'Hyperalign_v1/'
                    postfix = '_hyperaligned.nii'
                elif hyp==0 and radius == default_radius:
                    savename = savedir + 'group_data_roi' + str(roi) + '_radius' + str(radius) + '_nohyp.p'
                    dataloc = datadir
                    postfix = '_s0w_ME_denoised.nii'
                else:
                    continue
                if not os.path.exists(savename):
                    subfile = dataloc + CBUIDs[0][0] + postfix
                    obj = nifti_spheres_masker_allvox._ExtractionFunctor(seeds_=[coords[roi]], radius=radius,
                                                                         mask_img=None,
                                                                         allow_overlap=True, dtype=None)
                    subdata = obj(subfile)
                    subdata = subdata[0][0]
                    nsubjects = np.shape(CBUIDs)[0]
                    group_data = np.zeros([nsubjects, N, np.shape(subdata)[1]])

                    for count, s in enumerate(CBUIDs):
                        print(roi, count)
                        subfile = dataloc + s[0] + postfix
                        subdata = obj(subfile)
                        subdata = subdata[0][0]
                        R = np.eye(N) - np.matmul(filter, np.linalg.pinv(filter))
                        aY = np.matmul(R, subdata[0:N,:])
                        group_data[count, :, :] = aY

                    ISS = np.zeros([nsubjects, np.shape(group_data)[2]])

                    for s in range(0, nsubjects):
                        print(s)
                        subtest = np.setdiff1d(np.arange(0, nsubjects), s)
                        testdata = np.nanmean(group_data[subtest, :, :], 0)
                        traindata = group_data[s, :, :]
                        for r in range(0, np.shape(group_data)[2]):
                            ISS[s, r] = np.corrcoef(testdata[:, r], traindata[:, r])[0, 1]

                    with open(savename, 'wb') as output:
                        pickle.dump({'ISS': ISS, 'group_data': group_data}, output, pickle.HIGHEST_PROTOCOL)

def get_voxelnums(roilist, radiuslist,savedir):
    #get the number of voxels for each searchlight size
    voxelnum=np.zeros((len(roilist), len(radiuslist)))
    for roi in roilist:
        for rcount, radius in enumerate(radiuslist):
           file = open(savedir + 'group_data_roi' + str(roi) + '_radius' + str(radius) + '.p', 'rb')
           res = pickle.load(file)
           voxelnum[roi, rcount]=np.shape(res['ISS'])[1]
    return voxelnum

# detect the GS state boundaries in a particular searchlight, both for T-distance and WAC
def run_state_detection(kfold, roi, savedir, CV, radius, ISSth=-1, hyp=1, maxk:int=100, overwrite=False):
    if hyp == 0:
        outname = savedir + 'results_roi' + str(roi) + '_kfold' + str(kfold) + '_radius' + str(radius) + '_CV' + str(
            CV) + 'typeGS_nohyp.p'
        fname = savedir + 'group_data_roi' + str(roi) + '_radius' + str(radius) + '_nohyp.p'
    elif hyp == 1:
        outname = savedir + 'results_roi' + str(roi) + '_kfold' + str(kfold) + '_radius' + str(radius) + '_CV' + str(
            CV) + '_typeGS.p'
        fname = savedir + 'group_data_roi' + str(roi) + '_radius' + str(radius) + '.p'

    if ISSth != 0.35:
        outname=outname[0:-2] + 'ISS' + str(ISSth) + '.p'

    print(outname)
    if not os.path.exists(outname) or overwrite is True:
        file = open(fname, 'rb')
        res = pickle.load(file)
        ISS = res['ISS']
        ISSm = np.nanmean(ISS, 0)
        if hyp == 1:
            indices = np.squeeze(np.argwhere(ISSm >= ISSth))
        elif hyp==0:
            #make sure that only voxels that are hyperaligned are included in the analysis
            fname2 = savedir + 'group_data_roi' + str(roi) + '_radius' + str(radius) + '.p'
            file2 = open(fname2, 'rb')
            res2 = pickle.load(file2)
            ISS2 = res2['ISS']
            ISSm2 = np.nanmean(ISS2, 0)
            indices = np.intersect1d(np.squeeze(np.argwhere(ISSm >= ISSth)), np.squeeze(np.argwhere(ISSm2 >= -1)))

        group_data = res['group_data'][:, :, indices]
        nvox = len(indices)
        optimum_fold = np.zeros(kfold)
        optimum_wac_fold = np.zeros(kfold)
        fin_bounds_folds = np.zeros([kfold, np.shape(group_data)[1]])
        all_bounds_folds = np.zeros([kfold, maxk+1, np.shape(group_data)[1]])
        optimum_mdist_fold = np.zeros(kfold)

        fit_W = np.zeros([kfold, maxk+1])
        fit_Bcon = np.zeros([kfold, maxk+1])
        fit_Ball = np.zeros([kfold, maxk+1])
        tdist = np.zeros([kfold, maxk+1])
        wac = np.zeros([kfold, maxk+1])
        mdist = np.zeros([kfold, maxk+1])

        if kfold > 1:
            kf = KFold(n_splits=kfold, shuffle=True, random_state=1)

            count = -1
            for train_index, test_index in kf.split(np.arange(0, np.shape(group_data)[0])):
                count = count + 1
                traindata = np.mean(group_data[train_index, :, :], axis=0)
                if kfold == group_data.shape[0]:
                    testdata = np.squeeze(group_data[test_index, :, :])
                else:
                    testdata = np.mean(group_data[test_index, :, :], axis=0)

                if CV is True:
                    states = gsbs_extra.GSBS(x=traindata, y=testdata, kmax=maxk, outextra=True)
                else:
                    states = gsbs_extra.GSBS(x=testdata, kmax=maxk, outextra=True)

                states.fit()

                optimum_fold[count] = states.nstates
                optimum_wac_fold[count] = states.nstates_WAC
                optimum_mdist_fold[count] = states.nstates_mdist
                fin_bounds_folds[count, :] = states.bounds
                all_bounds_folds[count, :,:] = states.all_bounds

                fit_W[count, :] = states.all_m_W
                fit_Bcon[count, :] = states.all_m_Bcon
                fit_Ball[count, :] = states.all_m_Ball
                tdist[count, :] = states.tdists
                mdist[count, :] = states.mdist
                wac[count, :] = states.WAC

        else:
            traindata = np.squeeze(np.mean(group_data[:, :, :], axis=0))
            states = gsbs_extra.GSBS(x=traindata, kmax=maxk, outextra=True)
            states.fit()

            optimum_fold = states.nstates
            optimum_wac_fold = states.nstates_WAC
            optimum_mdist_fold = states.nstates_mdist
            fin_bounds_folds = states.bounds
            all_bounds_folds = states.all_bounds

            fit_W = states.all_m_W
            fit_Bcon = states.all_m_Bcon
            fit_Ball = states.all_m_Ball
            tdist = states.tdists
            mdist = states.mdist
            wac = states.WAC

        res = {'optimum_fold': optimum_fold, 'optimum_wac_fold': optimum_wac_fold,'all_bounds_folds': all_bounds_folds,
               'fin_bounds_folds': fin_bounds_folds, 'fit_W': fit_W,
               'fit_Bcon': fit_Bcon, 'fit_Ball': fit_Ball, 'tdist': tdist,
               'wac': wac, 'mdist':mdist, 'optimum_mdist_fold':optimum_mdist_fold, 'nvox':nvox}
        with open(outname, 'wb') as output:
            pickle.dump(res, output, pickle.HIGHEST_PROTOCOL)
    else:
        file = open(outname, 'rb')
        res = pickle.load(file)

    return res


# detect the HMM state boundaries for a fixed number of states
def run_state_detection_HMM(kfold, roi, savedir, CV, radius,  kvals, ISSth=-1, maxk=100, overwrite=False):
    outname = savedir + 'results_roi' + str(roi) + '_kfold' + str(kfold) + '_radius' + str(radius) + '_CV' + str(
        CV) + 'ISS' + str(ISSth) + '_typeHMM.p'

    if not os.path.exists(outname) or overwrite is True:
        file = open(savedir + 'group_data_roi' + str(roi) + '_radius' + str(radius) + '.p', 'rb')
        res = pickle.load(file)
        ISS = res['ISS']
        ISSm = np.nanmean(ISS, 0)
        indices = np.squeeze(np.argwhere(ISSm >= ISSth))
        group_data = res['group_data'][:, :, indices]


        if kfold > 1:
            LL_HMM = np.ones([kfold, maxk])*np.nan
            WAC_HMM = np.ones([kfold, maxk])*np.nan
            tdist_HMM = np.ones([kfold, maxk])*np.nan
            optimum_LL = np.zeros([kfold]).astype(int)
            optimum_WAC = np.zeros([kfold]).astype(int)
            optimum_tdist = np.zeros([kfold]).astype(int)
            hmm_bounds_LL = np.zeros([kfold, np.shape(group_data)[1]])
            hmm_bounds_split_LL = np.zeros([kfold, np.shape(group_data)[1]])
            hmm_bounds_WAC = np.zeros([kfold, np.shape(group_data)[1]])
            hmm_bounds_split_WAC = np.zeros([kfold, np.shape(group_data)[1]])
            hmm_bounds_tdist = np.zeros([kfold, np.shape(group_data)[1]])
            hmm_bounds_split_tdist = np.zeros([kfold, np.shape(group_data)[1]])
            hmm_bounds_fixK = np.zeros([kfold, len(kvals), np.shape(group_data)[1]])
            hmm_bounds_split_fixK = np.zeros([kfold, len(kvals), np.shape(group_data)[1]])

            kf = KFold(n_splits=kfold, shuffle=True, random_state=1)

            count = -1
            for train_index, test_index in kf.split(np.arange(0, np.shape(group_data)[0])):
                count = count + 1
                traindata = np.mean(group_data[train_index, :, :], axis=0)
                if kfold == group_data.shape[0]:
                    testdata = np.squeeze(group_data[test_index, :, :])
                else:
                    testdata = np.mean(group_data[test_index, :, :], axis=0)

                t = None
                ind = None
                for i in range(2, maxk):
                    print(roi, i)
                    if CV is True:
                        LL_HMM[count,i], WAC_HMM[count,i], tdist_HMM[count,i], _, t, ind = compute_fits_hmm(data=traindata, k=i, mindist=1, type='HMM', y=testdata, t1=t, ind1=ind)
                    else:
                        LL_HMM[count, i], WAC_HMM[count, i], tdist_HMM[count, i], _, t, ind = compute_fits_hmm(data=testdata, k=i, mindist=1,  type='HMM', y=None, t1=t, ind1=ind)

                x = testdata
                optimum_LL[count] = np.argmax(LL_HMM[count, 2:]).astype(int)+ 2
                optimum_WAC[count] = np.argmax(WAC_HMM[count, 2:]).astype(int)+ 2
                optimum_tdist[count] = np.argmax(tdist_HMM[count, 2:]).astype(int)+ 2
                _, _, _, hmm_bounds_LL[count, :], t, ind = compute_fits_hmm(data=x, k=optimum_LL[count], mindist=1, type='HMM', y=None, t1=t, ind1=ind)
                _, _, _, hmm_bounds_split_LL[count, :], t, ind = compute_fits_hmm(data=x, k=optimum_LL[count], mindist=1,type='HMMsplit', y=None, t1=t, ind1=ind)

                _, _, _, hmm_bounds_WAC[count, :], t, ind = compute_fits_hmm(data=x, k=optimum_WAC[count], mindist=1, type='HMM', y=None, t1=t, ind1=ind)
                _, _, _, hmm_bounds_split_WAC[count, :], t, ind = compute_fits_hmm(data=x, k=optimum_WAC[count], mindist=1,type='HMMsplit', y=None, t1=t, ind1=ind)
                _, _, _, hmm_bounds_tdist[count, :], t, ind = compute_fits_hmm(data=x, k=optimum_tdist[count], mindist=1, type='HMM', y=None, t1=t, ind1=ind)
                _, _, _, hmm_bounds_split_tdist[count, :], t, ind = compute_fits_hmm(data=x, k=optimum_tdist[count], mindist=1,type='HMMsplit', y=None, t1=t, ind1=ind)

                for kind, k in enumerate(kvals):
                    _, _, _, hmm_bounds_fixK[count,kind, :], t, ind = compute_fits_hmm(data=x, k=k, mindist=1,type='HMM', y=None, t1=t, ind1=ind)
                    _, _, _, hmm_bounds_split_fixK[count,kind, :], t, ind = compute_fits_hmm(data=x, k=k, mindist=1,type='HMMsplit', y=None, t1=t,ind1=ind)

        else:
            LL_HMM = np.ones([maxk])*np.nan
            WAC_HMM = np.ones([maxk])*np.nan
            tdist_HMM = np.ones([maxk])*np.nan
            hmm_bounds_fixK = np.zeros([len(kvals), np.shape(group_data)[1]])
            hmm_bounds_split_fixK = np.zeros([len(kvals), np.shape(group_data)[1]])

            traindata = np.squeeze(np.mean(group_data[:, :, :], axis=0))

            t = None; ind=None
            for i in range(2, maxk):
                print(roi, i)
                LL_HMM[i], WAC_HMM[i], tdist_HMM[i], _, t, ind = compute_fits_hmm(data=traindata, k=i, mindist=1, type='HMM', y=None, t1=t, ind1=ind)

            optimum_LL = np.argmax(LL_HMM[2:])+2
            optimum_WAC = np.argmax(WAC_HMM[2:]) + 2
            optimum_tdist = np.argmax(tdist_HMM[2:]) + 2

            _, _, _, hmm_bounds_LL, t, ind = compute_fits_hmm(data=traindata, k=optimum_LL, mindist=1, type='HMM', y=None, t1=t, ind1=ind)
            _, _, _, hmm_bounds_split_LL, t, ind = compute_fits_hmm(data=traindata, k=optimum_LL, mindist=1,type='HMMsplit', y=None, t1=t, ind1=ind)

            _, _, _, hmm_bounds_WAC, t, ind = compute_fits_hmm(data=traindata, k=optimum_WAC, mindist=1, type='HMM', y=None, t1=t, ind1=ind)
            _, _, _, hmm_bounds_split_WAC, t, ind = compute_fits_hmm(data=traindata, k=optimum_WAC, mindist=1,type='HMMsplit', y=None, t1=t, ind1=ind)

            _, _, _, hmm_bounds_tdist, t, ind = compute_fits_hmm(data=traindata, k=optimum_tdist, mindist=1, type='HMM', y=None, t1=t, ind1=ind)
            _, _, _, hmm_bounds_split_tdist, t, ind = compute_fits_hmm(data=traindata, k=optimum_tdist, mindist=1,type='HMMsplit', y=None, t1=t, ind1=ind)

            for kind, k in enumerate(kvals):
                _, _, _, hmm_bounds_fixK[kind, :], t, ind = compute_fits_hmm(data=traindata, k=k, mindist=1,type='HMM', y=None, t1=t, ind1=ind)
                _, _, _, hmm_bounds_split_fixK[kind, :], t, ind = compute_fits_hmm(data=traindata, k=k, mindist=1,type='HMMsplit', y=None, t1=t,ind1=ind)


        res={'optimum_LL':optimum_LL, 'optimum_WAC':optimum_WAC, 'optimum_tdist':optimum_tdist,'LL_HMM': LL_HMM, 'WAC_HMM':WAC_HMM, 'tdist_HMM':tdist_HMM, 'hmm_bounds_LL':hmm_bounds_LL, 'hmm_bounds_split_LL':hmm_bounds_split_LL, 'hmm_bounds_WAC':hmm_bounds_WAC,
                         'hmm_bounds_split_WAC':hmm_bounds_split_WAC, 'hmm_bounds_tdist':hmm_bounds_tdist, 'hmm_bounds_split_tdist':hmm_bounds_split_tdist, 'hmm_bounds_fixK':hmm_bounds_fixK, 'hmm_bounds_split_fixK':hmm_bounds_split_fixK}
        with open(outname, 'wb') as output:
            pickle.dump(res, output, pickle.HIGHEST_PROTOCOL)
    else:
        file = open(outname, 'rb')
        res = pickle.load(file)
    return res


def summarize_results_hypISS(roilist:np.ndarray, ISSlist:np.ndarray, kfold:int, savedir:str, CV:bool, radius:int, optimumvals):
    #think about when to match K

    hypISS_results=dict()
    hypISS_results['foldsim_sim_matchk_all'] = np.zeros([np.shape(roilist)[0], 2, np.shape(ISSlist)[0], kfold])
    hypISS_results['foldsim_simz_matchk_all'] = np.zeros([np.shape(roilist)[0], 2, np.shape(ISSlist)[0], kfold])
    hypISS_results['foldsim_simsub_matchk_all'] = np.zeros([np.shape(roilist)[0], 2, np.shape(ISSlist)[0], kfold])
    hypISS_results['optimum'] = np.zeros([np.shape(roilist)[0], 2, np.shape(ISSlist)[0], kfold])
    hypISS_results['nvox'] = np.zeros([np.shape(roilist)[0], 2, np.shape(ISSlist)[0]])

    for roicount, roi in enumerate(roilist):
        for hyp in range(0,2):
            for i, ISSth in enumerate(ISSlist):
                print(roicount, hyp, i)
                if hyp == 0:
                    outname = savedir + 'results_roi' + str(roi) + '_kfold' + str(kfold) + '_radius' + str(radius) + '_CV' + str(CV) + 'typeGS_nohyp.p'
                elif hyp == 1:
                    outname = savedir + 'results_roi' + str(roi) + '_kfold' + str(kfold) + '_radius' + str(radius) + '_CV' + str(CV) + '_typeGS.p'

                if ISSth != 0.35:
                    outname = outname[0:-2] + 'ISS' + str(ISSth) + '.p'

                file = open(outname, 'rb')
                res = pickle.load(file)
                hypISS_results['nvox'][roicount, hyp, i] = res['nvox']
                hypISS_results['optimum'][roicount, hyp, i,:]=res['optimum_fold']
                matchk_bounds_folds = res['all_bounds_folds'][:,optimumvals[roicount],:]
                matchk_bounds_folds[matchk_bounds_folds > 0] = 1

                hypISS_results['foldsim_sim_matchk_all'][roicount, hyp, i, :], hypISS_results['foldsim_simz_matchk_all'][roicount, hyp, i, :] = compute_reliability(matchk_bounds_folds)

    return hypISS_results


#reliability for matched-k based on t-distance

# load the results
# get reliability across folds for each number of folds
# get the optimum for each number of folds
def summarize_results(roilist, kfoldlist, savedir, CV, type, radiuslist, optimumvals, kvals, default_folds=15):
    # think about when to match K!

    GS_results = dict()
    list=['optimum_wac', 'optimum', 'foldsim_sim_matchk', 'foldsim_sim_matchk_std',
           'foldsim_simz_matchk', 'foldsim_simz_matchk_std', 'optimum_wac_sd', 'optimum_sd']
    for key in list:
        GS_results[key] = np.zeros([np.shape(roilist)[0], np.shape(kfoldlist)[0], np.shape(radiuslist)[0]])

    list=['fit_W', 'fit_Bcon', 'fit_Ball', 'tdist', 'wac']
    for key in list:
        GS_results[key] = np.zeros([np.shape(roilist)[0], default_folds, 101])

    GS_results['optimum_wac_all'] = np.zeros([np.shape(roilist)[0], default_folds])
    GS_results['optimum_all'] = np.zeros([np.shape(roilist)[0], default_folds])
    GS_results['bounds_matchk_folds'] = np.zeros([np.shape(roilist)[0], len(kvals) + 1, default_folds, 192])
    GS_results['foldsim_sim_matchk_all'] = np.zeros([np.shape(roilist)[0], len(kvals) + 1, default_folds])
    GS_results['foldsim_simz_matchk_all'] = np.zeros([np.shape(roilist)[0], len(kvals) + 1, default_folds])

    for roicount, roi in enumerate(roilist):
        klist = np.append(kvals, optimumvals[roicount])
        for radiuscount, radius in enumerate(radiuslist):
            for kcount, kfold in enumerate(kfoldlist):
                outname = savedir + 'results_roi' + str(roi) + '_kfold' + str(kfold) + '_radius' + str(
                    radius) + '_CV' + str(CV) + '_type' + type + 'ISS-1.p'
                file = open(outname, 'rb')
                res = pickle.load(file)
                optimum_fold = res['optimum_fold']
                optimum_wac_fold = res['optimum_wac_fold']

                if kfold==1:
                    matchk_bounds_folds = res['all_bounds_folds'][optimumvals[roicount], :]
                else:
                    matchk_bounds_folds = res['all_bounds_folds'][:,optimumvals[roicount],:]

                matchk_bounds_folds[matchk_bounds_folds > optimumvals[roicount]] = 0
                matchk_bounds_folds[matchk_bounds_folds > 0] = 1

                GS_results['optimum_wac'][roicount, kcount, radiuscount] = np.mean(optimum_wac_fold)
                GS_results['optimum'][roicount, kcount, radiuscount] = np.mean(optimum_fold)
                GS_results['optimum_wac_sd'][roicount, kcount, radiuscount] = np.std(optimum_wac_fold)
                GS_results['optimum_sd'][roicount, kcount, radiuscount] = np.std(optimum_fold)

                print([roicount, radiuscount, kcount])
                if kfold > 1:
                    sim, simz = compute_reliability(matchk_bounds_folds)
                    GS_results['foldsim_sim_matchk'][roicount, kcount, radiuscount] = np.mean(sim)
                    GS_results['foldsim_sim_matchk_std'][roicount, kcount, radiuscount] = np.std(sim)
                    GS_results['foldsim_simz_matchk'][roicount, kcount, radiuscount] = np.mean(simz)
                    GS_results['foldsim_simz_matchk_std'][roicount, kcount, radiuscount] = np.std(simz)

                if kfold == default_folds and radius == 8:
                    GS_results['optimum_wac_all'][roicount, :] = optimum_wac_fold
                    GS_results['optimum_all'][roicount, :] = optimum_fold
                    for kcount, kval in enumerate(klist):
                        matchk_bounds_folds = res['all_bounds_folds'][:,kval,:]
                        matchk_bounds_folds[matchk_bounds_folds > kval] = 0
                        matchk_bounds_folds[matchk_bounds_folds > 0] = 1
                        GS_results['foldsim_sim_matchk_all'][roicount, kcount, :], GS_results['foldsim_simz_matchk_all'][roicount, kcount, :] = compute_reliability(matchk_bounds_folds)
                        GS_results['bounds_matchk_folds'][roicount, kcount, :, :] = matchk_bounds_folds

                    GS_results['fit_W'][roicount, :, :] = res['fit_W']
                    GS_results['fit_Bcon'][roicount, :, :] = res['fit_Bcon']
                    GS_results['fit_Ball'][roicount, :, :] = res['fit_Ball']
                    GS_results['tdist'][roicount, :, :] = res['tdist']
                    GS_results['wac'][roicount, :, :] = res['wac']

    return GS_results

# load the results
# get reliability across folds for each number of folds
# get the optimum for each number of folds
def summarize_results_HMM(kfold, roilist, savedir, CV, radius, kvals, optimumvals, ISSth=-1):

    HMM_results = dict()

    HMM_results['bounds_matchk_folds'] = np.zeros([np.shape(roilist)[0], len(kvals) + 1, kfold, 192])
    HMM_results['bounds_matchk_folds_split'] = np.zeros([np.shape(roilist)[0], len(kvals) + 1, kfold, 192])

    list = ['optimum_WAC', 'optimum_LL', 'optimum_tdist','foldsim_sim_LL', 'foldsim_simz_LL', 'foldsim_sim_LL_split',  'foldsim_simz_LL_split',
         'foldsim_sim_WAC',  'foldsim_simz_WAC', 'foldsim_sim_WAC_split',  'foldsim_simz_WAC_split',
         'foldsim_sim_tdist',  'foldsim_simz_tdist', 'foldsim_sim_tdist_split', 'foldsim_simz_tdist_split']
    for key in list:
        HMM_results[key] = np.ones([np.shape(roilist)[0], kfold])

    list = ['tdists', 'WAC', 'LL']
    for key in list:
        HMM_results[key] = np.zeros([np.shape(roilist)[0], kfold, 100])

    list = ['foldsim_sim_klist', 'foldsim_simz_klist', 'foldsim_sim_klist_split',  'foldsim_simz_klist_split']
    for key in list:
        HMM_results[key] = np.zeros([np.shape(roilist)[0], len(kvals) + 1, kfold])

    for roicount, roi in enumerate(roilist):
        klist = np.append(kvals, optimumvals[roicount])
        outname = savedir + 'results_roi' + str(roi) + '_kfold' + str(kfold) + '_radius' + str(radius) + '_CV' + str(
            CV) + 'ISS' + str(ISSth) + '_typeHMM.p'
        file = open(outname, 'rb')
        res = pickle.load(file)

        HMM_results['optimum_WAC'][roicount, :] = res['optimum_WAC']
        HMM_results['optimum_LL'][roicount, :] = res['optimum_LL']
        HMM_results['optimum_tdist'][roicount, :] = res['optimum_tdist']
        for kcount, kval in enumerate(klist):
            #here, k is kept equal across all folds
            hmm_bounds = res['hmm_bounds_fixK'][:,kcount,:]
            HMM_results['foldsim_sim_klist'][roicount, kcount, :], HMM_results['foldsim_simz_klist'][roicount, kcount, :] = compute_reliability(hmm_bounds)
            HMM_results['bounds_matchk_folds'][roicount, kcount, :, :] = hmm_bounds
            hmm_bounds_split = res['hmm_bounds_split_fixK'][:,kcount,:]
            HMM_results['foldsim_sim_klist_split'][roicount, kcount, :], HMM_results['foldsim_simz_klist_split'][roicount, kcount, :] = compute_reliability(hmm_bounds_split)
            HMM_results['bounds_matchk_folds_split'][roicount, kcount, :, :] = hmm_bounds_split

        hmm_bounds = res['hmm_bounds_LL']
        HMM_results['foldsim_sim_LL'][roicount, :], HMM_results['foldsim_simz_LL'][roicount, :] = compute_reliability(hmm_bounds)
        hmm_bounds_split = res['hmm_bounds_split_LL']
        HMM_results['foldsim_sim_LL_split'][roicount, :], HMM_results['foldsim_simz_LL_split'][roicount, :] = compute_reliability(hmm_bounds_split)

        hmm_bounds = res['hmm_bounds_WAC']
        HMM_results['foldsim_sim_WAC'][roicount, :],  HMM_results['foldsim_simz_WAC'][roicount, :] = compute_reliability(hmm_bounds)
        hmm_bounds_split = res['hmm_bounds_split_WAC']
        HMM_results['foldsim_sim_WAC_split'][roicount, :],  HMM_results['foldsim_simz_WAC_split'][roicount, :] = compute_reliability(hmm_bounds_split)

        hmm_bounds = res['hmm_bounds_tdist']
        HMM_results['foldsim_sim_tdist'][roicount, :],  HMM_results['foldsim_simz_tdist'][roicount, :] = compute_reliability(hmm_bounds)
        hmm_bounds_split = res['hmm_bounds_split_tdist']
        HMM_results['foldsim_sim_tdist_split'][roicount, :], HMM_results['foldsim_simz_tdist_split'][roicount, :] = compute_reliability(hmm_bounds_split)

        HMM_results['tdists'][roicount, :, :] = res['tdist_HMM']
        HMM_results['WAC'][roicount, :, :] = res['WAC_HMM']
        HMM_results['LL'][roicount, :, :] = res['LL_HMM']

    return HMM_results


# compare averaging data first or averaging last
def average_first_or_last(roilist, savedir, radius=8, kfold=15):
    inds = np.squeeze(np.argwhere(np.triu(np.ones((192, 192)), 1).flatten() == 1))

    rel_avglast = np.zeros((len(roilist), kfold))
    rel_avgfirst = np.zeros((len(roilist), kfold))
    for roicount, roi in enumerate(roilist):
        file = open(savedir + 'group_data_roi' + str(roi) + '_radius' + str(radius) + '.p', 'rb')
        res = pickle.load(file)
        group_data = res['group_data'][:, :, :]

        matavg = np.zeros((kfold, 192, 192))
        matsingle = np.zeros((kfold, 192, 192))

        kf = KFold(n_splits=kfold, shuffle=True, random_state=1)

        count = -1
        for train_index, test_index in kf.split(np.arange(0, np.shape(group_data)[0])):
            count = count + 1
            matavg[count, :, :] = np.corrcoef(np.mean(group_data[test_index, :, :], axis=0))
            cm = np.zeros((len(test_index), 192, 192))
            for jcount, j in enumerate(test_index):
                cm[jcount, :, :] = np.corrcoef(group_data[j, :, :])
            matsingle[count, :, :] = np.mean(cm, axis=0)

        matsingle = matsingle.reshape(kfold, 192 * 192)
        matavg = matavg.reshape(kfold, 192 * 192)
        rel_avglast[roicount, :] = compute_reliability_pcor(matsingle[:, inds])
        rel_avgfirst[roicount, :] = compute_reliability_pcor(matavg[:, inds])
    return rel_avglast, rel_avgfirst


# get the optimal value of K for the default values
def optimalK(roilist, savedir, kfold, CV=False, type='GS', radius=8):
    optimum = np.zeros(len(roilist))
    for roicount, roi in enumerate(roilist):
        outname = savedir + 'results_roi' + str(roi) + '_kfold' + str(kfold) + '_radius' + str(
            radius) + '_CV' + str(CV) + '_type' + type + 'ISS-1.p'
        file = open(outname, 'rb')
        res = pickle.load(file)
        optimum[roicount] = np.mean(res['optimum_fold'])
    return optimum

#relation to behavioral event boundaries
def relate_events(HMMresults, GSresults):
    list = ['HMM_sim','HMM_simz','HMMsm_sim','HMMsm_simz','GS_sim','GS_simz']
    res_beh = dict()
    for key in list:
        res_beh[key] = np.ones((HMMresults['bounds_matchk_folds'].shape[0], HMMresults['bounds_matchk_folds'].shape[1],HMMresults['bounds_matchk_folds'].shape[2]))

    TR = 2.47
    onsets = loadmat('/home/lingee/wrkgrp/Cambridge_data/Movie_HMM/Results_method_paper/' + 'subjective_event_onsets.mat')['event_onsets']
    onsets = np.round((onsets+5)/TR).astype(int)
    beh = np.zeros((192,1))
    beh[onsets-1]=1
    beh=deltas_states(beh)
    for k in np.arange(0,HMMresults['bounds_matchk_folds'].shape[0]):
        for roi in np.arange(0,HMMresults['bounds_matchk_folds'].shape[1]):
            for f in np.arange(HMMresults['bounds_matchk_folds'].shape[2]):
                print(k,roi,f)
                res_beh['HMM_sim'][roi,k,f], res_beh['HMM_simz'][roi,k,f]= correct_fit_metric(deltas_states(HMMresults['bounds_matchk_folds'][roi,k,f]), beh, pflag=True)
                res_beh['HMMsm_sim'][roi,k,f], res_beh['HMMsm_simz'][roi,k,f]= correct_fit_metric(deltas_states(HMMresults['bounds_matchk_folds_split'][roi,k,f]), beh, pflag=True)
                res_beh['GS_sim'][roi,k,f], res_beh['GS_simz'][roi,k,f]= correct_fit_metric(deltas_states(GSresults['bounds_matchk_folds'][roi,k,f]), beh, pflag=True)
    return res_beh
