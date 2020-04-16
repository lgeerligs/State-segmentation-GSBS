import numpy as np
import os
import pickle
from state_boundary_detection import StateSegment
from sklearn.model_selection import KFold



# detect the GS state boundaries in a particular searchlight, both for T-distance and WAC
def run_state_detection(kfold, roi, savedir, CV, type, radius):
    outname = savedir + 'results_roi' + str(roi) + '_kfold' + str(kfold) + '_radius' + str(radius) + '_CV' + str(
        CV) + '_type' + type + '.p'
    if not os.path.exists(outname):
        file = open(savedir + 'group_data_roi' + str(roi) + '_radius' + str(radius) + '.p', 'rb')
        res = pickle.load(file)
        ISS = res['ISS']
        ISSm = np.nanmean(ISS, 0)
        indices = np.squeeze(np.argwhere(ISSm >= 0.35))
        group_data = res['group_data'][:, :, indices]

        optimum_fold = np.zeros(kfold)
        optimum_wac_fold = np.zeros(kfold)
        fin_bounds_folds = np.zeros([kfold, np.shape(group_data)[1]])
        all_bounds_folds = np.zeros([kfold, np.shape(group_data)[1]])

        fit_W = np.zeros([kfold, 151])
        fit_Bcon = np.zeros([kfold, 151])
        fit_Ball = np.zeros([kfold, 151])
        tdist = np.zeros([kfold, 151])
        wac = np.zeros([kfold, 151])

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
                    states = StateSegment(traindata, testdata, maxK=150)
                else:
                    states = StateSegment(testdata, testdata, maxK=150)

                states.train(CV, outextra=True, type=type)

                optimum_fold[count] = states.optimum
                optimum_wac_fold[count] = states.optimum_wac
                fin_bounds_folds[count, :] = states.fin_bounds
                all_bounds_folds[count, :] = states.all_bounds

                fit_W[count, :] = states.all_m_W
                fit_Bcon[count, :] = states.all_m_Bcon
                fit_Ball[count, :] = states.all_m_Ball
                tdist[count, :] = states.tdist
                wac[count, :] = states.wac

        else:
            traindata = np.squeeze(np.mean(group_data[:, :, :], axis=0))
            states = StateSegment(traindata, traindata, maxK=150)
            states.train(CV=False, outextra=True, type=type)

            optimum_fold = states.optimum
            optimum_wac_fold = states.optimum_wac
            fin_bounds_folds = states.fin_bounds
            all_bounds_folds = states.all_bounds

            fit_W = states.all_m_W
            fit_Bcon = states.all_m_Bcon
            fit_Ball = states.all_m_Ball
            tdist = states.tdist
            wac = states.wac

        res = {'optimum_fold': optimum_fold, 'optimum_wac_fold': optimum_wac_fold,
               'fin_bounds_folds': fin_bounds_folds, 'all_bounds_folds': all_bounds_folds, 'fit_W': fit_W,
               'fit_Bcon': fit_Bcon, 'fit_Ball': fit_Ball, 'tdist': tdist,
               'wac': wac}

        with open(outname, 'wb') as output:
            pickle.dump(res, output, pickle.HIGHEST_PROTOCOL)

    else:
        file = open(outname, 'rb')
        res = pickle.load(file)

    return res


# detect the HMM state boundaries for a fixed number of states
def run_state_detection_HMM(kfold, roi, savedir, CV, type, radius, kvals):
    outname = savedir + 'results_roi' + str(roi) + '_kfold' + str(kfold) + '_radius' + str(radius) + '_CV' + str(
        CV) + '_type' + type + '.p'
    if not os.path.exists(outname):
        file = open(savedir + 'group_data_roi' + str(roi) + '_radius' + str(radius) + '.p', 'rb')
        res = pickle.load(file)
        ISS = res['ISS']
        ISSm = np.nanmean(ISS, 0)
        indices = np.squeeze(np.argwhere(ISSm >= 0.35))
        group_data = res['group_data'][:, :, indices]

        hmm_bounds_folds = np.zeros([len(kvals), kfold, np.shape(group_data)[1]])

        for kcount, kval in enumerate(kvals):
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
                        states = StateSegment(traindata, testdata, maxK=kval)
                    else:
                        states = StateSegment(testdata, testdata, maxK=kval)
                    hmm_bounds_folds[kcount, count, :] = states.train_HMM(fixedK=True)

            else:
                traindata = np.squeeze(np.mean(group_data[:, :, :], axis=0))
                states = StateSegment(traindata, traindata, maxK=kval)
                hmm_bounds_folds[kcount, :] = states.train_HMM(fixedK=True)

        with open(outname, 'wb') as output:
            pickle.dump({'hmm_bounds_folds': hmm_bounds_folds}, output, pickle.HIGHEST_PROTOCOL)

    else:
        file = open(outname, 'rb')
        res = pickle.load(file)
        hmm_bounds_folds = res['hmm_bounds_folds']
    return hmm_bounds_folds


# subfunction to compute reliability, is used below
def LOO_reliability(data):
    indlist = np.arange(0, data.shape[0])
    reliability = np.zeros(len(indlist))
    for i in indlist:
        avgdata = np.mean(data[np.setdiff1d(indlist, i), :], axis=0)
        reliability[i] = np.corrcoef(avgdata, data[i, :])[0, 1]

    return reliability


# load the results
# get reliability across folds for each number of folds
# get the optimum for each number of folds
def summarize_results(roilist, kfoldlist, savedir, CV, type, radiuslist, optimumvals, kvals, default_folds=20):
    optimum_wac = np.zeros([np.shape(roilist)[0], np.shape(kfoldlist)[0], np.shape(radiuslist)[0]])
    optimum = np.zeros([np.shape(roilist)[0], np.shape(kfoldlist)[0], np.shape(radiuslist)[0]])
    foldsim = np.zeros([np.shape(roilist)[0], np.shape(kfoldlist)[0], np.shape(radiuslist)[0]])
    foldsim_matchk = np.zeros([np.shape(roilist)[0], np.shape(kfoldlist)[0], np.shape(radiuslist)[0]])
    foldsim_std = np.zeros([np.shape(roilist)[0], np.shape(kfoldlist)[0], np.shape(radiuslist)[0]])
    foldsim_matchk_std = np.zeros([np.shape(roilist)[0], np.shape(kfoldlist)[0], np.shape(radiuslist)[0]])

    optimum_wac_sd = np.zeros([np.shape(roilist)[0], np.shape(kfoldlist)[0], np.shape(radiuslist)[0]])
    optimum_sd = np.zeros([np.shape(roilist)[0], np.shape(kfoldlist)[0], np.shape(radiuslist)[0]])
    optimum_wac_max = np.zeros([np.shape(roilist)[0], np.shape(kfoldlist)[0], np.shape(radiuslist)[0]])
    optimum_max = np.zeros([np.shape(roilist)[0], np.shape(kfoldlist)[0], np.shape(radiuslist)[0]])

    optimum_wac_all = np.zeros([np.shape(roilist)[0], default_folds])
    optimum_all = np.zeros([np.shape(roilist)[0], default_folds])

    bounds_matchk_folds = np.zeros([np.shape(roilist)[0], len(kvals) + 1, default_folds, 192])
    foldsim_matchk_all = np.zeros([np.shape(roilist)[0], len(kvals) + 1, default_folds])

    for roicount, roi in enumerate(roilist):
        klist = np.append(kvals, optimumvals[roicount])
        for radiuscount, radius in enumerate(radiuslist):
            for kcount, kfold in enumerate(kfoldlist):
                outname = savedir + 'results_roi' + str(roi) + '_kfold' + str(kfold) + '_radius' + str(
                    radius) + '_CV' + str(CV) + '_type' + type + '.p'
                file = open(outname, 'rb')
                res = pickle.load(file)
                optimum_fold = res['optimum_fold']
                optimum_wac_fold = res['optimum_wac_fold']
                fin_bounds_folds = res['fin_bounds_folds']
                matchk_bounds_folds = res['all_bounds_folds']
                matchk_bounds_folds[matchk_bounds_folds > optimumvals[roicount]] = 0
                matchk_bounds_folds[matchk_bounds_folds > 0] = 1

                optimum_wac[roicount, kcount, radiuscount] = np.mean(optimum_wac_fold)
                optimum[roicount, kcount, radiuscount] = np.mean(optimum_fold)
                optimum_wac_sd[roicount, kcount, radiuscount] = np.std(optimum_wac_fold)
                optimum_sd[roicount, kcount, radiuscount] = np.std(optimum_fold)
                optimum_wac_max[roicount, kcount, radiuscount] = np.max(optimum_wac_fold)
                optimum_max[roicount, kcount, radiuscount] = np.max(optimum_fold)

                if kfold > 1:
                    simil = LOO_reliability(fin_bounds_folds)
                    foldsim[roicount, kcount, radiuscount] = np.mean(simil)
                    foldsim_std[roicount, kcount, radiuscount] = np.std(simil)

                    simil = LOO_reliability(matchk_bounds_folds)
                    foldsim_matchk[roicount, kcount, radiuscount] = np.mean(simil)
                    foldsim_matchk_std[roicount, kcount, radiuscount] = np.std(simil)

                if kfold == default_folds and radius == 8:
                    optimum_wac_all[roicount, :] = optimum_wac_fold
                    optimum_all[roicount, :] = optimum_fold
                    for kcount, kval in enumerate(klist):
                        matchk_bounds_folds = res['all_bounds_folds']
                        matchk_bounds_folds[matchk_bounds_folds > kval] = 0
                        matchk_bounds_folds[matchk_bounds_folds > 0] = 1
                        foldsim_matchk_all[roicount, kcount, :] = LOO_reliability(matchk_bounds_folds)
                        bounds_matchk_folds[roicount, kcount, :, :] = matchk_bounds_folds

    return optimum_wac, optimum, optimum_wac_sd, optimum_sd, foldsim, foldsim_matchk, foldsim_std, foldsim_matchk_std, optimum_wac_all, optimum_all, foldsim_matchk_all, bounds_matchk_folds, optimum_wac_max, optimum_max


# get the fit values for making the fit subplots
def get_fit_values(roilist, savedir, radius=8, kfold=10, CV=False, type='GS'):
    fit_W = np.zeros([len(roilist), kfold, 151])
    fit_Bcon = np.zeros([len(roilist), kfold, 151])
    fit_Ball = np.zeros([len(roilist), kfold, 151])
    tdist = np.zeros([len(roilist), kfold, 151])
    wac = np.zeros([len(roilist), kfold, 151])
    for roicount, roi in enumerate(roilist):
        outname = savedir + 'results_roi' + str(roi) + '_kfold' + str(kfold) + '_radius' + str(
            radius) + '_CV' + str(CV) + '_type' + type + '.p'
        file = open(outname, 'rb')
        res = pickle.load(file)
        fit_W[roicount, :, :] = res['fit_W']
        fit_Bcon[roicount, :, :] = res['fit_Bcon']
        fit_Ball[roicount, :, :] = res['fit_Ball']
        tdist[roicount, :, :] = res['tdist']
        wac[roicount, :, :] = res['wac']

    return fit_W, fit_Bcon, fit_Ball, tdist, wac


# compare averaging data first or averaging last
def average_first_or_last(roilist, savedir, radius=8, kfold=10):
    inds = np.squeeze(np.argwhere(np.triu(np.ones((192, 192)), 1).flatten() == 1))

    rel_avglast = np.zeros((len(roilist), kfold))
    rel_avgfirst = np.zeros((len(roilist), kfold))
    for roicount, roi in enumerate(roilist):
        file = open(savedir + 'group_data_roi' + str(roi) + '_radius' + str(radius) + '.p', 'rb')
        res = pickle.load(file)
        ISS = res['ISS']
        ISSm = np.nanmean(ISS, 0)
        indices = np.squeeze(np.argwhere(ISSm >= 0.35))
        group_data = res['group_data'][:, :, indices]

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
        rel_avglast[roicount, :] = LOO_reliability(matsingle[:, inds])
        rel_avgfirst[roicount, :] = LOO_reliability(matavg[:, inds])
    return rel_avglast, rel_avgfirst


# get the optimal value of K for the default values
def optimalK(roilist, savedir, kfold, CV=False, type='GS', radius=8):
    optimum = np.zeros(len(roilist))
    for roicount, roi in enumerate(roilist):
        outname = savedir + 'results_roi' + str(roi) + '_kfold' + str(kfold) + '_radius' + str(
            radius) + '_CV' + str(CV) + '_type' + type + '.p'
        file = open(outname, 'rb')
        res = pickle.load(file)
        optimum[roicount] = np.mean(res['optimum_fold'])
    return optimum


