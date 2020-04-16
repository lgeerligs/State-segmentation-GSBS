#this is a nilearn function that I adapted to return the signals from all voxels in the searchlight
from nilearn.input_data import nifti_spheres_masker_allvox
import numpy as np
import os
import pickle
import seaborn
import matplotlib.pyplot as plt
from scipy import stats
from operator import itemgetter
import pandas as pd
import real_data_analyses as rda

#load subject information
exec(open('load_subject_info_for_statedetection.py').read())
subin=np.argwhere(age<=50)[:,0]
CBUIDs=CBUID[subin]

#create dct, similar to spm_dctmtx
N=192
TR=2.47
n = np.arange(0,N)
C=np.zeros([np.shape(n)[0],N])
C[:,0] = np.ones([np.shape(n)[0]])/np.sqrt(N)
for k in np.arange(1,N):
    C[:,k] = np.sqrt(2/N)*np.cos(np.pi*(2*n+1)*(k-1)/(2*N))
nHP = np.int(np.floor(2*(N*TR)/(1/0.008))+1)
filter = C[:,0:nHP+1]

#load data
basedir = '/home/lingee/wrkgrp/Cambridge_data/Movie_HMM/'
datadir = basedir + 'Data_for_Donders/'
savedir = basedir + 'Results_method_paper/'
nsubjects = np.shape(CBUIDs)[0]

#create small ROIs (spheres)
coords=[(-4,-90,-2), (-50, -52,-8),(-42,-64,40),(0,54,22), (-44,-72,2)]
labels=['V1', 'inferior temporal', 'angular gyrus', 'medial prefrontal', 'V5'] #from neurosynth

kfoldlist=[1,2,5,10,15,20,np.shape(subin)[0]]
radiuslist=[6,8,10,12]

#select rois to use
roilist=np.arange(0,len(coords))
roilist = np.array([0,4,1,2,3]).astype('int')
roilabels = itemgetter(*roilist)(labels)

format = '.pdf'

default_rad_ind=np.argwhere(np.array(radiuslist)==8)[0][0]
default_folds=15
default_folds_ind=np.argwhere(np.array(kfoldlist)==default_folds)[0][0]
kvals = np.array([10,20, 30, 40])

#get the data for each of the rois
for radius in radiuslist:
    for roi in range(0, len(coords)):
        if not os.path.exists(savedir + 'group_data_roi' + str(roi) + '_radius' + str(radius) + '.p'):
            subfile = datadir + 'Hyperalign_v1/' + CBUIDs[0][0] + '_hyperaligned.nii'
            obj = nifti_spheres_masker_allvox._ExtractionFunctor(seeds_=[coords[roi]], radius=radius, mask_img=None,
                                                                 allow_overlap=True, dtype=None)
            subdata = obj(subfile)
            subdata = subdata[0][0]
            group_data = np.zeros([nsubjects, np.shape(subdata)[0], np.shape(subdata)[1]])

            for count, s in enumerate(CBUIDs):
                print(roi, count)
                subfile = datadir + 'Hyperalign_v1/' + s[0] + '_hyperaligned.nii'
                subdata=obj(subfile)
                subdata=subdata[0][0]
                R = np.eye(N) - np.matmul(filter,np.linalg.pinv(filter))
                aY = np.matmul(R, subdata)
                group_data[count, :, :] = aY

            ISS=np.zeros([nsubjects, np.shape(group_data)[2]])

            for s in range(0,nsubjects):
                print(s)
                subtest = np.setdiff1d(np.arange(0,nsubjects), s)
                testdata = np.nanmean(group_data[subtest,:,:],0)
                traindata = group_data[s,:,:]
                for r in range(0,np.shape(group_data)[2]):
                    ISS[s,r] = np.corrcoef(testdata[:,r], traindata[:,r])[0,1]

            with open(savedir + 'group_data_roi' + str(roi) + '_radius' + str(radius) + '.p', 'wb') as output:
                 pickle.dump({'ISS': ISS, 'group_data':group_data}, output, pickle.HIGHEST_PROTOCOL)

#get the number of voxels for each searchlight size
voxelnum=np.zeros((len(roilist), len(radiuslist)))
for roi in roilist:
    for rcount, radius in enumerate(radiuslist):
       file = open(savedir + 'group_data_roi' + str(roi) + '_radius' + str(radius) + '.p', 'rb')
       res = pickle.load(file)
       voxelnum[roi, rcount]=np.shape(res['ISS'])[1]

#run analyses for GS
for radius in radiuslist:
    for roicount, roi in enumerate(roilist):
        for k in kfoldlist:
            print(roi, k)
            rda.run_state_detection(kfold=k, roi=roi, savedir=savedir, CV=False, type='GS', radius=radius)

#get the optimal number of states for the default settings in each searchlight
optimumvals = rda.optimalK(roilist, savedir, kfold=default_folds)

#get results GS
optimum_wac, optimum, optimum_wac_sd, optimum_sd, foldsim,foldsim_matchk, foldsim_std, foldsim_matchk_std\
    , optimum_wac_all, optimum_all, foldsim_matchk_all, bounds_matchk_folds, optimum_wac_max, optimum_max\
    = rda.summarize_results(roilist, kfoldlist, savedir, False, 'GS', radiuslist, optimumvals, kvals, default_folds=default_folds)

#get HMM state boundaries for optimum and for a fixed set of k-vals
foldsim_matchk_HMM = np.zeros((len(roilist), len(kvals) + 1, default_folds))
hmm_bounds_folds = np.zeros((len(roilist), len(kvals) + 1, default_folds, 192))
for roicount, roi in enumerate(roilist):
        print(roi)
        hmm_bounds_folds[roicount,:,:,:] = rda.run_state_detection_HMM(kfold=default_folds, roi=roi, savedir=savedir, CV=False, type='HMM', radius=8, kvals= np.append(kvals, np.int(np.round(optimumvals[roicount]))))
        for kval in range(0,len(kvals)+1):
            foldsim_matchk_HMM[roicount,kval,:]=rda.LOO_reliability(hmm_bounds_folds[roicount,kval,:,:])


#show the distribution of state lengths for the optimal value of K for HMM versus GS.
stateLength=np.zeros((2,len(roilist),default_folds,192))
stateLength_list={}
for roi in range(0,len(roilist)):
    templist=np.zeros((2,default_folds,np.int(np.round(optimumvals[roi]))))
    for m in range(0,2):
        if m == 0:
            boundaries = bounds_matchk_folds[roi,len(kvals),:,:]
        elif m == 1:
            boundaries = hmm_bounds_folds[roi,len(kvals),:,:]
        for k in range(0,default_folds):
            states=np.zeros(192)
            for i in range(1,192):
                states[i]=states[i-1]+boundaries[k,i]
            for i in range(0, 192):
                stateLength[m, roi,k,i]=np.sum(states == states[i])
            for e in range(0, np.int(np.max(states))):
                ind=np.argmax(stateLength[m, roi,k,:] == e)
                templist[m,k,e]=stateLength[m, roi,k,ind]
    stateLength_list[roi]=templist


#show distribution of state lengths for HMM and GS
pal=seaborn.color_palette("Set2", 2)
SD=np.zeros((len(roilist),default_folds,2))
for roi in range(0,len(roilist)):
    SD[roi,:,0] = np.std(stateLength_list[roi][0,:,:], axis=1)
    SD[roi,:,1] = np.std(stateLength_list[roi][1, :, :],  axis=1)

plt.figure()
rois=np.repeat(np.expand_dims(np.arange(0,len(roilist)),1),default_folds, axis=1)
rois=np.repeat(np.expand_dims(rois,2),2, axis=2)
method=np.stack([np.ones((len(roilist),default_folds)),np.ones((len(roilist),default_folds))*2], axis=2)
d={'SD':SD.flatten(), 'method':method.flatten(), 'rois':rois.flatten()}
df=pd.DataFrame(d)
bp = seaborn.boxplot(data = df, x='rois', y='SD', hue='method', width=0.5, palette=pal)
handles, labels2=plt.gca().get_legend_handles_labels()
plt.legend(handles,['GS', 'HMM'], loc='upper right')
plt.xticks(np.arange(0, len(roilist)), roilabels)
plt.title('Effect of state segmentation method on SD of unit duration')
plt.savefig(savedir + 'SD_unit_length' + format, format=format[1:])

#compute statistical tests for SD of state duration
pval = np.zeros((len(roilist)))
#t-test reliability
for i in range(0, len(roilist)):
        co, pval[i] = stats.wilcoxon(SD[i,:,0],SD[i,:,1])

#show that the optimal number of states aligns with the anticipated cortical hierarchy
pal=seaborn.color_palette("Set2", 2)
plt.figure()
bp = seaborn.boxplot(np.repeat(np.arange(0,len(roilist)),default_folds), optimum_all.flatten(), width=0.5, color=pal[0])
handles, labels2=plt.gca().get_legend_handles_labels()
plt.xticks(np.arange(0,len(roilist)), roilabels)
plt.title('Hierarchy of timescales, Kfold=' + str(default_folds) + ', Radius=8')
plt.savefig(savedir + 'Timescale_hierarchy' + format, format=format)

plt.figure()
bp = seaborn.boxplot(np.repeat(np.arange(0,len(roilist)),default_folds), optimum_wac_all.flatten(), width=0.5, color=pal[0])
handles, labels2=plt.gca().get_legend_handles_labels()
plt.xticks(np.arange(0,len(roilist)),  roilabels)
plt.title('Hierarchy of timescales, Kfold=' + str(default_folds) + ', Radius=8, WBall')
plt.savefig(savedir + 'Timescale_hierarchy_wac' + format, format=format[1:])

#show the fit values
fit_W, fit_Bcon, fit_Ball, tdist, wac = rda.get_fit_values(roilist, savedir=savedir, radius=8, kfold=default_folds, CV=False, type='GS')

#plot fit for wac
pal=seaborn.color_palette("Set2", 6)
plt.rcParams['font.size']=10
maxK=150
f,ax=plt.subplots(len(roilist),1)
plt.setp(ax, yticks=[0,2,4], xlim=[0, 150])
for roi in range(0,len(roilist)):
    ax[roi].set_title(roilabels[roi])
    ax[roi].plot(np.arange(0, maxK + 1), fit_W[roi,:,:].mean(axis=0), color=pal[2])
    ax[roi].fill_between(np.arange(0, maxK+1), fit_W[roi,:,:].mean(axis=0)-fit_W[roi,:,:].std(axis=0), fit_W[roi,:,:].mean(axis=0)+fit_W[roi,:,:].std(axis=0), alpha=0.3, color=pal[2])
    ax[roi].plot(np.arange(0, maxK + 1), fit_Ball[roi,:,:].mean(axis=0), color=pal[4])
    ax[roi].fill_between(np.arange(0, maxK + 1), fit_Ball[roi,:,:].mean(axis=0) - fit_Ball[roi,:,:].std(axis=0),
                        fit_Ball[roi,:,:].mean(axis=0) + fit_Ball[roi,:,:].std(axis=0), alpha=0.3, color=pal[4])
    ax[roi].plot(wac[roi,:,:].mean(0), color=pal[1])
    ax[roi].set_ylabel('Correlation (z)')
plt.legend(['Within', 'Between - all', 'wac'])
plt.tight_layout()
plt.savefig(savedir + 'Fit_lines_real_data_wac' + format, format=format[1:])

#plot fit for T-distance
plt.rcParams['font.size']=10
maxK=150
f,ax=plt.subplots(len(roilist),1)
plt.setp(ax, yticks=[0,2,4], xlim=[0, 150])
for roi in range(0,len(roilist)):
    ax[roi].set_title(roilabels[roi])
    ax[roi].plot(np.arange(0, maxK + 1), fit_W[roi,:,:].mean(axis=0), color=pal[2])
    ax[roi].fill_between(np.arange(0, maxK+1), fit_W[roi,:,:].mean(axis=0)-fit_W[roi,:,:].std(axis=0), fit_W[roi,:,:].mean(axis=0)+fit_W[roi,:,:].std(axis=0), alpha=0.3, color=pal[2])
    ax[roi].plot(np.arange(0, maxK + 1), fit_Bcon[roi,:,:].mean(axis=0), color=pal[3])
    ax[roi].fill_between(np.arange(0, maxK + 1), fit_Bcon[roi,:,:].mean(axis=0) - fit_Bcon[roi,:,:].std(axis=0),
                        fit_Bcon[roi,:,:].mean(axis=0) + fit_Bcon[roi,:,:].std(axis=0), alpha=0.3, color=pal[3])
    ax2 = ax[roi].twinx()
    ax2.plot(tdist[roi,:,:].mean(axis=0), color=pal[0])
    ax[roi].set_ylabel('Correlation (z)')
    ax2.set_ylabel('T-value')
plt.legend(['Within', 'Between - con', 'T-distance'])
plt.tight_layout()
plt.savefig(savedir + 'Fit_lines_real_data_tdist' + format, format=format[1:])


#15-fold cross validation, radius=8 how does the reliability vary between HMM and GS
rois=np.repeat(np.expand_dims(np.arange(0,len(roilist)),1),default_folds, axis=1)
rois=np.repeat(np.expand_dims(rois,2),2, axis=2)
for k in range(0, len(kvals)+1):
    plt.figure()
    foldsims=np.stack([foldsim_matchk_all[:,k,:],foldsim_matchk_HMM[:,k,:]], axis=2)
    method=np.stack([np.ones((len(roilist),default_folds)),np.ones((len(roilist),default_folds))*2], axis=2)
    d={'data':foldsims.flatten(), 'method':method.flatten(), 'rois':rois.flatten()}
    df=pd.DataFrame(d)
    bp = seaborn.boxplot(data = df, x='rois', y='data', hue='method', width=0.5, palette=pal)
    handles, labels2=plt.gca().get_legend_handles_labels()
    plt.legend(handles,['GS', 'HMM'], loc='upper right')
    plt.xticks(np.arange(0, len(roilist)), roilabels)
    if k < len(kvals):
        plt.title('Effect of state segmentation method on reliability, k = ' + str(kvals[k]))
        plt.savefig(savedir + 'reliability_real_data_' + str(k) + format, format=format[1:])
    else:
        plt.title('Effect of state segmentation method on reliability')
        plt.savefig(savedir + 'reliability_real_data_tdist_' + format, format=format[1:])

#compute statistical tests to compare reliability
tval = np.zeros((len(roilist), len(kvals)+1))
pval = np.zeros((len(roilist), len(kvals)+1))
#t-test reliability
for i in range(0, len(roilist)):
    for k in range(0, len(kvals)+1):
        tval[i, k], pval[i,k] = stats.ttest_rel(np.arctan(foldsim_matchk_all[i,k,:]),np.arctan(foldsim_matchk_HMM[i,k,:]))


#15 fold cross validation, how does reliability vary with sphere size
pal=seaborn.color_palette("Set2", 5)
plt.figure()
for i in range(0,len(roilist)):
    plt.plot(radiuslist,foldsim_matchk[i,default_folds_ind-1,:].T,color=pal[i], linestyle='--', marker='o')
    plt.fill_between(x=radiuslist,y1=foldsim_matchk[i,default_folds_ind-1,:].T-(foldsim_matchk_std[i,default_folds_ind-1,:].T/np.sqrt(default_folds)),y2=foldsim_matchk[i,default_folds_ind-1,:].T+(foldsim_matchk_std[i,default_folds_ind-1,:].T/np.sqrt(default_folds)), facecolor=pal[i], alpha=0.5)
plt.xlabel('Radius')
plt.ylabel('Reliability')
plt.xticks(radiuslist)
plt.yticks([0.5, 0.7, 0.9])
plt.title('Effect of sphere size on reliability of boundaries, Kfold=' + str(default_folds))
plt.legend(loc='upper left', labels=roilabels)
plt.savefig(savedir + 'Reliability_spheresize' + format, format=format[1:])

numsubs=np.zeros(len(kfoldlist))
for kcount, k in enumerate(kfoldlist):
    numsubs[kcount]=265/k

#how does reliability vary with cross validation, radius=8?
pal=seaborn.color_palette("Set2", len(roilist))
plt.figure()
for i in range(0,len(roilist)):
    plt.plot(numsubs[1:],foldsim_matchk[i,1:,default_rad_ind].T,color=pal[i], linestyle='--', marker='o')
    plt.fill_between(x=numsubs[1:],y1=foldsim_matchk[i,1:,default_rad_ind].T-(foldsim_matchk_std[i,1:,default_rad_ind].T/np.sqrt(default_folds)),y2=foldsim_matchk[i,1:,default_rad_ind].T+(foldsim_matchk_std[i,1:,default_rad_ind].T/np.sqrt(default_folds)), facecolor=pal[i], alpha=0.5)
plt.xlabel('Number of averaged participants')
plt.ylabel('Reliability')
plt.title('Effect of no. averaged participants on reliability, radius=8')
plt.legend(loc='lower right', labels=roilabels)
plt.savefig(savedir + 'Reliability_averaging' + format, format=format[1:])

#how does reliability vary with cross validation, radius=8?
pal=seaborn.color_palette("Set2", len(roilist))
f, ax = plt.subplots(1)
for i in range(0,len(roilist)):
    ax.plot(numsubs[1:],foldsim_matchk[i,1:,default_rad_ind].T,color=pal[i], linestyle='--', marker='o')
    ax.fill_between(x=numsubs[1:],y1=foldsim_matchk[i,1:,default_rad_ind].T-(foldsim_matchk_std[i,1:,default_rad_ind].T/np.sqrt(default_folds)),y2=foldsim_matchk[i,1:,default_rad_ind].T+(foldsim_matchk_std[i,1:,default_rad_ind].T/np.sqrt(default_folds)), facecolor=pal[i], alpha=0.5)
plt.xlabel('Number of averaged participants')
plt.ylabel('Reliability')
plt.title('Effect of no. averaged participants on reliability, radius=8')
plt.legend(loc='lower right', labels=roilabels)
plt.setp(ax, xlim=[-1, 30])
plt.savefig(savedir + 'Reliability_averaging_zoom' + format, format=format[1:])


#15 fold cross validation, how does optimum vary with sphere size
pal=seaborn.color_palette("Set2", len(roilist))
plt.figure()
for i in range(0,len(roilist)):
    plt.plot(radiuslist,optimum[i,default_folds_ind,:].T,color=pal[i], linestyle='--', marker='o')
    plt.fill_between(x=radiuslist,y1=optimum[i,default_folds_ind,:].T-(optimum_sd[i,default_folds_ind,:].T/np.sqrt(default_folds)),y2=optimum[i,default_folds_ind,:].T+(optimum_sd[i,default_folds_ind,:].T/np.sqrt(default_folds)), facecolor=pal[i], alpha=0.5)
plt.xlabel('Radius')
plt.ylabel('Optimal number of units')
plt.xticks(radiuslist)
plt.legend(loc='upper left', labels=roilabels)
plt.title('Effect of sphere size on optimal number of units, Kfold' + str(default_folds))
plt.savefig(savedir + 'Optimum_spheresize' + format, format=format[1:])


#how does optimum vary with cross validation, radius=8?
pal=seaborn.color_palette("Set2", len(roilist))
f, ax = plt.subplots(1)
for i in range(0,len(roilist)):
    ax.plot(numsubs,optimum[i,:,default_rad_ind].T,color=pal[i], linestyle='--', marker='o')
    ax.fill_between(x=numsubs,y1=optimum[i,:,default_rad_ind].T-(optimum_sd[i,:,default_rad_ind].T/np.sqrt(default_folds)),y2=optimum[i,:,default_rad_ind].T+(optimum_sd[i,:,default_rad_ind].T/np.sqrt(default_folds)), facecolor=pal[i], alpha=0.5)
plt.xlabel('Number of averaged participants')
plt.ylabel('Optimal number of units')
plt.setp(ax, xlim=[-1, 55])
plt.title('Effect of no. averaged participants on optimal number of units, radius=8')
plt.savefig(savedir + 'Optimum_averaging_zoom' + format, format=format[1:])

#how does optimum vary with cross validation, radius=8?
pal=seaborn.color_palette("Set2", len(roilist))
plt.figure()
for i in range(0,len(roilist)):
    plt.plot(numsubs,optimum[i,:,default_rad_ind].T,color=pal[i], linestyle='--', marker='o')
    plt.fill_between(x=numsubs,y1=optimum[i,:,default_rad_ind].T-(optimum_sd[i,:,default_rad_ind].T/np.sqrt(default_folds)),y2=optimum[i,:,default_rad_ind].T+(optimum_sd[i,:,default_rad_ind].T/np.sqrt(default_folds)), facecolor=pal[i], alpha=0.5)
plt.xlabel('Number of averaged participants')
plt.ylabel('Optimal number of units')
plt.title('Effect of no. averaged participants on optimal number of units, radius=8')
plt.savefig(savedir + 'Optimum_averaging' + format, format=format[1:])



#show that averaging before gives more reliable time*time matrices than averaging after
#note - reliability is worse when we use fisher r-to-z
rel_avglast, rel_avgfirst = rda.average_first_or_last(roilist, savedir=savedir, kfold=default_folds)
pal=seaborn.color_palette("Set2", 2)
plt.figure()
rois = np.repeat(np.expand_dims(np.arange(0,len(roilist)),1),default_folds, axis=1)
rois = np.repeat(np.expand_dims(rois,2),2, axis=2)
rel_all = np.stack([rel_avglast,rel_avgfirst], axis=2)
method = np.stack([np.ones((len(roilist),default_folds))*2,np.ones((len(roilist),default_folds))], axis=2)
bp = seaborn.boxplot(rois.flatten(), rel_all.flatten(), hue=method.flatten(), width=0.5, palette=pal)
handles, labels2 = plt.gca().get_legend_handles_labels()
plt.legend(handles, ['average timeseries', 'average cmat'], loc='u pper right')
plt.xticks(np.arange(0,len(roilist)), roilabels)
plt.title('Time * time matrix reliability, effect of averaging timeseries or cmat')
plt.savefig(savedir + 'reliability_averaging_first_last' + format, format=format[1:])

#compute statistical tests for averagin first or last
tval = np.zeros((len(roilist)))
pval = np.zeros((len(roilist)))
#t-test reliability
for i in range(0, len(roilist)):
        tval[i], pval[i] = stats.ttest_rel(np.arctan(rel_avglast[i,:]),np.arctan(rel_avgfirst[i,:]))





