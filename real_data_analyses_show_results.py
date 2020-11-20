
import numpy as np
import os
import pickle
import seaborn
import matplotlib.pyplot as plt
from scipy import stats
from operator import itemgetter
import pandas as pd
import real_data_analyses as rda
from joblib import Parallel, delayed
from importlib import reload

#load subject information
exec(open('/home/lingee/wrkgrp/Cambridge_data/Scripts/Event_segmentation_python/Methods_paper/load_subject_info_for_statedetection.py').read())
subin=np.argwhere(age<=50)[:,0]
CBUIDs=CBUID[subin]

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

default_radius=8
default_rad_ind=np.argwhere(np.array(radiuslist)==default_radius)[0][0]
default_folds=15
default_folds_ind=np.argwhere(np.array(kfoldlist)==default_folds)[0][0]
kvals = np.array([10,20, 30, 40])

#extract all data
rda.get_data(radiuslist, coords, CBUIDs, savedir, datadir, default_radius)
voxelnum = rda.get_voxelnums(roilist, radiuslist,savedir)

#run analyses for GS for hyperalignment and ISS
res = Parallel(n_jobs=50)(delayed(rda.run_state_detection)(kfold=default_folds, roi=roi, savedir=savedir, CV=False, radius=default_radius, ISSth=ISSth, hyp=hyp, overwrite=True) for ISSth in [-1,0.25,0.35,0.45] for hyp in [0,1] for roi in roilist)

#run analyses for GS
res = Parallel(n_jobs=50)(delayed(rda.run_state_detection)(kfold=k, roi=roi, savedir=savedir, CV=False, radius=radius, overwrite=True) for radius in radiuslist for k in kfoldlist for roi in roilist)

#get the optimal number of states for the default settings in each searchlight
optimumvals = rda.optimalK(roilist, savedir, kfold=default_folds, CV=False, type='GS', radius=default_radius)
optimumvals = np.round(optimumvals).astype(int)

#run analyses for HMM
res = Parallel(n_jobs=5)(delayed(rda.run_state_detection_HMM)(kfold=default_folds, roi=roi, savedir=savedir, CV=False, radius=8, kvals = np.append(kvals, optimumvals[roicount]), overwrite=True) for roicount, roi in enumerate(roilist))
res = Parallel(n_jobs=5)(delayed(rda.run_state_detection_HMM)(kfold=default_folds, roi=roi, savedir=savedir, CV=True, radius=8, kvals = np.append(kvals, optimumvals[roicount]), overwrite=True) for roicount, roi in enumerate(roilist))


#get results GS
GSresults = rda.summarize_results(roilist, kfoldlist, savedir, False, 'GS', radiuslist, optimumvals, kvals, default_folds=default_folds)

#get results hypISS
ISSlist=[-1,0.25,0.35,0.45]
hypISS_results = rda.summarize_results_hypISS(roilist, ISSlist, default_folds, savedir, False, default_radius, optimumvals)

#get results HMM
HMMresults = rda.summarize_results_HMM(default_folds, roilist, savedir, False, default_radius, kvals, optimumvals, ISSth=-1)
HMMresultsCV = rda.summarize_results_HMM(default_folds, roilist, savedir, True, default_radius, kvals, optimumvals, ISSth=-1)

res_beh = rda.relate_events(HMMresults, GSresults,savedir)
# save all results
with open(savedir + 'GSresults.p', 'wb') as output:
    pickle.dump(GSresults, output, pickle.HIGHEST_PROTOCOL)
with open(savedir + 'hypISS_results.p', 'wb') as output:
    pickle.dump(hypISS_results, output, pickle.HIGHEST_PROTOCOL)
with open(savedir + 'HMMresults.p', 'wb') as output:
    pickle.dump(HMMresults, output, pickle.HIGHEST_PROTOCOL)
with open(savedir + 'HMMresultsCV.p', 'wb') as output:
    pickle.dump(HMMresultsCV, output, pickle.HIGHEST_PROTOCOL)
with open(savedir + 'res_beh.p', 'wb') as output:
    pickle.dump(res_beh, output, pickle.HIGHEST_PROTOCOL)

# load all results
file = open(savedir + 'GSresults.p', 'rb')
GSresults = pickle.load(file)
file = open(savedir + 'hypISS_results.p', 'rb')
hypISS_results = pickle.load(file)
file = open(savedir + 'HMMresults.p', 'rb')
HMMresults = pickle.load(file)
file = open(savedir + 'HMMresultsCV.p', 'rb')
HMMresultsCV = pickle.load(file)
file = open(savedir + 'res_beh.p', 'rb')
res_beh = pickle.load(file)

#show how reliability varies with hyperalignment options
pal=seaborn.color_palette("Set2", 6)
metric='sim'
X = np.tile(np.expand_dims(np.arange(len(roilist)), axis=1), (1, default_folds))
plt.figure()
bp = seaborn.boxplot(np.concatenate((X.flatten(), X.flatten(), X.flatten(), X.flatten(), X.flatten(), X.flatten())),
                     np.concatenate((hypISS_results['foldsim_' + metric + '_matchk_all'][:,0,0,:].flatten(), hypISS_results['foldsim_' + metric + '_matchk_all'][:,0,1,:].flatten(),
                                     hypISS_results['foldsim_' + metric + '_matchk_all'][:,1,0,:].flatten(), hypISS_results['foldsim_' + metric + '_matchk_all'][:,1,1,:].flatten(),
                                     hypISS_results['foldsim_' + metric + '_matchk_all'][:,1,2,:].flatten(),hypISS_results['foldsim_' + metric + '_matchk_all'][:,1,3,:].flatten())),
                     hue=np.concatenate((np.ones([default_folds * np.shape(X)[0]]), np.ones([default_folds * np.shape(X)[0]]) * 2, np.ones([default_folds * np.shape(X)[0]]) * 3,
                     np.ones([default_folds * np.shape(X)[0]]) * 4, np.ones([default_folds * np.shape(X)[0]]) * 5, np.ones([default_folds * np.shape(X)[0]]) * 6)), width=0.7, palette=pal)
handles, labels=plt.gca().get_legend_handles_labels()
labels=['NH - NT', 'NH - T=0.25', 'H - NT', 'H - T=0.25', 'H - T=0.35', 'H - T=0.45']
plt.legend(handles, labels, loc='lower left')
plt.xticks(np.arange(0, len(roilist)), roilabels)
plt.savefig(savedir + 'hyperalignment_ISS.pdf')

#show the number of voxels included with hyperalignment options
pal=seaborn.color_palette("Set2", 6)
metric='pcor'
X = np.arange(len(roilist))
plt.figure()
bp = seaborn.barplot(x=np.concatenate((X.flatten(), X.flatten(), X.flatten(), X.flatten(), X.flatten(), X.flatten())),
                     y=np.concatenate((hypISS_results['nvox'][:,0,0].flatten(), hypISS_results['nvox'][:,0,1].flatten(),
                                     hypISS_results['nvox'][:,1,0].flatten(), hypISS_results['nvox'][:,1,1].flatten(),
                                     hypISS_results['nvox'][:,1,2].flatten(),hypISS_results['nvox'][:,1,3].flatten())),
                     hue=np.concatenate((np.ones([np.shape(X)[0]]), np.ones([ np.shape(X)[0]]) * 2, np.ones([np.shape(X)[0]]) * 3,
                     np.ones([ np.shape(X)[0]]) * 4, np.ones([ np.shape(X)[0]]) * 5, np.ones([ np.shape(X)[0]]) * 6)), palette=pal)
# handles, labels=plt.gca().get_legend_handles_labels()
# labels=['NH - NT', 'NH - T=0.25', 'H - NT', 'H - T=0.25', 'H - T=0.35', 'H - T=0.45']
# plt.legend(handles, labels, loc='lower left')
plt.xticks(np.arange(0, len(roilist)), roilabels)
plt.savefig(savedir + 'hyperalignment_ISS_nvox.pdf')

#show how the optimal number of states aligns with the anticipated cortical hierarchy
#plot the number of detected states
pal=seaborn.color_palette("Set2", 7)
X = np.tile(np.expand_dims(np.array([0,1,2,3,4]), axis=0), (default_folds, 1)).T
plt.figure()
bp = seaborn.boxplot(np.concatenate((X.flatten(), X.flatten(), X.flatten(), X.flatten())),
                     np.concatenate((HMMresults['optimum_LL'].flatten(), HMMresultsCV['optimum_LL'].flatten(),
                                     GSresults['optimum_wac_all'].flatten(), HMMresults['optimum_WAC'].flatten())),
                     hue=np.concatenate((np.ones([default_folds * np.shape(X)[0]]),
                    np.ones([default_folds * np.shape(X)[0]]) * 2, np.ones([default_folds * np.shape(X)[0]]) * 3,
             np.ones([default_folds * np.shape(X)[0]]) * 4)), width=0.7, palette=pal[2:])
handles, labels=plt.gca().get_legend_handles_labels()
labels=['LL', 'LL_CV', 'wac', 'wac_HMM']
plt.legend(handles, labels, loc='lower left')
plt.xticks(np.arange(0,5),roilabels)
plt.savefig(savedir + 'nstates_f1.pdf')

#plot the number of detected states
pal=seaborn.color_palette("Set2", 10)
X = np.tile(np.expand_dims(np.array([0,1,2,3,4]), axis=0), (default_folds,1)).T
plt.figure()
bp = seaborn.boxplot(np.concatenate((X.flatten(), X.flatten())),
                     np.concatenate((GSresults['optimum_all'].flatten(), HMMresults['optimum_tdist'].flatten())),
                     hue=np.concatenate((np.ones([default_folds * np.shape(X)[0]]), np.ones([default_folds * np.shape(X)[0]]) * 2)), width=0.5, palette=pal)
handles, labels=plt.gca().get_legend_handles_labels()
plt.xticks(np.arange(0,5),roilabels)
labels=['T-distance', 'Tdist-HMM']
plt.legend(handles, labels, loc='lower left')
plt.savefig(savedir + 'nstates_f2.pdf')


#make the fitline plots
for roi in range(0,5):
    f,ax=plt.subplots(1,1)
    plt.setp(ax, xlim=[0, 100])
    ax.plot(np.arange(2, 100), GSresults['fit_W'][roi,:, 2:100].mean(0), color=pal[6])
    ax.fill_between(np.arange(2, 100), GSresults['fit_W'][roi,:, 2:100].mean(0)-GSresults['fit_W'][roi,:, 2:100].std(0), GSresults['fit_W'][roi,:, 2:100].mean(0)+GSresults['fit_W'][roi,:, 2:100].std(0), alpha=0.3, color=pal[6])
    ax.plot(np.arange(2, 100), GSresults['fit_Bcon'][roi,:, 2:100].mean(0), color=pal[7])
    ax.fill_between(np.arange(2, 100), GSresults['fit_Bcon'][roi,:, 2:100].mean(0) - GSresults['fit_Bcon'][roi,:, 2:100].std(0),
                        GSresults['fit_Bcon'][roi,:, 2:100].mean(0) + GSresults['fit_Bcon'][roi,:, 2:100].std(0), alpha=0.3, color=pal[7])
    ax.plot(np.arange(2, 100), GSresults['fit_Ball'][roi,:, 2:100].mean(0), color=pal[8])
    ax.fill_between(np.arange(2, 100), GSresults['fit_Ball'][roi,:, 2:100].mean(0) - GSresults['fit_Ball'][roi,:, 2:100].std(0),
                        GSresults['fit_Ball'][roi,:, 2:100].mean(0) + GSresults['fit_Ball'][roi,:, 2:100].std(0), alpha=0.3, color=pal[8])
    ax.set_ylabel('Correlation (z)')
    plt.tight_layout()
    plt.savefig(savedir + 'fitlines_components_str' + str(roi) + '.pdf')

    f, ax = plt.subplots(3, 1)
    plt.setp(ax, xlim=[0, 100])
    ax[0].set_title('Tdist')
    ax[0].plot(np.arange(2, 100), GSresults['tdist'][roi,:,2:100].mean(0), color=pal[0])
    ax[0].fill_between(np.arange(2, 100), GSresults['tdist'][roi,:,2:100].mean(0) - GSresults['tdist'][roi,:,2:100].std(0),
                       GSresults['tdist'][roi,:,2:100].mean(0) + GSresults['tdist'][roi,:,2:100].std(0), alpha=0.3, color=pal[0])
    ax[0].plot(np.arange(2, 100), HMMresults['tdists'][roi,:,2:100].mean(0), color=pal[1])
    ax[0].fill_between(np.arange(2, 100), HMMresults['tdists'][roi,:,2:100].mean(0) - HMMresults['tdists'][roi,:,2:100].std(0),
                       HMMresults['tdists'][roi,:,2:100].mean(0) + HMMresults['tdists'][roi,:,2:100].std(0), alpha=0.3, color=pal[1])
    ax[1].set_title('LL - HMM')
    ax[1].plot(np.arange(2, 100), HMMresults['LL'][roi,:,2:100].mean(0), color=pal[2])
    ax[1].fill_between(np.arange(2, 100), HMMresults['LL'][roi,:,2:100].mean(0) - HMMresults['LL'][roi,:,2:100].std(0),
                       HMMresults['LL'][roi,:,2:100].mean(0) + HMMresults['LL'][roi,:,2:100].std(0), alpha=0.3, color=pal[2])
    ax[1].plot(np.arange(2, 100), HMMresultsCV['LL'][roi,:,2:100].mean(0), color=pal[3])
    ax[1].fill_between(np.arange(2, 100), HMMresultsCV['LL'][roi,:,2:100].mean(0) - HMMresultsCV['LL'][roi,:,2:100].std(0),
                       HMMresultsCV['LL'][roi,:,2:100].mean(0) + HMMresultsCV['LL'][roi,:,2:100].std(0), alpha=0.3, color=pal[3])
    ax[2].set_title('WAC')
    ax[2].plot(np.arange(2, 100), GSresults['wac'][roi,:,2:100].mean(0), color=pal[4])
    ax[2].fill_between(np.arange(2, 100), GSresults['wac'][roi,:,2:100].mean(0) - GSresults['wac'][roi,:,2:100].std(0),
                       GSresults['wac'][roi,:,2:100].mean(0) + GSresults['wac'][roi,:,2:100].std(0), alpha=0.3, color=pal[4])
    ax[2].plot(np.arange(2, 100), HMMresults['WAC'][roi,:,2:100].mean(0), color=pal[5])
    ax[2].fill_between(np.arange(2, 100), HMMresults['WAC'][roi,:,2:100].mean(0) - HMMresults['WAC'][roi,:,2:100].std(0),
                       HMMresults['WAC'][roi,:,2:100].mean(0) + HMMresults['WAC'][roi,:,2:100].std(0), alpha=0.3, color=pal[5])
    plt.tight_layout()
    plt.savefig(savedir + 'fitlines_roi' + str(roi) + '.pdf')


#show the distribution of state lengths for the optimal value of K for HMM versus GS.
stateLength=np.zeros((3,len(roilist),default_folds,192))
stateLength_list={}
for roi in range(0,len(roilist)):
    templist=np.zeros((3,default_folds,np.int(np.round(optimumvals[roi]))))
    for m in range(0,3):
        if m == 0:
            boundaries = GSresults['bounds_matchk_folds'][roi,len(kvals),:,:]
        elif m == 1:
            boundaries = HMMresults['bounds_matchk_folds'][roi,len(kvals),:,:]
        elif m == 2:
            boundaries = HMMresults['bounds_matchk_folds_split'][roi, len(kvals), :, :]
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
pal=seaborn.color_palette("Set2", 3)
SD=np.zeros((len(roilist),default_folds,3))
for roi in range(0,len(roilist)):
    SD[roi,:,0] = np.std(stateLength_list[roi][0,:,:], axis=1)
    SD[roi,:,1] = np.std(stateLength_list[roi][1, :, :],  axis=1)
    SD[roi, :, 2] = np.std(stateLength_list[roi][2, :, :], axis=1)

plt.figure()
rois=np.repeat(np.expand_dims(np.arange(0,len(roilist)),1),default_folds, axis=1)
rois=np.repeat(np.expand_dims(rois,2),3, axis=2)
method=np.stack([np.ones((len(roilist),default_folds)),np.ones((len(roilist),default_folds))*2,np.ones((len(roilist),default_folds))*3], axis=2)
d={'SD':SD.flatten(), 'method':method.flatten(), 'rois':rois.flatten()}
df=pd.DataFrame(d)
bp = seaborn.boxplot(data = df, x='rois', y='SD', hue='method', width=0.6, palette=pal)
handles, labels2=plt.gca().get_legend_handles_labels()
plt.legend(handles,['GS', 'HMM', 'HMM-SM'], loc='upper right')
plt.xticks(np.arange(0, len(roilist)), roilabels)
plt.title('Effect of state segmentation method on SD of unit duration')
plt.savefig(savedir + 'SD_unit_length' + format, format=format[1:])

#compute statistical tests for SD of state duration
pval = np.zeros((len(roilist)))
pval2 = np.zeros((len(roilist)))
#t-test reliability
for i in range(0, len(roilist)):
        co, pval[i] = stats.wilcoxon(SD[i,:,0],SD[i,:,1])
        co, pval2[i] = stats.wilcoxon(SD[i,:,0],SD[i,:,2])

#15-fold cross validation, radius=8 how does the reliability vary between HMM and GS
pal=seaborn.color_palette("Set2", 3)
rois=np.repeat(np.expand_dims(np.arange(0,len(roilist)),1),default_folds, axis=1)
rois=np.repeat(np.expand_dims(rois,2),3, axis=2)
metric = 'sim'
for k in range(0, len(kvals)+1):
    plt.figure()
    foldsims=np.stack([GSresults['foldsim_'+ metric +'_matchk_all'][:,k,:],HMMresults['foldsim_'+ metric +'_klist'][:,k,:],HMMresults['foldsim_'+ metric +'_klist_split'][:,k,:]], axis=2)
    method=np.stack([np.ones((len(roilist),default_folds)),np.ones((len(roilist),default_folds))*2,np.ones((len(roilist),default_folds))*3], axis=2)
    d={'data':foldsims.flatten(), 'method':method.flatten(), 'rois':rois.flatten()}
    df=pd.DataFrame(d)
    bp = seaborn.boxplot(data = df, x='rois', y='data', hue='method', width=0.6, palette=pal)
    handles, labels2=plt.gca().get_legend_handles_labels()
    plt.legend(handles,['GS', 'HMM', 'HMM-SM'], loc='upper right')
    plt.xticks(np.arange(0, len(roilist)), roilabels)
    if k < len(kvals):
        plt.title('Effect of state segmentation method on reliability, k = ' + str(kvals[k]))
        plt.savefig(savedir + 'reliability_real_data_' + str(k) + metric + format, format=format[1:])
    else:
        plt.title('Effect of state segmentation method on reliability')
        plt.savefig(savedir + 'reliability_real_data_tdist_' + metric + format, format=format[1:])

#compute statistical tests to compare reliability
pval = np.zeros((len(roilist), len(kvals)+1))
pval2 = np.zeros((len(roilist), len(kvals)+1))
#test reliability
for i in range(0, len(roilist)):
    for k in range(0, len(kvals)+1):
            co, pval[i,k] = stats.wilcoxon(GSresults['foldsim_'+ metric +'_matchk_all'][i,k,:],HMMresults['foldsim_'+ metric +'_klist'][i,k,:])
            co, pval2[i,k] = stats.wilcoxon(GSresults['foldsim_'+ metric +'_matchk_all'][i,k,:],HMMresults['foldsim_'+ metric +'_klist_split'][i,k,:])

#15-fold cross validation, radius=8 how does the the relation with behavioral boundaries vary between HMM and GS
pal=seaborn.color_palette("Set2", 3)
rois=np.repeat(np.expand_dims(np.arange(0,len(roilist)),1),default_folds, axis=1)
rois=np.repeat(np.expand_dims(rois,2),3, axis=2)
metric = 'sim'
for k in range(0, len(kvals)+1):
    plt.figure()
    foldsims=np.stack([res_beh['GS_' + metric][:,k,:],res_beh['HMM_' + metric][:,k,:],res_beh['HMMsm_' + metric][:,k,:]], axis=2)
    method=np.stack([np.ones((len(roilist),default_folds)),np.ones((len(roilist),default_folds))*2,np.ones((len(roilist),default_folds))*3], axis=2)
    d={'data':foldsims.flatten(), 'method':method.flatten(), 'rois':rois.flatten()}
    df=pd.DataFrame(d)
    bp = seaborn.boxplot(data = df, x='rois', y='data', hue='method', width=0.6, palette=pal)
    handles, labels2=plt.gca().get_legend_handles_labels()
    plt.legend(handles,['GS', 'HMM', 'HMM-SM'], loc='upper right')
    plt.xticks(np.arange(0, len(roilist)), roilabels)
    # plt.hlines(0, -0.2, 4.2, linestyles='dotted')
    if k < len(kvals):
        plt.title('Effect of state segmentation method on relation to events, k = ' + str(kvals[k]))
        plt.savefig(savedir + 'association_eventboundaries_' + str(k) + metric + format, format=format[1:])
    else:
        plt.title('Effect of state segmentation method on relation to events')
        plt.savefig(savedir + 'association_eventboundaries_' + metric + format, format=format[1:])

#compute statistical tests to compare relation behavior
pval = np.zeros((len(roilist), len(kvals)+1))
pval2 = np.zeros((len(roilist), len(kvals)+1))
#t-test behavior
for i in range(0, len(roilist)):
    for k in range(0, len(kvals)+1):
            co,pval[i,k] = stats.wilcoxon(res_beh['GS_' + metric][i,k,:],res_beh['HMM_' + metric][i,k,:])
            co, pval2[i,k] = stats.wilcoxon(res_beh['GS_' + metric][i,k,:],res_beh['HMMsm_' + metric][i,k,:])



#15 fold cross validation, how does reliability vary with sphere size
pal=seaborn.color_palette("Set2", 5)
plt.figure()
for i in range(0,len(roilist)):
    plt.plot(radiuslist,GSresults['foldsim_sim_matchk'][i,default_folds_ind,:].T,color=pal[i], linestyle='--', marker='o')
    plt.fill_between(x=radiuslist,y1=GSresults['foldsim_sim_matchk'][i,default_folds_ind,:].T-(GSresults['foldsim_sim_matchk_std'][i,default_folds_ind,:].T/np.sqrt(default_folds)),y2=GSresults['foldsim_sim_matchk'][i,default_folds_ind,:].T+(GSresults['foldsim_sim_matchk_std'][i,default_folds_ind,:].T/np.sqrt(default_folds)), facecolor=pal[i], alpha=0.5)
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

#how does reliability vary with number of subjects averaged, radius=8?
metric = 'sim'
pal=seaborn.color_palette("Set2", len(roilist))
f,ax = plt.subplots(1)
for i in range(0,len(roilist)):
    ax.plot(numsubs[1:],GSresults['foldsim_'+ metric +'_matchk'][i,1:,default_rad_ind].T,color=pal[i], linestyle='--', marker='o')
    ax.fill_between(x=numsubs[1:],y1=GSresults['foldsim_'+ metric +'_matchk'][i,1:,default_rad_ind].T-(GSresults['foldsim_'+ metric +'_matchk_std'][i,1:,default_rad_ind].T/np.sqrt(default_folds)),y2=GSresults['foldsim_'+ metric +'_matchk'][i,1:,default_rad_ind].T+(GSresults['foldsim_'+ metric +'_matchk_std'][i,1:,default_rad_ind].T/np.sqrt(default_folds)), facecolor=pal[i], alpha=0.5)
plt.xlabel('Number of averaged participants')
plt.ylabel('Reliability')
plt.title('Effect of no. averaged participants on reliability, radius=8')
plt.legend(loc='lower right', labels=roilabels)
plt.savefig(savedir + 'Reliability_averaging' + format, format=format[1:])
#how does reliability vary with cross validation, radius=8? (zoom in)
plt.setp(ax, xlim=[-1, 30])
plt.savefig(savedir + 'Reliability_averaging_zoom' + format, format=format[1:])


#15 fold cross validation, how does optimum vary with sphere size
pal=seaborn.color_palette("Set2", len(roilist))
f,ax = plt.subplots(1)
for i in range(0,len(roilist)):
    ax.plot(radiuslist,GSresults['optimum'][i,default_folds_ind,:].T,color=pal[i], linestyle='--', marker='o')
    ax.fill_between(x=radiuslist,y1=GSresults['optimum'][i,default_folds_ind,:].T-(GSresults['optimum_sd'][i,default_folds_ind,:].T/np.sqrt(default_folds)),y2=GSresults['optimum'][i,default_folds_ind,:].T+(GSresults['optimum_sd'][i,default_folds_ind,:].T/np.sqrt(default_folds)), facecolor=pal[i], alpha=0.5)
plt.xlabel('Radius')
plt.ylabel('Optimal number of units')
plt.xticks(radiuslist)
plt.legend(loc='upper left', labels=roilabels)
plt.title('Effect of sphere size on optimal number of units, Kfold' + str(default_folds))
plt.savefig(savedir + 'Optimum_spheresize' + format, format=format[1:])


#how does optimum vary with number of subjects averaged, radius=8?
pal=seaborn.color_palette("Set2", len(roilist))
f,ax = plt.subplots(1)
for i in range(0,len(roilist)):
    ax.plot(numsubs[1:],GSresults['optimum'][i,1:,default_rad_ind].T,color=pal[i], linestyle='--', marker='o')
    ax.fill_between(x=numsubs[1:],y1=GSresults['optimum'][i,1:,default_rad_ind].T-(GSresults['optimum_sd'][i,1:,default_rad_ind].T/np.sqrt(default_folds)),y2=GSresults['optimum'][i,1:,default_rad_ind].T+(GSresults['optimum_sd'][i,1:,default_rad_ind].T/np.sqrt(default_folds)), facecolor=pal[i], alpha=0.5)
plt.xlabel('Number of averaged participants')
plt.ylabel('Optimal number of units')
plt.title('Effect of no. averaged participants on optimal number of units, radius=8')
plt.savefig(savedir + 'Optimum_averaging' + format, format=format[1:])
#zoom in
plt.setp(ax, xlim=[-1, 30])
plt.savefig(savedir + 'Optimum_averaging_zoom' + format, format=format[1:])


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
plt.legend(handles, ['average timeseries', 'average cmat'], loc='upper right')
plt.xticks(np.arange(0,len(roilist)), roilabels)
plt.title('Time * time matrix reliability, effect of averaging timeseries or cmat')
plt.savefig(savedir + 'reliability_averaging_first_last' + format, format=format[1:])

#compute statistical tests for averagin first or last
pval = np.zeros((len(roilist)))
for i in range(0, len(roilist)):
        co, pval[i] = stats.wilcoxon(rel_avglast[i,:],rel_avgfirst[i,:])

