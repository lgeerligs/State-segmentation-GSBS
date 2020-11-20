
import numpy as np
from joblib import Parallel, delayed
import seaborn
import matplotlib.pyplot as plt
from simulations_methods_comparison import Simulations
import pickle

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

savedir = '/home/lingee/wrkgrp/Cambridge_data/Movie_HMM/simulations/'

#default settings
nvox = 50
ntime = 200
nstates = 15
nsub = 1
group_std = 0
sub_std = 0.1
TR = 2.47
sub_evprob = 0
reps = 100
length_std = 1
maxK = 100
peak_delay = 6
peak_disp = 1
extime = 2
maxK=100
nstates_max = 100

#lists to loop over for simulations
length_std_list=[0.1, 1, 2]
nstates_list=[5,15,30]
sub_std_list = [1, 5, 10]
kfold_list = [1, 2, 20]
sub_evprob_list=[0.1, 0.2, 0.4]
peak_delay_list=[4, 6, 8]
peak_disp_list=[0.5, 1, 2]
CV_list = [True, False]

#run all the simulations
sim=Simulations(nvox=nvox, ntime=ntime, nstates=nstates, nsub=nsub, group_std=group_std, TR=TR, sub_evprob=sub_evprob, length_std=length_std,peak_delay=peak_delay,peak_disp=peak_disp, extime=extime, sub_std=sub_std, maxK=maxK)

#simulation 1, as shown in the paper
run_HMM=True
output_sim1 = Parallel(n_jobs=50)(delayed(sim.run_simulation_evlength)(length_std_list, nstates_list,run_HMM, rep) for rep in range(0, reps))
with open(savedir + 'output_sim1.mat', 'wb') as output:
    pickle.dump({'output_sim1': output_sim1}, output, pickle.HIGHEST_PROTOCOL)

# rerun Simulation 1 with a shorter TR, to identify if the finetune_width can be safely set to 1
run_HMM=False
output_sim1 = Parallel(n_jobs=50)(delayed(sim.run_simulation_evlength)(length_std_list, nstates_list,run_HMM, rep, TRfactor=0.5, finetune=0) for rep in range(0, reps))
with open(savedir + 'output_sim1_TR12_FT0.mat', 'wb') as output:
    pickle.dump({'output_sim1': output_sim1}, output, pickle.HIGHEST_PROTOCOL)
output_sim1 = Parallel(n_jobs=50)(delayed(sim.run_simulation_evlength)(length_std_list, nstates_list,run_HMM, rep, TRfactor=0.5, finetune=1) for rep in range(0, reps))
with open(savedir + 'output_sim1_TR12_FT1.mat', 'wb') as output:
    pickle.dump({'output_sim1': output_sim1}, output, pickle.HIGHEST_PROTOCOL)
output_sim1 = Parallel(n_jobs=50)(delayed(sim.run_simulation_evlength)(length_std_list, nstates_list,run_HMM, rep, TRfactor=0.5, finetune=-1) for rep in range(0, reps))
with open(savedir + 'output_sim1_TR12_FT-1.mat', 'wb') as output:
    pickle.dump({'output_sim1': output_sim1}, output, pickle.HIGHEST_PROTOCOL)

#simulation 2, with z-scoring for LL estimation
mindist=1; run_HMM=True; finetune=1; zs=True
output_sim2 = Parallel(n_jobs=50)(delayed(sim.run_simulation_compare_nstates)(nstates_list, mindist, run_HMM,finetune, zs, rep) for rep in range(0,reps))
with open(savedir + 'output_sim2_HMMzs.mat', 'wb') as output:
    pickle.dump({'output_sim2_HMMzs': output_sim2}, output, pickle.HIGHEST_PROTOCOL)

#simulation 2, without z-scoring for LL estimation (generated the results shown in the paper)
mindist=1; run_HMM=True; finetune=1; zs=False
output_sim2 = Parallel(n_jobs=50)(delayed(sim.run_simulation_compare_nstates)(nstates_list, mindist, run_HMM,finetune, zs, rep) for rep in range(0,reps))
with open(savedir + 'output_sim2.mat', 'wb') as output:
    pickle.dump({'output_sim2': output_sim2}, output, pickle.HIGHEST_PROTOCOL)

#simulation 2, leaving out TRs around the diagonal (supplementary figure 2).
mindist = 5; run_HMM=False; finetune=1; zs=False
output_sim2 = Parallel(n_jobs=50)(delayed(sim.run_simulation_compare_nstates)(nstates_list,  mindist, run_HMM, finetune,zs, rep) for rep in range(0, reps))
with open(savedir + 'output_sim2_supplementary.mat', 'wb') as output:
    pickle.dump({'output_sim2_supplementary': output_sim2}, output, pickle.HIGHEST_PROTOCOL)

#simulation 2, test if additional finetuning improves the estimate of the number of states (it does not).
mindist = 1; run_HMM=False; finetune=-1; zs=False
output_sim2 = Parallel(n_jobs=50)(delayed(sim.run_simulation_compare_nstates)(nstates_list,  mindist, run_HMM, finetune, zs, rep) for rep in range(0, reps))
with open(savedir + 'output_sim2_finetuneall.mat', 'wb') as output:
    pickle.dump({'output_sim2_finetuneall': output_sim2}, output, pickle.HIGHEST_PROTOCOL)

#simulation 3
nsub=20
output_sim3 = Parallel(n_jobs=50)(delayed(sim.run_simulation_sub_noise)(CV_list, sub_std_list, kfold_list, nsub, rep) for rep in range(0, reps))
with open(savedir + 'output_sim3.mat', 'wb') as output:
    pickle.dump({'output_sim3': output_sim3}, output, pickle.HIGHEST_PROTOCOL)

#simulation 4 with high noise
noise=5; nsub=20
output_sim4 = Parallel(n_jobs=50)(delayed(sim.run_simulation_sub_specific_states)(CV_list, sub_evprob_list, kfold_list, noise, nsub,rep) for rep in range(0,reps))
with open(savedir + 'output_sim4_high_noise.mat', 'wb') as output:
    pickle.dump({'output_sim4': output_sim4}, output, pickle.HIGHEST_PROTOCOL)

#simulation 4 with low noise, as reported in the paper
noise=0.1; nsub=20
output_sim4 = Parallel(n_jobs=50)(delayed(sim.run_simulation_sub_specific_states)(CV_list, sub_evprob_list, kfold_list, noise, nsub, rep) for rep in range(0,reps))
with open(savedir + 'output_sim4_low_noise.mat', 'wb') as output:
    pickle.dump({'output_sim4': output_sim4}, output, pickle.HIGHEST_PROTOCOL)

output_sim5 = Parallel(n_jobs=50)(delayed(sim.run_simulation_hrf_shape)(nstates_list, peak_delay_list, peak_disp_list, rep) for rep in range(0,reps))
with open(savedir + 'output_sim5.mat', 'wb') as output:
    pickle.dump({'output_sim5': output_sim5}, output, pickle.HIGHEST_PROTOCOL)

output_sim6 = Parallel(n_jobs=50)(delayed(sim.run_simulation_computation_time)(150, rep) for rep in range(0,reps))
with open(savedir + 'output_sim6.mat', 'wb') as output:
    pickle.dump({'output_sim6': output_sim6}, output, pickle.HIGHEST_PROTOCOL)


#show simulation 1
file = open(savedir + 'output_sim1.mat','rb'); res=pickle.load(file)
output_sim1=res['output_sim1']
res1=dict()
for key in output_sim1[0]:
    res1[key] = np.zeros(np.insert(np.asarray(np.shape(output_sim1[0][key])), 0, reps))
    for i in np.arange(0, reps):
        res1[key][i]=output_sim1[i][key]

#boundary accuracy
X = np.tile(np.expand_dims(np.array(length_std_list), axis=0), (reps, 1))
pal=seaborn.color_palette("Set2", 3)
ynames=['Adjusted accuracy', 'Z-scored accuracy']
plotnames=['Adjusted accuracy', 'Z-scored accuracy']
for n, nstates in enumerate(np.array(nstates_list)):
    print(n)
    for m,metric in enumerate(['sim', 'simz']):
        plt.figure()
        plt.rcParams['font.size']=18
        bp = seaborn.boxplot(np.concatenate((X.flatten(),X.flatten(),X.flatten())) , np.concatenate((res1[metric + '_GS'][:,:,n].flatten(),
                                                                                                     res1[metric + '_HMM'][:,:,n].flatten(),
                                                                                                     res1[metric + '_HMMsplit'][:,:,n].flatten())),
                               hue=np.concatenate((np.ones([np.shape(X.flatten())[0]]), np.ones([np.shape(X.flatten())[0]])*2, np.ones([np.shape(X.flatten())[0]])*3)), width=0.5,
                               palette=pal)
        #plt.setp(bp, ylim=[0.4, 1.05])
        plt.xlabel('SD of state length')
        plt.ylabel(ynames[m])
        plt.title(plotnames[m] + ', k =' + str(nstates))
        handles, labels=plt.gca().get_legend_handles_labels()
        labels=['GS', 'HMM', 'HMM_split']
        #plt.yticks(np.arange(0.4,1.1,0.2))
        plt.legend(handles, labels, loc='lower left')
        plt.savefig(savedir + 'Simulation1_split' + metric + str(nstates) +'.pdf')

#boundary distance
pal=seaborn.color_palette("Set2", 3)
for n, nstates in enumerate(np.array(nstates_list)[1:3], start=1):

    for idxl, l in enumerate(np.array(length_std_list)[1:3], start=1):
        print(idxl)
        plt.figure()
        plt.rcParams['font.size']=18

        y = np.abs(np.concatenate((res1['dists_GS'][:, idxl, n, 0:nstates].flatten(),
                                res1['dists_HMM'][:, idxl, n, 0:nstates].flatten(),
                                res1['dists_HMMsplit'][:, idxl, n, 0:nstates].flatten()), axis=0))
        y[y==0]=np.nan
        x = np.concatenate((np.ones([np.shape(res1['dists_GS'][:, idxl, n, 0:nstates].flatten())[0]]), np.ones([np.shape(res1['dists_GS'][:, idxl, n, 0:nstates].flatten())[0]]) * 2,
                                  np.ones([np.shape(res1['dists_GS'][:, idxl, n, 0:nstates].flatten())[0]]) * 3)).astype(int)
        bp = seaborn.countplot(x=y, hue=x,palette=pal, order=(np.arange(np.nanmin(y), np.nanmax(y)+1,1)).astype(int))
        #tick=np.sort(np.concatenate((np.arange(1, np.floor((np.nanmin(y)/5))*5-1,-5), np.arange(6, np.ceil((np.nanmax(y)/5)+1)*5,5)),0))
        #plt.xticks(tick,tick)

        plt.yticks(np.array([0, 0.01, 0.02, 0.03])*nstates*reps, np.array([0, 1, 2, 3]))
        plt.xlabel('Distance')
        plt.ylabel('Percentage')
        handles, labels=plt.gca().get_legend_handles_labels()
        labels=['GS', 'HMM', 'HMM_split']
        #plt.yticks(np.arange(0.4,1.1,0.2))
        plt.legend(handles, labels, loc='upper right')
        plt.savefig(savedir + 'Simulation1_split_dist_nstate' + str(nstates) + 'len' + str(l) + '.pdf')

#show results for simulation 1 with short TR and varying amounts of finetuning
file = open(savedir + 'output_sim1_TR12_FT0.mat','rb'); res=pickle.load(file)
output_sim1=res['output_sim1']
res1=dict()
for key in output_sim1[0]:
    res1[key] = np.zeros(np.insert(np.asarray(np.shape(output_sim1[0][key])), 0, reps))
    for i in np.arange(0, reps):
        res1[key][i]=output_sim1[i][key]
file = open(savedir + 'output_sim1_TR12_FT1.mat','rb'); res=pickle.load(file)
output_sim1=res['output_sim1']
res2=dict()
for key in output_sim1[0]:
    res2[key] = np.zeros(np.insert(np.asarray(np.shape(output_sim1[0][key])), 0, reps))
    for i in np.arange(0, reps):
        res2[key][i]=output_sim1[i][key]
file = open(savedir + 'output_sim1_TR12_FT-1.mat','rb'); res=pickle.load(file)
output_sim1=res['output_sim1']
res3=dict()
for key in output_sim1[0]:
    res3[key] = np.zeros(np.insert(np.asarray(np.shape(output_sim1[0][key])), 0, reps))
    for i in np.arange(0, reps):
        res3[key][i]=output_sim1[i][key]

#boundary accuracy
X = np.tile(np.expand_dims(np.array(length_std_list), axis=0), (reps, 1))
pal=seaborn.color_palette("Set2", 3)
ynames=['Adjusted accuracy', 'Z-scored accuracy']
plotnames=['Adjusted accuracy', 'Z-scored accuracy']
for n, nstates in enumerate(np.array(nstates_list)):
    print(n)
    for m,metric in enumerate(['sim', 'simz']):
        plt.figure()
        plt.rcParams['font.size']=18
        bp = seaborn.boxplot(np.concatenate((X.flatten(),X.flatten(),X.flatten())) , np.concatenate((res1[metric + '_GS'][:,:,n].flatten(),
                                                                                                     res2[metric + '_GS'][:,:,n].flatten(),
                                                                                         res3[metric + '_GS'][:,:,n].flatten())),
                               hue=np.concatenate((np.ones([np.shape(X.flatten())[0]]), np.ones([np.shape(X.flatten())[0]])*2,
                                                   np.ones([np.shape(X.flatten())[0]])*3)), width=0.5,
                               palette=pal)
        #plt.setp(bp, ylim=[0.4, 1.05])
        plt.xlabel('SD of state length')
        plt.ylabel(ynames[m])
        plt.title(plotnames[m] + ', k =' + str(nstates))
        handles, labels=plt.gca().get_legend_handles_labels()
        labels=['Finetune - none', 'Finetune - 1TR', 'Finetune - all TRs']
        #plt.yticks(np.arange(0.4,1.1,0.2))
        plt.legend(handles, labels, loc='lower left')
        plt.savefig(savedir + 'Simulation1_supplemenatary_shortTR_finetune' + metric + str(nstates) +'.pdf')


#show simulation 2
name =''#_supplementary'#'_finetuneall'#'_HMMzs'
file = open(savedir + 'output_sim2' + name + '.mat','rb'); res=pickle.load(file)
output_sim2=res['output_sim2' + name]
res2=dict()
for key in output_sim2[0]:
    res2[key] = np.zeros(np.insert(np.asarray(np.shape(output_sim2[0][key])), 0, reps))
    for i in np.arange(0, reps):
        res2[key][i]=output_sim2[i][key]

#plot the fit for unknown k
metric = 'sim'
pal=seaborn.color_palette("Set2", 8)
for n, nstates in enumerate(np.array(nstates_list)[0:3], start=0):
    #X = np.tile(np.expand_dims(np.array(nstates_list[n]), axis=0), (reps, 1))
    plt.figure()
    bp = seaborn.boxplot(x=np.concatenate((np.ones([reps,1]),np.ones([reps,1]) * 1, np.ones([reps,1]) * 1,
                                         np.ones([reps,1]) * 2, np.ones([reps,1]) * 2, np.ones([reps,1]) * 3,
                                        np.ones([reps,1]) * 3, np.ones([reps,1]) * 3)).flatten(),
                         y=np.concatenate((res2[metric + '_GS_tdist'][:,n].flatten(), res2[metric + '_HMM_tdist'][:,n].flatten(), res2[metric + '_HMMsplit_tdist'][:,n].flatten(),
                                         res2[metric + '_HMM_LL'][:,n].flatten(), res2[metric + '_HMMsplit_LL'][:,n].flatten(),
                                         res2[metric + '_GS_WAC'][:,n].flatten(), res2[metric + '_HMM_WAC'][:,n].flatten(), res2[metric + '_HMMsplit_WAC'][:,n].flatten(),
                                         )), hue=np.concatenate((np.ones([reps,1]),np.ones([reps,1]) * 2, np.ones([reps,1]) * 3,
                                         np.ones([reps,1]) * 2, np.ones([reps,1]) * 3, np.ones([reps,1]) * 1,
                                        np.ones([reps,1]) * 2, np.ones([reps,1]) * 3)).flatten(), width=0.6, palette=pal)
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = ['GSBS', 'HMM', 'HMM-s']
    plt.legend(handles, labels, loc='upper left')

    plt.xticks(np.arange(0,3), ['T-distance', 'LL', 'WAC'])
    plt.savefig(savedir + 'Simulation2_' + metric + str(nstates) + '.pdf')


#plot the number of detected states
pal=seaborn.color_palette("Set2", 9)
for n, nstates in enumerate(np.array(nstates_list)[0:3], start=0):
    #X = np.tile(np.expand_dims(np.array(nstates_list[n]), axis=0), (reps, 1))
    plt.figure()
    bp = seaborn.boxplot(x=np.concatenate((np.ones([reps,1])*1,np.ones([reps,1]) * 1, np.ones([reps,1]) * 2,
                                         np.ones([reps,1]) * 3, np.ones([reps,1]) * 3)).flatten(),
                         y=np.concatenate((res2['optimum_tdist'][:,n].flatten(), res2['optimum_tdist_HMM'][:,n].flatten(), res2['optimum_LL_HMM'][:,n].flatten(),
                                         res2['optimum_wac'][:,n].flatten(), res2['optimum_WAC_HMM'][:,n].flatten())),
                         hue=np.concatenate((np.ones([reps,1]),np.ones([reps,1]) * 2, np.ones([reps,1]) * 2,
                                         np.ones([reps,1]) * 1, np.ones([reps,1]) * 2)).flatten(), width=0.6, palette=pal)
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = ['GSBS', 'HMM']
    plt.hlines(nstates, -0.2, 2.2, linestyles='dotted')
    plt.legend(handles, labels, loc='upper left')

    plt.xticks(np.arange(0,3), ['T-distance', 'LL', 'WAC'])
    plt.savefig(savedir + 'Simulation2_' + 'nstates' + str(nstates) + '.pdf')

#plot the number of detected states for alternative metrics (at reviewer request)
pal=seaborn.color_palette("Set2", 3)
plt.rcParams['font.size']=18
X = np.tile(np.expand_dims(np.array(nstates_list), axis=0), (reps, 1))
plt.figure()
bp = seaborn.boxplot(np.concatenate((X.flatten(), X.flatten(), X.flatten())),
                       np.concatenate((res2['optimum_tdist'][:,:].flatten(), res2['optimum_mdist'][:,:].flatten(), res2['optimum_meddist'][:,:].flatten())), hue=np.concatenate(
        (np.ones([reps * np.shape(X)[1]]), np.ones([reps * np.shape(X)[1]]) * 2, np.ones([reps * np.shape(X)[1]]) * 3)), width=0.5, palette=pal)
plt.xlabel('True number of states')
plt.ylabel('Estimated number of states')
plt.hlines(30, 1.7, 2.3, linestyles='dotted')
plt.hlines(15, 0.7, 1.3, linestyles='dotted')
plt.hlines(5, -0.3, 0.3, linestyles='dotted')
#plt.axis([-0.3, 2.3, 0 ,45])
handles, labels=plt.gca().get_legend_handles_labels()
labels=['T-distance', 'Mean', 'Median']
plt.legend(handles, labels, loc='upper left')
plt.tight_layout()
plt.savefig(savedir + 'Simulation2_metrics_4reviewer.png')

#make the line fit curves
for d2, nstates in enumerate(np.array(nstates_list)[1:3], start=1):

    f,ax=plt.subplots(1,1)
    plt.setp(ax, xlim=[0, 60])
    ax.set_title('k = ' + str(nstates_list[d2]))
    ax.plot(np.arange(2, 60), res2['fit_W_mean'][:, d2, 2:60].mean(0), color=pal[6])
    ax.fill_between(np.arange(2, 60), res2['fit_W_mean'][:,d2,2:60].mean(0)-res2['fit_W_std'][:,d2,2:60].mean(0), res2['fit_W_mean'][:,d2,2:60].mean(0)+res2['fit_W_std'][:,d2,2:60].mean(0), alpha=0.3, color=pal[6])
    ax.plot(np.arange(2, 60), res2['fit_Bcon_mean'][:,d2,2:60].mean(0), color=pal[7])
    ax.fill_between(np.arange(2, 60), res2['fit_Bcon_mean'][:,d2,2:60].mean(0) - res2['fit_Bcon_std'][:,d2,2:60].mean(0),
                        res2['fit_Bcon_mean'][:,d2,2:60].mean(0) + res2['fit_Bcon_std'][:,d2,2:60].mean(0), alpha=0.3, color=pal[7])
    ax.plot(np.arange(2, 60), res2['fit_Ball_mean'][:,d2,2:60].mean(0), color=pal[8])
    ax.fill_between(np.arange(2, 60), res2['fit_Ball_mean'][:,d2,2:60].mean(0) - res2['fit_Ball_std'][:,d2,2:60].mean(0),
                        res2['fit_Ball_mean'][:,d2,2:60].mean(0) + res2['fit_Ball_std'][:,d2,2:60].mean(0), alpha=0.3, color=pal[8])
    ax.set_ylabel('Correlation (z)')
    plt.tight_layout()
    plt.legend(['Within', 'Between - con', 'Between-all'])
    plt.savefig(savedir + 'Simulation2_fitlines_components_GSBS_k' + str(nstates) + '.pdf')

for d2, nstates in enumerate(np.array(nstates_list)[1:3], start=1):

    f,ax=plt.subplots(3,1)
    plt.setp(ax, xlim=[0, 60])
    ax[0].set_title('Tdist')
    ax[0].plot(np.arange(2, 60), res2['tdist'][:, d2, 2:60].mean(0), color=pal[0])
    ax[0].fill_between(np.arange(2, 60), res2['tdist'][:, d2, 2:60].mean(0) - res2['tdist'][:, d2, 2:60].std(0),
                    res2['tdist'][:, d2, 2:60].mean(0) + res2['tdist'][:, d2, 2:60].std(0), alpha=0.3, color=pal[0])
    ax[0].plot(np.arange(2, 60), res2['tdist_HMM'][:, d2, 2:60].mean(0), color=pal[1])
    ax[0].fill_between(np.arange(2, 60), res2['tdist_HMM'][:, d2, 2:60].mean(0) - res2['tdist_HMM'][:, d2, 2:60].std(0),
                    res2['tdist_HMM'][:, d2, 2:60].mean(0) + res2['tdist_HMM'][:, d2, 2:60].std(0), alpha=0.3, color=pal[1])
    ax[1].set_title('LL - HMM')
    ax[1].plot(np.arange(2, 60), res2['LL_HMM'][:, d2, 2:60].mean(0), color=pal[1])
    ax[1].fill_between(np.arange(2, 60), res2['LL_HMM'][:, d2, 2:60].mean(0) - res2['LL_HMM'][:, d2, 2:60].std(0),
                    res2['LL_HMM'][:, d2, 2:60].mean(0) + res2['LL_HMM'][:, d2, 2:60].std(0), alpha=0.3, color=pal[1])
    ax[2].set_title('WAC')
    ax[2].plot(np.arange(2, 60), res2['wac'][:, d2, 2:60].mean(0), color=pal[0])
    ax[2].fill_between(np.arange(2, 60), res2['wac'][:, d2, 2:60].mean(0) - res2['wac'][:, d2, 2:60].std(0),
                    res2['wac'][:, d2, 2:60].mean(0) + res2['wac'][:, d2, 2:60].std(0), alpha=0.3, color=pal[0])
    ax[2].plot(np.arange(2, 60), res2['WAC_HMM'][:, d2, 2:60].mean(0), color=pal[1])
    ax[2].fill_between(np.arange(2, 60), res2['WAC_HMM'][:, d2, 2:60].mean(0) - res2['WAC_HMM'][:, d2, 2:60].std(0),
                    res2['WAC_HMM'][:, d2, 2:60].mean(0) + res2['WAC_HMM'][:, d2, 2:60].std(0), alpha=0.3, color=pal[1])
    plt.tight_layout()
    plt.savefig(savedir + 'Simulation2_fitlines_all5_k' + str(nstates) + '.pdf')

#show simulation 2 - supplementary results, ignoring 4 TRs around diagonal - all values of K in one plot
name ='_supplementary'
file = open(savedir + 'output_sim2' + name + '.mat','rb'); res=pickle.load(file)
output_sim2=res['output_sim2' + name]
res2=dict()
for key in output_sim2[0]:
    res2[key] = np.zeros(np.insert(np.asarray(np.shape(output_sim2[0][key])), 0, reps))
    for i in np.arange(0, reps):
        res2[key][i]=output_sim2[i][key]

pal=seaborn.color_palette("Set2", 2)
plt.rcParams['font.size']=18
X = np.tile(np.expand_dims(np.array(nstates_list), axis=0), (reps, 1))
plt.figure()
bp = seaborn.boxplot(np.concatenate((X.flatten(), X.flatten())),
                       np.concatenate((res2['optimum_tdist'][:,:].flatten(), res2['optimum_wac'][:,:].flatten())), hue=np.concatenate(
        (np.ones([reps * np.shape(X)[1]]), np.ones([reps * np.shape(X)[1]]) * 2)), width=0.5, palette=pal)
plt.xlabel('True number of states')
plt.ylabel('Estimated number of states')
plt.hlines(30, 1.7, 2.3, linestyles='dotted')
plt.hlines(15, 0.7, 1.3, linestyles='dotted')
plt.hlines(5, -0.3, 0.3, linestyles='dotted')
plt.setp(bp,yticks=np.arange(0,90,20))
#plt.axis([-0.3, 2.3, 0 ,45])
handles, labels=plt.gca().get_legend_handles_labels()
labels=['T-distance', 'wac']
plt.legend(handles, labels, loc='upper left')
plt.savefig(savedir + 'Simulation2' + name + '.pdf')

## show simulation 3
file = open(savedir + 'output_sim3.mat','rb'); res=pickle.load(file)
output_sim3=res['output_sim3']
res3=dict()
for key in output_sim3[0]:
    res3[key] = np.zeros(np.insert(np.asarray(np.shape(output_sim3[0][key])), 0, reps))
    for i in np.arange(0, reps):
        res3[key][i]=output_sim3[i][key]

#number of states
X = np.tile(np.expand_dims(np.array(sub_std_list), axis=0), (reps, 1))
pal=seaborn.color_palette("Set2", 5)
plt.figure()
plt.rcParams['font.size']=18
metric='optimum'
#remove values >26 to improve plot visibility
#only the categories with means higher than 26 contain the values 26 or higher
res3[metric][res3[metric][:]>26]=26
bp = seaborn.boxplot(np.concatenate((X.flatten(),X.flatten(),X.flatten(),X.flatten(),X.flatten())) , np.concatenate((res3[metric][:,1,:,0].flatten(),
                                                                                             res3[metric][:,0,:,1].flatten(),
                                                                                             res3[metric][:,1,:,1].flatten(),
                                                                                             res3[metric][:,0,:,2].flatten(),
                                                                                             res3[metric][:,1,:,2].flatten())),
                       hue=np.concatenate((np.ones([np.shape(X.flatten())[0]]), np.ones([np.shape(X.flatten())[0]])*2, np.ones([np.shape(X.flatten())[0]])*3
                                           , np.ones([np.shape(X.flatten())[0]])*4, np.ones([np.shape(X.flatten())[0]])*5)), width=0.6,palette=pal)
#plt.setp(bp, ylim=[0.4, 1.05])
plt.xlabel('Noise SD')
plt.ylabel('Estimated number of states')
plt.hlines(15, -0.3, 2.3, linestyles='dotted')
plt.title('Estimated number of states' + ', k = 15')
handles, labels=plt.gca().get_legend_handles_labels()
labels=['avg all', '2-fold CV', 'avg half', 'LOO CV', 'no avg or CV']
#plt.yticks(np.arange(0.4,1.1,0.2))
plt.legend(handles, labels, loc='upper left')
plt.savefig(savedir + 'Simulation3_estimateK.pdf')

#accuracy
X = np.tile(np.expand_dims(np.array(sub_std_list), axis=0), (reps, 1))
pal=seaborn.color_palette("Set2", 5)
plt.figure()
plt.rcParams['font.size']=18
metric='sim_GS_fixK'
bp = seaborn.boxplot(np.concatenate((X.flatten(),X.flatten(),X.flatten())) , np.concatenate((res3[metric][:,1,:,0].flatten(),
                                                                                             res3[metric][:,1,:,1].flatten(),
                                                                                             res3[metric][:,1,:,2].flatten())),
                       hue=np.concatenate((np.ones([np.shape(X.flatten())[0]]),  np.ones([np.shape(X.flatten())[0]])*3
                                           , np.ones([np.shape(X.flatten())[0]])*5)), width=0.5,palette=[pal[0], pal[2], pal[4]])
#plt.setp(bp, ylim=[0.4, 1.05])
plt.xlabel('Noise SD')
plt.ylabel('Adjusted accuracy')
plt.title('Adjusted accuracy' + ', k = 15')
handles, labels=plt.gca().get_legend_handles_labels()
labels=['avg all',  'avg half', 'no avg or CV']
#plt.yticks(np.arange(0.4,1.1,0.2))
plt.legend(handles, labels, loc='lower left')
plt.savefig(savedir + 'Simulation3_accuracy.pdf')



# show simulation 4
file = open(savedir + 'output_sim4_low_noise.mat','rb'); res=pickle.load(file)
output_sim4=res['output_sim4']
res4=dict()
for key in output_sim4[0]:
    res4[key] = np.zeros(np.insert(np.asarray(np.shape(output_sim4[0][key])), 0, reps))
    for i in np.arange(0, reps):
        res4[key][i]=output_sim4[i][key]

#number of states
X = np.tile(np.expand_dims(np.array(sub_evprob_list), axis=0), (reps, 1))
pal=seaborn.color_palette("Set2", 5)
plt.figure()
plt.rcParams['font.size']=18
metric='optimum'
bp = seaborn.boxplot(np.concatenate((X.flatten(),X.flatten(),X.flatten(),X.flatten())) , np.concatenate((res4[metric][:,1,:,0].flatten(),
                                                                                             res4[metric][:,0,:,1].flatten(),
                                                                                             res4[metric][:,1,:,1].flatten(),
                                                                                             res4[metric][:,0,:,2].flatten())),
                       hue=np.concatenate((np.ones([np.shape(X.flatten())[0]]), np.ones([np.shape(X.flatten())[0]])*2, np.ones([np.shape(X.flatten())[0]])*3
                                           , np.ones([np.shape(X.flatten())[0]])*4)), width=0.5,palette=pal)
plt.xlabel('Proportion of unique states')
plt.ylabel('Estimated number of states')
plt.hlines(15, -0.3, 2.3, linestyles='dotted')
plt.title('Estimated number of states' + ', k = 15')
handles, labels=plt.gca().get_legend_handles_labels()
labels=['avg all', '2-fold CV', 'avg half', 'LOO CV']
#plt.yticks(np.arange(0.4,1.1,0.2))
plt.legend(handles, labels, loc='upper left')
plt.savefig(savedir + 'Simulation4_estimateK.pdf')

#accuracy
X = np.tile(np.expand_dims(np.array(sub_evprob_list), axis=0), (reps, 1))
pal=seaborn.color_palette("Set2", 5)
plt.figure()
plt.rcParams['font.size']=18
metric='sim_GS_fixK'
#indices order = rep, CV(yes,no), sub_std, kfold
bp = seaborn.boxplot(np.concatenate((X.flatten(),X.flatten())), np.concatenate((res4[metric][:,1,:,0].flatten(),
                                                                                res4[metric][:,1,:,1].flatten())),
                       hue=np.concatenate((np.ones([np.shape(X.flatten())[0]]),  np.ones([np.shape(X.flatten())[0]])*3)), width=0.5,palette=[pal[0], pal[2]])
#plt.setp(bp, ylim=[0.4, 1.05])
plt.xlabel('Proportion of unique states')
plt.ylabel('Adjusted accuracy')
plt.title('Adjusted accuracy' + ', k = 15')
handles, labels=plt.gca().get_legend_handles_labels()
labels=['avg all',  'avg half']
#plt.yticks(np.arange(0.4,1.1,0.2))
plt.legend(handles, labels, loc='lower left')
plt.savefig(savedir + 'Simulation4_accuracy.pdf')




# show simulation 5
file = open(savedir + 'output_sim5.mat','rb'); res=pickle.load(file)
output_sim5=res['output_sim5']
res5=dict()
for key in output_sim5[0]:
    res5[key] = np.zeros(np.insert(np.asarray(np.shape(output_sim5[0][key])), 0, reps))
    for i in np.arange(0, reps):
        res5[key][i]=output_sim5[i][key]


X = np.tile(np.expand_dims(np.array(peak_delay_list), axis=0), (reps, 1))

pal=seaborn.color_palette("Set2", 5)
mypal ={0.5:pal[0], 1:pal[2], 2:pal[4]}
for j in range(0, len(nstates_list)):
    plt.figure()
    bp = seaborn.boxplot(x=np.concatenate((X.flatten(),X.flatten(),X.flatten())), y=res5['optimum'][:,j,:,:].flatten(),
                         hue=np.concatenate((np.ones([np.shape(X.flatten())[0]]), np.ones([np.shape(X.flatten())[0]])*2, np.ones([np.shape(X.flatten())[0]])*3)), width=0.6, palette=pal)
    bottom, top = bp.get_ylim()
    plt.setp(bp, yticks=np.arange(0, 50, 2), ylim=[bottom, top])
    plt.title('K-estimation  - nstates = ' + str(nstates_list[j]))
    bottom, top = bp.get_xlim()
    plt.hlines(nstates_list[j], bottom, top, linestyles='dotted')
    plt.savefig(savedir + 'Simulation5_effect_HRF_nstates' + str(nstates_list[j]) + '.pdf')




# show simulation 6
file = open(savedir + 'output_sim6.mat','rb'); res=pickle.load(file)
output_sim6=res['output_sim6']
res6=dict()
for key in output_sim6[0]:
    res6[key] = np.zeros(np.insert(np.asarray(np.shape(output_sim6[0][key])), 0, reps))
    for i in np.arange(0, reps):
        res6[key][i]=output_sim6[i][key]

pal=seaborn.color_palette("Set2", 3)
plt.figure()
for i in np.arange(0,3):
    if i==0:
        X=res6['duration_GSBS']/60
    elif i==1:
        X = res6['duration_HMM_fixK']/60
    elif i==2:
        X = res6['duration_HMMsm_fixK']/60
    plt.plot(np.arange(0,150),np.mean(X,0),color=pal[i], linestyle='--', marker='o')
    plt.fill_between(x=np.arange(0,150),y1=np.mean(X,0)-(np.std(X,0)/np.sqrt(reps)),y2=np.mean(X,0)+(np.std(X,0)/np.sqrt(reps)), facecolor=pal[i], alpha=0.5)

plt.xlabel('Number of states (k)')
plt.ylabel('Time (minutes)')
plt.legend(loc='upper left', labels=['GSBS', 'HMM', 'HMMsm'])
plt.title('Effect of boundary detection method on computational time (fixed k)')
plt.savefig(savedir + 'Simulation6_computation_time_fixedK.pdf')

pal=seaborn.color_palette("Set2", 3)
plt.figure()
for i in np.arange(0,3):
    if i==0:
        X=res6['duration_GSBS']/60/60
    elif i==1:
        X = res6['duration_HMM_estK']/60/60
    elif i == 2:
        X = res6['duration_HMMsm_estK'] / 60 / 60
    plt.plot(np.arange(0,150),np.mean(X,0),color=pal[i], linestyle='--', marker='o')
    plt.fill_between(x=np.arange(0,150),y1=np.mean(X,0)-(np.std(X,0)/np.sqrt(reps)),y2=np.mean(X,0)+(np.std(X,0)/np.sqrt(reps)), facecolor=pal[i], alpha=0.5)

plt.xlabel('Number of states (k)')
plt.ylabel('Time (hours)')
plt.legend(loc='upper left', labels=['GSBS', 'HMM', 'HMMsm'])
plt.title('Effect of boundary detection method on computational time (estimate k)')
plt.savefig(savedir + 'Simulation6_computation_time_estK.pdf')