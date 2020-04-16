
import numpy as np
from joblib import Parallel, delayed
import seaborn
import matplotlib.pyplot as plt
from simulations_methods_comparison import Simulations
import pickle

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

savedir = '/home/lingee/wrkgrp/Cambridge_data/Movie_HMM/simulations/'

nvox = 50
ntime = 200
nstates = 15
nsub = 1
group_std = 0
sub_std = 0.1
TR = 2.47
sub_evprob = 0
reps = 100
length_std = 0.5
maxK = 100
peak_delay = 6
peak_disp = 1
extime =2
maxK=100

length_std_list=[0.1, 0.5, 1]
nstates_list=[5,15,30]
group_std_list=[0.1]
train  = [1, 5, 10]
kfold_list = [1, 2, 20]
sub_evprob_list=[0.1, 0.2, 0.4]
peak_delay_list=[4, 6, 8]
peak_disp_list=[0.5, 1, 2]
CV_list = [True, False]

#run all the simulations
sim=Simulations(nvox=nvox, ntime=ntime, nstates=nstates, nsub=nsub, group_std=group_std, TR=TR, sub_evprob=sub_evprob, length_std=length_std,peak_delay=peak_delay,peak_disp=peak_disp, extime=extime, sub_std=sub_std, maxK=maxK)

output_sim1 = Parallel(n_jobs=50)(delayed(sim.run_simulation_evlength)(length_std_list, rep) for rep in range(0, reps))
with open(savedir + 'output_sim1.mat', 'wb') as output:
    pickle.dump({'output_sim1': output_sim1}, output, pickle.HIGHEST_PROTOCOL)

mindist=1
output_sim2 = Parallel(n_jobs=50)(delayed(sim.run_simulation_compare_fit)(nstates_list,group_std_list, mindist, rep) for rep in range(0,reps))
with open(savedir + 'output_sim2.mat', 'wb') as output:
    pickle.dump({'output_sim2': output_sim2}, output, pickle.HIGHEST_PROTOCOL)

mindist = 5
output_sim2 = Parallel(n_jobs=50)(delayed(sim.run_simulation_compare_fit)(nstates_list, group_std_list, mindist, rep) for rep in range(0, reps))
with open(savedir + 'output_sim2_supplementary.mat', 'wb') as output:
    pickle.dump({'output_sim2_supplementary': output_sim2}, output, pickle.HIGHEST_PROTOCOL)

output_sim3 = Parallel(n_jobs=50)(delayed(sim.run_simulation_sub_noise)(CV_list, train, kfold_list, rep) for rep in range(0, reps))
with open(savedir + 'output_sim3.mat', 'wb') as output:
    pickle.dump({'output_sim3': output_sim3}, output, pickle.HIGHEST_PROTOCOL)

output_sim4 = Parallel(n_jobs=50)(delayed(sim.run_simulation_sub_specific_states)(CV_list, sub_evprob_list, kfold_list, rep) for rep in range(0,reps))
with open(savedir + 'output_sim4.mat', 'wb') as output:
    pickle.dump({'output_sim4': output_sim4}, output, pickle.HIGHEST_PROTOCOL)

output_sim5 = Parallel(n_jobs=50)(delayed(sim.run_simulation_hrf_shape)(nstates_list, peak_delay_list, peak_disp_list, rep) for rep in range(0,reps))
with open(savedir + 'output_sim5.mat', 'wb') as output:
    pickle.dump({'output_sim5': output_sim5}, output, pickle.HIGHEST_PROTOCOL)



#show simulation 1
file = open(savedir + 'output_sim1.mat','rb'); res=pickle.load(file)
output_sim1=res['output_sim1']
GS_sim = np.zeros(np.insert(np.asarray(np.shape(output_sim1[1][0])), 0, reps))
HMM_sim = np.zeros(np.shape(GS_sim))
X = np.zeros(np.shape(GS_sim))
GS_bounds = np.zeros(np.insert(np.asarray(np.shape(output_sim1[1][2])), 0, reps))
HMM_bounds = np.zeros(np.shape(GS_bounds))
real_bounds = np.zeros(np.shape(GS_bounds))
for i in np.arange(0,reps):
    GS_sim[i] = output_sim1[i][0]
    HMM_sim[i] = output_sim1[i][1]
    GS_bounds[i] = output_sim1[i][2]
    HMM_bounds[i] = output_sim1[i][3]
    real_bounds[i] = output_sim1[i][4]
    X[i] = length_std_list

pal=seaborn.color_palette("Set2", 2)
plt.figure()
plt.rcParams['font.size']=18
bp = seaborn.boxplot(np.concatenate((X.flatten(),X.flatten())) , np.concatenate((GS_sim.flatten(), HMM_sim.flatten())),
                       hue=np.concatenate((np.ones([reps*np.shape(X)[1]]), np.ones([reps*np.shape(X)[1]])*2)), width=0.3,
                       palette=pal)
plt.setp(bp, ylim=[0.4, 1.05])
plt.xlabel('SD of state length')
plt.ylabel('Boundary accuracy (r)')
handles, labels=plt.gca().get_legend_handles_labels()
labels=['GS', 'HMM']
plt.yticks(np.arange(0.4,1.1,0.2))
plt.legend(handles, labels, loc='lower left')
plt.savefig(savedir + 'Simulation1.pdf')

cat=1
idx=np.argmin(GS_sim[:,cat])
print(GS_sim[idx,cat])
print(HMM_sim[idx,cat])
plt.figure()
plt.vlines(np.where(GS_bounds[idx,cat,:]), 0, 1, color=pal[0])
plt.vlines(np.where(HMM_bounds[idx,cat,:]), 1, 2, color=pal[1])
plt.vlines(np.where(real_bounds[idx,cat,:]), 0.5, 1.5, color='k')
plt.axis('off')
plt.savefig(savedir + 'Simulation1_example_bounds_poorGS.pdf')


cat=2
idx=np.argmin(HMM_sim[:,cat])
print(GS_sim[idx,cat])
print(HMM_sim[idx,cat])
plt.figure()
plt.vlines(np.where(GS_bounds[idx,cat,:]), 0, 1, color=pal[0])
plt.vlines(np.where(HMM_bounds[idx,cat,:]), 1, 2, color=pal[1])
plt.vlines(np.where(real_bounds[idx,cat,:]), 0.5, 1.5, color='k')
plt.axis('off')
plt.savefig(savedir + 'Simulation1_example_poorHMM.pdf')





#show simulation 2
name ='_supplementary'#'''#'_supplementary' # or ''
file = open(savedir + 'output_sim2' + name + '.mat','rb'); res=pickle.load(file)
output_sim2=res['output_sim2' + name]
optimum = np.zeros(np.insert(np.asarray(np.shape(output_sim2[1][0])), 0, reps))
optimum_wac = np.zeros(np.insert(np.asarray(np.shape(output_sim2[1][1])), 0, reps))
tdist = np.zeros(np.insert(np.asarray(np.shape(output_sim2[1][2])), 0, reps))
wac = np.zeros(np.insert(np.asarray(np.shape(output_sim2[1][3])), 0, reps))
fit_W_mean = np.zeros(np.insert(np.asarray(np.shape(output_sim2[1][2])), 0, reps))
fit_W_std = np.zeros(np.insert(np.asarray(np.shape(output_sim2[1][2])), 0, reps))
fit_Ball_mean = np.zeros(np.insert(np.asarray(np.shape(output_sim2[1][2])), 0, reps))
fit_Ball_std = np.zeros(np.insert(np.asarray(np.shape(output_sim2[1][2])), 0, reps))
fit_Bcon_mean = np.zeros(np.insert(np.asarray(np.shape(output_sim2[1][2])), 0, reps))
fit_Bcon_std = np.zeros(np.insert(np.asarray(np.shape(output_sim2[1][2])), 0, reps))

X1 = np.zeros([reps, np.shape(output_sim2[1][0])[0]])
X2 = np.zeros([reps, np.shape(output_sim2[1][0])[1]])
for i in np.arange(0,reps):
    optimum[i,:,:] = output_sim2[i][0]
    optimum_wac[i,:,:] = output_sim2[i][1]
    tdist[i,:,:] = output_sim2[i][2]
    wac[i, :, :] = output_sim2[i][3]
    fit_W_mean[i, :, :] = output_sim2[i][4]
    fit_W_std[i, :, :] = output_sim2[i][5]
    fit_Ball_mean[i, :, :] = output_sim2[i][6]
    fit_Ball_std[i, :, :] = output_sim2[i][7]
    fit_Bcon_mean[i, :, :] = output_sim2[i][8]
    fit_Bcon_std[i, :, :] = output_sim2[i][9]
    X1[i, :] = nstates_list
    X2[i, :] = group_std_list

pal=seaborn.color_palette("Set2", 2)
plt.rcParams['font.size']=18
for j in np.arange(0, len(group_std_list)):
    plt.figure()
    bp = seaborn.boxplot(np.concatenate((X1.flatten(), X1.flatten())),
                           np.concatenate((optimum[:,:,j].flatten(), optimum_wac[:,:,j].flatten())), hue=np.concatenate(
            (np.ones([reps * np.shape(X1)[1]]), np.ones([reps * np.shape(X1)[1]]) * 2)), width=0.5, palette=pal)
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

#make the tdist line plots
#dim 1 = nstates
#dim 2 = group_std
plt.rcParams['font.size']=14
pal=seaborn.color_palette("Set2", 6)
f,ax=plt.subplots(3,1)
plt.setp(ax, yticks=[0,2,4], xlim=[0, 60])
d1=0
for d2 in np.arange(0,3,1):
    ax[d2].set_title('k = ' + str(nstates_list[d2]))
    ax[d2].plot(np.arange(0, maxK + 1), fit_W_mean[:, d2, d1, :].mean(0), color=pal[2])
    ax[d2].fill_between(np.arange(0, maxK+1), fit_W_mean[:,d2,d1,:].mean(0)-fit_W_std[:,d2,d1,:].mean(0), fit_W_mean[:,d2,d1,:].mean(0)+fit_W_std[:,d2,d1,:].mean(0), alpha=0.3, color=pal[2])
    ax[d2].plot(np.arange(0, maxK + 1), fit_Ball_mean[:, d2, d1, :].mean(0), color=pal[4])
    ax[d2].fill_between(np.arange(0, maxK + 1), fit_Ball_mean[:, d2, d1, :].mean(0) - fit_Ball_std[:, d2, d1, :].mean(0),
                        fit_Ball_mean[:, d2, d1, :].mean(0) + fit_Ball_std[:, d2, d1, :].mean(0), alpha=0.3, color=pal[4])
    ax[d2].plot(wac[:, d2, d1, :].mean(0), color=pal[1])
    ax[d2].set_ylabel('Correlation (z)')
plt.legend(['Within', 'Between - all', 'wac'])
plt.tight_layout()
plt.savefig(savedir + 'Simulation2_fitlines_wac' + name + '.pdf')

f,ax=plt.subplots(3,1)
plt.setp(ax, yticks=[0,2,4], xlim=[0, 60])
d1=0
for d2 in np.arange(0,3,1):
    ax[d2].set_title('k = ' + str(nstates_list[d2]))
    ax[d2].plot(np.arange(0, maxK + 1), fit_W_mean[:, d2, d1, :].mean(0), color=pal[2])
    ax[d2].fill_between(np.arange(0, maxK+1), fit_W_mean[:,d2,d1,:].mean(0)-fit_W_std[:,d2,d1,:].mean(0), fit_W_mean[:,d2,d1,:].mean(0)+fit_W_std[:,d2,d1,:].mean(0), alpha=0.3, color=pal[2])
    ax[d2].plot(np.arange(0, maxK + 1), fit_Bcon_mean[:, d2, d1, :].mean(0), color=pal[3])
    ax[d2].fill_between(np.arange(0, maxK + 1), fit_Bcon_mean[:, d2, d1, :].mean(0) - fit_Bcon_std[:, d2, d1, :].mean(0),
                        fit_Bcon_mean[:, d2, d1, :].mean(0) + fit_Bcon_std[:, d2, d1, :].mean(0), alpha=0.3, color=pal[3])
    ax2 = ax[d2].twinx()
    ax2.plot(tdist[:, d2, d1, :].mean(0), color=pal[0])
    ax[d2].set_ylabel('Correlation (z)')
    ax2.set_ylabel('T-value')
plt.tight_layout()
plt.legend(['Within', 'Between - con', 'T-distance'])
plt.savefig(savedir + 'Simulation2_fitlines_Tdist' + name + '.pdf')



## show simulation 3
file = open(savedir + 'output_sim3.mat','rb'); res=pickle.load(file)
output_sim3=res['output_sim3']

optimum_wac = np.zeros(np.insert(np.asarray(np.shape(output_sim3[1][0])), 0, reps))
optimum = np.zeros(np.insert(np.asarray(np.shape(output_sim3[1][1])), 0, reps))
GS_sim = np.zeros(np.insert(np.asarray(np.shape(output_sim3[1][2])), 0, reps))
GS_sim_fixK = np.zeros(np.insert(np.asarray(np.shape(output_sim3[1][3])), 0, reps))
tdist = np.zeros(np.insert(np.asarray(np.shape(output_sim3[1][4])), 0, reps))
wac = np.zeros(np.insert(np.asarray(np.shape(output_sim3[1][5])), 0, reps))

X1 = np.zeros(np.insert(np.asarray(np.shape(output_sim3[1][0])), 0, reps))
X2 = np.zeros(np.insert(np.asarray(np.shape(output_sim3[1][0])), 0, reps))
X3 = np.zeros(np.insert(np.asarray(np.shape(output_sim3[1][0])), 0, reps))
for i in np.arange(0,reps):
    optimum_wac[i,:,:] = output_sim3[i][0]
    optimum[i,:,:] = output_sim3[i][1]
    GS_sim[i, :, :] = output_sim3[i][2]
    GS_sim_fixK[i, :, :] = output_sim3[i][3]
    tdist[i,:,:] = output_sim3[i][4]
    wac[i, :, :] = output_sim3[i][5]
    for j in range(0, len(CV_list)):
        X1[i, j, :, :] = CV_list[j]
    for j in range(0, len(train)):
        X2[i, :, j, :] = train[j]
    for j in range(0, len(kfold_list)):
        X3[i, :, :, j] = kfold_list[j]



CV_listd = np.double(X1)
f=plt.figure()
X=X2.flatten()
Y=optimum.flatten()
cat=X3.flatten()-(CV_listd.flatten()*0.5)
Y1=np.copy(Y)
#Y1[(X>4)&(cat>19.6)]=25
Y1[(X>6)&(cat>19)]=29
pal=seaborn.color_palette("Set2", 5)
mypal ={1:pal[0], 1.5:pal[1], 2:pal[2],19.5:pal[3], 20:pal[4]}
bp = seaborn.boxplot(X[cat!=0.5],Y1[cat!=0.5] , hue=cat[cat!=0.5], width=0.6, palette=mypal)
plt.setp(bp,yticks=np.arange(0,30,5), ylim=[5, 30])
plt.hlines(nstates, -0.3, 2.3, linestyles='dotted')
handles, labels=plt.gca().get_legend_handles_labels()
labels=['avg all', '2-fold CV', 'avg half', 'LOO CV', 'no avg or CV']
plt.legend(handles, labels, loc='upper left')
plt.savefig(savedir + 'Simulation3_estimateK.pdf')

plt.figure()
X=X2[:,1,:,:].flatten()
Y=GS_sim_fixK[:,1,:,:].flatten()
cat=X3[:,1,:,:].flatten()
bp = seaborn.boxplot(X, Y , hue=cat, width=0.4, palette=mypal)
plt.setp(bp, ylim=[-0.05, 1])
handles, labels=plt.gca().get_legend_handles_labels()
labels=['avg all', 'avg half', 'no avg']
plt.legend(handles, labels, loc='lower left')
plt.savefig(savedir + 'Simulation3_estimatebounds.pdf')



#plot the tdist lines
pal=seaborn.color_palette("Set2", 5)
mypal ={1:pal[0], 1.5:pal[1], 2:pal[2],19.5:pal[3], 20:pal[4]}
cat=X3.flatten()-(CV_listd.flatten()*0.5)
X=X2.flatten()
Y=tdist.reshape(-1,101)
f,ax=plt.subplots(3,1)
plt.setp(ax, xlim=[0, 70])
vals=np.unique(cat)
valsX=np.unique(X)
for d1,xval in enumerate(valsX):
    for d2,type in enumerate(vals):
        if type>0.5:
           inds=np.argwhere((cat==type)&(X==xval))
           ax[d1].set_title('Noise SD = ' + str(xval))
           ax[d1].plot(np.arange(0, maxK + 1),np.squeeze(Y[inds, :].mean(0)), color=mypal[type])
           ax[d1].set_ylabel('T-value')
           bottom, top = ax[d1].get_ylim()
           ax[d1].vlines(nstates, 0, top, linestyles='dotted')
plt.tight_layout()
plt.savefig(savedir + 'Simulation3_fitlines.pdf')





# show simulation 4
file = open(savedir + 'output_sim4.mat','rb'); res=pickle.load(file)
output_sim4=res['output_sim4']

optimum_wac = np.zeros(np.insert(np.asarray(np.shape(output_sim4[1][0])), 0, reps))
optimum = np.zeros(np.insert(np.asarray(np.shape(output_sim4[1][1])), 0, reps))
GS_sim = np.zeros(np.insert(np.asarray(np.shape(output_sim4[1][2])), 0, reps))
GS_sim_fixK = np.zeros(np.insert(np.asarray(np.shape(output_sim4[1][3])), 0, reps))
tdist = np.zeros(np.insert(np.asarray(np.shape(output_sim4[1][4])), 0, reps))
wac = np.zeros(np.insert(np.asarray(np.shape(output_sim4[1][5])), 0, reps))

X1 = np.zeros(np.insert(np.asarray(np.shape(output_sim4[1][0])), 0, reps))
X2 = np.zeros(np.insert(np.asarray(np.shape(output_sim4[1][0])), 0, reps))
X3 = np.zeros(np.insert(np.asarray(np.shape(output_sim4[1][0])), 0, reps))
for i in np.arange(0,reps):
    optimum_wac[i,:,:] = output_sim4[i][0]
    optimum[i,:,:] = output_sim4[i][1]
    GS_sim[i, :, :] = output_sim4[i][2]
    GS_sim_fixK[i, :, :] = output_sim4[i][3]
    tdist[i,:,:] = output_sim4[i][4]
    wac[i, :, :] = output_sim4[i][5]
    for j in range(0, len(CV_list)):
        X1[i, j, :, :] = CV_list[j]
    for j in range(0, len(sub_evprob_list)):
        X2[i, :, j, :] = sub_evprob_list[j]
    for j in range(0, len(kfold_list)):
        X3[i, :, :, j] = kfold_list[j]

V_listd = np.double(X1)
f=plt.figure()
X=X2.flatten()
Y=optimum.flatten()
cat=X3.flatten()-(CV_listd.flatten()*0.5)
Y1=np.copy(Y)
pal=seaborn.color_palette("Set2", 5)
mypal ={1:pal[0], 1.5:pal[1], 2:pal[2],19.5:pal[3], 20:pal[4]}
bp = seaborn.boxplot(X[cat!=0.5],Y1[cat!=0.5] , hue=cat[cat!=0.5], width=0.6, palette=mypal)
plt.setp(bp,yticks=np.arange(0,30,5), ylim=[3, 20])
plt.hlines(nstates, -0.3, 2.3, linestyles='dotted')
handles, labels=plt.gca().get_legend_handles_labels()
labels=['avg all', '2-fold CV', 'avg half', 'LOO CV', 'no avg or CV']
plt.legend(handles, labels, loc='lower left')
plt.savefig(savedir + 'Simulation4_estimateK.pdf')

plt.figure()
X=X2[:,1,:,:].flatten()
Y=GS_sim_fixK[:,1,:,:].flatten()
cat=X3[:,1,:,:].flatten()
bp = seaborn.boxplot(X, Y , hue=cat, width=0.4, palette=mypal)
plt.setp(bp, ylim=[0.35, 1])
handles, labels=plt.gca().get_legend_handles_labels()
labels=['avg all', 'avg half', 'no avg']
plt.legend(handles, labels, loc='lower left')
plt.savefig(savedir + 'Simulation4_estimatebounds.pdf')




# show simulation 5
file = open(savedir + 'output_sim5.mat','rb'); res=pickle.load(file)
output_sim5=res['output_sim5']

GS_sim = np.zeros(np.insert(np.asarray(np.shape(output_sim5[1][0])), 0, reps))
GS_sim_fixK = np.zeros(np.insert(np.asarray(np.shape(output_sim5[1][1])), 0, reps))
HMM_sim = np.zeros(np.insert(np.asarray(np.shape(output_sim5[1][2])), 0, reps))
optimum = np.zeros(np.insert(np.asarray(np.shape(output_sim5[1][3])), 0, reps))

X1 = np.zeros(np.insert(np.asarray(np.shape(output_sim5[1][0])), 0, reps))
X2 = np.zeros(np.insert(np.asarray(np.shape(output_sim5[1][0])), 0, reps))
X3 = np.zeros(np.insert(np.asarray(np.shape(output_sim5[1][0])), 0, reps))

for i in np.arange(0,reps):
    GS_sim[i,:,:] = output_sim5[i][0]
    GS_sim_fixK[i,:,:] = output_sim5[i][1]
    HMM_sim[i, :, :] = output_sim5[i][2]
    optimum[i, :, :] = output_sim5[i][3]

    for j in range(0, len(nstates_list)):
        X1[i, j, :, :] = nstates_list[j]
    for j in range(0, len(peak_delay_list)):
        X2[i, :, j, :] = peak_delay_list[j]
    for j in range(0, len(peak_disp_list)):
        X3[i, :, :, j] = peak_disp_list[j]

pal=seaborn.color_palette("Set2", 5)
mypal ={0.5:pal[0], 1:pal[2], 2:pal[4]}
for j in range(0, len(nstates_list)):
    plt.figure()
    bp = seaborn.boxplot(X2[:,j,:,:].flatten(), optimum[:,j,:,:].flatten(), hue=X3[:,j,:,:].flatten(), width=0.3, palette=mypal)
    bottom, top = bp.get_ylim()
    plt.setp(bp, yticks=np.arange(0, 50, 2), ylim=[bottom, top])
    plt.title('K-estimation  - nstates = ' + str(nstates_list[j]))
    bottom, top = bp.get_xlim()
    plt.hlines(nstates_list[j], bottom, top, linestyles='dotted')
    plt.savefig(savedir + 'Simulation5_effect_HRF_nstates' + str(nstates_list[j]) + '.pdf')


