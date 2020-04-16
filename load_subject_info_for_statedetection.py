import scipy.io as io
import numpy as np

subdata = io.loadmat('/home/lingee/wrkgrp/Cambridge_data/FC_analysis/ME_ICA/subs_MEICA')
ICA_dat = io.loadmat('/home/lingee/wrkgrp/Cambridge_data/FC_analysis/ME_ICA/subs_MEICA_icadata.mat')

#select subjects to include
ICA_dat['rem']=ICA_dat['rejall'].T/ICA_dat['tot'].T
subin=np.nonzero((ICA_dat['rem']<0.8)&(subdata['rms_tot'] <(np.mean(subdata['rms_tot'])+2*np.std(subdata['rms_tot']))))[0]

ICA_dat['var']=ICA_dat['var'][0][subin]
ICA_dat['tot']=ICA_dat['tot'][0][subin]
ICA_dat['rejall']=ICA_dat['rejall'][0][subin]
ICA_dat['rejnb']=ICA_dat['rejnb'][0][subin]
ICA_dat['rem']=ICA_dat['rem'][subin]

age=subdata['age'][subin]
gender=subdata['gender'][subin]
rel_rms=subdata['rel_rms'][subin,:]
rms_tot=subdata['rms_tot'][subin]
rms_max=subdata['rms_max'][subin]
rms_skew=subdata['rms_skew'][subin]
rms_totlarge=subdata['rms_totlarge'][subin]
CCID=subdata['CCID'][subin]
CBUID=subdata['CBUID'][0][subin]