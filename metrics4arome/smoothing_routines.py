#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:21:37 2022

@author: brochetc

smoothing routines for spectra

"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

data_dir='/home/brochetc/Bureau/Thèse/présentations_thèse/images_des_entrainements/'

output_dir=data_dir

params=[11,21,31,51,71]
def param_slide(param, rads):
    res=np.array([param*(t**(2.6)) for t in rads])
    print((res/res.max()).max())
    return res/res.max()  

var=['u', 'v', 't2m']
def ema(series, param):
    res=series.copy()
    T=series.shape[0]
    for t in range(T-1):
        res[t+1]=series[t]*param[t]+(1-param[t])*series[t+1]
    return res

if __name__=="__main__":
    data_r=pickle.load(open(data_dir+'spectral_66048.p', 'rb'))['spectral']
    data_f=pickle.load(open(data_dir+'spectral_fake_66048_1.p', 'rb'))['spectral']
    
    UniRad=data_r[0][0]
    spectr_r=data_r[0][1]
    spectr_f=data_f[0][1]
    
    for p in params:
        print(p)
        
        for i in range(3):
            fig=plt.figure(figsize=(10,10))
            par=param_slide(p, UniRad)
            filtered_r=savgol_filter(spectr_r[i,:],p,1)
            filtered_f=savgol_filter(spectr_f[i,:],p,1)
            plt.plot(UniRad,filtered_r, label='PEARO')
            plt.plot(UniRad, filtered_f,label='GAN')
            plt.legend()
            plt.yscale('log')
            plt.xscale('log')
            plt.xlabel('Wavenumber (normalized)')
            plt.ylabel(var[i])
            fig.tight_layout()
            #st.set_y(0.98)
            fig.subplots_adjust(top=0.95)
            plt.savefig(output_dir+'compar_psd_smooth'+str(p)+var[i]+'_plot.png')
            plt.show()