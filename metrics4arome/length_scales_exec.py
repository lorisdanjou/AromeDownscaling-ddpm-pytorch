#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 13:52:03 2022

@author: brochetc

Executable code for length scale testing

"""

import length_scales as ls
import numpy as np
import random
from glob import glob
import matplotlib.pyplot as plt

DATA_DIR='/home/brochetc/Bureau/These/presentations_these/Premiere_annee/images_des_entrainements/echantillons/'
DATA_DIR_F='/home/brochetc/Bureau/These/presentations_these/Premiere_annee/images_des_entrainements/echantillons/'
#output_dir='/scratch/mrmn/brochetc/GAN_2D/Set_13/resnet_128_wgan-hinge_64_64_1_0.001_0.001/Instance_1/log/'
CI=[78,206,55,183]



def load_batch(path,number,CI,Shape=(3,128,128), option='fake'):
    
    if option=='fake':
        
        list_files=glob(path+'_Fsample_*.npy')

        Mat=np.zeros((number, Shape[0], Shape[1], Shape[2]), dtype=np.float32)
        
        list_inds=random.sample(list_files, number)
        for i in range(number):
            Mat[i]=np.load(list_inds[i])[:,:Shape[1],:Shape[2]].astype(np.float32)
            
    elif option=='real':
        
        list_files=glob(path+'_sample*')
        Shape=np.load(list_files[0])[1:4,CI[0]:CI[1], CI[2]:CI[3]].shape
        Mat=np.zeros((number, Shape[0], Shape[1], Shape[2]), dtype=np.float32)
        
        list_inds=random.sample(list_files, number)
        for i in range(number):
            Mat[i]=np.load(list_inds[i])[1:4,CI[0]:CI[1], CI[2]:CI[3]].astype(np.float32)
            
            
        Means=np.load(path+'mean_with_orog.npy')[1:4].reshape(1,3,1,1).astype(np.float32)
        Maxs=np.load(path+'max_with_orog.npy')[1:4].reshape(1,3,1,1).astype(np.float32)
        Mat=(0.95)*(Mat-Means)/Maxs
        

    return Mat

N_tests=1
sca = 2.5

for n in range(N_tests):
    
    M_real=load_batch(DATA_DIR,158,CI, option='real')
   
    M_fake=load_batch(DATA_DIR_F,64, CI, option='fake')
    
    print(M_real.shape, M_fake.shape)
  
    #### length scales
    eps_real =ls.get_normalized_field(M_real)
    g_real = ls.get_metric_tensor(eps_real)
    ls_real = ls.correlation_length(g_real, sca)
    ani_real =ls.compute_anisotropy(g_real, sca)
    
    eps_fake = ls.get_normalized_field(M_fake)
    g_fake = ls.get_metric_tensor(eps_fake)
    ls_fake = ls.correlation_length(g_fake, sca)
    ani_fake = ls.compute_anisotropy(g_fake, sca)
    
    ######### PLOTS ############
    
    plt.imshow(ls_real[0], origin = 'lower')
    plt.colorbar()
    plt.show()
    
    plt.imshow(ls_fake[0], origin = 'lower')
    plt.colorbar()
    plt.show()
    
    delta0 = np.abs(ls_fake[0]-ls_real[0])/ls_real[0]
    print(delta0.min(), delta0.max(), delta0.mean())

    plt.imshow(np.abs(ls_fake[0]-ls_real[0])/ls_real[0], origin = 'lower', cmap = 'plasma')
    plt.colorbar()
    plt.show()
    
    ############

    plt.imshow(ls_real[1], origin = 'lower')
    plt.colorbar()
    plt.show()
    
    plt.imshow(ls_fake[1], origin = 'lower')
    plt.colorbar()
    plt.show()
    
    
    delta1 = np.abs(ls_fake[1]-ls_real[1])/ls_real[1]
    print(delta1.min(), delta1.max(), delta1.mean())
    
    plt.imshow(np.abs(ls_fake[1]-ls_real[1])/ls_real[1], origin = 'lower', cmap = 'plasma')
    plt.colorbar()
    plt.show()
    
    ##############
    
    plt.imshow(ls_real[2], origin = 'lower')
    plt.colorbar()
    plt.show()
    
    plt.imshow(ls_fake[2], origin = 'lower')
    plt.colorbar()
    plt.show()
    
    delta2 = np.abs(ls_fake[2]-ls_real[2])/ls_real[2]
    print(delta2.min(), delta2.max(), delta2.mean())
    plt.imshow(np.abs(ls_fake[2]-ls_real[2])/ls_real[2], origin = 'lower', cmap = 'plasma')
    plt.colorbar()
    plt.show()
    
    #############
    
    """plt.imshow(ani_fake[0], origin = 'lower')
    plt.colorbar()
    plt.show()
    
    plt.imshow(ani_real[0]*ls_real[0], origin = 'lower')
    plt.colorbar()
    plt.show()
    
    plt.imshow(ani_fake[1], origin = 'lower')
    plt.colorbar()
    plt.show()
    
    plt.imshow(ani_real[1]*ls_real[1], origin = 'lower')
    plt.colorbar()
    plt.show()
    
    plt.imshow(ani_fake[2], origin = 'lower')
    plt.colorbar()
    plt.show()
    
    plt.imshow(ani_real[2]*ls_real[2], origin = 'lower')
    plt.colorbar()
    plt.show()"""