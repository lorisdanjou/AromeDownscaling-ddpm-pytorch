#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 15:43:02 2022

@author: brochetc

Testing a 'pure RMSE' criterion to verify GAN innovation

Using Nearest Neightbour Descent ?

"""

import numpy as np
from glob import glob
import random

N_samples_real = 66048

N_samples_fake = 16384

data_dir_r = '/scratch/mrmn/brochetc/GAN_2D/Sud_Est_Baselines_IS_1_1.0_0_0_0_0_0_256_done/'
data_dir_f = '/scratch/mrmn/brochetc/GAN_2D/Set_38/resnet_128_wgan-hinge_64_8_1_0.0001_0.0001/Instance_1/samples/'

Means = np.load(data_dir_r+'mean_with_orog.npy')[1:4].reshape(3,1,1)
Maxs = np.load(data_dir_r+'max_with_orog.npy')[1:4].reshape(3,1,1)

CI =[78,206,55,183]

real_files = glob(data_dir_r+'_sample*.npy')

def load_batch(file_list, number, option = 'fake'):

    if option=='fake' :
        
        assert number <= 16384  # maximum number of generated samples
        
        print(number)
        
        print('loading shape')
        Shape = np.load(file_list[0]).shape
        
        ## case : isolated files (no batching)
        if len(Shape)==3:
            
            Mat = np.zeros((number, Shape[0], Shape[1], Shape[2]), dtype=np.float32)
            
            list_inds=random.sample(file_list, number)
            
            for i in range(number) :
                
                print('fake file index',i)
                
                Mat[i]=np.load(list_inds[i]).astype(np.float32)
        
        ## case : batching -> select the right number of files to get enough samples
        elif len(Shape)==4:
            
            batch = Shape[0]
                        
            if batch > number : # one file is enough
                
                indices = random.sample(range(batch), number)
                k = random.randint(0,len(file_list)-1)
                
                Mat=np.load(file_list[k])[indices]
            
            else : #select multiple files and fill the number
            
                Mat=np.zeros((number, Shape[1], Shape[2], Shape[3]), \
                                                         dtype=np.float32)
                
                list_inds=random.sample(file_list, number//batch)
                
                for i in range(number//batch) :
                    Mat[i*batch: (i+1)*batch]=\
                    np.load(list_inds[i]).astype(np.float32)
                    
                if number%batch !=0 :
                    remain_inds = random.sample(range(batch),number%batch)
                
                    Mat[i*batch :] = np.load(list_inds[i+1])[remain_inds].astype(np.float32)
        
    elif option=='real' :
        Shape=(3,
               CI[1]-CI[0],
               CI[3]-CI[2])
        
        Mat=np.zeros((number, Shape[0], Shape[1], Shape[2]), dtype=np.float32)
        
        list_inds=random.sample(file_list, number) # randomly drawing samples
        
        for i in range(number):
            
            Mat[i]=np.load(list_inds[i])[1:4,CI[0]:CI[1],
                                           CI[2]:CI[3]].astype(np.float32)
        
    return Mat

def normalize(BigMat, scale, Mean, Max):
    
    """
    
    Normalize samples with specific Mean and max + rescaling
    
    Inputs :
        
        BigMat : ndarray, samples to rescale
        
        scale : float, scale to set maximum amplitude of samples
        
        Mean, Max : ndarrays, must be broadcastable to BigMat
        
    Returns :
        
        res : ndarray, same dimensions as BigMat
    
    """
    
    res= scale*(BigMat-Mean)/(Max)

    return  res


fake_files = glob(data_dir_f+'_FsampleChunk_52500*.npy')

fake_batch = load_batch(fake_files, N_samples_fake, option = 'fake')

real_batch = load_batch(real_files, N_samples_real, option = 'real')

min_dist =1e4
i_min= -1
j_min = -1


    
real_batch = normalize(real_batch,0.95, 
                          Means, Maxs)

for i, file in enumerate(real_files) :

    
    comp = np.sqrt((fake_batch-real_batch[i])**2).mean(axis = (1,2,3))
    
    if i%32==0 : print(i)
    
    if min_dist>comp.min():
        min_dist = comp.min()
        i_min = i
        j_min = np.argmin(comp)
        fake_data_nearest = fake_batch[j_min]
        real_data_nearest = real_batch[i_min]
        print(i_min, j_min, min_dist)
    
    if i%128==0 : 
        np.save('./fake_closest.npy', fake_data_nearest)
        np.save('./real_closest.npy', real_data_nearest)
