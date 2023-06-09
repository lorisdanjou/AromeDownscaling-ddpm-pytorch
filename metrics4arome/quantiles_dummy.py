#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 15:09:26 2022

@author: brochetc

quantiles computation

"""

import numpy as np
from random import shuffle

data_dir ='/home/mrmn/brochetc/scratch_link/GAN_2D/Sud_Est_Baselines_IS_1_1.0_0_0_0_0_0_256_done/'
data_dir_log = data_dir #'/home/mrmn/brochetc/scratch_link/GAN_2D/Set_38/resnet_128_wgan-hinge_64_8_1_0.001_0.001/Instance_6/log/'  #/home/mrmn/brochetc/scratch_link/GAN_2D/Sud_Est_Baselines_IS_1_1.0_0_0_0_0_0_256_done/'
#data_dir_log = data_dir #'/home/mrmn/brochetc/scratch_link/GAN_2D/Set_38/resnet_128_wgan-hinge_64_8_1_0.001_0.001/Instance_6/log/'
step = 58500

data = []

Mat =np.zeros((64*1024,3,128,128))

Means, Maxs = np.load(data_dir+'mean_with_orog.npy')[1:4].reshape(3,1,1), np.load(data_dir+'max_with_orog.npy')[1:4].reshape(3,1,1)
li = list(range(66048))
shuffle(li)
for ind,i in enumerate(li[:64*1024]) :
    if ind%2048 ==0 : print(ind)
    Mat[ind] = np.load(data_dir+'_sample{}.npy'.format(i))[1:4,78:206,55:183].astype(np.float32)
    
Mat = 0.95*(Mat-Means)/Maxs

dataq01 = np.quantile(Mat, 0.01, axis=0)
dataq10 = np.quantile(Mat, 0.1, axis =0)
dataq50 = np.quantile(Mat,0.5, axis = 0)
dataq90 = np.quantile(Mat, 0.9, axis =0)
dataq99 = np.quantile(Mat, 0.99, axis =0)
datavar = np.var(Mat, axis = 0)

np.save(data_dir_log + 'var_{}_norm.npy'.format(step), datavar)
np.save(data_dir_log + 'Q01_{}_norm.npy'.format(step), dataq01)
np.save(data_dir_log + 'Q10_{}_norm.npy'.format(step), dataq10)
np.save(data_dir_log + 'Q50_{}_norm.npy'.format(step), dataq50)
np.save(data_dir_log + 'Q90_{}_norm.npy'.format(step), dataq90)
np.save(data_dir_log + 'Q99_{}_norm.npy'.format(step), dataq99)

"""

data_dir ='/home/mrmn/brochetc/scratch_link/GAN_2D/Set_38/resnet_128_wgan-hinge_64_8_1_0.001_0.001/Instance_6/samples/'  #/home/mrmn/brochetc/scratch_link/GAN_2D/Sud_Est_Baselines_IS_1_1.0_0_0_0_0_0_256_done/'
data_dir_log = data_dir


Mat = np.zeros((64*1024,3,128,128))

for i in range(64) :
    if i%16==0 : print(i)
    Mat[1024*i:1024*(i+1)] = np.load(data_dir+'_Fsample_{}_{}.npy'.format(step,i)).astype(np.float32)

print(Mat.shape)

dataq01 = np.quantile(Mat, 0.01, axis=0)
dataq10 = np.quantile(Mat, 0.1, axis =0)
dataq50 = np.quantile(Mat,0.5, axis = 0)
dataq90 = np.quantile(Mat, 0.9, axis =0)
dataq99 = np.quantile(Mat, 0.99, axis =0)
datavar = np.var(Mat, axis = 0)

np.save('var_{}.npy'.format(step), datavar)
np.save(data_dir_log + 'Q01_{}.npy'.format(step), dataq01)
np.save(data_dir_log + 'Q10_{}.npy'.format(step), dataq10)
np.save(data_dir_log + 'Q50_{}.npy'.format(step), dataq50)
np.save(data_dir_log + 'Q90_{}.npy'.format(step), dataq90)
np.save(data_dir_log + 'Q99_{}.npy'.format(step), dataq99)
"""

