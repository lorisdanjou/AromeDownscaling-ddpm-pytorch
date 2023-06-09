#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 17:56:58 2022

@author: brochetc

Structure functions plot

"""

import matplotlib.pyplot as plt
import numpy as np


def plot_structure(data,p_list, scale, var_names, output_dir='./', add_name=''):
    """
    plot data as outputted by structure_functions.structure_function 
    
    Inputs :
        data : numpy array, shape : C x R x P
                with P the number of orders
                
        p_list : iterable, contains the values of the structure orders
        
        scale : float : scalar value to set the unit of one pixel
                (eg for AROME 1 pix= 1.3km)
                
        var_names : list(str) : list of the variable names
        
        output_dir : str, saving directory, default to current
        
        add_name : str, additional naming parameters, default empty
    """
    assert data.shape[0]==len(var_names)
    channels=data.shape[0]
    
    X_range=scale*np.arange(data.shape[1])
    
    fig, axs=plt.subplots(1,channels, figsize=(10,6))
    
    for i in range(channels):
        for j,p in enumerate(p_list) :
            axs[i].plot(np.log2(X_range[1:]),np.log2(data[i,1:,j]), \
               label='p={}'.format(p))
            
        axs[i].set_xlabel('Length (km, log 10 scale)')
        axs[i].title.set_text(var_names[i])
        
        if i==0:
            axs[i].set_ylabel('Structure functions')
    plt.legend(bbox_to_anchor=(1.02,0.5), loc='center left')
    fig.tight_layout()
    plt.savefig(output_dir+'structure_functions'+add_name+'.png')

    return 0

def plot_structure_compare(real_data, fake_data, p_list, scale, var_names, \
                           output_dir='./', add_name=''):
    """
    plot data as outputted by structure_functions.structure_function 
    
    Inputs :
        real_data, fake_data : numpy arrays, shape : C x R x P
                with P the number of orders
                
        p_list : iterable, contains the values of the structure orders
        
        scale : float : scalar value to set the unit of one pixel
                (eg for AROME 1 pix= 1.3km)
                
        var_names : list(str) : list of the variable names
        
        output_dir : str, saving directory, default to current
        
        add_name : str, additional naming parameters, default empty
    """
    assert real_data.shape[0]==len(var_names)
    channels=real_data.shape[0]
    
    X_range=scale*np.arange(real_data.shape[1])
    
    fig, axs=plt.subplots(1,channels, figsize=(10,6))
    
    for i in range(channels):
        for j,p in enumerate(p_list) :
            axs[i].plot(np.log2(X_range[1:]),np.log2(real_data[i,1:,j]), \
                '-',label='PEARO, p={}'.format(p))
            axs[i].plot(np.log2(X_range[1:]), np.log2(fake_data[i,1:,j]),\
               '--',label='GAN, p={}'.format(p))
            
        axs[i].set_xlabel('Length scale (km, log 2 scale)')
        axs[i].title.set_text(var_names[i])
        
        if i==0:
            axs[i].set_ylabel('Structure functions')
    plt.legend(bbox_to_anchor=(1.02,0.5), loc='center left')
    fig.tight_layout()
    plt.savefig(output_dir+'structure_functions_compare'+add_name+'.png')

    return 0

def plot_increments(real_data, fake_data,scale, var_names,\
                    output_dir='./', add_name=''):
    """
    plot data as outputted by structure_functions.increments
    
    Inputs :
        real_data, fake_data : numpy arrays, shape : C x R x h x w
                with P the number of orders and h,w the spatial dimensions
                
        p_list : iterable, contains the values of the structure orders
        
        scale : float : scalar value to set the unit of one pixel
                (eg for AROME 1 pix= 1.3km)
                
        var_names : list(str) : list of the variable names
        
        output_dir : str, saving directory, default to current
        
        add_name : str, additional naming parameters, default empty
    """
    
    assert real_data.shape[0]==len(var_names)
    
    channels=real_data.shape[0]
    H=real_data.shape[-2]
    W=real_data.shape[-1]
    X_range=scale*np.arange(real_data.shape[1])
    
    
    for i in range(channels):
        fig, axs=plt.subplots(H,W, figsize=(20,20))
        if H>1 and W>1:
            for h in range(H):
                for w in range(W):
    
                    axs[H-h-1,w].plot(np.log2(X_range[1:]), np.log2(real_data[i,1:,h,w]),\
                               'r-',label='PEARO',
                               )
                    axs[H-h-1,w].plot(np.log2(X_range[1:]), np.log2(fake_data[i,1:,h,w]),\
                               'b-',label='GAN',
                               )

                    
            handles, labels= axs[0,0].get_legend_handles_labels()
        else :
            axs.plot(np.log2(X_range[1:]), np.log2(real_data[i,1:,0,0]),\
                               'r-',label='PEARO',
                               )
            axs.plot(np.log2(X_range[1:]), np.log2(fake_data[i,1:,0,0]),\
                               'b-',label='GAN',
                               )
            handles, labels=axs.get_legend_handles_labels()
           
        fig.supxlabel('Length scale (km, log 10 scale)',size='x-large')
        fig.supylabel('Increments magnitude of {}'.format(var_names[i]),size='x-large')
        
        fig.legend(handles, labels,bbox_to_anchor=(0.95,0.5), loc='center left')
        fig.subplots_adjust(bottom=0.05,top=0.9, left=0.05, right=0.9)


        fig.tight_layout()
        plt.savefig(output_dir+'increments_'+var_names[i]+'_'+add_name+'.png')
        plt.close()
    return 0
