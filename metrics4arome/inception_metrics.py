#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:15:28 2022

@author: brochetc

Works around Inception_v3 model for metrics design and score analysis


Code largely taken from 
Maximilian Seitzer https://github.com/mseitzer/pytorch-fid
Ibtesam Ahmed https://www.kaggle.com/code/ibtesama/gan-in-pytorch-with-fid/notebook
yaroslavvb https://github.com/pytorch/pytorch/issues/25481

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import scipy.linalg as linalg


inceptionPath='/home/mrmn/brochetc/gan4arome_aliasing/gan_horovod/metrics/inception_v3_weights'
    
class InceptionV3(nn.Module):
    """
    Pretrained InceptionV3 network returning feature maps
    taken and (slightly) adapted from Maximilian Seitzer, 2020
    
    """

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True, #specifying True increase performance for extra cost
                 normalize_input=False,
                 wpath='/home/mrmn/brochetc/gan4arome_aliasing/gan_horovod/metrics/inception_v3_weights',
                 requires_grad=False):
        
        super(InceptionV3, self).__init__()
        print('initializing Inception v3')
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        
        inception = models.inception_v3()
        inception.load_state_dict(torch.load(wpath))

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (-1, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp



def symsqrt(a, cond=-1, return_rank=False):
    """Computes the symmetric square root of a positive definite matrix
    
    WARNING : yields unstable results
    
    """

    s, u = torch.linalg.eigh(a)
    cond_dict = {torch.float32: 1e3 * 1.1920929e-07,\
                 torch.float64: 1E6 * 2.220446049250313e-16}

    if cond in [None, -1]:
        cond = cond_dict[a.dtype]

    above_cutoff = (abs(s) > cond * torch.max(abs(s)))

    psigma_diag = torch.sqrt(s[above_cutoff])
    u = u[:, above_cutoff]

    B = u @ torch.diag(psigma_diag) @ u.t()
    if return_rank:
        return B, len(psigma_diag)
    else:
        return B

def activation_statistics(batch,model):
    with torch.no_grad():
        pred=model(batch)[0]
        pred=pred.view(pred.shape[0], pred.shape[1])
    mu=torch.mean(pred, dim=0)
    N=pred.shape[1]
    assert N>0
    sigma=(1/(N-1))*(pred-mu).t()@(pred-mu)
    
    if not torch.isfinite(sigma).all():
        raise RuntimeError('Singular value encountered in correlation')
    
    return mu, sigma
    

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-3, module=np):
    """Numpy/Pytorch implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    module is the name of used module (either torch or numpy as np)
    # PLEASE NOTE : torch computation of matrix square root is unstable and does not handle
    complex numbers -> frequently produces Nans
    # numpy (as np) module should be preferred
    
    Stable version by Dougal J. Sutherland
    
    """
    if module==np:
        mu1=mu1.detach().cpu().numpy()
        mu2=mu2.detach().cpu().numpy()
        sigma1=sigma1.detach().cpu().numpy()
        sigma2=sigma2.detach().cpu().numpy()
        
    mu1 = module.atleast_1d(mu1)
    mu2 = module.atleast_1d(mu2)

    sigma1 = module.atleast_2d(sigma1)
    sigma2 = module.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    if module==np:
        covmean,_=linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = module.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        if np.iscomplexobj(covmean):
            print("complex value detected")
            #print(np.diagonal(covmean).imag.max())
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real
        
    elif module==torch:
        covmean=symsqrt(sigma1@sigma2)
        if not torch.isfinite(covmean).any():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = module.eye(sigma1.shape[0]) * eps
            covmean = symsqrt((sigma1 + offset)@(sigma2 + offset))
    

    tr_covmean = module.trace(covmean)
    res=(module.linalg.norm(diff)**2 + module.trace(sigma1) +
            module.trace(sigma2) - 2 * tr_covmean)
    print(res)
    return res
    
def loadInceptionV3(path):
    """
    Instantiate a pretrained version of inception v3 
    whose outputs are Pool3 feature maps
    
    weights downloaded from 
        https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth
    weights assumed already downloaded and located at "path"
    
    Inputs :
        
        path  -> absolute path string to Inception_v3 weights
    
    """
    
    
    Inception_v3=InceptionV3(wpath=path)
    
    return Inception_v3

class FIDclass():
    
    def __init__(self, path):
        self.count=0
        self.path=path
    def FID(self, real_samples, fake_samples):
        if self.count==0:
            self.model=loadInceptionV3(self.path)
        self.count+=1
        if torch.cuda.is_available():
            real_samples=real_samples.cuda()
            fake_samples=fake_samples.cuda()
            MODEL=self.model.cuda()
            
        mu1, sigma1=activation_statistics(real_samples, MODEL) #conversion to numpy done internally
        mu2, sigma2=activation_statistics(fake_samples, MODEL) #conversion to numpy done internally
        fid0=calculate_frechet_distance(mu1, sigma1, mu2, sigma2, module=np)
        return torch.tensor(fid0, dtype=torch.float32).reshape(1)
    
    


    
