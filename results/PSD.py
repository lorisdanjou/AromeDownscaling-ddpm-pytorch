import numpy as np
from scipy.fftpack import dct
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl



def dct2D(x):
    """
    2D dct transform for 2D (square) numpy array
    or for each sample b of BxNxN numpy array
    """
    assert x.ndim in [2,3]
    if x.ndim==3:
        res=dct(dct(x.transpose((0,2,1)), norm='ortho').transpose((0,2,1)),\
                norm='ortho')
    else :
        res=dct(dct(x.T, norm='ortho').T, norm='ortho')
    return res


def dct_var(x):
    """
    compute the bidirectional variance spectrum of the (square) numpy array x
    """
    N=x.shape[-1]
    
    fx=dct2D(x)
    Sigma=(1/N**2)*fx**2
    return Sigma


def radial_bin_dct(dct_sig,center):
    y, x= np.indices(dct_sig.shape)
    r=np.sqrt((x-center[0])**2+(y-center[1])**2)
    r=r.astype(int)
    
    Rmax=min(x.max(),y.max(),r.max())//2
    
    #### double binning for dct
    dct=dct_sig.ravel()[2*r.ravel()]+0.5*dct_sig.ravel()[2*r.ravel()-1]\
                                    +0.5*dct_sig.ravel()[2*r.ravel()+1]
    
    tbin=np.bincount(r.ravel()[r.ravel()<Rmax], dct[r.ravel()<Rmax])
    nr=np.bincount(r.ravel()[r.ravel()<Rmax])
    
    radial_profile=tbin/nr
    
    return radial_profile


def PowerSpectralDensitySlow(x):
    """
    compute the radially-binned, sample-averaged power spectral density 
    and radially-binned, sample-standardized power spectral density
    of the data x
    
    Inputs :
        x : numpy array, shape is B x N x N
    
    Returns :
        out : numpy array, shape is (Rmax,3), defined in radial_bin_dct function
               [:, 0] : contains average spectrum
               [:, 1] : contains q90 of spectrum
               [:, 2] : contains q10 of spectrum
               
    Slow but should be more robust
    """
    sig = dct_var(x)

    center = (sig.shape[1]//2, sig.shape[2]//2)
    N_samples = sig.shape[0]
    out_list = []
    
    for i in range(N_samples):
        out_list.append(radial_bin_dct(sig[i], center))
        
    out_list = np.array(out_list)
    out = out_list.mean(axis=0)
    out_90 = np.quantile(out_list,0.9, axis=0)
    out_10 = np.quantile(out_list,0.1, axis=0)
    return np.concatenate((np.expand_dims(out, axis=-1),\
                            np.expand_dims(out_90, axis=-1),\
                            np.expand_dims(out_10, axis=-1)),axis=-1)


def PowerSpectralDensity(x):
    """
    compute the radially-averaged, sample-averaged power spectral density 
    of the data x
    Inputs :
        x : numpy array, shape is B x C x N x N
    Return :
        out : numpy array, shape is (C, Rmax), with R_max defined in radial_bin_dct function
    """
    out_list = []
    channels = x.shape[1]    
    
    for c in range(channels) :
        x_c = x[:,c,:,:]
        sig = dct_var(x_c).mean(axis=0)
    
        center = (sig.shape[0]//2, sig.shape[1]//2)
        out_list.append(radial_bin_dct(sig, center))
    
    out=np.concatenate([np.expand_dims(o, axis = 0) for o in out_list], axis = 0)
    return out


def PSD(results_df):
    y_pred = np.zeros((len(results_df), 1, results_df.y_pred[0].shape[0], results_df.y_pred[0].shape[1]))
    y_test = np.zeros((len(results_df), 1, results_df.y_test[0].shape[0], results_df.y_test[0].shape[1]))
    baseline = np.zeros((len(results_df), 1, results_df.baseline[0].shape[0], results_df.baseline[0].shape[1]))
    X_test = np.zeros((len(results_df), 1, results_df.X_test[0].shape[0], results_df.X_test[0].shape[1]))
    for k in range(len(results_df)):
        y_test[k, 0, :, :] = results_df.y_test[k]
        y_pred[k, 0, :, :] = results_df.y_pred[k]
        baseline[k, 0, :, :] = results_df.baseline[k]
        X_test[k, 0, :, :] = results_df.X_test[k]
        
    psd_pred =  PowerSpectralDensity(y_pred)
    psd_test =  PowerSpectralDensity(y_test)
    psd_baseline =  PowerSpectralDensity(baseline)
    psd_X_test =  PowerSpectralDensity(X_test)
    psd_df = pd.DataFrame(
        {'psd_test': psd_test[0, :],
        'psd_pred' : psd_pred[0, :],
        'psd_baseline' : psd_baseline[0, :],
        'psd_X_test' : psd_X_test[0, :]}
    )
    return psd_df


def plot_PSDs(psd_df, output_dir):
    fig, ax = plt.subplots()
    ax.grid()
    ax.plot(psd_df.psd_test, color='k', label='PSD Arome500m')
    ax.plot(psd_df.psd_baseline, color='gray', label='PSD fullpos', linestyle="dashed")
    ax.plot(psd_df.psd_pred, color='r', label='PSD DDPM')

    ax.loglog()
    ax.legend()
    ax.set_title('PSDs')
    ax.set_xlabel("$k$ [$km^{-1}$]")

    fig.savefig(output_dir + 'PSDs.png')


def synthesis_PSDs(expes_names, psds_df, output_dir, several_inputs=False):
    if not several_inputs:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(22, 11))
        colormap = plt.get_cmap('cool')
        axs[1].set_prop_cycle(mpl.cycler(color=[colormap(k) for k in np.linspace(0, 1, len(psds_df))]))
        for i in range(len(expes_names)):
            psd_df = psds_df[i]
            axs[1].plot(psd_df.psd_pred, label=expes_names[i])
        axs[1].plot(psd_df.psd_test, color="k", label="Arome500m")
        axs[1].grid()
        axs[1].loglog()
        axs[1].legend()
        axs[1].set_xlabel("$k$ [$km^{-1}$]")

        axs[0].grid()
        axs[0].plot(psd_df.psd_X_test, color="b", linestyle="dashed", label="Arome2km5")
        axs[0].plot(psd_df.psd_baseline, color="r", linestyle="dashed", label="fullpos")
        axs[0].plot(psd_df.psd_test, color="k", label="Arome500m")
        axs[0].loglog()
        axs[0].legend()
        axs[0].set_xlabel("$k$ [$km^{-1}$]")

        fig.suptitle("PSDs", fontsize=30)
        
        fig.savefig(output_dir + 'PSDs.png', bbox_inches='tight')
    else:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(22, 11))
        colormap = plt.get_cmap('cool')
        axs[0].set_prop_cycle(mpl.cycler(color=[colormap(k) for k in np.linspace(0, 1, len(psds_df))]))
        axs[1].set_prop_cycle(mpl.cycler(color=[colormap(k) for k in np.linspace(0, 1, len(psds_df))]))
        for i in range(len(expes_names)):
            psd_df = psds_df[i]
            axs[1].plot(psd_df.psd_pred, label="DDPM {}".format(expes_names[i]))
            axs[0].plot(psd_df.psd_X_test, label="Arome2km5 {}".format(expes_names[i]))
        axs[1].plot(psd_df.psd_test, color="k", label="Arome500m")
        axs[1].grid()
        axs[1].loglog()
        axs[1].legend()
        axs[1].set_xlabel("$k$ [$km^{-1}$]")

        axs[0].grid()
        axs[0].plot(psd_df.psd_baseline, color="k", linestyle="dashed", label="fullpos")
        axs[0].plot(psd_df.psd_test, color="k", label="Arome500m")
        axs[0].loglog()
        axs[0].legend()
        axs[0].set_xlabel("$k$ [$km^{-1}$]")

        fig.suptitle("PSDs", fontsize=30)
        
        fig.savefig(output_dir + 'PSDs.png', bbox_inches='tight')