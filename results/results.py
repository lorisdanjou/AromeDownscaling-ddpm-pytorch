import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy.stats as sc
from skimage.metrics import structural_similarity
from metrics4arome.spectrum_analysis import *
from metrics4arome.length_scales import *

def get_ind_terre_mer_500m():
    filepath = '/cnrm/recyf/Data/users/danjoul/dataset/static_G9KP_SURFIND.TERREMER.npy'
    return np.load(filepath)

'''
Metrics
'''
def mse(a, b):
    return (a - b)**2

def mae(a, b):
    return (np.abs(a - b))

def biais(a, b):
    return a - b

def ssim(a, b):
    ssim_m, ssim_map = structural_similarity(
        a,
        b, 
        data_range=b.max() - b.min(),
        win_size=None,
        full=True
    )
    return ssim_map

'''
Load Data
'''
def load_results(working_dir, dates_test, echeances, resample, data_test_location, baseline_location, param='t2m'):
    y_pred_df = pd.read_pickle(working_dir + 'y_pred.csv')
    results_df = pd.DataFrame(
        {'dates' : [],
        'echeances' : [],
        'X_test' : [],
        'baseline' : [],
        'y_pred' : [],
        'y_test' : []}
    )
    for i_d, d in enumerate(dates_test):
        # Load X_test :
        try:
            if resample == 'c':
                filepath_X_test = data_test_location + 'oper_c_' + d.isoformat() + 'Z_' + param + '.npy'
            elif resample == 'r':
                filepath_X_test = data_test_location + 'oper_r_' + d.isoformat() + 'Z_' + param + '.npy'
            elif resample == 'bl':
                filepath_X_test = data_test_location + 'oper_bl_' + d.isoformat() + 'Z_' + param + '.npy'
            elif resample == 'bc':
                filepath_X_test = data_test_location + 'oper_bc_' + d.isoformat() + 'Z_' + param + '.npy'
            else:
                raise ValueError("resample mal défini")
        # try:
        #     if resample == 'c':
        #         filepath_X_test = data_test_location + 'oper_c_' + d.isoformat() + 'Z_' + param + '.npy'
        #     else:
        #         filepath_X_test = data_test_location + 'oper_r_' + d.isoformat() + 'Z_' + param + '.npy'
            X_test = np.load(filepath_X_test)
            if resample in ['bl', 'bc']:
                X_test = np.pad(X_test, ((5,4), (2,5), (0,0)), mode='edge')
        except FileNotFoundError:
            print('missing day (X): ' + d.isoformat())
            X_test = None

        # Load baseline : 
        # filepath_baseline = baseline_location + 'GG9B_' + d.isoformat() + 'Z_' + param + '.npy'
        # baseline = np.load(filepath_baseline)
        try:
            filepath_baseline = baseline_location + 'GG9B_' + d.isoformat() + 'Z_' + param + '.npy'
            baseline = np.load(filepath_baseline)
        except FileNotFoundError:
            print('missing day (b): ' + d.isoformat())
            baseline = None

        # Load y_test : 
        try:
            filepath_y_test = data_test_location + 'G9L1_' + d.isoformat() + 'Z_' + param + '.npy'
            y_test = np.load(filepath_y_test)
        except FileNotFoundError:
            print('missing day (y): ' + d.isoformat())
            y_test = None

        for i_ech, ech in enumerate(echeances):
            try:
                results_d_ech = pd.DataFrame(
                    {'dates' : [dates_test[i_d].isoformat()],
                    'echeances' : [echeances[i_ech]],
                    'X_test' : [X_test[:, :, i_ech]],
                    'baseline' : [baseline[:, :, i_ech]],
                    'y_pred' : y_pred_df[y_pred_df.dates == d.isoformat()][y_pred_df.echeances == ech][param].to_list(),
                    'y_test' : [y_test[:, :, i_ech]]}
                )
            except TypeError:
                results_d_ech = pd.DataFrame(
                    {'dates' : [],
                    'echeances' : [],
                    'X_test' : [],
                    'baseline' : [],
                    'y_pred' : [],
                    'y_test' : []}
                )
            results_df = pd.concat([results_df, results_d_ech])
    return results_df.reset_index(drop=True)


'''
Get scores global/terre/mer
'''
def datewise_scores(results_df, metric, metric_name):
    metric_df = pd.DataFrame(
        {'dates' : [],
        'echeances' : [],
        metric_name + '_baseline_map' : [],
        metric_name + '_y_pred_map' : [],
        metric_name + '_baseline_mean' : [],
        metric_name + '_y_pred_mean' : []}
    )
    for i in range(len(results_df)):
        metric_i = pd.DataFrame(
            {'dates' : [results_df.dates[i]],
            'echeances' : [results_df.echeances[i]],
            metric_name + '_baseline_map' : [metric(results_df.baseline[i], results_df.y_test[i])],
            metric_name + '_y_pred_map' : [metric(results_df.y_pred[i], results_df.y_test[i])],
            metric_name + '_baseline_mean' : [np.mean(metric(results_df.baseline[i], results_df.y_test[i]))],
            metric_name + '_y_pred_mean' : [np.mean(metric(results_df.y_pred[i], results_df.y_test[i]))]}
        )
        metric_df = pd.concat([metric_df, metric_i])
    return metric_df.reset_index(drop=True)


def datewise_scores_terre(results_df, metric, metric_name):
    metric_df = datewise_scores(results_df, metric, metric_name)
    ind_terre_mer = get_ind_terre_mer_500m()
    metric_terre_df = pd.DataFrame(
        {'dates' : [],
        'echeances' : [],
        metric_name + '_baseline_map' : [],
        metric_name + '_y_pred_map' : [],
        metric_name + '_baseline_mean' : [],
        metric_name + '_y_pred_mean' : []}
    )
    for i in range(len(results_df)):
        metric_i = pd.DataFrame(
            {'dates' : [results_df.dates[i]],
            'echeances' : [results_df.echeances[i]],
            metric_name + '_baseline_map' : [metric_df[metric_name + '_baseline_map'][i]*ind_terre_mer],
            metric_name + '_y_pred_map' : [metric_df[metric_name + '_y_pred_map'][i]*ind_terre_mer],
            metric_name + '_baseline_mean' : [np.sum(metric_df[metric_name + '_baseline_map'][i]*ind_terre_mer)/np.sum(ind_terre_mer)],
            metric_name + '_y_pred_mean' : [np.sum(metric_df[metric_name + '_y_pred_map'][i]*ind_terre_mer)/np.sum(ind_terre_mer)]}
        )
        metric_terre_df = pd.concat([metric_terre_df, metric_i])
    return metric_terre_df.reset_index(drop=True)


def datewise_scores_mer(results_df, metric, metric_name):
    metric_df = datewise_scores(results_df, metric, metric_name)
    ind_terre_mer = get_ind_terre_mer_500m()
    metric_mer_df = pd.DataFrame(
        {'dates' : [],
        'echeances' : [],
        metric_name + '_baseline_map' : [],
        metric_name + '_y_pred_map' : [],
        metric_name + '_baseline_mean' : [],
        metric_name + '_y_pred_mean' : []}
    )
    for i in range(len(results_df)):
        metric_i = pd.DataFrame(
            {'dates' : [results_df.dates[i]],
            'echeances' : [results_df.echeances[i]],
            metric_name + '_baseline_map' : [metric_df[metric_name + '_baseline_map'][i]*(1-ind_terre_mer)],
            metric_name + '_y_pred_map' : [metric_df[metric_name + '_y_pred_map'][i]*(1-ind_terre_mer)],
            metric_name + '_baseline_mean' : [np.sum(metric_df[metric_name + '_baseline_map'][i]*(1-ind_terre_mer))/np.sum((1-ind_terre_mer))],
            metric_name + '_y_pred_mean' : [np.sum(metric_df[metric_name + '_y_pred_map'][i]*(1-ind_terre_mer))/np.sum((1-ind_terre_mer))]}
        )
        metric_mer_df = pd.concat([metric_mer_df, metric_i])
    return metric_mer_df.reset_index(drop=True)


def datewise_wasserstein_distance(results_df):
    wasserstein_df = pd.DataFrame(
        {'dates' : [],
        'echeances' : [],
        'datewise_wasserstein_distance_baseline' : [],
        'datewise_wasserstein_distance_pred' : []}
    )
    for i in range(len(results_df)):
        dist_pred = np.reshape(results_df.y_pred[i], -1)
        dist_baseline = np.reshape(results_df.baseline[i], -1)
        dist_test = np.reshape(results_df.y_test[i], -1)
        
        wasserstein_i = pd.DataFrame(
            {'dates' : [results_df.dates[i]],
            'echeances' : [results_df.echeances[i]],
            'datewise_wasserstein_distance_baseline' : [sc.wasserstein_distance(dist_baseline, dist_test)],
            'datewise_wasserstein_distance_pred' : [sc.wasserstein_distance(dist_pred, dist_test)]}
        )
        wasserstein_df = pd.concat([wasserstein_df, wasserstein_i])
    return wasserstein_df.reset_index(drop=True)


def datewise_wasserstein_distance_terre(results_df):
    ind_terre_mer = np.reshape(get_ind_terre_mer_500m(), -1)
    wasserstein_df = pd.DataFrame(
        {'dates' : [],
        'echeances' : [],
        'datewise_wasserstein_distance_baseline' : [],
        'datewise_wasserstein_distance_pred' : []}
    )
    for i in range(len(results_df)):
        dist_pred_global = np.reshape(results_df.y_pred[i], -1)
        dist_baseline_global = np.reshape(results_df.baseline[i], -1)
        dist_test_global = np.reshape(results_df.y_test[i], -1)

        dist_pred = []
        dist_baseline = []
        dist_test = []

        for k in range(len(ind_terre_mer)):
            if ind_terre_mer[k] == 1:
                dist_pred.append(dist_pred_global[k])
                dist_test.append(dist_test_global[k])
                dist_baseline.append(dist_baseline_global[k])      
        wasserstein_i = pd.DataFrame(
            {'dates' : [results_df.dates[i]],
            'echeances' : [results_df.echeances[i]],
            'datewise_wasserstein_distance_baseline' : [sc.wasserstein_distance(dist_baseline, dist_test)],
            'datewise_wasserstein_distance_pred' : [sc.wasserstein_distance(dist_pred, dist_test)]}
        )
        wasserstein_df = pd.concat([wasserstein_df, wasserstein_i])
    return wasserstein_df.reset_index(drop=True)


def datewise_wasserstein_distance_mer(results_df):
    ind_terre_mer = np.reshape(get_ind_terre_mer_500m(), -1)
    wasserstein_df = pd.DataFrame(
        {'dates' : [],
        'echeances' : [],
        'datewise_wasserstein_distance_baseline' : [],
        'datewise_wasserstein_distance_pred' : []}
    )
    for i in range(len(results_df)):
        dist_pred_global = np.reshape(results_df.y_pred[i], -1)
        dist_baseline_global = np.reshape(results_df.baseline[i], -1)
        dist_test_global = np.reshape(results_df.y_test[i], -1)

        dist_pred = []
        dist_baseline = []
        dist_test = []

        for k in range(len(ind_terre_mer)):
            if ind_terre_mer[k] == 0:
                dist_pred.append(dist_pred_global[k])
                dist_test.append(dist_test_global[k])
                dist_baseline.append(dist_baseline_global[k])      
        wasserstein_i = pd.DataFrame(
            {'dates' : [results_df.dates[i]],
            'echeances' : [results_df.echeances[i]],
            'datewise_wasserstein_distance_baseline' : [sc.wasserstein_distance(dist_baseline, dist_test)],
            'datewise_wasserstein_distance_pred' : [sc.wasserstein_distance(dist_pred, dist_test)]}
        )
        wasserstein_df = pd.concat([wasserstein_df, wasserstein_i])
    return wasserstein_df.reset_index(drop=True)


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


def corr_len(results_df):
    y_pred = np.zeros((len(results_df), 1, results_df.y_pred[0].shape[0], results_df.y_pred[0].shape[1]))
    y_test = np.zeros((len(results_df), 1, results_df.y_test[0].shape[0], results_df.y_test[0].shape[1]))
    baseline = np.zeros((len(results_df), 1, results_df.baseline[0].shape[0], results_df.baseline[0].shape[1]))

    for k in range(len(results_df)):
        y_test[k, 0, :, :] = results_df.y_test[k]
        y_pred[k, 0, :, :] = results_df.y_pred[k]
        baseline[k, 0, :, :] = results_df.baseline[k]

    corr_len_pred = length_scale(y_pred, sca=2.5)
    corr_len_test = length_scale(y_test, sca=2.5)
    corr_len_baseline = length_scale(baseline, sca=2.5)

    corr_len_df = pd.DataFrame(
        {'corr_len_test': [corr_len_test[0, :, :]],
        'corr_len_pred' : [corr_len_pred[0, :, :]],
        'corr_len_baseline' : [corr_len_baseline[0, :, :]]}
    )
    return corr_len_df


def correlation(results_df):
    corr_df = pd.DataFrame(
        [],
        columns=['dates', 'echeances', 'r_baseline', 'r_pred']
    )
    for i in range(len(results_df)):
        r_baseline, _ = sc.pearsonr(results_df.y_test.iloc[i].reshape(-1), results_df.baseline.iloc[i].reshape(-1))
        r_pred    , _ = sc.pearsonr(results_df.y_test.iloc[i].reshape(-1), results_df.y_pred.iloc[i].reshape(-1))
        corr_df.loc[len(corr_df)] = [
            results_df.dates.iloc[i],
            results_df.echeances.iloc[i],
            r_baseline,
            r_pred
        ]
    return corr_df


def correlation_terre(results_df): # using masks
    ind_terre_mer = get_ind_terre_mer_500m()
    corr_df = pd.DataFrame(
        [],
        columns=['dates', 'echeances', 'r_baseline', 'r_pred']
    )
    for i in range(len(results_df)):
        y_test = np.ma.masked_array(results_df.y_test.iloc[i], (1-ind_terre_mer))
        y_pred = np.ma.masked_array(results_df.y_pred.iloc[i], (1-ind_terre_mer))
        baseline = np.ma.masked_array(results_df.baseline.iloc[i], (1-ind_terre_mer))

        r_baseline, _ = sc.pearsonr(
            baseline.reshape(-1),
            y_test.reshape(-1)
        )
        r_pred    , _ = sc.pearsonr(
            y_pred.reshape(-1),
            y_test.reshape(-1)
        )
        corr_df.loc[len(corr_df)] = [
            results_df.dates.iloc[i],
            results_df.echeances.iloc[i],
            r_baseline,
            r_pred
        ]
    return corr_df


def correlation_mer(results_df): # using masks
    ind_terre_mer = get_ind_terre_mer_500m()
    corr_df = pd.DataFrame(
        [],
        columns=['dates', 'echeances', 'r_baseline', 'r_pred']
    )
    for i in range(len(results_df)):
        y_test = np.ma.masked_array(results_df.y_test.iloc[i], ind_terre_mer)
        y_pred = np.ma.masked_array(results_df.y_pred.iloc[i], ind_terre_mer)
        baseline = np.ma.masked_array(results_df.baseline.iloc[i], ind_terre_mer)

        r_baseline, _ = sc.pearsonr(
            baseline.reshape(-1),
            y_test.reshape(-1)
        )
        r_pred    , _ = sc.pearsonr(
            y_pred.reshape(-1),
            y_test.reshape(-1)
        )
        corr_df.loc[len(corr_df)] = [
            results_df.dates.iloc[i],
            results_df.echeances.iloc[i],
            r_baseline,
            r_pred
        ]
    return corr_df




'''
Plots
'''
def plot_results(results_df, param,  output_dir):
    for i in range(10):
        fig, axs = plt.subplots(nrows=1,ncols=4, figsize = (28, 7))
        data = [results_df.X_test[i], results_df.baseline[i], results_df.y_pred[i], results_df.y_test[i]]
        images = []
        for j in range(4):
            images.append(axs[j].imshow(data[j]))
            axs[j].label_outer()
        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        for im in images:
            im.set_norm(norm)
        axs[0].set_title('X_test')
        axs[1].set_title('baseline')
        axs[2].set_title('y_pred')
        axs[3].set_title('y_test')
        fig.colorbar(images[0], ax=axs)
        plt.savefig(output_dir + 'results_' + str(i) + '_' + param + '.png', bbox_inches='tight')


def plot_score_maps(results_df, metric, metric_name, output_dir, cmap='coolwarm'):
    for i in range(10):
        metric_df = datewise_scores(results_df, metric, metric_name)
        fig, axs = plt.subplots(nrows=1,ncols=2, figsize = (25, 12))
        images = []
        data = [metric_df[metric_name + '_baseline_map'][i], metric_df[metric_name + '_y_pred_map'][i]]
        for j in range(len(data)):
            im = axs[j].imshow(data[j], cmap=cmap)
            images.append(im)
            axs[j].label_outer()
        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        for im in images:
            im.set_norm(norm)
        axs[0].set_title('baseline global')
        axs[1].set_title('pred global')
        fig.colorbar(images[0], ax=axs)
        plt.savefig(output_dir + metric_name + str(i) + '_map.png')


def plot_unique_score_map(results_df, metric, metric_name, output_dir, cmap='coolwarm'):
    metric_df = datewise_scores(results_df, metric, metric_name)
    metric_baseline = metric_df[metric_name + '_baseline_map'].mean()
    metric_y_pred   = metric_df[metric_name + '_baseline_map'].mean()
    fig, axs = plt.subplots(nrows=1,ncols=2, figsize = (25, 12))
    images = []
    data = [metric_baseline, metric_y_pred]
    for j in range(len(data)):
        im = axs[j].imshow(data[j], cmap=cmap)
        images.append(im)
        axs[j].label_outer()
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)
    mean = metric_baseline.mean()
    axs[0].set_title('baseline ' + metric_name  + ' ' + f'{mean:.2f}')
    mean = metric_y_pred.mean()
    axs[1].set_title('pred ' + metric_name  + ' ' + f'{mean:.2f}')
    fig.colorbar(images[0], ax=axs)
    plt.savefig(output_dir + metric_name + '_unique_map.png')


def plot_distrib(results_df, metric, metric_name, output_dir):
    score_baseline = datewise_scores(results_df, metric, metric_name)[metric_name + '_baseline_mean']
    score_baseline_terre = datewise_scores_terre(results_df, metric, metric_name)[metric_name + '_baseline_mean']
    score_baseline_mer = datewise_scores_mer(results_df, metric, metric_name)[metric_name + '_baseline_mean']
    score_pred = datewise_scores(results_df, metric, metric_name)[metric_name + '_y_pred_mean']
    score_pred_terre = datewise_scores_terre(results_df, metric, metric_name)[metric_name + '_y_pred_mean']
    score_pred_mer = datewise_scores_mer(results_df, metric, metric_name)[metric_name + '_y_pred_mean']
    
    D_baseline = np.zeros((len(score_baseline), 3))
    for i in range(len(score_baseline)):
        D_baseline[i, 0] = score_baseline[i]
        D_baseline[i, 1] = score_baseline_terre[i]
        D_baseline[i, 2] = score_baseline_mer[i]

    D_pred = np.zeros((len(score_pred), 3))
    for i in range(len(score_pred)):
        D_pred[i, 0] = score_pred[i]
        D_pred[i, 1] = score_pred_terre[i]
        D_pred[i, 2] = score_pred_mer[i]

    D = np.concatenate([D_baseline, D_pred], axis=1)
    # labels = ['global', 'terre', 'mer']
    labels = ['global_baseline', 'terre_baseline', 'mer_baseline', 'global_pred', 'terre_pred', 'mer_pred']

    
    fig, ax = plt.subplots(figsize=(10, 11))
    plt.grid()
    VP = ax.boxplot(D, positions=[3, 6, 9, 12, 15, 18], widths=1.5, patch_artist=True,
                    showmeans=True, meanline=True, showfliers=False,
                    medianprops={"color": "white", "linewidth": 0.5},
                    boxprops={"facecolor": "C0", "edgecolor": "white",
                            "linewidth": 0.5},
                    whiskerprops={"color": "C0", "linewidth": 1.5},
                    capprops={"color": "C0", "linewidth": 1.5},
                    meanprops = dict(linestyle='--', linewidth=2.5, color='purple'),
                    labels=labels)
    ax.set_title(metric_name + ' distribution')
    ax.tick_params(axis='x', rotation=45)

    plt.savefig(output_dir + 'distribution_' +  metric_name + '.png')


def plot_datewise_wasserstein_distance_distrib(results_df, output_dir):    
    wd_df = datewise_wasserstein_distance(results_df)
    wd_df_baseline = wd_df['datewise_wasserstein_distance_baseline']
    wd_df_pred = wd_df['datewise_wasserstein_distance_pred']
    wd_df_terre = datewise_wasserstein_distance_terre(results_df)
    wd_df_baseline_terre = wd_df_terre['datewise_wasserstein_distance_baseline']
    wd_df_pred_terre = wd_df_terre['datewise_wasserstein_distance_pred']
    wd_df_mer = datewise_wasserstein_distance_mer(results_df)
    wd_df_baseline_mer = wd_df_mer['datewise_wasserstein_distance_baseline']
    wd_df_pred_mer = wd_df_mer['datewise_wasserstein_distance_pred']

    D_baseline = np.zeros((len(wd_df), 3))
    for i in range(len(wd_df_baseline)):
        D_baseline[i, 0] = wd_df_baseline[i]
        D_baseline[i, 1] = wd_df_baseline_terre[i]
        D_baseline[i, 2] = wd_df_baseline_mer[i]

    D_pred = np.zeros((len(wd_df), 3))
    for i in range(len(wd_df_pred)):
        D_pred[i, 0] = wd_df_pred[i]
        D_pred[i, 1] = wd_df_pred_terre[i]
        D_pred[i, 2] = wd_df_pred_mer[i]
        
    D = np.concatenate([D_baseline, D_pred], axis=1)
    labels = ['global_baseline', 'terre_baseline', 'mer_baseline', 'global_pred', 'terre_pred', 'mer_pred']

    fig, ax = plt.subplots(figsize=(10, 11))
    plt.grid()
    VP = ax.boxplot(D, positions=[3, 6, 9, 12, 15, 18], widths=1.5, patch_artist=True,
                    showmeans=True, meanline=True, showfliers=False,
                    medianprops={"color": "white", "linewidth": 0.5},
                    boxprops={"facecolor": "C0", "edgecolor": "white",
                            "linewidth": 0.5},
                    whiskerprops={"color": "C0", "linewidth": 1.5},
                    capprops={"color": "C0", "linewidth": 1.5},
                    meanprops = dict(linestyle='--', linewidth=2.5, color='purple'),
                    labels=labels)
    ax.set_title('datewise wasserstein distance distribution')
    ax.tick_params(axis='x', rotation=45)

    plt.savefig(output_dir + 'distribution_wd.png')


def plot_PSDs(results_df, output_dir):
    psd_df = PSD(results_df)

    fig, ax = plt.subplots()
    ax.grid()
    ax.plot(psd_df.psd_test, color='r', label='psd_test')
    ax.plot(psd_df.psd_pred, color='b', label='psd_pred')
    ax.plot(psd_df.psd_baseline, color='g', label='psd_baseline')
    ax.loglog()
    ax.legend()
    ax.set_title('PSDs')

    fig.savefig(output_dir + 'PSDs.png')


def plot_cor_len(results_df, output_dir):
    corr_len_df = corr_len(results_df)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize = (28, 7))
    data = [corr_len_df.corr_len_baseline[0], corr_len_df.corr_len_pred[0], corr_len_df.corr_len_test[0]]
    images = []
    for i in range(3):
        images.append(axs[i].imshow(data[i]))
        axs[i].label_outer()
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)
    axs[0].set_title('baseline')
    axs[1].set_title('y_pred')
    axs[2].set_title('y_test')
    fig.colorbar(images[0], ax=axs)
    plt.savefig(output_dir + 'correlation_length_maps.png', bbox_inches='tight')


def plot_corr_distrib(results_df, output_dir):
    corr_df                = correlation(results_df)
    corr_df_pred           = corr_df['r_pred']
    corr_df_baseline       = corr_df['r_baseline']
    corr_df_terre          = correlation_terre(results_df)
    corr_df_pred_terre     = corr_df_terre['r_pred']
    corr_df_baseline_terre = corr_df_terre['r_baseline']
    corr_df_mer            = correlation_mer(results_df)
    corr_df_pred_mer       = corr_df_mer['r_pred']
    corr_df_baseline_mer   = corr_df_mer['r_baseline']

    D_baseline = np.zeros((len(corr_df), 3))
    for i in range(len(corr_df_baseline)):
        D_baseline[i, 0] = corr_df_baseline[i]
        D_baseline[i, 1] = corr_df_baseline_terre[i]
        D_baseline[i, 2] = corr_df_baseline_mer[i]

    D_pred = np.zeros((len(corr_df), 3))
    for i in range(len(corr_df_pred)):
        D_pred[i, 0] = corr_df_pred[i]
        D_pred[i, 1] = corr_df_pred_terre[i]
        D_pred[i, 2] = corr_df_pred_mer[i]
        
    D = np.concatenate([D_baseline, D_pred], axis=1)
    labels = ['global_baseline', 'terre_baseline', 'mer_baseline', 'global_pred', 'terre_pred', 'mer_pred']

    fig, ax = plt.subplots(figsize=(10, 11))
    plt.grid()
    VP = ax.boxplot(D, positions=[3, 6, 9, 12, 15, 18], widths=1.5, patch_artist=True,
                    showmeans=True, meanline=True, showfliers=False,
                    medianprops={"color": "white", "linewidth": 0.5},
                    boxprops={"facecolor": "C0", "edgecolor": "white",
                            "linewidth": 0.5},
                    whiskerprops={"color": "C0", "linewidth": 1.5},
                    capprops={"color": "C0", "linewidth": 1.5},
                    meanprops = dict(linestyle='--', linewidth=2.5, color='purple'),
                    labels=labels)
    ax.set_title('pearson correlation distribution')
    ax.tick_params(axis='x', rotation=45)

    plt.savefig(output_dir + 'distribution_corr.png')