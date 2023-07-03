import numpy as np
import pandas as pd
import scipy.stats as sc
import utils
import matplotlib.pyplot as plt


def wasserstein_distance(a, b):
    dist_a = np.reshape(a, -1)
    dist_b = np.reshape(b, -1)
    return sc.wasserstein_distance(dist_a, dist_b)


# compute WD for the whole results dataframe
def compute_datewise_WD(results_df):
    wasserstein_df = pd.DataFrame(
        {'dates' : [],
        'echeances' : [],
        'datewise_wasserstein_distance_baseline' : [],
        'datewise_wasserstein_distance_pred' : []}
    )
    for i in range(len(results_df)):
        wasserstein_df.loc[len(wasserstein_df)] = [
            results_df.dates.iloc[i],
            results_df.echeances.iloc[i],
            wasserstein_distance(results_df.baseline.iloc[i], results_df.y_test.iloc[i]),
            wasserstein_distance(results_df.y_pred.iloc[i], results_df.y_test.iloc[i])
        ]
    return wasserstein_df


def compute_datewise_WD_terre(results_df):
    ind_terre_mer = np.reshape(utils.get_ind_terre_mer_500m(), -1)
    wasserstein_df = pd.DataFrame(
        {'dates' : [],
        'echeances' : [],
        'datewise_wasserstein_distance_baseline' : [],
        'datewise_wasserstein_distance_pred' : []}
    )
    for i in range(len(results_df)):
        dist_test     = ind_terre_mer * np.reshape(results_df.y_test.iloc[i], -1)
        dist_pred     = ind_terre_mer * np.reshape(results_df.y_pred.iloc[i], -1)
        dist_baseline = ind_terre_mer * np.reshape(results_df.baseline.iloc[i], -1)

        wasserstein_df.loc[len(wasserstein_df)] = [
            results_df.dates.iloc[i],
            results_df.echeances.iloc[i],
            sc.wasserstein_distance(dist_baseline, dist_test),
            sc.wasserstein_distance(dist_pred, dist_test)
        ]
    return wasserstein_df


def compute_datewise_WD_mer(results_df):
    ind_terre_mer = np.reshape(utils.get_ind_terre_mer_500m(), -1)
    wasserstein_df = pd.DataFrame(
        {'dates' : [],
        'echeances' : [],
        'datewise_wasserstein_distance_baseline' : [],
        'datewise_wasserstein_distance_pred' : []}
    )
    for i in range(len(results_df)):
        dist_test     = (1 - ind_terre_mer) * np.reshape(results_df.y_test.iloc[i], -1)
        dist_pred     = (1 - ind_terre_mer) * np.reshape(results_df.y_pred.iloc[i], -1)
        dist_baseline = (1 - ind_terre_mer) * np.reshape(results_df.baseline.iloc[i], -1)

        wasserstein_df.loc[len(wasserstein_df)] = [
            results_df.dates.iloc[i],
            results_df.echeances.iloc[i],
            sc.wasserstein_distance(dist_baseline, dist_test),
            sc.wasserstein_distance(dist_pred, dist_test)
        ]
    return wasserstein_df


# plot distributions
def plot_datewise_wasserstein_distance_distrib(wd_df, wd_df_terre, wd_df_mer, output_dir):    
    wd_df_baseline = wd_df['datewise_wasserstein_distance_baseline']
    wd_df_pred = wd_df['datewise_wasserstein_distance_pred']
    wd_df_baseline_terre = wd_df_terre['datewise_wasserstein_distance_baseline']
    wd_df_pred_terre = wd_df_terre['datewise_wasserstein_distance_pred']
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


def synthesis_wasserstein_distance_distrib(expes_names, wds_df, wds_df_terre, wds_df_mer, output_dir):
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(20, 15))    
    D = []
    D_terre = []
    D_mer = []
    labels = list(expes_names) + ['baseline']
    for i in range(len(wds_df)):
        wd_df                = wds_df[i]
        wd_df_baseline       = wd_df['datewise_wasserstein_distance_baseline']
        wd_df_pred           = wd_df['datewise_wasserstein_distance_pred']
        wd_df_terre          = wds_df_terre[i]
        wd_df_baseline_terre = wd_df_terre['datewise_wasserstein_distance_baseline']
        wd_df_pred_terre     = wd_df_terre['datewise_wasserstein_distance_pred']
        wd_df_mer            = wds_df_mer[i]
        wd_df_baseline_mer   = wd_df_mer['datewise_wasserstein_distance_baseline']
        wd_df_pred_mer       = wd_df_mer['datewise_wasserstein_distance_pred']
        
        D.append(wd_df_pred)
        D_terre.append(wd_df_pred_terre)
        D_mer.append(wd_df_pred_mer)
    D.append(wd_df_baseline)
    D_terre.append(wd_df_baseline_terre)
    D_mer.append(wd_df_baseline_mer)
    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
    VP = axs[0].boxplot(D, positions=range(0, 3*(len(expes_names)+1), 3), widths=1.5, patch_artist=True,
                    showmeans=True, meanline=True, showfliers=False,
                    medianprops={"color": "white", "linewidth": 0.5},
                    boxprops={"facecolor": "C0", "edgecolor": "white",
                            "linewidth": 0.5},
                    whiskerprops={"color": "C0", "linewidth": 1.5},
                    capprops={"color": "C0", "linewidth": 1.5},
                    meanprops = dict(linestyle='--', linewidth=2.5, color='purple'),
                    labels=labels)
    VP = axs[1].boxplot(D_terre, positions=range(0, 3*(len(expes_names)+1), 3), widths=1.5, patch_artist=True,
                    showmeans=True, meanline=True, showfliers=False,
                    medianprops={"color": "white", "linewidth": 0.5},
                    boxprops={"facecolor": "C0", "edgecolor": "white",
                            "linewidth": 0.5},
                    whiskerprops={"color": "C0", "linewidth": 1.5},
                    capprops={"color": "C0", "linewidth": 1.5},
                    meanprops = dict(linestyle='--', linewidth=2.5, color='purple'),
                    labels=labels)
    VP = axs[2].boxplot(D_mer, positions=range(0, 3*(len(expes_names)+1), 3), widths=1.5, patch_artist=True,
                    showmeans=True, meanline=True, showfliers=False,
                    medianprops={"color": "white", "linewidth": 0.5},
                    boxprops={"facecolor": "C0", "edgecolor": "white",
                            "linewidth": 0.5},
                    whiskerprops={"color": "C0", "linewidth": 1.5},
                    capprops={"color": "C0", "linewidth": 1.5},
                    meanprops = dict(linestyle='--', linewidth=2.5, color='purple'),
                    labels=labels)
    axs[0].set_title('Wasserstein distance distribution')
    axs[1].set_title('Wasserstein distance terre distribution')
    axs[2].set_title('Wasserstein distance mer distribution')
    # axs.tick_params(axis='x', rotation=90)
    plt.savefig(output_dir + 'synthesis_distributions_wd.png', bbox_inches='tight')
