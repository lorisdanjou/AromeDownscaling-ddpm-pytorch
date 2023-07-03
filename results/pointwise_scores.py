import numpy as np
from skimage.metrics import structural_similarity
import pandas as pd
import utils
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs


# score functions
def mse(a, b):
    return (a - b)**2

def mae(a, b):
    return (np.abs(a - b))

def bias(a, b):
    return a - b

def ssim(a, b):
    _, ssim_map = structural_similarity(
        a,
        b, 
        data_range=b.max() - b.min(),
        win_size=None,
        full=True
    )
    return ssim_map


# compute pointwise scores for a dataframe
def compute_score(results_df, metric, metric_name):
    metric_df = pd.DataFrame(
        {'dates' : [],
        'echeances' : [],
        metric_name + '_baseline_map' : [],
        metric_name + '_y_pred_map' : [],
        metric_name + '_baseline_mean' : [],
        metric_name + '_y_pred_mean' : []}
    )
    for i in range(len(results_df)):
        metric_df.loc[len(metric_df)] = [
            results_df.dates.iloc[i],
            results_df.echeances.iloc[i],
            metric(results_df.baseline.iloc[i], results_df.y_test.iloc[i]),
            metric(results_df.y_pred.iloc[i], results_df.y_test.iloc[i]),
            np.mean(metric(results_df.baseline.iloc[i], results_df.y_test.iloc[i])),
            np.mean(metric(results_df.y_pred.iloc[i], results_df.y_test.iloc[i])),
        ]
    return metric_df


def compute_score_terre(metric_df, metric_name):
    ind_terre_mer = utils.get_ind_terre_mer_500m()
    metric_terre_df = pd.DataFrame(
        {'dates' : [],
        'echeances' : [],
        metric_name + '_baseline_map' : [],
        metric_name + '_y_pred_map' : [],
        metric_name + '_baseline_mean' : [],
        metric_name + '_y_pred_mean' : []}
    )
    for i in range(len(metric_df)):
        metric_terre_df.loc[len(metric_terre_df)] = [
            metric_df.dates[i],
            metric_df.echeances[i],
            metric_df[metric_name + '_baseline_map'][i]*ind_terre_mer,
            metric_df[metric_name + '_y_pred_map'][i]*ind_terre_mer,
            np.sum(metric_df[metric_name + '_baseline_map'][i]*ind_terre_mer)/np.sum(ind_terre_mer),
            np.sum(metric_df[metric_name + '_y_pred_map'][i]*ind_terre_mer)/np.sum(ind_terre_mer)
        ]
    return metric_terre_df


def compute_score_mer(metric_df, metric_name):
    ind_terre_mer = utils.get_ind_terre_mer_500m()
    metric_mer_df = pd.DataFrame(
        {'dates' : [],
        'echeances' : [],
        metric_name + '_baseline_map' : [],
        metric_name + '_y_pred_map' : [],
        metric_name + '_baseline_mean' : [],
        metric_name + '_y_pred_mean' : []}
    )
    for i in range(len(metric_df)):
        metric_mer_df.loc[len(metric_mer_df)] = [
            metric_df.dates[i],
            metric_df.echeances[i],
            metric_df[metric_name + '_baseline_map'][i]*(1 - ind_terre_mer),
            metric_df[metric_name + '_y_pred_map'][i]*(1 - ind_terre_mer),
            np.sum(metric_df[metric_name + '_baseline_map'][i]*(1 - ind_terre_mer))/np.sum((1 - ind_terre_mer)),
            np.sum(metric_df[metric_name + '_y_pred_map'][i]*(1 - ind_terre_mer))/np.sum((1 - ind_terre_mer))
        ]
    return metric_mer_df


def plot_score_maps(metric_df, output_dir, metric_name, unit, cmap="viridis", n=10):
    for i in range(n):
        fig = plt.figure(figsize=[25, 12])
        fig.suptitle(metric_name, fontsize=30)
        axs = []
        for j in range(2):
            axs.append(fig.add_subplot(1, 2, j+1, projection=ccrs.PlateCarree()))
            axs[j].set_extent(utils.IMG_EXTENT)
            axs[j].coastlines(resolution='10m', color='black', linewidth=1)

        data = [metric_df[metric_name + '_baseline_map'][i], metric_df[metric_name + '_y_pred_map'][i]]
        images = []
        for j in range(2):
            images.append(axs[j].imshow(data[j], cmap=cmap, origin='upper', extent=utils.IMG_EXTENT, transform=ccrs.PlateCarree()))
            axs[j].label_outer()
        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        for im in images:
            im.set_norm(norm)
        axs[0].set_title("fullpos", fontdict={"fontsize": 20})
        axs[1].set_title("DDPM", fontdict={"fontsize": 20})
        fig.colorbar(images[0], ax=axs, label="{} [{}]".format(metric_name, unit))
        plt.savefig(output_dir + metric_name + str(i) + '_map.png', bbox_inches="tight")


def plot_unique_score_map(metric_df, output_dir, metric_name, unit, cmap="viridis", n=10):
    metric_baseline = metric_df[metric_name + '_baseline_map'].mean()
    metric_y_pred   = metric_df[metric_name + '_y_pred_map'].mean()
    fig = plt.figure(figsize=[25, 12])
    fig.suptitle(metric_name, fontsize=30)
    axs = []
    for j in range(2):
        axs.append(fig.add_subplot(1, 2, j+1, projection=ccrs.PlateCarree()))
        axs[j].set_extent(utils.IMG_EXTENT)
        axs[j].coastlines(resolution='10m', color='black', linewidth=1)

    data = [metric_baseline, metric_y_pred]
    images = []
    for j in range(2):
        images.append(axs[j].imshow(data[j], cmap=cmap, origin='upper', extent=utils.IMG_EXTENT, transform=ccrs.PlateCarree()))
        axs[j].label_outer()
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)
    axs[0].set_title("fullpos", fontdict={"fontsize": 20})
    axs[1].set_title("DDPM", fontdict={"fontsize": 20})
    fig.colorbar(images[0], ax=axs, label="{} [{}]".format(metric_name, unit))
    plt.savefig(output_dir + metric_name + '_unique_map.png', bbox_inches="tight")


def plot_distrib(metric_df, metric_name, output_dir):
    score_baseline = metric_df[metric_name + '_baseline_mean']
    score_baseline_terre = compute_score_terre(metric_df, metric_name)[metric_name + '_baseline_mean']
    score_baseline_mer = compute_score_mer(metric_df, metric_name)[metric_name + '_baseline_mean']
    score_pred = metric_df[metric_name + '_y_pred_mean']
    score_pred_terre = compute_score_terre(metric_df, metric_name)[metric_name + '_y_pred_mean']
    score_pred_mer = compute_score_mer(metric_df, metric_name)[metric_name + '_y_pred_mean']
    
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


def plot_synthesis_scores(expes_names, metrics_df, output_dir, metric_name, unit, cmap="viridis"):
    n_expes = len(metrics_df)
    fig = plt.figure(figsize=[5*n_expes, 9])
    fig.suptitle(metric_name, fontsize=30)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    axs = []
    for j in range(n_expes + 1):
        axs.append(fig.add_subplot(2, n_expes, j+1, projection=ccrs.PlateCarree()))
        axs[j].set_extent(utils.IMG_EXTENT)
        axs[j].coastlines(resolution='10m', color='black', linewidth=1)

    data = [metrics_df[j][metric_name + "_y_pred_map"].mean() for j in range(n_expes)] + \
        [metrics_df[0][metric_name + "_baseline_map"].mean()]
    images = []
    for j in range(1 + n_expes):
        images.append(axs[j].imshow(data[j], cmap=cmap, origin='upper', extent=utils.IMG_EXTENT, transform=ccrs.PlateCarree()))
        axs[j].label_outer()
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)
    for j in range(n_expes):
        axs[j].set_title("DDPM {}".format(expes_names[j]))
    axs[-1].set_title('fullpos')
    fig.colorbar(images[0], ax=axs, label="{} [{}]".format(metric_name, unit))
    plt.savefig(output_dir + 'synthesis_map_{}.png'.format(metric_name), bbox_inches='tight')


def synthesis_score_distribs(expes_names, metrics_df, metrics_df_terre, metrics_df_mer, output_dir, metric_name):
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(20, 15))    
    D = []
    D_terre = []
    D_mer = []
    labels = expes_names + ['baseline']
    for i in range(len(metrics_df)):
        metric_df = metrics_df[i]
        metric_df_terre = metrics_df_terre[i]
        metric_df_mer   = metrics_df_mer[i]
        score_baseline       = metric_df[metric_name + '_baseline_mean']
        score_pred           = metric_df[metric_name + '_y_pred_mean']
        score_baseline_terre = metric_df_terre[metric_name + '_baseline_mean']
        score_pred_terre     = metric_df_terre[metric_name + '_y_pred_mean']
        score_baseline_mer   = metric_df_mer[metric_name + '_baseline_mean']
        score_pred_mer       = metric_df_mer[metric_name + '_y_pred_mean']
        
        D.append(score_pred)
        D_terre.append(score_pred_terre)
        D_mer.append(score_pred_mer)
    D.append(score_baseline)
    D_terre.append(score_baseline_terre)
    D_mer.append(score_baseline_mer)
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
    axs[0].set_title(metric_name + ' distribution')
    axs[1].set_title(metric_name + ' terre distribution')
    axs[2].set_title(metric_name + ' mer distribution')
    # axs.tick_params(axis='x', rotation=90)
    plt.savefig(output_dir + 'synthesis_distributions_' +  metric_name + '.png', bbox_inches='tight')