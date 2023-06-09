import numpy as np
import pandas as pd
from results.results import *
from bronx.stdtypes.date import daterangex as rangex
import matplotlib.pyplot as plt
from matplotlib import colors
import warnings

warnings.filterwarnings("ignore")



def synthesis_maps(expes_names, expes_results, output_dir, full=False):
    for k in range(10):
        if full:
            fig, axs = plt.subplots(nrows=4, ncols=len(expes_names), figsize=(5*len(expes_names), 16))
            images = []
            for j in range(len(expes_names)):
                name = expes_names[j]
                results_df = expes_results[j]
                data = [results_df.X_test[k], results_df.baseline[k], results_df.y_pred[k], results_df.y_test[k]]
                for i in range(len(data)):
                    im = axs[i, j].imshow(data[i], cmap='viridis')
                    images.append(im)
                    axs[i, j].label_outer()
                axs[0, j].set_title('X_test ' + name)
                axs[1, j].set_title('baseline ' + name)
                axs[2, j].set_title('y_pred ' + name)
                axs[3, j].set_title('y_test ' + name)

            vmin = min(image.get_array().min() for image in images)
            vmax = max(image.get_array().max() for image in images)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            for im in images:
                im.set_norm(norm)
            fig.colorbar(images[0], ax=axs)
            plt.savefig(output_dir + 'synthesis_' + str(k) + '_map.png', bbox_inches='tight')
        else:
            fig, axs = plt.subplots(nrows=1, ncols=len(expes_names), figsize=(5*len(expes_names), 4))
            images = []
            for j in range(len(expes_names)):
                name = expes_names[j]
                results_df = expes_results[j]
                im = axs[j].imshow(results_df.y_pred[k], cmap='viridis')
                images.append(im)
                axs[j].label_outer()
                axs[j].set_title('y_pred ' + name)

            vmin = min(image.get_array().min() for image in images)
            vmax = max(image.get_array().max() for image in images)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            for im in images:
                im.set_norm(norm)
            fig.colorbar(images[0], ax=axs)
            plt.savefig(output_dir + 'synthesis_' + str(k) + '_map.png', bbox_inches='tight')


def synthesis_score_maps(expes_names, expes_results, output_dir, metric, metric_name, cmap='coolwarm'):
    for k in range(10):
        fig, axs = plt.subplots(nrows=1, ncols=len(expes_names)+1, figsize=(5*len(expes_names), 5))
        images = []
        for j in range(len(expes_names)):
            name = expes_names[j]
            results_df = expes_results[j]
            metric_df  = datewise_scores(results_df, metric, metric_name)
            im = axs[j].imshow(metric_df[metric_name + '_y_pred_map'][k], cmap=cmap)
            images.append(im)
            axs[j].label_outer()
            axs[j].set_title(metric_name + ' y_pred ' + name)
        im = axs[j+1].imshow(metric_df[metric_name + '_baseline_map'][k], cmap=cmap)
        images.append(im)
        axs[j+1].label_outer()
        axs[j+1].set_title(metric_name + ' baseline ')

        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        for im in images:
            im.set_norm(norm)
        fig.colorbar(images[0], ax=axs)
        plt.savefig(output_dir + metric_name + '_' + str(k) + '_map.png', bbox_inches='tight')


def synthesis_unique_score_map(expes_names, expes_results, output_dir, metric, metric_name, cmap='coolwarm'):
    fig, axs = plt.subplots(nrows=1, ncols=len(expes_names)+1, figsize=(5*len(expes_names), 5))
    images = []
    for j in range(len(expes_names)):
        name = expes_names[j]
        results_df = expes_results[j]
        metric_df  = datewise_scores(results_df, metric, metric_name)
        im = axs[j].imshow(metric_df[metric_name + '_y_pred_map'].mean(), cmap=cmap)
        images.append(im)
        axs[j].label_outer()
        mean = metric_df[metric_name + '_y_pred_map'].mean().mean()
        axs[j].set_title(metric_name + ' y_pred ' + name + ' ' + f'{mean:.2f}')
    im = axs[j+1].imshow(metric_df[metric_name + '_baseline_map'].mean(), cmap=cmap)
    images.append(im)
    axs[j+1].label_outer()
    mean = metric_df[metric_name + '_baseline_map'].mean().mean()
    axs[j+1].set_title(metric_name + ' baseline ' + ' ' + f'{mean:.2f}')

    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)
    fig.colorbar(images[0], ax=axs)
    plt.savefig(output_dir + metric_name + '_' + '_map.png', bbox_inches='tight')


def synthesis_score_distribs(expes_names, expes_results, output_dir, metric, metric_name):
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(20, 15))    
    D = []
    D_terre = []
    D_mer = []
    labels = expes_names + ['baseline']
    for i in range(len(expes_names)):
        results_df = expes_results[i]
        score_baseline = datewise_scores(results_df, metric, metric_name)[metric_name + '_baseline_mean']
        score_pred = datewise_scores(results_df, metric, metric_name)[metric_name + '_y_pred_mean']
        score_baseline_terre = datewise_scores_terre(results_df, metric, metric_name)[metric_name + '_baseline_mean']
        score_pred_terre = datewise_scores_terre(results_df, metric, metric_name)[metric_name + '_y_pred_mean']
        score_baseline_mer = datewise_scores_mer(results_df, metric, metric_name)[metric_name + '_baseline_mean']
        score_pred_mer = datewise_scores_mer(results_df, metric, metric_name)[metric_name + '_y_pred_mean']
        
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


def synthesis_wasserstein_distance_distrib(expes_names, expes_results, output_dir):
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(20, 15))    
    D = []
    D_terre = []
    D_mer = []
    labels = list(expes_names) + ['baseline']
    for i in range(len(expes_names)):
        results_df = expes_results[i]

        wd_df = datewise_wasserstein_distance(results_df)
        wd_df_baseline = wd_df['datewise_wasserstein_distance_baseline']
        wd_df_pred = wd_df['datewise_wasserstein_distance_pred']
        wd_df_terre = datewise_wasserstein_distance_terre(results_df)
        wd_df_baseline_terre = wd_df_terre['datewise_wasserstein_distance_baseline']
        wd_df_pred_terre = wd_df_terre['datewise_wasserstein_distance_pred']
        wd_df_mer = datewise_wasserstein_distance_mer(results_df)
        wd_df_baseline_mer = wd_df_mer['datewise_wasserstein_distance_baseline']
        wd_df_pred_mer = wd_df_mer['datewise_wasserstein_distance_pred']
        
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


def synthesis_PSDs(expes_names, expes_results, output_dir):
    fig, axs = plt.subplots(nrows=len(expes_names), ncols=1, figsize=(10, 4*len(expes_names)))
    for i in range(len(expes_names)):
        results_df = expes_results[i]
        psd = PSD(results_df)

        axs[i].grid()
        axs[i].plot(psd.psd_test, color='r', label='psd_y_test')
        axs[i].plot(psd.psd_pred, color='b', label='psd_pred')
        axs[i].plot(psd.psd_baseline, color='g', label='psd_baseline')
        axs[i].plot(psd.psd_X_test, color='m', label='psd_X_test')
        axs[i].loglog()
        
        axs[i].set_title('PSDs ' + expes_names[i])
    axs[0].legend()
    fig.savefig(output_dir + 'PSDs.png', bbox_inches='tight')


def synthesis_corr_len(expes_names, expes_results, output_dir):
    cmap = colors.ListedColormap([(0, 0.2, 1, alpha) for alpha in np.linspace(0, 1, 8)])
    fig, axs = plt.subplots(nrows=1, ncols=len(expes_names)+2, figsize=(7*len(expes_names), 5))
    images = []
    for j in range(len(expes_names)):
        name = expes_names[j]
        results_df  = expes_results[j]
        corr_len_df = corr_len(results_df)
        
        im = axs[j].imshow(corr_len_df.corr_len_pred[0], cmap=cmap)
        images.append(im)
        axs[j].label_outer()
        axs[j].set_title('corr length y_pred ' + name)
    im = axs[j+1].imshow(corr_len_df.corr_len_baseline[0], cmap=cmap)
    images.append(im)
    axs[j+1].label_outer()
    axs[j+1].set_title('correlation length baseline')
    im = axs[j+2].imshow(corr_len_df.corr_len_test[0], cmap=cmap)
    images.append(im)
    axs[j+1].label_outer()
    axs[j+2].set_title('correlation length test')

    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)
    fig.colorbar(images[0], ax=axs)
    plt.savefig(output_dir + 'correlation_length_map.png', bbox_inches='tight')


def synthesis_corr_distrib(expes_names, expes_results, output_dir):
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(20, 15))    
    D = []
    D_terre = []
    D_mer = []
    labels = list(expes_names) + ['baseline']
    for i in range(len(expes_names)):
        results_df = expes_results[i]

        corr_df                = correlation(results_df)
        corr_df_baseline       = corr_df['r_baseline']
        corr_df_pred           = corr_df['r_pred']
        corr_df_terre          = correlation_terre(results_df)
        corr_df_baseline_terre = corr_df_terre['r_baseline']
        corr_df_pred_terre     = corr_df_terre['r_pred']
        corr_df_mer            = correlation_mer(results_df)
        corr_df_baseline_mer   = corr_df_mer['r_baseline']
        corr_df_pred_mer       = corr_df_mer['r_pred']
        
        D.append(corr_df_pred)
        D_terre.append(corr_df_pred_terre)
        D_mer.append(corr_df_pred_mer)
    D.append(corr_df_baseline)
    D_terre.append(corr_df_baseline_terre)
    D_mer.append(corr_df_baseline_mer)
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
    axs[0].set_title('pearson distribution')
    axs[1].set_title('pearson terre distribution')
    axs[2].set_title('pearson mer distribution')
    # axs.tick_params(axis='x', rotation=90)
    plt.savefig(output_dir + 'synthesis_distributions_corr.png', bbox_inches='tight')