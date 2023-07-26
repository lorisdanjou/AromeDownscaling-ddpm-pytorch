import numpy as np
import pandas as pd
import utils
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs


def compute_pointwise_mean(ens_df, n_members, params_out):
    """
    Returns a dataframe containing arrays of pointwise means for each sample in the ensemble

    Args:
        ens_df (DataFrame): dataframe containing results
        n_members (int): number of members in the ensemble
        params_out (list): parameters (str) predicted by the model

    Returns:
        DataFrame: a dataframe containing all the pointwise mean maps in the ensemble for each sample
    """
    stat_df = pd.DataFrame(
        [],
        columns=["dates", "echeances"] + [p for p in params_out]
    )
    for i in range(len(ens_df)):
        values = []
        for i_p, p in enumerate(params_out):
            array_p = np.array([ens_df[p + "_" + str(j + 1)].iloc[i] for j in range(n_members)])
            values.append(array_p.mean(axis=0))
        stat_df.loc[len(stat_df)] = [ens_df.dates.iloc[i], ens_df.echeances.iloc[i]] + values
    return stat_df


def compute_pointwise_std(ens_df, n_members, params_out):
    """
    Returns a dataframe containing arrays of pointwise means for each sample in the ensemble

    Args:
        ens_df (DataFrame): dataframe containing results
        n_members (int): number of members in the ensemble
        params_out (list): parameters (str) predicted by the model

    Returns:
        DataFrame: a dataframe containing all the pointwise std maps in the ensemble for each sample
    """
    stat_df = pd.DataFrame(
        [],
        columns=["dates", "echeances"] + [p for p in params_out]
    )
    for i in range(len(ens_df)):
        values = []
        for i_p, p in enumerate(params_out):
            array_p = np.array([ens_df[p + "_" + str(j + 1)].iloc[i] for j in range(n_members)])
            values.append(array_p.std(axis=0))
        stat_df.loc[len(stat_df)] = [ens_df.dates.iloc[i], ens_df.echeances.iloc[i]] + values
    return stat_df


def compute_pointwise_Q5(ens_df, n_members, params_out):
    """
    Returns a dataframe containing arrays of pointwise Q5 for each sample in the ensemble

    Args:
        ens_df (DataFrame): dataframe containing results
        n_members (int): number of members in the ensemble
        params_out (list): parameters (str) predicted by the model

    Returns:
        DataFrame: a dataframe containing all the pointwise Q5 maps in the ensemble for each sample
    """
    stat_df = pd.DataFrame(
        [],
        columns=["dates", "echeances"] + [p for p in params_out]
    )
    for i in range(len(ens_df)):
        values = []
        for i_p, p in enumerate(params_out):
            array_p = np.array([ens_df[p + "_" + str(j + 1)].iloc[i] for j in range(n_members)])
            values.append(np.percentile(array_p, 5, axis=0))
        stat_df.loc[len(stat_df)] = [ens_df.dates.iloc[i], ens_df.echeances.iloc[i]] + values
    return stat_df


def compute_pointwise_Q95(ens_df, n_members, params_out):
    """
    Returns a dataframe containing arrays of pointwise Q95 for each sample in the ensemble

    Args:
        ens_df (DataFrame): dataframe containing results
        n_members (int): number of members in the ensemble
        params_out (list): parameters (str) predicted by the model

    Returns:
        DataFrame: a dataframe containing all the pointwise Q95 maps in the ensemble for each sample
    """
    stat_df = pd.DataFrame(
        [],
        columns=["dates", "echeances"] + [p for p in params_out]
    )
    for i in range(len(ens_df)):
        values = []
        for i_p, p in enumerate(params_out):
            array_p = np.array([ens_df[p + "_" + str(j + 1)].iloc[i] for j in range(n_members)])
            values.append(np.percentile(array_p, 95, axis=0))
        stat_df.loc[len(stat_df)] = [ens_df.dates.iloc[i], ens_df.echeances.iloc[i]] + values
    return stat_df


def plot_unique_all_stats_ensemble(mean_df, std_df, Q5_df, Q95_df, output_dir, param):
    """
    Plots all the statistics on the same figure (pointwise mean for all the samples)

    Args:
        mean_df (DataFrame): a dataframe that contains the mean values
        std_df (DataFrame): a dataframe that contains the std values
        Q5_df (DataFrame): a dataframe that contais the Q5 values
        Q95_df (DataFrame): a dataframe that contains the Q95 values
        output_dir (st): output directory
        param (str): studied parameter
    """
    all_stats_df = pd.DataFrame(
        [],
        columns = []
    )
    all_stats_df = pd.concat([all_stats_df, mean_df[["dates", "echeances"]]], axis=1)
    all_stats_df = pd.concat([all_stats_df, mean_df[[param]].rename(columns={param:param + "_mean"})], axis=1)
    all_stats_df = pd.concat([all_stats_df,  std_df[[param]].rename(columns={param:param + "_std"})], axis=1)
    all_stats_df = pd.concat([all_stats_df,   Q5_df[[param]].rename(columns={param:param + "_Q5"})], axis=1)
    all_stats_df = pd.concat([all_stats_df,  Q95_df[[param]].rename(columns={param:param + "_Q95"})], axis=1)

    fig = plt.figure(figsize=[25, 5])
    axs = []
    for j in range(4):
        axs.append(fig.add_subplot(1, 4, j+1, projection=ccrs.PlateCarree()))
        axs[j].set_extent(utils.IMG_EXTENT)
        axs[j].coastlines(resolution='10m', color='black', linewidth=1)

    images = []

    im = axs[0].imshow(all_stats_df[param + "_mean"].mean(), cmap="viridis", origin='upper', 
                        extent=utils.IMG_EXTENT, transform=ccrs.PlateCarree())
    images.append(im)
    axs[0].label_outer()
    im = axs[1].imshow(all_stats_df[param + "_std"].mean(), cmap="plasma", origin='upper', 
                        extent=utils.IMG_EXTENT, transform=ccrs.PlateCarree())
    fig.colorbar(im, ax=axs[1])
    im = axs[2].imshow(all_stats_df[param + "_Q5"].mean(), cmap="viridis", origin='upper', 
                        extent=utils.IMG_EXTENT, transform=ccrs.PlateCarree())
    images.append(im)
    axs[2].label_outer()
    im = axs[3].imshow(all_stats_df[param + "_Q95"].mean(), cmap="viridis", origin='upper', 
                        extent=utils.IMG_EXTENT, transform=ccrs.PlateCarree())
    images.append(im)
    axs[3].label_outer()        
    
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)

    fig.colorbar(images[0], ax=axs[0])
    fig.colorbar(images[0], ax=axs[2])
    fig.colorbar(images[0], ax=axs[3])

    axs[0].set_title("mean")
    axs[1].set_title("std")
    axs[2].set_title("Q5")
    axs[3].set_title("Q95")

    plt.savefig(output_dir + 'all_stats_unique_' + param + '.png', bbox_inches='tight')


def synthesis_all_stats_ensemble(
    mean_df,
    std_df,
    Q5_df,
    Q95_df,
    output_dir,
    param,
    n=42
):
    """
    Plots for each day all stats for all echeances (DDPM + Arome)
    """
    all_stats_df = pd.DataFrame(
        [],
        columns = []
    )
    all_stats_df = pd.concat([all_stats_df, mean_df[["dates", "echeances"]]], axis=1)
    all_stats_df = pd.concat([all_stats_df, mean_df[[param + m for m in ["_arome", "_ddpm"]]].rename(columns={param+ m:param + m + "_mean" for m in ["_arome", "_ddpm"]})], axis=1)
    all_stats_df = pd.concat([all_stats_df,  std_df[[param + m for m in ["_arome", "_ddpm"]]].rename(columns={param+ m:param + m + "_std" for m in ["_arome", "_ddpm"]})], axis=1)
    all_stats_df = pd.concat([all_stats_df,   Q5_df[[param + m for m in ["_arome", "_ddpm"]]].rename(columns={param+ m:param + m + "_Q5" for m in ["_arome", "_ddpm"]})], axis=1)
    all_stats_df = pd.concat([all_stats_df,  Q95_df[[param + m for m in ["_arome", "_ddpm"]]].rename(columns={param+ m:param + m + "_Q95" for m in ["_arome", "_ddpm"]})], axis=1)

    dates = mean_df.dates.drop_duplicates().values
    echeances = mean_df.echeances.drop_duplicates().values

    for i_d, d in enumerate(dates):
        fig = plt.figure(figsize=[25, 11*len(echeances)])
        axs = []
        for j in range(8 * len(echeances)):
            axs.append(fig.add_subplot(2*len(echeances), 4, j+1, projection=ccrs.PlateCarree()))
            axs[j].set_extent(utils.IMG_EXTENT)
            axs[j].coastlines(resolution='10m', color='black', linewidth=1)

        images = []
        stds = []
        
        for i_ech, ech in enumerate(echeances):
            im = axs[0 + 8*i_ech].imshow(all_stats_df[(all_stats_df.dates == d)][(all_stats_df.echeances == ech)][param + "_ddpm_mean"].iloc[0], cmap="viridis", origin='upper', 
                                extent=utils.IMG_EXTENT, transform=ccrs.PlateCarree())
            images.append(im)
            axs[0 + 8*i_ech].label_outer()
            im = axs[1 + 8*i_ech].imshow(all_stats_df[(all_stats_df.dates == d)][(all_stats_df.echeances == ech)][param + "_ddpm_std"].iloc[0], cmap="plasma", origin='upper', 
                                extent=utils.IMG_EXTENT, transform=ccrs.PlateCarree())
            stds.append(im)
            axs[1 + 8*i_ech].label_outer()
            im = axs[2 + 8*i_ech].imshow(all_stats_df[(all_stats_df.dates == d)][(all_stats_df.echeances == ech)][param + "_ddpm_Q5"].iloc[0], cmap="viridis", origin='upper', 
                                extent=utils.IMG_EXTENT, transform=ccrs.PlateCarree())
            images.append(im)
            axs[2 + 8*i_ech].label_outer()
            im = axs[3 + 8*i_ech].imshow(all_stats_df[(all_stats_df.dates == d)][(all_stats_df.echeances == ech)][param + "_ddpm_Q95"].iloc[0], cmap="viridis", origin='upper', 
                                extent=utils.IMG_EXTENT, transform=ccrs.PlateCarree())
            images.append(im)
            axs[3 + 8*i_ech].label_outer()    

            im = axs[4 + 8*i_ech].imshow(all_stats_df[(all_stats_df.dates == d)][(all_stats_df.echeances == ech)][param + "_arome_mean"].iloc[0], cmap="viridis", origin='upper', 
                                extent=utils.IMG_EXTENT, transform=ccrs.PlateCarree())
            images.append(im)
            axs[4 + 8*i_ech].label_outer()
            im = axs[5 + 8*i_ech].imshow(all_stats_df[(all_stats_df.dates == d)][(all_stats_df.echeances == ech)][param + "_arome_std"].iloc[0], cmap="plasma", origin='upper', 
                                extent=utils.IMG_EXTENT, transform=ccrs.PlateCarree())
            stds.append(im)
            axs[1 + 8*i_ech].label_outer()
            im = axs[6 + 8*i_ech].imshow(all_stats_df[(all_stats_df.dates == d)][(all_stats_df.echeances == ech)][param + "_arome_Q5"].iloc[0], cmap="viridis", origin='upper', 
                                extent=utils.IMG_EXTENT, transform=ccrs.PlateCarree())
            images.append(im)
            axs[6 + 8*i_ech].label_outer()
            im = axs[7 + 8*i_ech].imshow(all_stats_df[(all_stats_df.dates == d)][(all_stats_df.echeances == ech)][param + "_arome_Q95"].iloc[0], cmap="viridis", origin='upper', 
                                extent=utils.IMG_EXTENT, transform=ccrs.PlateCarree())
            images.append(im)
            axs[7 + 8*i_ech].label_outer()        
        
        # same scale for all mean, Q5, Q95 images
        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        for im in images:
            im.set_norm(norm)

        # another scale for std images
        vmin = min(std.get_array().min() for std in stds)
        vmax = max(std.get_array().max() for std in stds)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        for std in stds:
            std.set_norm(norm)


        for i_ech, ech in enumerate(echeances):
            fig.colorbar(images[0], ax=axs[0 + 8*i_ech])
            fig.colorbar(images[0], ax=axs[2 + 8*i_ech])
            fig.colorbar(images[0], ax=axs[3 + 8*i_ech])
            fig.colorbar(images[0], ax=axs[4 + 8*i_ech])
            fig.colorbar(images[0], ax=axs[6 + 8*i_ech])
            fig.colorbar(images[0], ax=axs[7 + 8*i_ech])
            fig.colorbar(stds[0], ax=axs[1 + 8*i_ech])
            fig.colorbar(stds[0], ax=axs[5 + 8*i_ech])

            axs[0 + 8*i_ech].set_title("mean DDPM, +" + str(ech) + "h")
            axs[1 + 8*i_ech].set_title("std DDPM, +" + str(ech) + "h")
            axs[2 + 8*i_ech].set_title("Q5 DDPM, +" + str(ech) + "h")
            axs[3 + 8*i_ech].set_title("Q95 DDPM, +" + str(ech) + "h")
            axs[4 + 8*i_ech].set_title("mean Arome, +" + str(ech) + "h")
            axs[5 + 8*i_ech].set_title("std Arome, +" + str(ech) + "h")
            axs[6 + 8*i_ech].set_title("Q5 Arome, +" + str(ech) + "h")
            axs[7 + 8*i_ech].set_title("Q95 Arome, +" + str(ech) + "h")

        plt.savefig(output_dir + 'all_stats_synthesis_unique_' + param + "_" + d + '.png', bbox_inches='tight')


def synthesis_unique_all_stats_ensemble(
    mean_df,
    std_df,
    Q5_df,
    Q95_df,
    output_dir,
    param,

):
    """
    Plots a unique map for each stat (DDPM + Arome)
    """
    all_stats_df = pd.DataFrame(
        [],
        columns = []
    )
    all_stats_df = pd.concat([all_stats_df, mean_df[["dates", "echeances"]]], axis=1)
    all_stats_df = pd.concat([all_stats_df, mean_df[[param + m for m in ["_arome", "_ddpm"]]].rename(columns={param+ m:param + m + "_mean" for m in ["_arome", "_ddpm"]})], axis=1)
    all_stats_df = pd.concat([all_stats_df,  std_df[[param + m for m in ["_arome", "_ddpm"]]].rename(columns={param+ m:param + m + "_std" for m in ["_arome", "_ddpm"]})], axis=1)
    all_stats_df = pd.concat([all_stats_df,   Q5_df[[param + m for m in ["_arome", "_ddpm"]]].rename(columns={param+ m:param + m + "_Q5" for m in ["_arome", "_ddpm"]})], axis=1)
    all_stats_df = pd.concat([all_stats_df,  Q95_df[[param + m for m in ["_arome", "_ddpm"]]].rename(columns={param+ m:param + m + "_Q95" for m in ["_arome", "_ddpm"]})], axis=1)

    fig = plt.figure(figsize=[25, 11])
    axs = []
    for j in range(8):
        axs.append(fig.add_subplot(2, 4, j+1, projection=ccrs.PlateCarree()))
        axs[j].set_extent(utils.IMG_EXTENT)
        axs[j].coastlines(resolution='10m', color='black', linewidth=1)

    images = []
    stds = []

    im = axs[0].imshow(all_stats_df[param + "_ddpm_mean"].mean(), cmap="viridis", origin='upper', 
                        extent=utils.IMG_EXTENT, transform=ccrs.PlateCarree())
    images.append(im)
    axs[0].label_outer()
    im = axs[1].imshow(all_stats_df[param + "_ddpm_std"].mean(), cmap="plasma", origin='upper', 
                        extent=utils.IMG_EXTENT, transform=ccrs.PlateCarree())
    stds.append(im)
    axs[1].label_outer()
    im = axs[2].imshow(all_stats_df[param + "_ddpm_Q5"].mean(), cmap="viridis", origin='upper', 
                        extent=utils.IMG_EXTENT, transform=ccrs.PlateCarree())
    images.append(im)
    axs[2].label_outer()
    im = axs[3].imshow(all_stats_df[param + "_ddpm_Q95"].mean(), cmap="viridis", origin='upper', 
                        extent=utils.IMG_EXTENT, transform=ccrs.PlateCarree())
    images.append(im)
    axs[3].label_outer()    

    im = axs[4].imshow(all_stats_df[param + "_arome_mean"].mean(), cmap="viridis", origin='upper', 
                        extent=utils.IMG_EXTENT, transform=ccrs.PlateCarree())
    images.append(im)
    axs[4].label_outer()
    im = axs[5].imshow(all_stats_df[param + "_arome_std"].mean(), cmap="plasma", origin='upper', 
                        extent=utils.IMG_EXTENT, transform=ccrs.PlateCarree())
    stds.append(im)
    axs[1].label_outer()
    im = axs[6].imshow(all_stats_df[param + "_arome_Q5"].mean(), cmap="viridis", origin='upper', 
                        extent=utils.IMG_EXTENT, transform=ccrs.PlateCarree())
    images.append(im)
    axs[6].label_outer()
    im = axs[7].imshow(all_stats_df[param + "_arome_Q95"].mean(), cmap="viridis", origin='upper', 
                        extent=utils.IMG_EXTENT, transform=ccrs.PlateCarree())
    images.append(im)
    axs[7].label_outer()        
    
    # same scale for all mean, Q5, Q95 images
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)

    # another scale for std images
    vmin = min(std.get_array().min() for std in stds)
    vmax = max(std.get_array().max() for std in stds)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for std in stds:
        std.set_norm(norm)

    fig.colorbar(images[0], ax=axs[0])
    fig.colorbar(images[0], ax=axs[2])
    fig.colorbar(images[0], ax=axs[3])
    fig.colorbar(images[0], ax=axs[4])
    fig.colorbar(images[0], ax=axs[6])
    fig.colorbar(images[0], ax=axs[7])
    fig.colorbar(stds[0], ax=axs[1])
    fig.colorbar(stds[0], ax=axs[5])

    axs[0].set_title("mean DDPM")
    axs[1].set_title("std DDPM")
    axs[2].set_title("Q5 DDPM")
    axs[3].set_title("Q95 DDPM")
    axs[4].set_title("mean Arome")
    axs[5].set_title("std Arome")
    axs[6].set_title("Q5 Arome")
    axs[7].set_title("Q95 Arome")

    plt.savefig(output_dir + 'all_stats_synthesis_unique_' + param + '.png', bbox_inches='tight')


def synthesis_stat_distrib(mean_df, std_df, Q5_df, Q95_df, output_dir, param):
    """
    Plots all stats distributions (Arome + DDPM)
    """
    D = np.zeros((len(mean_df), 6))
    for i in range(len(mean_df)):
        D[i, 0] = mean_df[param + "_ddpm"].iloc[i].mean()
        D[i, 1] = Q5_df[param + "_ddpm"].iloc[i].mean()
        D[i, 2] = Q95_df[param + "_ddpm"].iloc[i].mean()
        D[i, 3] = mean_df[param + "_arome"].iloc[i].mean()
        D[i, 4] = Q5_df[param + "_arome"].iloc[i].mean()
        D[i, 5] = Q95_df[param + "_arome"].iloc[i].mean()
    labels = ['mean DDPM', 'Q5 DDPM', 'Q95 DDPM', 'mean Arome', 'Q5 Arome', 'Q95 Arome']

    D_std = np.zeros((len(mean_df), 2))
    for i in range(len(mean_df)):
        D_std[i, 0] = std_df[param + "_ddpm"].iloc[i].mean()
        D_std[i, 1] = std_df[param + "_arome"].iloc[i].mean()
    labels_std = ['std DDPM', 'std Arome']

    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    axs[0].grid()
    VP = axs[0].boxplot(D, positions=[3, 6, 9, 12, 15, 18], widths=1.5, patch_artist=True,
                    showmeans=True, meanline=True, showfliers=False,
                    medianprops={"color": "white", "linewidth": 0.5},
                    boxprops={"facecolor": "C0", "edgecolor": "white",
                            "linewidth": 0.5},
                    whiskerprops={"color": "C0", "linewidth": 1.5},
                    capprops={"color": "C0", "linewidth": 1.5},
                    meanprops = dict(linestyle='--', linewidth=2.5, color='purple'),
                    labels=labels)
    axs[0].tick_params(axis='x', rotation=45)

    axs[1].grid()
    VP = axs[1].boxplot(D_std, positions=[3, 6], widths=0.5, patch_artist=True,
                    showmeans=True, meanline=True, showfliers=False,
                    medianprops={"color": "white", "linewidth": 0.5},
                    boxprops={"facecolor": "C0", "edgecolor": "white",
                            "linewidth": 0.5},
                    whiskerprops={"color": "C0", "linewidth": 1.5},
                    capprops={"color": "C0", "linewidth": 1.5},
                    meanprops = dict(linestyle='--', linewidth=2.5, color='purple'),
                    labels=labels_std)
    axs[1].tick_params(axis='x', rotation=45)

    plt.savefig(output_dir + 'synthesis_distributions.png')


def plot_std_echeance(std_df, output_dir, param):
    """
    Plots std versus echeance
    """
    dates = std_df.dates.drop_duplicates().values
    echeances = std_df.echeances.drop_duplicates().values

    for d in dates:
        fig, ax = plt.subplots(figsize=[10, 10])
        values_ddpm = [std_df[std_df.dates == d][std_df.echeances == ech][param + "_ddpm"].iloc[0].mean() for ech in echeances]
        values_arome = [std_df[std_df.dates == d][std_df.echeances == ech][param + "_arome"].iloc[0].mean() for ech in echeances]

        ax.plot(echeances, values_ddpm, color='k', label="DDPM")
        ax.plot(echeances, values_arome, color='r', label="Arome")
        ax.legend()
        ax.grid()

        fig.savefig(output_dir + "std_echeance_" + d + ".png")
