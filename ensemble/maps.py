import utils
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import math


def plot_maps_ensemble(ens_df, output_dir, param, unit, n_members, n=42, cmap="viridis"):
    """
    Plots fields given by each member of the ensemble

    Args:
        ens_df (DataFrame): dataframe containing the fields predicted by the model
        output_dir (str): output directory
        param (str): parameter to plot
        unit (str): unit of the considered parameter
        n_members (int): number of members
        n (int, optional): number of images to plot. Defaults to 10.
        cmap (str, optional): colormap. Defaults to "viridis".
    """
    k = math.floor(math.sqrt(n_members)) # size of the figure (number of plots / side)
    for i in range(n):
        fig = plt.figure(figsize=[5*(k+1), 4*(k+1)])
        axs = []
        for j in range(n_members):
            axs.append(fig.add_subplot(k+1, k+1, j+1, projection=ccrs.PlateCarree()))
            axs[j].set_extent(utils.IMG_EXTENT)
            axs[j].coastlines(resolution='10m', color='black', linewidth=1)

        axs.append(fig.add_subplot(k+1, k+1, k**2 + k + 1, projection=ccrs.PlateCarree()))
        axs[n_members].set_extent(utils.IMG_EXTENT)
        axs[n_members].coastlines(resolution='10m', color='black', linewidth=1)

        axs.append(fig.add_subplot(k+1, k+1, k**2 + k + 2, projection=ccrs.PlateCarree()))
        axs[n_members + 1].set_extent(utils.IMG_EXTENT)
        axs[n_members + 1].coastlines(resolution='10m', color='black', linewidth=1)

        data = [ens_df[param + "_" + str(j + 1)].iloc[i] for j in range(n_members)]
        images = []
        for j in range(n_members):
            images.append(axs[j].imshow(data[j], cmap=cmap, origin='upper', extent=utils.IMG_EXTENT, transform=ccrs.PlateCarree()))
            axs[j].label_outer()
            axs[j].set_title(str(j + 1), fontdict={"fontsize": 20})

        images.append(axs[n_members].imshow(ens_df[param + "_X"].iloc[i], cmap=cmap, origin='upper', extent=utils.IMG_EXTENT, transform=ccrs.PlateCarree()))
        axs[n_members].label_outer()
        axs[n_members].set_title("Arome 2km5", fontdict={"fontsize": 20})
        
        images.append(axs[n_members+1].imshow(ens_df[param + "_y"].iloc[i], cmap=cmap, origin='upper', extent=utils.IMG_EXTENT, transform=ccrs.PlateCarree()))
        axs[n_members + 1].label_outer()
        axs[n_members + 1].set_title("Arome 500m", fontdict={"fontsize": 20})
        
        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        for im in images:
            im.set_norm(norm)
        
        fig.colorbar(images[0], ax=axs, label="{} [{}]".format(param, unit))
        plt.savefig(output_dir + 'results_' + str(i) + '_' + param + '.png', bbox_inches='tight')