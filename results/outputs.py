import numpy as np
import pandas as pd
import utils
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs


def load_results(y_pred_path, resample, data_test_location, baseline_location, param='t2m'):
    y_pred_df = pd.read_pickle(y_pred_path)
    dates_test = y_pred_df.dates.drop_duplicates().values

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
                filepath_X_test = data_test_location + 'oper_c_' + d + 'Z_' + param + '.npy'
            elif resample == 'r':
                filepath_X_test = data_test_location + 'oper_r_' + d + 'Z_' + param + '.npy'
            elif resample == 'bl':
                filepath_X_test = data_test_location + 'oper_bl_' + d + 'Z_' + param + '.npy'
            elif resample == 'bc':
                filepath_X_test = data_test_location + 'oper_bc_' + d + 'Z_' + param + '.npy'
            else:
                raise NotImplementedError

            X_test = np.load(filepath_X_test)
            if resample in ['bl', 'bc']:
                X_test = np.pad(X_test, ((5,4), (2,5), (0,0)), mode='edge')
        except FileNotFoundError:
            print('missing day (X): ' + d)
            X_test = None

        # Load baseline : 
        try:
            filepath_baseline = baseline_location + 'GG9B_' + d + 'Z_' + param + '.npy'
            baseline = np.load(filepath_baseline)
        except FileNotFoundError:
            print('missing day (b): ' + d)
            baseline = None

        # Load y_test : 
        try:
            filepath_y_test = data_test_location + 'G9L1_' + d + 'Z_' + param + '.npy'
            y_test = np.load(filepath_y_test)
        except FileNotFoundError:
            print('missing day (y): ' + d)
            y_test = None

        echeances = y_pred_df[y_pred_df.dates == d].echeances.drop_duplicates().values

        for i_ech, ech in enumerate(echeances):
            if (X_test is not None) and (y_test is not None) and (baseline is not None):
                results_df.loc[len(results_df)] = [
                    dates_test[i_d],
                    echeances[i_ech],
                    X_test[:, :, i_ech],
                    baseline[:, :, i_ech],
                    y_pred_df[y_pred_df.dates == d][y_pred_df.echeances == ech][param].to_list()[0],
                    y_test[:, :, i_ech]
                ]

    return results_df


def plot_maps(results_df, output_dir, param, unit, cmap="viridis", n=10):
    for i in range(n):
        fig = plt.figure(figsize=[25, 6])
        axs = []
        for j in range(4):
            axs.append(fig.add_subplot(1, 4, j+1, projection=ccrs.PlateCarree()))
            axs[j].set_extent(utils.IMG_EXTENT)
            axs[j].coastlines(resolution='10m', color='black', linewidth=1)

        data = [results_df.X_test[i], results_df.baseline[i], results_df.y_pred[i], results_df.y_test[i]]
        images = []
        for j in range(4):
            images.append(axs[j].imshow(data[j], cmap=cmap, origin='upper', extent=utils.IMG_EXTENT, transform=ccrs.PlateCarree()))
            axs[j].label_outer()
        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        for im in images:
            im.set_norm(norm)
        axs[0].set_title('Arome2km5', fontdict={"fontsize": 20})
        axs[1].set_title('fullpos', fontdict={"fontsize": 20})
        axs[2].set_title('DDPM', fontdict={"fontsize": 20})
        axs[3].set_title('Arome500m', fontdict={"fontsize": 20})
        fig.colorbar(images[0], ax=axs, label="{} [{}]".format(param, unit))
        plt.savefig(output_dir + 'results_' + str(i) + '_' + param + '.png', bbox_inches='tight')


def plot_synthesis_maps(expes_names, expes_results, output_dir, param, unit, cmap="viridis", n=10, several_inputs=False):
    n_expes = len(expes_results)
    if not several_inputs:
        for i in range(n):
            fig = plt.figure(figsize=[max(5*n_expes, 15), 9])
            fig.subplots_adjust(wspace=0.1, hspace=0.1)
            axs = []
            for j in range(n_expes + 3):
                axs.append(fig.add_subplot(2, max(3, n_expes), j+1, projection=ccrs.PlateCarree()))
                axs[j].set_extent(utils.IMG_EXTENT)
                axs[j].coastlines(resolution='10m', color='black', linewidth=1)

            data = [expes_results[j].y_pred.iloc[i]for j in range(n_expes)] + \
                [expes_results[0].X_test.iloc[i], expes_results[0].baseline.iloc[i], expes_results[0].y_test.iloc[i]]
            images = []
            for j in range(3 + n_expes):
                images.append(axs[j].imshow(data[j], cmap=cmap, origin='upper', extent=utils.IMG_EXTENT, transform=ccrs.PlateCarree()))
                axs[j].label_outer()
            vmin = min(image.get_array().min() for image in images)
            vmax = max(image.get_array().max() for image in images)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            for im in images:
                im.set_norm(norm)
            for j in range(n_expes):
                axs[j].set_title("DDPM {}".format(expes_names[j]))
            axs[-3].set_title('Arome2km5')
            axs[-2].set_title('fullpos')
            axs[-1].set_title('Arome500m')
            fig.colorbar(images[0], ax=axs, label="{} [{}]".format(param, unit))
            plt.savefig(output_dir + 'synthesis_' + str(i) + '_map.png', bbox_inches='tight')
    else:
        for i in range(n):
            fig = plt.figure(figsize=[max(5*n_expes, 10), 12])
            fig.subplots_adjust(wspace=0.1, hspace=0.1)
            axs = []
            for j in range(2 * n_expes + 2):
                axs.append(fig.add_subplot(3, max(2, n_expes), j+1, projection=ccrs.PlateCarree()))
                axs[j].set_extent(utils.IMG_EXTENT)
                axs[j].coastlines(resolution='10m', color='black', linewidth=1)

            data = [expes_results[j].X_test.iloc[i]for j in range(n_expes)] + \
                [expes_results[j].y_pred.iloc[i]for j in range(n_expes)] + \
                [expes_results[0].baseline.iloc[i], expes_results[0].y_test.iloc[i]]
            images = []
            for j in range(2 * n_expes + 2):
                images.append(axs[j].imshow(data[j], cmap=cmap, origin='upper', extent=utils.IMG_EXTENT, transform=ccrs.PlateCarree()))
                axs[j].label_outer()
            vmin = min(image.get_array().min() for image in images)
            vmax = max(image.get_array().max() for image in images)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            for im in images:
                im.set_norm(norm)
            for j in range(n_expes):
                axs[j].set_title("Arome2km5 {}".format(expes_names[j]))
            for j in range(n_expes):
                axs[j + n_expes].set_title("DDPM {}".format(expes_names[j]))
            axs[-2].set_title('fullpos')
            axs[-1].set_title('Arome500m')
            fig.colorbar(images[0], ax=axs, label="{} [{}]".format(param, unit))
            plt.savefig(output_dir + 'synthesis_' + str(i) + '_map.png', bbox_inches='tight')