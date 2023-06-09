import random
import numpy as np
import pandas as pd
from preprocessing.load_data import get_arrays_cols, param_to_array
from patchify import patchify, unpatchify
from math import gcd


# Dans le cas de deux df de même tailles
def extract_patches(X_df, y_df, patch_h, patch_w, n_patches):
    """
    Randomly extracts n_patches of size (patch_h, patch_w) in each X and y samples.
    Inputs : 
        X_df : A pandas dataframe dontaining the input samples (loaded with preprocessing.load_data)
        y_df : A pandas dataframe dontaining the output samples 
        patch_h : height of the patch
        patch_w : width of the patch
        n_patches : number of patches to extract in each sample
    Outputs : 
        2 new pandas dataframs containing the patches
    """

    arrays_cols_X = get_arrays_cols(X_df)
    arrays_cols_y = get_arrays_cols(y_df)
    img_h = X_df[arrays_cols_X[0]][0].shape[0]
    img_w = X_df[arrays_cols_X[0]][0].shape[1]

    X_df_out = pd.DataFrame(
        [],
        columns = X_df.columns
    )
    y_df_out = pd.DataFrame(
        [],
        columns = y_df.columns
    )

    X = np.zeros((len(arrays_cols_X), img_h, img_w))
    y = np.zeros((len(arrays_cols_y), img_h, img_w))
    for i in range(len(X_df)):
        for i_x in range(len(arrays_cols_X)):
            X[i_x, :, :] = X_df[arrays_cols_X[i_x]][i]
        for i_y in range(len(arrays_cols_y)):
            y[i_y, :, :] = y_df[arrays_cols_y[i_y]][i]
        
        for k in range(n_patches):
            x_center = random.randint(0+int(patch_w/2),img_w-int(patch_w/2))
            y_center = random.randint(0+int(patch_h/2),img_h-int(patch_h/2))
            patch_X = X[:,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
            patch_y = y[:,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]

            # ajout d'une ligne dans le dataset:
            values_X = []
            values_y = []
            for j in range(patch_X.shape[0]):
                values_X.append(patch_X[j, :, :].reshape((patch_X.shape[1], patch_X.shape[2])))
            for j in range(patch_y.shape[0]):
                values_y.append(patch_y[j, :, :].reshape((patch_y.shape[1], patch_y.shape[2])))
            X_df_i = pd.DataFrame(
                [[X_df.dates[i], X_df.echeances[i]] + values_X],
                columns=X_df.columns
            )
            y_df_i = pd.DataFrame(
                [[y_df.dates[i], y_df.echeances[i]] + values_y],
                columns=y_df.columns
            )
            X_df_out = pd.concat([X_df_out, X_df_i])
            y_df_out = pd.concat([y_df_out, y_df_i])
    return X_df_out.reset_index(drop=True), y_df_out.reset_index(drop=True)


def extract_patches_patchify(df, patch_size):
    """
    Uses the Patchify package to extract patches in each samples of a dataset. The goal is then to be able to rebuild the dataset withs the patches
    Inputs : 
        df : a dataframe containing the samples (inputs or outputs)
        patch_size : the size of a side of the patch
    Output : 
        A new dataframe
    """
    channels = get_arrays_cols(df)
    img_h = df[channels[0]][0].shape[0]
    img_w = df[channels[0]][0].shape[1]

    # définition datafram vide de sortie:
    df_out = pd.DataFrame(
        [],
        columns = df.columns
    )

    step = min(gcd(patch_size, img_h), gcd(patch_size, img_w))

    for i in range(len(df)):
        # un dataframe par validité pour contenir les patchs
        df_i = pd.DataFrame(
            [],
            columns = []
        )
        for i_c, c in enumerate(channels):
            X = df[c][i]
            # extraction des patchs
            patches = patchify(X, (patch_size, patch_size), step=step)
            patches = patches.reshape((-1, patch_size, patch_size))

            # création d'un dataframe à une colonne contenant tous les patchs pour une validité et un channel donné
            values_c = []
            for j in range(patches.shape[0]):
                values_c.append([patches[j, :, :]])
            df_c = pd.DataFrame(
                values_c,
                columns=[c]
            )

            # concaténation de tous les channels pour une validité donnée
            df_i = pd.concat([df_i, df_c], axis=1)
            # ajout des dates et échéances
            dates_ech = pd.DataFrame([], columns = ['dates', 'echeances'])
            for _ in range(len(df_i)):
                dates_ech.loc[len(dates_ech)] = [df.dates[i], df.echeances[i]]

        df_i = pd.concat([dates_ech, df_i], axis=1)

        df_out = pd.concat([df_out, df_i], axis=0)

    return df_out.reset_index(drop=True)


def rebuild_from_patchify(df_patches, img_h, img_w):
    """
    Rebuilds the images given a dataset of patches
    Inputs:
        df_patches : a pandas dataframe containing the patches
        img_h : height of the full image
        img_w : width of the full image
    Output : 
        A pandas dataframe containing the full images
    """
    channels = get_arrays_cols(df_patches)
    patch_size = df_patches[channels[0]][0].shape[0]
    img_example = np.zeros((img_h, img_w))
    step = min(gcd(patch_size, img_h), gcd(patch_size, img_w))
    shape_patches = patchify(img_example, (patch_size, patch_size), step=step).shape

    df_out = pd.DataFrame(
        [],
        columns=df_patches.columns
    )

    dates = df_patches.dates.drop_duplicates().to_list()
    echeances = df_patches.echeances.drop_duplicates().to_list()

    for i_d, d in enumerate(dates):
        for i_ech, ech in enumerate(echeances):
            values = []
            for i_c, c in enumerate(channels):
                df_patches_i = df_patches.loc[(df_patches.echeances==ech) & (df_patches.dates==d)]
                patches = np.zeros((len(df_patches_i), patch_size, patch_size))
                for j in range(len(df_patches_i)):
                    patches[j, :, :] = df_patches_i[c].iloc[j]
                patches = patches.reshape(shape_patches)
                img = unpatchify(patches, (img_h, img_w))
                values.append(img)
            
            df_out_i = pd.DataFrame(
                [[d, ech] + values],
                columns = df_patches.columns
            )
            df_out = pd.concat([df_out, df_out_i], axis=0)

    return df_out.reset_index(drop=True)


# padding for patchify
def pad_for_patchify(df):
    df_out = df.copy()
    channels = get_arrays_cols(df)
    for c in channels:
        for i in range(len(df_out)):
            df_out[c][i] = np.pad(df_out[c][i], ((21,21), (27,26)), mode='reflect')
    return df_out

def crop_for_patchify(df):
    df_out = df.copy()
    channels = get_arrays_cols(df)
    for c in channels:
        for i in range(len(df_out)):
            df_out[c][i] = df_out[c][i][21:-21, 27:-26]
    return df_out

def get_n_patches(img_h, img_w, patch_size):
    img = np.zeros((img_h, img_w))
    step = min(gcd(patch_size, img_h), gcd(patch_size, img_w))
    patches = patchify(img,  (patch_size, patch_size), step=step)

    return step, patches.shape[0]*patches.shape[1]