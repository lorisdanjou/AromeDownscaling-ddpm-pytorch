import numpy as np
from numpy.random import randint
import pandas as pd
from preprocessing.load_data import get_arrays_cols


def random_flip(X_df, y_df, frac=0.1):
    """
    Extracts a fraction of the dataset and applies random flips on each sample
    """
    indexes = [randint(len(X_df)) for _ in np.arange(frac*len(X_df))]

    X_frac_df = X_df.loc[indexes].copy()
    y_frac_df = y_df.loc[indexes].copy()

    X_flip_df = pd.DataFrame([], columns=X_df.columns)
    y_flip_df = pd.DataFrame([], columns=y_df.columns)

    for i in range(len(X_frac_df)):
        values_X = [X_frac_df.dates.iloc[i], X_frac_df.echeances.iloc[i]]
        values_y = [y_frac_df.dates.iloc[i], y_frac_df.echeances.iloc[i]]
        rand = randint(3)
        for i_c, c in enumerate(get_arrays_cols(X_frac_df)):
            if rand == 0:
                X_c_flip = X_frac_df[c].iloc[i][::, ::-1]
            elif rand == 1:
                X_c_flip = X_frac_df[c].iloc[i][::-1, ::]
            elif rand == 2:
                X_c_flip = X_frac_df[c].iloc[i][::-1, ::-1]

            values_X.append(X_c_flip)
        X_flip_df.loc[len(X_flip_df)] = values_X

        for i_c, c in enumerate(get_arrays_cols(y_frac_df)):
            if rand == 0:
                y_c_flip = y_frac_df[c].iloc[i][::, ::-1]
            elif rand == 1:
                y_c_flip = y_frac_df[c].iloc[i][::-1, ::]
            elif rand == 2:
                y_c_flip = y_frac_df[c].iloc[i][::-1, ::-1]
            values_y.append(y_c_flip)
        y_flip_df.loc[len(y_flip_df)] = values_y

    y_flip_df.head()

    X_aug_df = pd.concat([X_df, X_flip_df], axis=0).reset_index(drop=True)
    y_aug_df = pd.concat([y_df, y_flip_df], axis=0).reset_index(drop=True)

    return X_aug_df, y_aug_df


def random_rot(X_df, y_df, frac=0.1):
    """
    Extracts a fraction of the dataset and applies rotations on each sample
    """
    indexes = [randint(len(X_df)) for _ in np.arange(frac*len(X_df))]

    X_frac_df = X_df.loc[indexes]
    y_frac_df = y_df.loc[indexes]

    X_rot_df = pd.DataFrame([], columns=X_df.columns)
    y_rot_df = pd.DataFrame([], columns=y_df.columns)

    for i in range(len(X_frac_df)):
        values_X = [X_frac_df.dates.iloc[i], X_frac_df.echeances.iloc[i]]
        values_y = [y_frac_df.dates.iloc[i], y_frac_df.echeances.iloc[i]]
        for i_c, c in enumerate(get_arrays_cols(X_frac_df)):
            X_c_rot = np.rot90(X_frac_df[c].iloc[i], k=2)

            values_X.append(X_c_rot)
        X_rot_df.loc[len(X_rot_df)] = values_X

        for i_c, c in enumerate(get_arrays_cols(y_frac_df)):
            y_c_rot = np.rot90(y_frac_df[c].iloc[i], k=2)
            values_y.append(y_c_rot)
        y_rot_df.loc[len(y_rot_df)] = values_y

    y_rot_df.head()

    X_aug_df = pd.concat([X_df, X_rot_df], axis=0).reset_index(drop=True)
    y_aug_df = pd.concat([y_df, y_rot_df], axis=0).reset_index(drop=True)
    
    return X_aug_df, y_aug_df