import numpy as np
import pandas as pd
from data.load_data import get_arrays_cols, param_to_array



# Normalisation between -1 and 1
def get_max_abs(X_df, working_dir):
    """
    Get the absolute maximun of the columns containing arrays of X
    ! the channels of X must begin by the channels of y (in the same order)
    Input : X dataframe
    Output: List of absolute maximums 
    """
    max_abs_out = []
    arrays_cols = get_arrays_cols(X_df)
    for i_c, c in enumerate(arrays_cols):
        X_c = param_to_array(X_df[c])
        max_abs_out.append(np.abs(X_c).max())
    np.save(working_dir + 'max_abs_X.npy', max_abs_out, allow_pickle=True)

def normalisation(df, working_dir):
    """
    Normalise a dataframe between -1 and 1
    Input : X or y dataframe
    Output : normalised copy of the dataframe
    """
    df_norm = df.copy()
    arrays_cols = get_arrays_cols(df_norm)
    max_abs_X = np.load(working_dir + 'max_abs_X.npy')
    for i in range(len(df_norm)):
        for i_c, c in enumerate(arrays_cols):
            df_norm[c][i] = df_norm[c][i] / max_abs_X[i_c]
    return df_norm

def denormalisation(df, working_dir):
    """
    Denormalise a dataframe
    Input : X or y dataframe
    Output : denormalised copy of the dataframe
    """
    df_den = df.copy()
    arrays_cols = get_arrays_cols(df_den)
    max_abs_X = np.load(working_dir + 'max_abs_X.npy')
    for i in range(len(df_den)):
        for i_c, c in enumerate(arrays_cols):
            df_den[c][i] = df_den[c][i] * max_abs_X[i_c]
    return df_den


# Standardisation
def get_mean(X_df, working_dir):
    """
    Get the mean of the columns containing arrays of X
    ! the channels of X must begin by the channels of y (in the same order)
    Input : X dataframe
    Output: List of means
    """
    mean_out_X = []
    arrays_cols_X = get_arrays_cols(X_df)
    for i_c, c in enumerate(arrays_cols_X):
        X_c = param_to_array(X_df[c])
        mean_out_X.append(X_c.mean())
    np.save(working_dir + 'mean_X.npy', mean_out_X, allow_pickle=True)


def get_std(X_df, working_dir):
    """
    Get the standard deviations of the columns containing arrays of X
    ! the channels of X must begin by the channels of y (in the same order)
    Input : X dataframe
    Output: List of standard deviations
    """
    std_out_X = []
    arrays_cols_X = get_arrays_cols(X_df)
    for i_c, c in enumerate(arrays_cols_X):
        X_c = param_to_array(X_df[c])
        std_out_X.append(X_c.std())
    np.save(working_dir + 'std_X.npy', std_out_X, allow_pickle=True)


def standardisation(X_df, working_dir):
    """
    Standardise a dataframe (* - mean) / std
    Input : X or y dataframe
    Output : standardised copy of the dataframe
    """
    X_df_norm = X_df.copy()
    arrays_cols = get_arrays_cols(X_df_norm)
    mean_X = np.load(working_dir + 'mean_X.npy')
    std_X = np.load(working_dir + 'std_X.npy')
    for i in range(len(X_df_norm)):
        for i_c, c in enumerate(arrays_cols):
            if std_X[i_c] < 1e-9:
                raise ValueError('std = 0') 
            X_df_norm[c][i] = (X_df_norm[c][i] - mean_X[i_c]) / std_X[i_c]
    return X_df_norm


def destandardisation(df, working_dir):
    """
    Destandardise a dataframe
    Input : X or y dataframe
    Output : destandardised copy of the dataframe
    """
    df_norm = df.copy()
    arrays_cols = get_arrays_cols(df_norm)
    mean_X = np.load(working_dir + 'mean_X.npy')
    std_X = np.load(working_dir + 'std_X.npy')
    for i in range(len(df_norm)):
        for i_c, c in enumerate(arrays_cols):
            df_norm[c][i] = (df_norm[c][i] * std_X[i_c]) + mean_X[i_c]
    return df_norm


# def get_mean_both(X_df, y_df, working_dir):
#     mean_out_X = []
#     arrays_cols_X = get_arrays_cols(X_df)
#     for i_c, c in enumerate(arrays_cols_X):
#         X_c = param_to_array(X_df[c])
#         mean_out_X.append(X_c.mean())
#     np.save(working_dir + 'mean_X.npy', mean_out_X, allow_pickle=True)
#     mean_out_y = []
#     arrays_cols_y = get_arrays_cols(y_df)
#     for i_c, c in enumerate(arrays_cols_y):
#         y_c = param_to_array(y_df[c])
#         mean_out_y.append(y_c.mean())
#     np.save(working_dir + 'mean_y.npy', mean_out_y, allow_pickle=True)


# def get_std_both(X_df, y_df, working_dir):
#     std_out_X = []
#     arrays_cols_X = get_arrays_cols(X_df)
#     for i_c, c in enumerate(arrays_cols_X):
#         X_c = param_to_array(X_df[c])
#         std_out_X.append(X_c.std())
#     np.save(working_dir + 'std_X.npy', std_out_X, allow_pickle=True)
#     std_out_y = []
#     arrays_cols_y = get_arrays_cols(y_df)
#     for i_c, c in enumerate(arrays_cols_y):
#         y_c = param_to_array(y_df[c])
#         std_out_y.append(y_c.std())
#     np.save(working_dir + 'std_y.npy', std_out_y, allow_pickle=True)


# def standardisation_both(X_df, y_df, working_dir):
#     X_df_norm = X_df.copy()
#     arrays_cols_X = get_arrays_cols(X_df_norm)
#     mean_X = np.load(working_dir + 'mean_X.npy')
#     std_X = np.load(working_dir + 'std_X.npy')
#     for i in range(len(X_df_norm)):
#         for i_c, c in enumerate(arrays_cols_X):
#             if std_X[i_c] < 1e-9:
#                 raise ValueError('std = 0') 
#             X_df_norm[c][i] = (X_df_norm[c][i] - mean_X[i_c]) / std_X[i_c]
#     y_df_norm = y_df.copy()
#     arrays_cols_y = get_arrays_cols(y_df_norm)
#     mean_y = np.load(working_dir + 'mean_y.npy')
#     std_y = np.load(working_dir + 'std_y.npy')
#     for i in range(len(y_df_norm)):
#         for i_c, c in enumerate(arrays_cols_y):
#             if std_y[i_c] < 1e-9:
#                 raise ValueError('std = 0') 
#             y_df_norm[c][i] = (y_df_norm[c][i] - mean_y[i_c]) / std_y[i_c]
#     return X_df_norm, y_df_norm


# def destandardisation_both(X_df, y_df, working_dir):
#     X_df_norm = X_df.copy()
#     arrays_cols_X = get_arrays_cols(X_df_norm)
#     mean_X = np.load(working_dir + 'mean_X.npy')
#     std_X = np.load(working_dir + 'std_X.npy')
#     for i in range(len(X_df_norm)):
#         for i_c, c in enumerate(arrays_cols_X):
#             X_df_norm[c][i] = (X_df_norm[c][i] * std_X[i_c]) + mean_X[i_c]
#     y_df_norm = y_df.copy()
#     arrays_cols_y = get_arrays_cols(y_df_norm)
#     mean_y = np.load(working_dir + 'mean_y.npy')
#     std_y = np.load(working_dir + 'std_y.npy')
#     for i in range(len(y_df_norm)):
#         for i_c, c in enumerate(arrays_cols_y):
#             y_df_norm[c][i] = (y_df_norm[c][i] * std_y[i_c]) + mean_y[i_c]
#     return X_df_norm, y_df_norm


# def get_mean_df(df):
#     mean_out = []
#     arrays_cols = get_arrays_cols(df)
#     for i_c, c in enumerate(arrays_cols):
#         array_c = param_to_array(df[c])
#         mean_out.append(array_c.mean())
#     return mean_out


# def get_std_df(df):
#     std_out = []
#     arrays_cols = get_arrays_cols(df)
#     for i_c, c in enumerate(arrays_cols):
#         array_c = param_to_array(df[c])
#         std_out.append(array_c.std())
#     return std_out


# def standardisation_df(df, mean, std):
#     df_norm = df.copy()
#     arrays_cols = get_arrays_cols(df_norm)
#     for i in range(len(df_norm)):
#         for i_c, c in enumerate(arrays_cols):
#             if std[i_c] < 1e-9:
#                 raise ValueError('std = 0') 
#             df_norm[c][i] = (df_norm[c][i] - mean[i_c]) / std[i_c]
#     return df_norm


# def destandardisation_df(df, mean, std):
#     df_norm = df.copy()
#     arrays_cols = get_arrays_cols(df_norm)
#     for i in range(len(df_norm)):
#         for i_c, c in enumerate(arrays_cols):
#             if std[i_c] < 1e-9:
#                 raise ValueError('std = 0') 
#             df_norm[c][i] = df_norm[c][i] * std[i_c] + mean[i_c]
#     return df_norm


def standardisation_sample(df):
    df_norm = df.copy()
    columns_mean_std = []
    arrays_cols = get_arrays_cols(df_norm)
    for i_c, c in enumerate(arrays_cols):
        columns_mean_std.append('mean_' + c)
        columns_mean_std.append('std_' + c)
    df_mean_std = pd.DataFrame(
        [], 
        columns = columns_mean_std
    )
    for i in range(len(df_norm)):
        means_stds = []
        for i_c, c in enumerate(arrays_cols):
            mean_c = df_norm[c][i].mean()
            std_c  = df_norm[c][i].std()
            print(std_c)
            means_stds.append(mean_c)
            means_stds.append(std_c)
            df_norm[c][i] = (df_norm[c][i] - mean_c) / std_c
        df_mean_std_i = pd.DataFrame(
            [means_stds], 
            columns = columns_mean_std
        )
        df_mean_std = pd.concat([df_mean_std, df_mean_std_i])    
    df_mean_std = df_mean_std.reset_index(drop=True)
    df_norm = pd.concat([df_norm, df_mean_std], axis=1)
    return df_norm

def destandardisation_sample(df):
    df_den = df.copy()
    arrays_cols = get_arrays_cols(df_den)
    for i in range(len(df_den)):
        for i_c, c in enumerate(arrays_cols):
            mean_c = df_den['mean_' + c][i]
            std_c  = df_den['std_' + c][i]
            df_den[c][i] = df_den[c][i] * std_c + mean_c 
    cols_to_drop = []
    for c in arrays_cols:
        cols_to_drop.append('mean_' + c)
        cols_to_drop.append('std_' + c)
    return df_den.drop(cols_to_drop, axis=1)


# MinMax normalisation
def get_min(X_df, working_dir):
    """
    Get the min of the columns containing arrays of X
    ! the channels of X must begin by the channels of y (in the same order)
    Input : X dataframe
    Output: List of mins
    """
    min_out = []
    arrays_cols = get_arrays_cols(X_df)
    for i_c, c in enumerate(arrays_cols):
        X_c = param_to_array(X_df[c])
        min_out.append(X_c.min())
    np.save(working_dir + 'min_X.npy', min_out, allow_pickle=True)


def get_max(X_df, working_dir):
    """
    Get the max of the columns containing arrays of X
    ! the channels of X must begin by the channels of y (in the same order)
    Input : X dataframe
    Output: List of maxs
    """
    max_out = []
    arrays_cols = get_arrays_cols(X_df)
    for i_c, c in enumerate(arrays_cols):
        X_c = param_to_array(X_df[c])
        max_out.append(X_c.max())
    np.save(working_dir + 'max_X.npy', max_out, allow_pickle=True)


def min_max_norm(df, working_dir):
    """
    Normalise a dataframe
    Input : X or y dataframe
    Output : min-max normalised copy of the dataframe
    """
    df_norm = df.copy()
    arrays_cols = get_arrays_cols(df_norm)
    min_X = np.load(working_dir + 'min_X.npy')
    max_X = np.load(working_dir + 'max_X.npy')
    for i in range(len(df_norm)):
        for i_c, c in enumerate(arrays_cols):
            if (max_X[i_c] - min_X[i_c]) < 1e-9:
                raise ValueError('min - max = 0') 
            df_norm[c][i] = (df_norm[c][i] - min_X[i_c]) / (max_X[i_c] - min_X[i_c])
    return df_norm


def min_max_denorm(df, working_dir):
    """
    Destandardise a dataframe
    Input : X or y dataframe
    Output : min-max denormalised copy of the dataframe
    """
    df_norm = df.copy()
    arrays_cols = get_arrays_cols(df_norm)
    min_X = np.load(working_dir + 'min_X.npy')
    max_X = np.load(working_dir + 'max_X.npy')
    for i in range(len(df_norm)):
        for i_c, c in enumerate(arrays_cols):
            df_norm[c][i] = df_norm[c][i]  * (max_X[i_c] - min_X[i_c]) + min_X[i_c]
    return df_norm


# Mean normalisation

def mean_norm(df, working_dir):
    """
    Normalise a dataframe
    Input : X or y dataframe
    Output : mean normalised copy of the dataframe
    """
    df_norm = df.copy()
    arrays_cols = get_arrays_cols(df_norm)
    min_X = np.load(working_dir + 'min_X.npy')
    max_X = np.load(working_dir + 'max_X.npy')
    mean_X =np.load(working_dir + 'mean_X.npy')
    for i in range(len(df_norm)):
        for i_c, c in enumerate(arrays_cols):
            if (max_X[i_c] - min_X[i_c]) < 1e-9:
                raise ValueError('min - max = 0') 
            df_norm[c][i] = (df_norm[c][i] - mean_X[i_c]) / (max_X[i_c] - min_X[i_c])
    return df_norm


def mean_denorm(df, working_dir):
    """
    Destandardise a dataframe
    Input : X or y dataframe
    Output : mean denormalised copy of the dataframe
    """
    df_norm = df.copy()
    arrays_cols = get_arrays_cols(df_norm)
    min_X = np.load(working_dir + 'min_X.npy')
    max_X = np.load(working_dir + 'max_X.npy')
    mean_X =np.load(working_dir + 'mean_X.npy')
    for i in range(len(df_norm)):
        for i_c, c in enumerate(arrays_cols):
            df_norm[c][i] = df_norm[c][i]  * (max_X[i_c] - min_X[i_c]) + mean_X[i_c]
    return df_norm