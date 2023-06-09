import numpy as np
from os.path import exists
import pandas as pd
from bronx.stdtypes.date import daterangex as rangex


def get_shape_500m():
    field_500m = np.load('/cnrm/recyf/Data/users/danjoul/dataset/data_train/G9KP_2021-01-01T00:00:00Z_rr.npy')
    return field_500m[:, :, 0].shape


def get_shape_2km5(resample='c'):
    if resample == 'c':
        field_2km5 = np.load('/cnrm/recyf/Data/users/danjoul/dataset/data_train/oper_c_2021-01-01T00:00:00Z_rr.npy')
        return field_2km5[:, :, 0].shape
    else:
        field_2km5 = np.load('/cnrm/recyf/Data/users/danjoul/dataset/data_train/oper_r_2021-01-01T00:00:00Z_rr.npy')
        return field_2km5[:, :, 0].shape


def load_X(dates, echeances, params, data_location, data_static_location='', static_fields=[], resample='r'):
    """
    Loads all the inputs of the NN in a pandas dataframe
    Inputs:
    Outputs : pansdas dataframe
    """
    if resample in ['bl', 'bc']:
        domain_shape = [205, 260]
    elif resample =='r':
        domain_shape = get_shape_500m()
    else:
        domain_shape = get_shape_2km5()
    data = pd.DataFrame(
        [], 
        columns = ['dates', 'echeances'] + params
    )
    for i_d, d in enumerate(dates):
        # chargement des données
        X_d = np.zeros([len(echeances), domain_shape[0], domain_shape[1], len(params) + len(static_fields)], dtype=np.float32)
        try:
            for i_p, p in enumerate(params):
                if resample == 'c':
                    filepath_X = data_location + 'oper_c_' + d.isoformat() + 'Z_' + p + '.npy'
                    X_d[:, :, :, i_p] = np.load(filepath_X).transpose([2, 0, 1])
                elif resample == 'r':
                    filepath_X = data_location + 'oper_r_' + d.isoformat() + 'Z_' + p + '.npy'
                    X_d[:, :, :, i_p] = np.load(filepath_X).transpose([2, 0, 1])
                elif resample == 'bl':
                    filepath_X = data_location + 'oper_bl_' + d.isoformat() + 'Z_' + p + '.npy'
                    X_d[:, :, :, i_p] = np.load(filepath_X).transpose([2, 0, 1])
                elif resample == 'bc':
                    filepath_X = data_location + 'oper_bc_' + d.isoformat() + 'Z_' + p + '.npy'
                    X_d[:, :, :, i_p] = np.load(filepath_X).transpose([2, 0, 1])
                else:
                    raise ValueError("resample mal défini")
        # champs statiques
        # ! erreur chargement bc bl des champs statiques à corriger
            for i_s, s in enumerate(static_fields):
                if resample in ['r', 'bl', 'bc']:
                    filepath_static = data_static_location + 'static_G9KP_' + s + '.npy'
                elif resample =='c':
                    filepath_static = data_static_location + 'static_oper_c_' + s + '.npy'
                else:
                    raise ValueError("resample mal défini")
                
                for i_ech, ech in enumerate(echeances):
                    X_d[i_ech, :, :, len(params) + i_s] = np.load(filepath_static)

            if resample in ['bl', 'bc']:
                X_d = np.pad(X_d, ((0,0), (5,4), (2,5), (0,0)), mode='edge')
        except FileNotFoundError:
            print('missing day (X): ' + d.isoformat())
            X_d = None
        
        for i_ech, ech in enumerate(echeances):
            values_i = []
            try:
                for i_p, p in enumerate(params + static_fields):
                    values_i.append(X_d[i_ech, :, :, i_p])
                data_i = pd.DataFrame(
                    [[d.isoformat(), ech] + values_i], 
                    columns = ['dates', 'echeances'] + params + static_fields
                )
            except TypeError:
                for i_p, p in enumerate(params + static_fields):
                    values_i.append(None)
                data_i = pd.DataFrame(
                    [[d.isoformat(), ech] + values_i], 
                    columns = ['dates', 'echeances'] + params + static_fields
                )

            data = pd.concat([data, data_i])
    return data.reset_index(drop=True)


def load_y(dates, echeances, params, data_location):
    """
    Loads all the outputs of the NN in a pandas dataframe
    Inputs:
    Outputs : pansdas dataframe
    """
    data = pd.DataFrame(
        [], 
        columns = ['dates', 'echeances'] + params
    )
    domain_shape_out = get_shape_500m()
    for i_d, d in enumerate(dates):
        # chargement des données
        y_d = np.zeros([len(echeances), domain_shape_out[0], domain_shape_out[1], len(params)], dtype=np.float32)
        try:
            for i_p, p in enumerate(params):
                filepath_y = data_location + 'G9L1_' + d.isoformat() + 'Z_' + p + '.npy'
                if exists(filepath_y):
                    y_d[:, :, :, i_p] = np.load(filepath_y).transpose([2, 0, 1])
                else:
                    filepath_y = data_location + 'G9KP_' + d.isoformat() + 'Z_' + p + '.npy'
                    y_d[:, :, :, i_p] = np.load(filepath_y).transpose([2, 0, 1])
        except FileNotFoundError:
            print('missing day (y): ' + d.isoformat())
            y_d = None
        
        # création du dataframe
        for i_ech, ech in enumerate(echeances):
            values_i = []
            try:
                for i_p, p in enumerate(params):
                    values_i.append(y_d[i_ech, :, :, i_p])
                data_i = pd.DataFrame(
                    [[d.isoformat(), ech] + values_i], 
                    columns = ['dates', 'echeances'] + params
                )
            except TypeError:
                for i_p, p in enumerate(params):
                    values_i.append(None)
                data_i = pd.DataFrame(
                    [[d.isoformat(), ech] + values_i], 
                    columns = ['dates', 'echeances'] + params
                )

            data = pd.concat([data, data_i])
    return data.reset_index(drop=True)


def delete_missing_days(X_df, y_df):
    """
    Deletes all the missing days in both X and y
    Inputs :
        X_df : a pandas dataframe representing X 
        y_df : a pandas dataframe representing y 
    Outputs:
        X_df : a pandas dataframe representing X without missing days
        y_df : a pandas dataframe representing y without missing days
    """
    X_df_out = X_df.copy()
    y_df_out = y_df.copy()
    nan_indices_y = y_df_out[y_df_out.isna().any(axis=1)].index
    y_df_out = y_df_out.drop(index=nan_indices_y, axis = 0)
    X_df_out = X_df_out.drop(index=nan_indices_y, axis = 0)
    nan_indices_X = X_df_out[X_df_out.isna().any(axis=1)].index
    X_df_out = X_df_out.drop(index=nan_indices_X, axis = 0)
    y_df_out = y_df_out.drop(index=nan_indices_X, axis = 0)

    return X_df_out.reset_index(drop=True), y_df_out.reset_index(drop=True)


def get_arrays_cols(df):
    arrays_cols = []
    for c in df.columns:
        if type(df[c].iloc[0]) == np.ndarray:
            arrays_cols.append(c)
    return arrays_cols


def pad(df):
    """
    Pad the data in order to make its size compatible with the unet
    Input : dataframe
    Output : dataframe with padding
    """
    df_out = df.copy()
    arrays_cols = get_arrays_cols(df_out)
    for c in arrays_cols:
        for i in range(len(df_out)):
            df_out[c][i] = np.pad(df_out[c][i], ((5,5), (2,3)), mode='reflect')
    return df_out


def crop(df):
    """
    Crop the data
    Input : dataframe
    Output : cropped dataframe
    """
    df_out = df.copy()
    for c in df_out.columns:
        if type(df_out[c][0]) == np.ndarray:
            for i in range(len(df_out)):
                df_out[c][i] = df_out[c][i][5:-5, 2:-3]
    return df_out


def param_to_array(arrays_serie):
    """
    Transforms a pandas series into a big numpy array of shape B x H x W
    """
    array = np.zeros((len(arrays_serie), arrays_serie[0].shape[0], arrays_serie[0].shape[1]), dtype=np.float32)
    for i in range(len(arrays_serie)):
        array[i, :, :] = arrays_serie[i]
    return array


def df_to_array(df):
    """
    transforms a pandas dataframe into a big numpy array of shape B x H x W x C
    """
    arrays_cols = get_arrays_cols(df)
            
    array = np.zeros((len(df), df[arrays_cols[0]].iloc[0].shape[0], df[arrays_cols[0]].iloc[0].shape[1], len(arrays_cols)), dtype=np.float32)

    for i in range(len(df)):
        for i_c, c in enumerate(arrays_cols):
            array[i, :, :, i_c] = df[c].iloc[i]
    return array