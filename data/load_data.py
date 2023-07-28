import numpy as np
from os.path import exists
import pandas as pd
import utils


def load_X(dates, echeances, params, data_location, data_static_location='', static_fields=[], resample='r'):
    """
    Loads all the inputs of the NN in a pandas dataframe
    Inputs:
    Outputs : pansdas dataframe
    """
    if resample in ['bl', 'bc']:
        domain_shape = [205, 260]
    elif resample =='r':
        domain_shape = utils.get_shape_500m()
    else:
        domain_shape = utils.get_shape_2km5()
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
    domain_shape_out = utils.get_shape_500m()
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