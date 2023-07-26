import os
import numpy as np
import pandas as pd
import utils
from datetime import datetime, timedelta


def load_ensemble_ddpm(working_dir, n_members, params):
    """
    Loads the results of an ensemble

    Args:
        working_dir (str): filepath to the root of the ensemble
        n_members (int): number of members in the ensemble
        params (list): parameters (str) predicted by the model

    Returns:
        DataFrame: a dataframe that contains all the members for all the parameters in the ensemble
    """
    ens_df = pd.DataFrame(
        [],
        columns = []
    )

    for i_m in range(n_members):
        path = os.path.join(
            working_dir, str(i_m + 1), "y_pred.csv"
        )
        y_m = pd.read_pickle(path)
        if i_m == 0:
            y_m = y_m.rename(columns={p:p + "_" + str(i_m + 1) for p in params})
        else:
            y_m = y_m.rename(columns={p:p + "_" + str(i_m + 1) for p in params}).drop(columns=["dates", "echeances"])
        
        ens_df = pd.concat([ens_df, y_m], axis=1)
    
    return ens_df


def select_indices_echeances(full_ech, real_ech):
    """
    Selects the good indices corresponding to the good echeances
    """
    indices = []
    for ech in real_ech:
        for i_fech, fech in enumerate(full_ech):
            if ech == fech:
                indices.append(i_fech)
                break
    return indices


def load_ensemble_arome(dates, echeances, params, n_members, data_location):
    """
    Loads a dataframe containing results of a PE-Arome ensemble (2,5km)

    Args:
        dates (list): list of dates (str)
        echeances (list): list of echeances (int)
        params (list): list of parameters
        n_members (int): number of members
        data_location (str): filepath to directory containing .npy files

    Returns:
        DataFrame: dataframe containing all the needed fields
    """
    ens_df = pd.DataFrame(
        [], 
        columns = ['dates', 'echeances'] + [p + "_" + str(j + 1) for p in params for j in range(n_members)]
    )
    domain_shape = utils.get_shape_2km5()
    
    for i_d, d in enumerate(dates):
        ens_d = np.zeros((n_members, len(echeances), domain_shape[0], domain_shape[1], len(params)))
        # charger tous les membres de tous les paramètres
        try:
            for i_m in range(n_members):
                for i_p, p in enumerate(params):
                    filepath = data_location + str(i_m + 1) + '/GC81_' + d + 'Z_' + p + '.npy'
                    ens_d[i_m, :, :, :, i_p] = np.load(filepath).transpose([2, 0, 1])
        except FileNotFoundError:
            print('missing day : ' + d)
            ens_d = None
        
        if ens_d is not None:
            for i_ech, ech in enumerate(echeances):
                values = [ens_d[i_m, i_ech, :, :, i_p] for i_p in range(len(params)) for i_m in range(n_members)]
                ens_df.loc[len(ens_df)] = [d, ech] + values
        
    return ens_df


def correct_dates_for_arome(arome_ens_df):
    """
    Corrects dates for the real Arome ensemble

    Args:
        arome_ens_df (DataFrame): Arome ensemble data loaded with load_ensemble_arome()

    Returns:
        DataFrame: dataframe with corrected dates and echeances (to match the situations with the ddpm ensemble)
    """
    arome_ens_corrected = pd.DataFrame(
        [],
        columns=arome_ens_df.columns
    )
    channels = utils.get_arrays_cols(arome_ens_df)
    delta = timedelta(hours = 6)
    for i in range(len(arome_ens_df)):
        values = [arome_ens_df[channels[j]].iloc[i] for j in range(len(channels))]
        date_ech = [
            (datetime.fromisoformat(arome_ens_df.dates.iloc[i]) + delta).isoformat(),
            arome_ens_df.echeances.iloc[i] - 6
        ]
        arome_ens_corrected.loc[len(arome_ens_corrected)] = date_ech + values 
    return arome_ens_corrected


def add_input_output(ens_df, resample, data_test_location, params):
    """
    Adds inputs and outputs used to generate the ensemble to the results

    Args:
        ens_df (DataFrame): contains the results
        resample (str): interpolation
        data_test_location (st): filepath to the dataset
        params (list): list of params (str)

    Raises:
        NotImplementedError: 

    Returns:
        DataFrame: contains the results and the inputs/outputs
    """
    dates = ens_df.dates.drop_duplicates().values

    input_output_df = pd.DataFrame(
        [],
        columns = ["dates", "echeances"] + [p + "_X" for p in params] + [p + "_y" for p in params]
    )

    for i_d, d in enumerate(dates):
        echeances = ens_df[ens_df.dates == d].echeances.drop_duplicates().values
        values_X = []
        values_y = []
        for param in params:
            # Load X
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

                X = np.load(filepath_X_test)
                if resample in ['bl', 'bc']:
                    X = np.pad(X, ((5,4), (2,5), (0,0)), mode='edge')
                values_X.append(X)
            except FileNotFoundError:
                print('missing day (X): ' + d)

            # Load y
            try:
                filepath_y_test = data_test_location + 'G9L1_' + d + 'Z_' + param + '.npy'
                y = np.load(filepath_y_test)
                values_y.append(y)
            except FileNotFoundError:
                print('missing day (y): ' + d)
        
        values_X = np.array(values_X)
        values_y = np.array(values_y)
        indices = select_indices_echeances(utils.FULL_ECHEANCES, echeances)

        for i_ech, ech in enumerate(echeances):
            if (len(values_X) == len(params)) and (len(values_y) == len(params)):
                values_X_ech = [values_X[i][:, :, indices[i_ech]] for i in range(len(values_X))]
                values_y_ech = [values_y[i][:, :, indices[i_ech]] for i in range(len(values_y))]
                input_output_df.loc[len(input_output_df)] = [dates[i_d], echeances[i_ech]] + \
                    values_X_ech + values_y_ech

    return pd.merge(ens_df, input_output_df, how="inner", on=["dates", "echeances"])#.dropna()#.reset_index(drop=True)


def group_ensembles(arome_df, ddpm_df):
    """
    Groups two DataFrame containing ensemble samples / stats into a single DataFrame
    """
    unique_df = pd.DataFrame(
        [],
        columns=[]
    )

    unique_df = ddpm_df.merge(arome_df, how="outer", on=["dates", "echeances"], suffixes=("_ddpm", "_arome"))

    return unique_df.dropna().reset_index(drop=True)