import numpy as np
import pandas as pd
from bronx.stdtypes.date import daterangex as rangex
from data.load_data import load_X, load_y, get_arrays_cols

def min_max_scale(x, x_ref):
    normalized_x = (x - x.min()) / (x.max() - x.min())
    return normalized_x * (x_ref.max() - x_ref.min()) + x_ref.min()

def mean_std_scale(x, x_ref):
    normalized_x = (x - x.mean()) / x.std()
    return normalized_x * x_ref.std() + x_ref.mean()

def postprocess_df(df, df_ref, postproc_opt):
    out_df = pd.DataFrame(
        [],
        columns=df.columns
    )
    channels = get_arrays_cols(df)
    for i in range(len(df)):
        row = [df.dates.iloc[i], df.echeances.iloc[i]]
        for i_c, c in enumerate(channels):
            ref = df_ref[df_ref.dates == row[0]][df_ref.echeances == row[1]][c].iloc[0] 
            if postproc_opt["method"] == "minmax":
                postprocessed_array = min_max_scale(df[c].iloc[i], ref)
            elif postproc_opt["method"] == "std":
                postprocessed_array = mean_std_scale(df[c].iloc[i], ref)
            row.append(postprocessed_array)
        out_df.loc[len(out_df)] = row
    return out_df


if __name__ == "__main__":
    import argparse
    import json
    from collections import OrderedDict
    import os
    import warnings
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_ddpm.jsonc',
                        help='JSON file for configuration')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)

    # parse configs
    args = parser.parse_args()
    opt_path = args.config

    # remove comments starting with '//'
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)


    # load X_test & y_test
    data_train_location  = opt["data_loading"]["data_train_location"]
    data_valid_location  = opt["data_loading"]["data_valid_location"]
    data_test_location   = opt["data_loading"]["data_test_location"]
    data_static_location = opt["data_loading"]["data_static_location"]
    dates_train          = rangex(opt["data_loading"]["dates_train"])
    dates_valid          = rangex(opt["data_loading"]["dates_valid"])
    dates_test           = rangex(opt["data_loading"]["dates_test"])
    echeances            = opt["data_loading"]["echeances"]
    params_in            = opt["data_loading"]["params_in"]
    params_out           = opt["data_loading"]["params_out"]
    static_fields        = opt["data_loading"]["static_fields"]
    interp               = opt["data_loading"]["interp"]

    if opt["data_loading"]["config"] == "optimisation": # the test dataset is not used
        X_test_df = load_X(
            dates_valid, 
            echeances,
            params_in,
            data_valid_location,
            data_static_location,
            static_fields = static_fields,
            resample=interp
        )

        y_test_df = load_y(
            dates_valid,
            echeances,
            params_out,
            data_valid_location
        )

    elif opt["data_loading"]["config"] =="test": # the whole dataset is used
        X_test_df = load_X(
            dates_test, 
            echeances,
            params_in,
            data_test_location,
            data_static_location,
            static_fields = static_fields,
            resample=interp
        )

        y_test_df = load_y(
            dates_test,
            echeances,
            params_out,
            data_test_location
        )

    else:
        raise NotImplementedError

    # load y_pred
    y_pred_path = os.path.join(opt["path"]["working_dir"], "y_pred.csv")
    y_pred_df = pd.read_pickle(y_pred_path)

    # postprocess y_pred
    postproc_opt = opt["postprocessing"]
    y_pred_postproc_df = postprocess_df(y_pred_df, X_test_df, postproc_opt)

    # save y_pred
    y_postproc_path = os.path.join(opt["path"]["working_dir"], "y_pred_postproc.csv")
    pd.to_pickle(y_pred_postproc_df, y_postproc_path)