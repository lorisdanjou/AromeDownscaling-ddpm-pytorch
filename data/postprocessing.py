import numpy as np
import pandas as pd
import data.load_data as ld

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
    channels = ld.get_arrays_cols(df)
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

    # load y_pred

    # postprocess y_pred

    # save y_pred