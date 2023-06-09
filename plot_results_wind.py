import numpy as np
import pandas as pd
from results.results import *
from bronx.stdtypes.date import daterangex as rangex
import matplotlib.pyplot as plt
from matplotlib import colors
import warnings

warnings.filterwarnings("ignore")

data_train_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_train/'
data_valid_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_test/'
data_test_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_test/'
data_static_location = '/cnrm/recyf/Data/users/danjoul/dataset/'
baseline_location = '/cnrm/recyf/Data/users/danjoul/dataset/baseline/test/'

# ========== Setup
# params = ["t2m", "rr", "rh2m", "tpw850", "ffu", "ffv", "tcwv", "sp", "cape", "hpbl", "ts", "toa","tke","u700","v700","u500","v500", "u10", "v10"]
params = ['t2m']
static_fields = []
dates_train = rangex([
    '2020070100-2021053100-PT24H'
])
dates_valid = rangex([
    '2021080100-2021083100-PT24H',
    '2021100100-2021103100-PT24H',
    '2021100100-2021123100-PT24H',
    '2022020100-2022022800-PT24H',
    '2022040100-2022043000-PT24H',
    '2022060100-2022063000-PT24H'
])
dates_test = rangex([
    '2021070100-2021073100-PT24H',
    '2021090100-2021093000-PT24H',
    '2021110100-2021113000-PT24H',
    '2022030100-2022033100-PT24H',
    '2022050100-2022053100-PT24H'
])
echeances = range(6, 37, 3)

resample = 'r'
echeances = range(6, 37, 3)
working_dir = '/cnrm/recyf/Data/users/danjoul/unet_experiments/wind/losses/custom_loss/0.6-5/'


# ========== Load Data
results_df_u = load_results(working_dir, dates_valid, echeances, resample, data_test_location, baseline_location, param='u10')
results_df_v = load_results(working_dir, dates_valid, echeances, resample, data_test_location, baseline_location, param='v10')


# ========== U
# ========== Plots
plot_results(results_df_u, 'u10', working_dir)
plot_unique_score_map(results_df_u, mae, 'mae', working_dir)
plot_unique_score_map(results_df_u, biais, 'biais', working_dir)
plot_distrib(results_df_u, mse, 'mse', working_dir)
plot_distrib(results_df_u, mae, 'mae', working_dir)
# plot_datewise_wasserstein_distance_distrib(results_df_u, working_dir)
# plot_cor_len(results_df_u, working_dir)


# ========== Print mean scores
# mse_global_df = datewise_scores(results_df_u, mse, 'mse') 
# mse_terre_df  = datewise_scores_terre(results_df_u, mse, 'mse')
# mse_mer_df    = datewise_scores_mer(results_df_u, mse, 'mse')

# print('mse:')
# print('  global:')
# print('    baseline : ' + str(mse_global_df['mse_baseline_mean'].mean()))
# print('    prediction : ' + str(mse_global_df['mse_y_pred_mean'].mean()))
# print('  terre:')
# print('    baseline : ' + str(mse_terre_df['mse_baseline_mean'].mean()))
# print('    prediction : ' + str(mse_terre_df['mse_y_pred_mean'].mean()))
# print('  mer:')
# print('    baseline : ' + str(mse_mer_df['mse_baseline_mean'].mean()))
# print('    prediction : ' + str(mse_mer_df['mse_y_pred_mean'].mean()))


# ========== Correlation length
# corr_len_df = corr_len(results_df_u)

# print('correlation lenght :')
# print('baseline : ' + str(corr_len_df.corr_len_baseline[0]))
# print('pred : ' + str(corr_len_df.corr_len_pred[0]))
# print('test : ' + str(corr_len_df.corr_len_test[0]))



# ========== V
# ========== Plots
plot_results(results_df_v, 'v10', working_dir)
plot_unique_score_map(results_df_v, mae, 'mae', working_dir)
plot_unique_score_map(results_df_v, biais, 'biais', working_dir)
plot_distrib(results_df_v, mse, 'mse', working_dir)
plot_distrib(results_df_v, mae, 'mae', working_dir)
# plot_datewise_wasserstein_distance_distrib(results_df_v, working_dir)
# plot_cor_len(results_df_v, working_dir)


# ========== Print mean scores
# mse_global_df = datewise_scores(results_df_v, mse, 'mse') 
# mse_terre_df  = datewise_scores_terre(results_df_v, mse, 'mse')
# mse_mer_df    = datewise_scores_mer(results_df_v, mse, 'mse')

# print('mse:')
# print('  global:')
# print('    baseline : ' + str(mse_global_df['mse_baseline_mean'].mean()))
# print('    prediction : ' + str(mse_global_df['mse_y_pred_mean'].mean()))
# print('  terre:')
# print('    baseline : ' + str(mse_terre_df['mse_baseline_mean'].mean()))
# print('    prediction : ' + str(mse_terre_df['mse_y_pred_mean'].mean()))
# print('  mer:')
# print('    baseline : ' + str(mse_mer_df['mse_baseline_mean'].mean()))
# print('    prediction : ' + str(mse_mer_df['mse_y_pred_mean'].mean()))


# ========== Correlation length
# corr_len_df = corr_len(results_df_v)

# print('correlation lenght :')
# print('baseline : ' + str(corr_len_df.corr_len_baseline[0]))
# print('pred : ' + str(corr_len_df.corr_len_pred[0]))
# print('test : ' + str(corr_len_df.corr_len_test[0]))


    