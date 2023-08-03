import os
import utils
import torch
import argparse
import core.logger as Logger
import logging
import data as Data
from data.normalisations import destandardisation, denormalisation, min_max_denorm, mean_denorm
from data.postprocessing import postprocess_df
import model as Model
import core.metrics as Metrics
import matplotlib.pyplot as plt
import pandas as pd
from time import perf_counter



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_example.jsonc',
                        help='JSON file for configuration')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)

    # parse configs
    args = parser.parse_args()

    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # create dirs
    for _, item in opt["path"].items():
        os.makedirs(item, exist_ok=True)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'log', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))

    # loading in preprocessing data
    X_train_df, y_train_df, X_valid_df, y_valid_df, X_test_df, y_test_df = Data.load_data(opt["data_loading"])
    X_train_df, y_train_df, X_valid_df, y_valid_df, X_test_df, y_test_df = Data.preprocess_data(
        opt,
        X_train_df,
        y_train_df,
        X_valid_df,
        y_valid_df,
        X_test_df,
        y_test_df
    )
    
    test_ds  = Data.create_dataset(X_test_df, y_test_df)

    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # load weights
    if opt["path"]["resume_state"] is not None:
        diffusion.load_network()
    else:
        load_path = os.path.join(opt['path']['checkpoint'], 'best_model.pth')
        diffusion.load_best_model(load_path)

    # inference
    y_pred_df = pd.DataFrame(
        [],
        columns=y_test_df.columns
    )
    channels = utils.get_arrays_cols(y_test_df)

    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')

    indices = range(opt["inference"]["i_min"], opt["inference"]["i_max"], opt["inference"]["step"])
    test_subset = torch.utils.data.Subset(test_ds, indices)
    test_loader = Data.create_dataloader(test_subset)
    for i, test_data in enumerate(test_loader):

        t3 = perf_counter()
        diffusion.feed_data(test_data)
        diffusion.test(continous=False)
        t4 = perf_counter()

        if opt["benchmark"]:
            logger.info("Inference time: {:.2f}s".format(t4 - t3))

        sr_img = Metrics.tensor2image(diffusion.SR)
        y_pred_i = [y_test_df.dates.iloc[i], y_test_df.echeances.iloc[i]]
        for i_c, c in enumerate(channels):
            y_pred_i.append(sr_img[:, :, i_c])
        y_pred_df.loc[len(y_pred_df)] = y_pred_i

        if i % opt["inference"]["save_freq"] == 0:
            y_pred_df.to_pickle(opt["path"]["working_dir"] + 'y_pred_norm.csv')
            logger.info("Step {}: results saved.".format(i))

        # image plot
        if i % opt["inference"]["print_freq"] == 0:
            fig, ax = plt.subplots()
            im = ax.imshow(sr_img[:, :, 0])
            fig.colorbar(im, ax=ax)
            plt.savefig(opt["path"]["infer_results"] + "image_{}.png".format(i))


    # denormmalisation
    if opt["preprocessing"]["normalisation"] is not None:
        if opt["preprocessing"]["normalisation"] == "standardisation":
            y_pred_df = destandardisation(y_pred_df, opt["path"]["working_dir"])
            X_test_df = destandardisation(X_test_df, opt["path"]["working_dir"])
        elif opt["preprocessing"]["normalisation"] == "normalisation":
            y_pred_df = denormalisation(y_pred_df, opt["path"]["working_dir"])
            X_test_df = denormalisation(X_test_df, opt["path"]["working_dir"])
        elif opt["preprocessing"]["normalisation"] == "minmax":
            y_pred_df = min_max_denorm(y_pred_df, opt["path"]["working_dir"])
            X_test_df = min_max_denorm(X_test_df, opt["path"]["working_dir"])
        elif opt["preprocessing"]["normalisation"] == "mean":
            y_pred_df = mean_denorm(y_pred_df, opt["path"]["working_dir"])
            X_test_df = mean_denorm(X_test_df, opt["path"]["working_dir"])

    # postprocessing 
    postproc_opt = opt["postprocessing"]
    if postproc_opt is not None:
        y_pred_df = postprocess_df(y_pred_df, X_test_df, postproc_opt)

    y_pred_df.to_pickle(opt["path"]["working_dir"] + 'y_pred.csv')

    logger.info("End of inference.")