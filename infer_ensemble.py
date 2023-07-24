import os
import torch
import argparse
import core.logger as Logger
import logging
from tensorboardX import SummaryWriter
import data as Data
from data.load_data import crop
from data.normalisations import destandardisation, denormalisation, min_max_denorm, mean_denorm
from data.postprocessing import postprocess_df
import model as Model
import core.metrics as Metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_ddpm.jsonc',
                        help='JSON file for configuration')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)

    # parse configs
    args = parser.parse_args()

    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # loading and preprocessing data
    X_ens_df, X_train_df = Data.load_data_ensemble(opt["data_loading"], opt["ensemble"]["DDPM"])
    X_ens_df = Data.preprocess_data_ensemble(opt, X_ens_df, X_train_df)

    ens_ds  = Data.create_dataset_ensemble(X_ens_df)

    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # load weights
    if opt["path"]["resume_state"] is not None:
        diffusion.load_network()
    else:
        load_path = os.path.join(opt['path']['weights'], 'best_model.pth')
        diffusion.load_best_model(load_path)

    # inference
    for i in range(opt["ensemble"]["DDPM"]["n_members"]):
        output_dir = os.path.join(
            opt["path"]["working_dir"], str(i + 1) + "/"
        )
        print(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir + "images/", exist_ok=True)

        channels = opt["data_loading"]["params_out"]
        y_pred_df = pd.DataFrame(
            [],
            columns=["dates", "echeances"] + channels
        )
        
        diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')

        test_loader = Data.create_dataloader(ens_ds)

        for j, test_data in enumerate(test_loader):

            diffusion.feed_data(test_data)
            diffusion.test(continous=False)

            sr_img = Metrics.tensor2image(diffusion.SR)
            y_pred_j = [X_ens_df.dates.iloc[j], X_ens_df.echeances.iloc[j]]
            for i_c, c in enumerate(channels):
                y_pred_j.append(sr_img[:, :, i_c])
            y_pred_df.loc[len(y_pred_df)] = y_pred_j

            if j % opt["inference"]["save_freq"] == 0:
                y_pred_df.to_pickle(output_dir + 'y_pred_norm.csv')
                logger.info("Step {}: results saved.".format(j))

            # image plot
            if j % opt["inference"]["print_freq"] == 0:
                fig, ax = plt.subplots()
                im = ax.imshow(sr_img[:, :, 0])
                fig.colorbar(im, ax=ax)
                plt.savefig(output_dir + "images/" + "image_{}.png".format(j))
                logger.info("Figure saved")

        if opt["preprocessing"]["normalisation"] is not None:
            if opt["preprocessing"]["normalisation"] == "standardisation":
                y_pred_df = destandardisation(y_pred_df, opt["path"]["working_dir"])
            elif opt["preprocessing"]["normalisation"] == "normalisation":
                y_pred_df = denormalisation(y_pred_df, opt["path"]["working_dir"])
            elif opt["preprocessing"]["normalisation"] == "minmax":
                y_pred_df = min_max_denorm(y_pred_df, opt["path"]["working_dir"])
            elif opt["preprocessing"]["normalisation"] == "mean":
                y_pred_df = mean_denorm(y_pred_df, opt["path"]["working_dir"])
        y_pred_df = crop(y_pred_df)

        # postprocessing 
        postproc_opt = opt["postprocessing"]
        if postproc_opt is not None:
            y_pred_df = postprocess_df(y_pred_df, X_ens_df, postproc_opt)

        y_pred_df.to_pickle(output_dir + 'y_pred.csv')

        logger.info("End of member ", i)

    logger.info("End of inference")





















    


    
    