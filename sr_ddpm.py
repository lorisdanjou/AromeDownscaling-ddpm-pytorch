import os
import torch
import argparse
import core.logger as Logger
import logging
from tensorboardX import SummaryWriter
import data as Data
from data.load_data import get_arrays_cols, crop
from data.normalisations import destandardisation
import model as Model
import core.metrics as Metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



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


    train_ds = Data.create_dataset(X_train_df, y_train_df)
    valid_ds = Data.create_dataset(X_valid_df, y_valid_df)#, data_len=opt["training"]["data_len"])
    test_ds  = Data.create_dataset(X_test_df, y_test_df)


    train_loader = Data.create_dataloader(train_ds, training_opt=opt["training"], phase="train")
    valid_loader = Data.create_dataloader(valid_ds, training_opt=opt["training"])
    test_loader  = Data.create_dataloader(test_ds, training_opt=opt["training"])
    logger.info('Initial Dataset Finished')

    # for i, val_data in enumerate(valid_loader):
    #     fig, ax = plt.subplots()
    #     im = ax.imshow(val_data["HR"][0, 0, :, :].numpy())
    #     fig.colorbar(im, ax=ax)
    #     plt.savefig('/cnrm/recyf/Data/users/danjoul/ddpm/experiments/test/hr_{}.png'.format(i))


    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')


    # Training
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt["training"]["n_iter"]

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']["train"], schedule_phase="train")

    val_steps = []
    val_loss = []
    while current_step < n_iter:
        current_epoch += 1
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > n_iter:
                break
            diffusion.feed_data(train_data)
            diffusion.optimize_parameters()
            # log
            if current_step % opt["training"]["print_freq"] == 0:
                logs = diffusion.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                    current_epoch, current_step)
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    tb_logger.add_scalar(k, v, current_step)
                logger.info(message)


            # validation
            if current_step % opt["training"]["val_freq"] == 0:
                result_path = '{}/{}'.format(opt['path']
                                                ['results'], current_epoch)
                os.makedirs(result_path, exist_ok=True)

                diffusion.set_new_noise_schedule(
                    opt['model']['beta_schedule']['val'], schedule_phase='val')

                indices = np.random.randint(0, len(valid_ds), opt["training"]["data_len"])
                valid_subset = torch.utils.data.Subset(valid_ds, indices)
                valid_loader = Data.create_dataloader(valid_subset)
                for i, val_data in enumerate(valid_loader):
                    diffusion.feed_data(val_data)
                    diffusion.test(continous=False)

                    # evaluate model
                    hr_img = Metrics.tensor2image(val_data["HR"]) # see function update visuals 
                    sr_img = Metrics.tensor2image(diffusion.SR)                    
                    mae  = Metrics.score_value(hr_img, sr_img, "mae")
                    mse  = Metrics.score_value(hr_img, sr_img, "mse")
                    logger.info("[mae, mse]: [{:.3f}, {:.3f}]".format(mae, mse))

                    # intermediate image plot
                    fig, ax = plt.subplots()
                    im = ax.imshow(sr_img[0, :, :])
                    fig.colorbar(im, ax=ax)
                    plt.savefig(opt["path"]["results"] + "{}_{:.2f}.png".format(current_step, mse))

                    val_loss.append(mse)
                    val_steps.append(current_step)

                diffusion.set_new_noise_schedule(
                    opt['model']['beta_schedule']['train'], schedule_phase='train')


            if current_step % opt["training"]["save_checkpoint_freq"] == 0:
                logger.info('Saving models and training states.')
                diffusion.save_network(current_epoch, current_step)

    # save model
    logger.info('End of training.')

    # training curves
    loss_curve = plt.figure()
    plt.plot(val_steps, val_loss)
    plt.title('model val mse')
    plt.ylabel('mse')
    plt.xlabel('iter')
    plt.savefig(opt["path"]["working_dir"] + 'RMSE_curve.png')

    # inference
    y_pred_df = y_test_df.copy()
    channels = get_arrays_cols(y_pred_df)

    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')
    for i, test_data in enumerate(test_loader):
        diffusion.feed_data(test_data)
        diffusion.test(continous=False)
        out_sr = diffusion.SR.cpu().numpy()
        for i_c, c in enumerate(channels):
            y_pred_df[c][i] = out_sr[i_c, :, :]

        # image plot
        if i % opt["inference"]["print_freq"] == 0:
            fig, ax = plt.subplots()
            im = ax.imshow(out_sr[0, :, :])
            fig.colorbar(im, ax=ax)
            plt.savefig(opt["path"]["results"] + "image_{}.png".format(i))



    y_pred_df = destandardisation(y_pred_df, opt["path"]["working_dir"])
    y_pred_df = crop(y_pred_df)

    y_pred_df.to_pickle(opt["path"]["working_dir"] + 'y_pred.csv')

    logger.info("End of inference.")
