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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
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
    logger = logging.getLogger("base")
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


    train_ds = Data.create_dataset(X_train_df, y_train_df)
    valid_ds = Data.create_dataset(X_valid_df, y_valid_df)#, data_len=opt["training"]["data_len"])
    test_ds  = Data.create_dataset(X_test_df, y_test_df)


    train_loader = Data.create_dataloader(train_ds, training_opt=opt["training"], phase="train")
    # valid_loader = Data.create_dataloader(valid_ds, training_opt=opt["training"])
    # test_loader  = Data.create_dataloader(test_ds, training_opt=opt["training"])
    logger.info('Initial Dataset Finished')

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
    mse = math.inf
    best_mse = math.inf
    while current_step < n_iter:
        current_epoch += 1
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > n_iter:
                break
            t0 = perf_counter()
            diffusion.feed_data(train_data)
            diffusion.optimize_parameters()
            # log
            if current_step % opt["training"]["print_freq"] == 0:
                message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                    current_epoch, current_step)
                logger.info(message)
            t1 = perf_counter()


            # validation
            if current_step % opt["training"]["val_freq"] == 0:
                result_path = '{}/{}'.format(opt['path']
                                                ['eval_results'], current_epoch)
                os.makedirs(result_path, exist_ok=True)

                diffusion.set_new_noise_schedule(
                    opt['model']['beta_schedule']['val'], schedule_phase='val')

                indices = np.random.randint(0, len(valid_ds), opt["training"]["data_len"])
                valid_subset = torch.utils.data.Subset(valid_ds, indices)
                valid_loader = Data.create_dataloader(valid_subset)
                list_mse = []
                list_mae = []
                for i, val_data in enumerate(valid_loader):
                    diffusion.feed_data(val_data)
                    diffusion.test(continous=False)

                    # evaluate model
                    hr_img = Metrics.tensor2image(val_data["HR"]) # see function update visuals 
                    sr_img = Metrics.tensor2image(diffusion.SR)                   
                    mae = Metrics.score_value(hr_img, sr_img, "mae")
                    mse = Metrics.score_value(hr_img, sr_img, "mse")
                    list_mse.append(mse)
                    list_mae.append(mae)

                    # intermediate image plot
                    fig, ax = plt.subplots()
                    im = ax.imshow(sr_img[:, :, 0])
                    fig.colorbar(im, ax=ax)
                    plt.savefig(result_path + "/eval_{}_{}_{:.2f}.png".format(current_step, i, mse))

                # save best model
                eval_mse = np.array(list_mse).mean()
                eval_mae = np.array(list_mae).mean()
                logger.info("Eval [mae, mse] : [{:.2f}, {:.2f}]".format(eval_mae, eval_mse))
                if eval_mse < best_mse:
                    diffusion.save_best_model()
                    best_mse = eval_mse

                # training curves
                val_loss.append(eval_mse)
                val_steps.append(current_step)
                loss_curve = plt.figure()
                plt.plot(val_steps, val_loss)
                plt.title('model val mse')
                plt.ylabel('mse')
                plt.xlabel('iter')
                plt.savefig(opt["path"]["working_dir"] + 'MSE_curve.png')


                diffusion.set_new_noise_schedule(
                    opt['model']['beta_schedule']['train'], schedule_phase='train')


            if current_step % opt["training"]["save_checkpoint_freq"] == 0:
                logger.info('Saving models and training states.')
                diffusion.save_network(current_epoch, current_step)
            t2 = perf_counter()

            if opt["benchmark"]:
                logger.info("Training time: {:.2f}s".format(t1 - t0))
                logger.info("Eval time: {:.2f}s".format(t2 - t1))

    # save model
    logger.info('End of training.')

    if opt["inference"]["enable"]:

        # load best model
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
        
        y_pred_df = utils.crop(y_pred_df)
        y_pred_df.to_pickle(opt["path"]["working_dir"] + 'y_pred.csv')

        logger.info("End of inference.")