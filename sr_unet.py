import argparse
from bronx.stdtypes.date import daterangex as rangex
from sklearn.model_selection import train_test_split
from preprocessing.load_data import *
from preprocessing.normalisations import *
from preprocessing.patches import *
from unet.architectures import unet_maker
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizer_v2.adam import Adam
from training.losses import mse_terre_mer, modified_mse, rmse_k
from training.generator import DataGenerator
import matplotlib.pyplot as plt
from time import perf_counter

import json
from collections import OrderedDict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    args = parser.parse_args()
    opt_path = args.config
    # remove comments starting with '//'
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)
    
    # ========== Setup
    t0 = perf_counter()
    data_train_location  = opt["dataset"]["data_train_location"]
    data_valid_location  = opt["dataset"]["data_valid_location"]
    data_test_location   = opt["dataset"]["data_test_location"]
    data_static_location = opt["dataset"]["data_static_location"] 
    params_in = opt["dataset"]["params_in"]
    params_out = opt["dataset"]["params_out"]
    static_fields = opt["dataset"]["static_fields"]
    dates_train = rangex(opt["dataset"]["dates_train"])
    dates_valid = rangex(opt["dataset"]["dates_valid"])
    dates_test  = rangex(opt["dataset"]["dates_test"])
    interp = opt["dataset"]["interp"]
    echeances = opt["dataset"]["echeances"]
    LR, batch_size, epochs = opt["training"]["learning_rate"], opt["training"]["batch_size"], opt["training"]["n_epochs"]
    output_dir = opt["output_dir"]

    t1 = perf_counter()
    print('setup time = ' + str(t1-t0))


    # ========== Load data
    if opt["dataset"]["config"] == "optimisation": # the test dataset is not used
        X_train_df = load_X(
            dates_train, 
            echeances,
            params_in,
            data_train_location,
            data_static_location,
            static_fields = static_fields,
            resample=interp
        )

        X_test_df = load_X(
            dates_valid, 
            echeances,
            params_in,
            data_valid_location,
            data_static_location,
            static_fields = static_fields,
            resample=interp
        )

        y_train_df = load_y(
            dates_train,
            echeances,
            params_out,
            data_train_location
        )

        y_test_df = load_y(
            dates_valid,
            echeances,
            params_out,
            data_valid_location
        )
        # split train set
        X_train_df, X_valid_df, y_train_df, y_valid_df = train_test_split(
            X_train_df, y_train_df, test_size=int(0.2*len(X_train_df)))

    elif opt["dataset"]["config"] =="test": # the whole dataset is used
        X_train_df = load_X(
            dates_train, 
            echeances,
            params_in,
            data_train_location,
            data_static_location,
            static_fields = static_fields,
            resample=interp
        )

        X_valid_df = load_X(
            dates_valid, 
            echeances,
            params_in,
            data_valid_location,
            data_static_location,
            static_fields = static_fields,
            resample=interp
        )

        X_test_df = load_X(
            dates_test, 
            echeances,
            params_in,
            data_test_location,
            data_static_location,
            static_fields = static_fields,
            resample=interp
        )

        y_train_df = load_y(
            dates_train,
            echeances,
            params_out,
            data_train_location
        )

        y_valid_df = load_y(
            dates_valid,
            echeances,
            params_out,
            data_valid_location
        )

        y_test_df = load_y(
            dates_test,
            echeances,
            params_out,
            data_test_location
        )
    
    else:
        raise NotImplementedError

    t2 = perf_counter()
    print('loading time = ' + str(t2-t1))


    # ========== Preprocessing
    # remove missing days
    X_train_df, y_train_df = delete_missing_days(X_train_df, y_train_df)
    X_valid_df, y_valid_df = delete_missing_days(X_valid_df, y_valid_df)
    X_test_df , y_test_df  = delete_missing_days(X_test_df, y_test_df)

    # pad data
    if opt["training"]["patches"] is not None:
        if opt["training"]["patches"] == "random":
            if opt["inference"]["patches"] is None:
                X_valid_df, y_valid_df = pad(X_valid_df), pad(y_valid_df)
                X_test_df , y_test_df  = pad(X_test_df),  pad(y_test_df)
            elif opt["inference"]["patches"] == "patchify":
                X_valid_df, y_valid_df = pad_for_patchify(X_valid_df), pad_for_patchify(y_valid_df)
                X_test_df, y_test_df = pad_for_patchify(X_test_df), pad_for_patchify(y_test_df)
        elif opt["training"]["patches"] == "patchify":
            X_train_df, y_train_df = pad_for_patchify(X_train_df), pad_for_patchify(y_train_df)
            if opt["inference"]["patches"] is None:
                X_valid_df, y_valid_df = pad(X_valid_df), pad(y_valid_df)
                X_test_df , y_test_df  = pad(X_test_df),  pad(y_test_df)
            elif opt["inference"]["patches"] == "patchify":
                X_valid_df, y_valid_df = pad_for_patchify(X_valid_df), pad_for_patchify(y_valid_df)
                X_test_df, y_test_df = pad_for_patchify(X_test_df), pad_for_patchify(y_test_df)
        # else:
        #     raise NotImplementedError
    else:
        X_train_df, y_train_df = pad(X_train_df), pad(y_train_df)
        X_valid_df, y_valid_df = pad(X_valid_df), pad(y_valid_df)
        X_test_df , y_test_df  = pad(X_test_df),  pad(y_test_df)

    # Normalisation:
    if opt["training"]["normalisation"] == "standardisation":
        get_mean(X_train_df, output_dir)
        get_std(X_train_df, output_dir)
        X_train_df, y_train_df = standardisation(X_train_df, output_dir), standardisation(y_train_df, output_dir)
        X_valid_df, y_valid_df = standardisation(X_valid_df, output_dir), standardisation(y_valid_df, output_dir)
        X_test_df , y_test_df  = standardisation(X_test_df, output_dir) , standardisation(y_test_df, output_dir)
    elif opt["training"]["normalisation"] == "normalisation":
        get_max_abs(X_train_df, output_dir)
        X_train_df, y_train_df = normalisation(X_train_df, output_dir), normalisation(y_train_df, output_dir)
        X_valid_df, y_valid_df = normalisation(X_valid_df, output_dir), normalisation(y_valid_df, output_dir)
        X_test_df , y_test_df  = normalisation(X_test_df, output_dir) , normalisation(y_test_df, output_dir)
    elif opt["training"]["normalisation"] == "minmax":
        get_min(X_train_df, output_dir)
        get_max(X_train_df, output_dir)
        X_train_df, y_train_df = min_max_norm(X_train_df, output_dir), min_max_norm(y_train_df, output_dir)
        X_valid_df, y_valid_df = min_max_norm(X_valid_df, output_dir), min_max_norm(y_valid_df, output_dir)
        X_test_df , y_test_df  = min_max_norm(X_test_df, output_dir) , min_max_norm(y_test_df, output_dir)
    elif opt["training"]["normalisation"] == "mean":
        get_min(X_train_df, output_dir)
        get_max(X_train_df, output_dir)
        get_mean(X_train_df, output_dir)
        X_train_df, y_train_df = mean_norm(X_train_df, output_dir), mean_norm(y_train_df, output_dir)
        X_valid_df, y_valid_df = mean_norm(X_valid_df, output_dir), mean_norm(y_valid_df, output_dir)
        X_test_df , y_test_df  = mean_norm(X_test_df, output_dir) , mean_norm(y_test_df, output_dir)
    else:
        raise NotImplementedError

    # patches
    if opt["training"]["patches"] is not None:
        patch_size = opt["training"]["patch_size"]
        if opt["training"]["patches"] == "random":
            n_patches  = opt["training"]["n_patches"]
            X_train_df, y_train_df = extract_patches(X_train_df, y_train_df, patch_size, patch_size, n_patches)
            if opt["inference"]["patches"] is None:
                pass
            elif opt["inference"]["patches"] == "patchify":
                img_h = X_train_df.t2m[0].shape[0]
                img_w = X_train_df.t2m[0].shape[1]
                X_valid_df, y_valid_df = extract_patches_patchify(X_valid_df, patch_size), extract_patches_patchify(y_valid_df, patch_size)
                X_test_df, y_test_df = extract_patches_patchify(X_test_df, patch_size), extract_patches_patchify(y_test_df, patch_size)
        elif opt["training"]["patches"] == "patchify":
            img_h = X_train_df.t2m[0].shape[0]
            img_w = X_train_df.t2m[0].shape[1]
            X_train_df, y_train_df = extract_patches_patchify(X_train_df, patch_size), extract_patches_patchify(y_train_df, patch_size)
            if opt["inference"]["patches"] is None:
                pass
            elif opt["inference"]["patches"] == "patchify":
                X_valid_df, y_valid_df = extract_patches_patchify(X_valid_df, patch_size), extract_patches_patchify(y_valid_df, patch_size)
                X_test_df, y_test_df = extract_patches_patchify(X_test_df, patch_size), extract_patches_patchify(y_test_df, patch_size)
        else:
            raise NotImplementedError


    # generators
    train_generator = DataGenerator(X_train_df, y_train_df, batch_size)
    valid_generator = DataGenerator(X_valid_df, y_valid_df, batch_size)
    X_test , y_test = df_to_array(X_test_df) , df_to_array(y_test_df)

    t3 = perf_counter()
    print('preprocessing time = ' + str(t3-t2))

    # ========== Model definition
    unet = unet_maker((None, None, X_test.shape[3]), output_channels=len(params_out))
    print('unet creation ok')

    # ========== Training
    if opt["training"]["loss"] == "mse":
        loss = "mse"
    elif opt["training"]["loss"] == "hybrid":
        loss = mse_terre_mer(opt["training"]["frac"])
    elif opt["training"]["loss"] == "hybrid":
        loss = modified_mse(opt["training"]["tau"], opt["training"]["eps"])
    else:
        raise NotImplementedError

    unet.compile(optimizer=Adam(learning_rate=LR), loss=loss, metrics=[rmse_k], run_eagerly=opt["training"]["run_eagerly"])  
    print('compilation ok')

    model_name = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=4, verbose=1), ## we set some callbacks to reduce the learning rate during the training
                EarlyStopping(monitor='val_loss', patience=15, verbose=1),               ## Stops the fitting if val_loss does not improve after 15 iterations
                ModelCheckpoint(output_dir + model_name, monitor='val_loss', verbose=1, save_best_only=True)] ## Save only the best model

    history = unet.fit(train_generator, 
            batch_size=batch_size, epochs=epochs,  
            validation_data=valid_generator, 
            callbacks = callbacks,
            verbose=2, shuffle=True)

    unet.summary()
    print(history.history.keys())

    # ========== Curves
    # summarize history for accuracy
    accuracy_curve = plt.figure()
    plt.plot(history.history['rmse_k'])
    plt.plot(history.history['val_rmse_k'])
    plt.semilogy()
    plt.title('model rmse_k')
    plt.ylabel('rmse_k')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(output_dir + 'RMSE_curve.png')
    # summarize history for loss
    loss_curve = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.semilogy()
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(output_dir + 'Loss_curve.png')

    t4 = perf_counter()
    print('training time = ' + str(t4-t3))


    # ========== Prediction
    y_pred = unet.predict(X_test)
    print(y_pred.shape)

    t5 = perf_counter()
    print('predicting time = ' + str(t5-t4))

    # ========== Postprocessing
    y_pred_df = y_test_df.copy()
    arrays_cols = get_arrays_cols(y_pred_df)
    for i in range(len(y_pred_df)):
        for i_c, c in enumerate(arrays_cols):
            y_pred_df[c][i] = y_pred[i, :, :, i_c]

    if opt["inference"]["patches"] is not None:
        if opt["inference"]["patches"] == "patchify":
            y_pred_df = rebuild_from_patchify(y_pred_df, img_h, img_w)
            y_pred_df = crop_for_patchify(y_pred_df)

    if opt["training"]["normalisation"] == "standardisation":
        y_pred_df = destandardisation(y_pred_df, output_dir)
    elif opt["training"]["normalisation"] == "normalisation":
        y_pred_df = denormalisation(y_pred_df, output_dir)
    elif opt["training"]["normalisation"] == "minmax":
        y_pred_df = min_max_denorm(y_pred_df, output_dir)
    elif opt["training"]["normalisation"] == "mean":
        y_pred_df = mean_denorm(y_pred_df, output_dir)
    
    y_pred_df = crop(y_pred_df)
    y_pred_df.to_pickle(output_dir + 'y_pred.csv')