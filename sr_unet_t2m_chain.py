import numpy as np 
import random as rn
from bronx.stdtypes.date import daterangex as rangex
from sklearn.model_selection import train_test_split
from unet.architectures import *
from training.imports4training import *
from training.generator import DataGenerator
import matplotlib.pyplot as plt
from preprocessing.load_data import *
from preprocessing.normalisations import *
from time import perf_counter
# import warnings
# warnings.filterwarnings("ignore")
plt.switch_backend('agg')

physical_devices = tf.config.list_physical_devices()
print(physical_devices)

t0 = perf_counter()

data_train_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_train/'
data_valid_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_test/'
data_test_location = '/cnrm/recyf/Data/users/danjoul/dataset/data_test/'
data_static_location = '/cnrm/recyf/Data/users/danjoul/dataset/'
baseline_location = '/cnrm/recyf/Data/users/danjoul/dataset/baseline/'
model_name = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'

def experiment(params_in, params_out, static_fields, output_dir, resample='r', LR=0.005, batch_size=32, epochs=100):
    # ========== Setup
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

    t1 = perf_counter()
    print('setup time = ' + str(t1-t0))


    # ========== Load data
    X_train_df = load_X(
        dates_train, 
        echeances,
        params_in,
        data_train_location,
        data_static_location,
        static_fields = static_fields,
        resample=resample
    )

    X_test_df = load_X(
        dates_valid, 
        echeances,
        params_in,
        data_valid_location,
        data_static_location,
        static_fields = static_fields,
        resample=resample
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
    t2 = perf_counter()
    print('loading time = ' + str(t2-t1))


    # ========== Preprocessing
    # split train set
    X_train_df, X_valid_df, y_train_df, y_valid_df = train_test_split(X_train_df, y_train_df, test_size=int(0.2*len(X_train_df)))

    # remove missing days
    X_train_df, y_train_df = delete_missing_days(X_train_df, y_train_df)
    X_valid_df, y_valid_df = delete_missing_days(X_valid_df, y_valid_df)
    X_test_df , y_test_df  = delete_missing_days(X_test_df, y_test_df)

    # pad data
    X_train_df, y_train_df = pad(X_train_df), pad(y_train_df)
    X_valid_df, y_valid_df = pad(X_valid_df), pad(y_valid_df)
    X_test_df , y_test_df  = pad(X_test_df),  pad(y_test_df)

    # Normalisation:
    get_mean(X_train_df, output_dir)
    get_std(X_train_df, output_dir)
    X_train_df, y_train_df = standardisation(X_train_df, output_dir), standardisation(y_train_df, output_dir)
    X_valid_df, y_valid_df = standardisation(X_valid_df, output_dir), standardisation(y_valid_df, output_dir)
    X_test_df , y_test_df  = standardisation(X_test_df, output_dir) , standardisation(y_test_df, output_dir)


    train_generator = DataGenerator(X_train_df, y_train_df, batch_size)
    valid_generator = DataGenerator(X_valid_df, y_valid_df, batch_size)
    X_test , y_test = df_to_array(X_test_df) , df_to_array(y_test_df)

    t3 = perf_counter()
    print('preprocessing time = ' + str(t3-t2))


    # ========== Model definition
    unet = unet_maker(X_test[0, :, :, :].shape, output_channels=len(params_out))
    print('unet creation ok')

    # ========== Training
    unet.compile(optimizer=Adam(learning_rate=LR), loss='mse', metrics=[rmse_k], run_eagerly=True)  
    print('compilation ok')
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
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(output_dir + 'RMSE_curve.png')
    # summarize history for loss
    loss_curve = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.semilogy()
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
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

    y_pred_df = destandardisation(y_pred_df, output_dir)
    y_pred_df = crop(y_pred_df)

    y_pred_df.to_pickle(output_dir + 'y_pred.csv')


base_output_dir = '/cnrm/recyf/Data/users/danjoul/unet_experiments/t2m/params/'

params_in = ['t2m']
params_out = ['t2m']
static_fields = []
experiment(
    params_in,
    params_out,
    static_fields,
    base_output_dir + 't2m/',
    resample='r',
    LR=0.005,
    batch_size=32,
    epochs=100
)

params_in = ['t2m', 'cape']
params_out = ['t2m']
static_fields = []
experiment(
    params_in,
    params_out,
    static_fields,
    base_output_dir + 'cape/',
    resample='r',
    LR=0.005,
    batch_size=32,
    epochs=100
)

params_in = ['t2m', 'tke']
params_out = ['t2m']
static_fields = []
experiment(
    params_in,
    params_out,
    static_fields,
    base_output_dir + 'tke/',
    resample='r',
    LR=0.005,
    batch_size=32,
    epochs=100
)

params_in = ['t2m', 'toa']
params_out = ['t2m']
static_fields = []
experiment(
    params_in,
    params_out,
    static_fields,
    base_output_dir + 'toa/',
    resample='r',
    LR=0.005,
    batch_size=32,
    epochs=100
)

params_in = ['t2m', 'ts']
params_out = ['t2m']
static_fields = []
experiment(
    params_in,
    params_out,
    static_fields,
    base_output_dir + 'ts/',
    resample='r',
    LR=0.005,
    batch_size=32,
    epochs=100
)

params_in = ['t2m', 'u10', 'v10']
params_out = ['t2m']
static_fields = []
experiment(
    params_in,
    params_out,
    static_fields,
    base_output_dir + 'uv10/',
    resample='r',
    LR=0.005,
    batch_size=32,
    epochs=100
)

params_in = ['t2m']
params_out = ['t2m']
static_fields = ['SURFGEOPOTENTIEL']
experiment(
    params_in,
    params_out,
    static_fields,
    base_output_dir + 'SURFGEOPOTENTIEL/',
    resample='r',
    LR=0.005,
    batch_size=32,
    epochs=100
)

params_in = ['t2m']
params_out = ['t2m']
static_fields = ['SURFIND.TERREMER']
experiment(
    params_in,
    params_out,
    static_fields,
    base_output_dir + 'SURFIND.TERREMER/',
    resample='r',
    LR=0.005,
    batch_size=32,
    epochs=100
)

params_in = ['t2m']
params_out = ['t2m']
static_fields = ['SFX.BATHY']
experiment(
    params_in,
    params_out,
    static_fields,
    base_output_dir + 'SFX.BATHY/',
    resample='r',
    LR=0.005,
    batch_size=32,
    epochs=100
)

params_in = ['t2m','cape', 'tke', 'toa', 'ts', 'u10', 'v10']
params_out = ['t2m']
static_fields = ['SURFGEOPOTENTIEL', 'SURFIND.TERREMER', 'SFX.BATHY']
experiment(
    params_in,
    params_out,
    static_fields,
    base_output_dir + 'all_params/',
    resample='r',
    LR=0.005,
    batch_size=32,
    epochs=100
)