import data.load_data as ld
import data.normalisations as norm
from .pytorch_dataset import PyTorchDataset
from torch.utils.data import DataLoader

def load_data(data_loading_opt):
    data_train_location  = data_loading_opt["data_train_location"]
    data_valid_location  = data_loading_opt["data_valid_location"]
    data_test_location   = data_loading_opt["data_test_location"]
    data_static_location = data_loading_opt["data_static_location"]
    dates_train          = data_loading_opt["dates_train"]
    dates_valid          = data_loading_opt["dates_valid"]
    dates_test           = data_loading_opt["fates_test"]
    echeances            = data_loading_opt["echeances"]
    params_in            = data_loading_opt["params_in"]
    params_out           = data_loading_opt["params_out"]
    static_fields        = data_loading_opt["static_fields"]
    interp               = data_loading_opt["interp"]

    if data_loading_opt["config"] == "optimisation": # the test dataset is not used
        from sklearn.model_selection import train_test_split   
        X_train_df = ld.load_X(
            dates_train, 
            echeances,
            params_in,
            data_train_location,
            data_static_location,
            static_fields = static_fields,
            resample=interp
        )

        X_test_df = ld.load_X(
            dates_valid, 
            echeances,
            params_in,
            data_valid_location,
            data_static_location,
            static_fields = static_fields,
            resample=interp
        )

        y_train_df = ld.load_y(
            dates_train,
            echeances,
            params_out,
            data_train_location
        )

        y_test_df = ld.load_y(
            dates_valid,
            echeances,
            params_out,
            data_valid_location
        )
        # split train set
        X_train_df, X_valid_df, y_train_df, y_valid_df = train_test_split(
            X_train_df, y_train_df, test_size=int(0.2*len(X_train_df)))

    elif data_loading_opt["config"] =="test": # the whole dataset is used
        X_train_df = ld.load_X(
            dates_train, 
            echeances,
            params_in,
            data_train_location,
            data_static_location,
            static_fields = static_fields,
            resample=interp
        )

        X_valid_df = ld.load_X(
            dates_valid, 
            echeances,
            params_in,
            data_valid_location,
            data_static_location,
            static_fields = static_fields,
            resample=interp
        )

        X_test_df = ld.load_X(
            dates_test, 
            echeances,
            params_in,
            data_test_location,
            data_static_location,
            static_fields = static_fields,
            resample=interp
        )

        y_train_df = ld.load_y(
            dates_train,
            echeances,
            params_out,
            data_train_location
        )

        y_valid_df = ld.load_y(
            dates_valid,
            echeances,
            params_out,
            data_valid_location
        )

        y_test_df = ld.load_y(
            dates_test,
            echeances,
            params_out,
            data_test_location
        )
    
    else:
        raise NotImplementedError

    return X_train_df, y_train_df, X_valid_df, y_valid_df, X_test_df, y_test_df

def  preprocess_data(opt, X_train_df, y_train_df, X_valid_df, y_valid_df, X_test_df, y_test_df):
    training_opt = opt["preprocessing"]
    output_dir = opt["path"]["working_dir"]

    # remove missing days
    X_train_df, y_train_df = ld.delete_missing_days(X_train_df, y_train_df)
    X_valid_df, y_valid_df = ld.delete_missing_days(X_valid_df, y_valid_df)
    X_test_df , y_test_df  = ld.delete_missing_days(X_test_df, y_test_df)

    # pad data
    X_train_df, y_train_df = ld.pad(X_train_df), ld.pad(y_train_df)
    X_valid_df, y_valid_df = ld.pad(X_valid_df), ld.pad(y_valid_df)
    X_test_df , y_test_df  = ld.pad(X_test_df),  ld.pad(y_test_df)

    # Normalisation:
    if training_opt["normalisation"] == "standardisation":
        norm.get_mean(X_train_df, output_dir)
        norm.get_std(X_train_df, output_dir)
        X_train_df, y_train_df = norm.standardisation(X_train_df, output_dir), norm.standardisation(y_train_df, output_dir)
        X_valid_df, y_valid_df = norm.standardisation(X_valid_df, output_dir), norm.standardisation(y_valid_df, output_dir)
        X_test_df , y_test_df  = norm.standardisation(X_test_df, output_dir) , norm.standardisation(y_test_df, output_dir)
    elif opt["training"]["normalisation"] == "normalisation":
        norm.get_max_abs(X_train_df, output_dir)
        X_train_df, y_train_df = norm.normalisation(X_train_df, output_dir), norm.normalisation(y_train_df, output_dir)
        X_valid_df, y_valid_df = norm.normalisation(X_valid_df, output_dir), norm.normalisation(y_valid_df, output_dir)
        X_test_df , y_test_df  = norm.normalisation(X_test_df, output_dir) , norm.normalisation(y_test_df, output_dir)
    elif opt["training"]["normalisation"] == "minmax":
        norm.get_min(X_train_df, output_dir)
        norm.get_max(X_train_df, output_dir)
        X_train_df, y_train_df = norm.min_max_norm(X_train_df, output_dir), norm.min_max_norm(y_train_df, output_dir)
        X_valid_df, y_valid_df = norm.min_max_norm(X_valid_df, output_dir), norm.min_max_norm(y_valid_df, output_dir)
        X_test_df , y_test_df  = norm.min_max_norm(X_test_df, output_dir) , norm.min_max_norm(y_test_df, output_dir)
    elif opt["training"]["normalisation"] == "mean":
        norm.get_min(X_train_df, output_dir)
        norm.get_max(X_train_df, output_dir)
        norm.get_mean(X_train_df, output_dir)
        X_train_df, y_train_df = norm.mean_norm(X_train_df, output_dir), norm.mean_norm(y_train_df, output_dir)
        X_valid_df, y_valid_df = norm.mean_norm(X_valid_df, output_dir), norm.mean_norm(y_valid_df, output_dir)
        X_test_df , y_test_df  = norm.mean_norm(X_test_df, output_dir) , norm.mean_norm(y_test_df, output_dir)
    else:
        raise NotImplementedError

    return X_train_df, y_train_df, X_valid_df, y_valid_df, X_test_df, y_test_df


def create_dataset(X_df, y_df):
    return PyTorchDataset(X_df, y_df)

def create_dataloader(dataset, training_opt, phase=None):
    if phase == "train":
        return DataLoader(
            dataset,
            batch_size=training_opt['batch_size'],
            shuffle=training_opt['use_shuffle'],
            num_workers=training_opt['num_workers'],
            pin_memory=True
        )
    else:
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )
