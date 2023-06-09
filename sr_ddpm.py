import torch
import argparse
import core.logger as Logger
import logging
from tensorboardX import SummaryWriter
import data as Data
from data.pytorch_dataset import PyTorchDataset





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

    train_ds = PyTorchDataset(X_train_df, y_train_df)
    valid_ds = PyTorchDataset(X_valid_df, y_valid_df)
    test_ds  = PyTorchDataset(X_test_df, y_test_df)

    train_loader = Data.create_dataloader(train_ds, opt["training"], phase="train")
    valid_loader = Data.create_dataloader(valid_ds, opt["training"])
    test_loader  = Data.create_dataloader(test_ds, opt["training"])

    logger.info('Initial Dataset Finished')

    print(train_ds.__get_item__(0))
    print(train_loader)
    