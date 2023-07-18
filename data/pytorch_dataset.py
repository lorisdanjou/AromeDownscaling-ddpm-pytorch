import torch
from torch.utils.data import Dataset
import data.load_data as ld
import pandas as pd
import numpy as np
import numpy.random
import matplotlib.pyplot as plt


class PyTorchDataset(Dataset):
    def __init__(self, X_df, y_df, data_len=-1):
        self.X_df = X_df
        self.y_df = y_df
        self.data_len = data_len

    def __len__(self):
        if self.data_len < 0:
            return len(self.X_df)
        else: 
            return self.data_len

    def __getitem__(self, index):
        X_df_item = pd.DataFrame(
            [],
            columns=self.X_df.columns
        )
        X_df_item.loc[len(X_df_item)] = self.X_df.loc[index]
        y_df_item = pd.DataFrame(
            [],
            columns=self.y_df.columns
        )
        y_df_item.loc[len(y_df_item)] = self.y_df.loc[index]

        X_array = ld.df_to_array(X_df_item)
        y_array = ld.df_to_array(y_df_item)

        X_array = X_array[0, :, :, :].transpose((2,0,1))
        y_array = y_array[0, :, :, :].transpose((2,0,1))

        X = torch.from_numpy(X_array)
        y = torch.from_numpy(y_array)

        return {"SR": X, "HR": y}
    

class EnsembleDataset(Dataset):
    def __init__(self, X_df, data_len=-1):
        self.X_df = X_df
        self.data_len = data_len

    def __len__(self):
        if self.data_len < 0:
            return len(self.X_df)
        else: 
            return self.data_len

    def __getitem__(self, index):
        X_df_item = pd.DataFrame(
            [],
            columns=self.X_df.columns
        )
        X_df_item.loc[len(X_df_item)] = self.X_df.loc[index]

        X_array = ld.df_to_array(X_df_item)

        X_array = X_array[0, :, :, :].transpose((2,0,1))

        X = torch.from_numpy(X_array)
        return {"SR": X}
    
        
