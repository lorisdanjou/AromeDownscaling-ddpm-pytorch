import torch
from torch.utils.data import Dataset
import data.load_data as ld
import pandas as pd


class PyTorchDataset(Dataset):
    def __init__(self, X_df, y_df):
        self.X_df = X_df
        self.y_df = y_df

    def __len__(self):
        return len(self.X_df)

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
        X = torch.from_numpy(X_array)
        y = torch.from_numpy(y_array)

        return [X, y]
        
