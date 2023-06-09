import torch
from torch.utils.data import Dataset
import data.load_data as ld


class PyTorchDataset(Dataset):
    def __init__(self, X_df, y_df):
        self.X_df = X_df
        self.y_df = y_df

    def __len__(self):
        return len(self.X_df)

    def __getitem__(self, index):
        X_array = ld.df_to_array(self.X_df.loc[index])
        y_array = ld.df_to_array(self.y_df.loc[index])
        X = torch.from_numpy(X_array)
        y = torch.from_numpy(y_array)
        
