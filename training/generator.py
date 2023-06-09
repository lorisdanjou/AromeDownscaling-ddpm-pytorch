import numpy as np
from tensorflow.keras.utils import Sequence
from preprocessing.load_data import *


class DataGenerator(Sequence):
    """Generates batches of data without all the dataset in the memory."""

    def __init__(self, X_df, y_df, batch_size, shuffle=True):
        self.X_df = X_df
        self.y_df = y_df
        self.size = len(self.X_df)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Returns the number of batches per epoch"""
        return int(np.floor(self.size / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        # Find list of IDs
        X = df_to_array(self.X_df.loc[indexes])
        y = df_to_array(self.y_df.loc[indexes])
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(self.size)
        if self.shuffle:
            np.random.shuffle(self.indexes)
