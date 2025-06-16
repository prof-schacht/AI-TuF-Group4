import pandas as pd
import tensorflow as tf
from typing import List, Optional
from sklearn.preprocessing import MinMaxScaler


class LoadAndPrepareData:

    """
    Args:
        filepath: Pfad zur .txt-Datei
        sep: Separator im file
        na_values: Zeichen für Missing Values im file
        resample_rule: Pandas-Aggregierungs-Regel (z.B. "h"=Stunde)
        split_ratios: Anteile für Train/Val/Test
        window_size: Länge der Input-Sequenz (Zeitschritte)
        horizon: Länge der Vorhersage (Zeitschritte)
        batch_size: Batch-Größe
    """
    def __init__(self,
                 filepath: str,
                 sep: str = ";",
                 na_values: Optional[List[str]] = None,
                 resample_rule: str = "h",
                 split_ratios: tuple = (0.7, 0.15, 0.15),
                 window_size: int = 24,
                 horizon: int = 6,
                 batch_size: int = 32):

        self.filepath = filepath
        self.sep = sep
        if na_values is None:
            self.na_values = ["?"]
        else:
            self.na_values = na_values
        self.resample_rule = resample_rule
        self.split_ratios = split_ratios
        self.window_size = window_size
        self.horizon = horizon
        self.batch_size = batch_size
        self.date_col: str = "Date"
        self.time_col: str = "Time"

        # Placeholder
        self.df = None
        self.df_resampled = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.test_scaled = None
        self.val_scaled = None
        self.train_scaled = None
        self.scaler = MinMaxScaler()

    def load_and_clean(self):
        self.df = pd.read_csv(
            self.filepath,
            sep=self.sep,
            na_values=self.na_values,
            low_memory=False
        )

        self.df['datetime'] = pd.to_datetime(
            self.df[self.date_col] + ' ' + self.df[self.time_col],
            dayfirst=True
        )

        self.df.set_index('datetime', inplace=True)
        self.df.drop(columns=[self.date_col, self.time_col], inplace=True)
        self.df.interpolate(method='time', inplace=True)

    def resample(self):
        self.df_resampled = self.df.resample(self.resample_rule).mean()

    def split(self):
        n = len(self.df_resampled)
        r1, r2, _ = self.split_ratios
        i1 = int(n * r1)
        i2 = i1 + int(n * r2)
        self.train_df = self.df_resampled.iloc[:i1]
        self.val_df = self.df_resampled.iloc[i1:i2]
        self.test_df = self.df_resampled.iloc[i2:]

    def scale(self):
        self.train_scaled = self.scaler.fit_transform(self.train_df)
        self.val_scaled = self.scaler.transform(self.val_df)
        self.test_scaled = self.scaler.transform(self.test_df)

    def make_tf_dataset(self, data_array):
        ds = tf.data.Dataset.from_tensor_slices(data_array)
        ds = ds.window(self.window_size + self.horizon,
                       shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(self.window_size + self.horizon))
        ds = ds.map(lambda w: (w[:self.window_size], w[self.window_size:]))
        ds = ds.shuffle(1000).batch(self.batch_size)
        return ds.prefetch(tf.data.AUTOTUNE)

    def get_datasets(self):
        self.load_and_clean()
        self.resample()
        self.split()
        self.scale()

        training_ds = self.make_tf_dataset(self.train_scaled)
        validation_ds = self.make_tf_dataset(self.val_scaled)
        testing_ds = self.make_tf_dataset(self.test_scaled)

        # print("Train scaled  → min, max:", self.train_scaled.min(), self.train_scaled.max())
        # print("Valid scaled  → min, max:", self.val_scaled.min(),   self.val_scaled.max())
        # print("Test  scaled  → min, max:", self.test_scaled.min(),  self.test_scaled.max())

        return training_ds, validation_ds, testing_ds

    def get_scaler(self):
        return self.scaler


# Temporary test
# TODO: remove
if __name__ == "__main__":
    print("Initialize...")
    loader = LoadAndPrepareData(
        filepath="../../data/household_power_consumption.txt",
        window_size=24,
        horizon=6,
        batch_size=32
    )
    print("Load Data...")
    train_ds, val_ds, test_ds = loader.get_datasets()

    print("Result:")
    print(train_ds.element_spec)
