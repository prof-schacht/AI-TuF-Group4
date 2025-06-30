import pandas as pd
from typing import Optional, List
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import logging
logger = logging.getLogger(__name__)


class LoadAndPrepareData:

    """
    Args:
        filepath: Path to the .txt file
        sep: Separator in the file
        na_values: Character for missing values in the file
        resample_rule: Pandas aggregation rule (e.g., "h" = hour)
        split_ratios: Ratios for train/val/test
        window_size: Length of the input sequence (time steps)
        horizon: Length of the prediction (time steps)
        batch_size: Batch size
    """
    def __init__(self,
                 filepath: str,
                 sep: str = ";",
                 na_values: Optional[List[str]] = None,
                 resample_rule: str = "h",
                 split_ratios: tuple = (0.7, 0.15, 0.15),
                 window_size: int = 24,
                 horizon: int = 6,
                 batch_size: int = 32,
                 test_mode: bool = False,
                 test_subset: int = None):

        # In test mode, override batch_size and resample_rule for robustness
        if test_mode:
            if test_subset is None:
                test_subset = 1000
            logger.info(f"[LoadAndPrepareData] test_mode: batch_size={batch_size}, resample_rule={resample_rule}, test_subset={test_subset}")

        self.filepath = filepath
        self.sep = sep
        if na_values is None:
            self.na_values = ["?"]
        else:
            self.na_values = na_values
        self.resample_rule = resample_rule
        self.test_subset = test_subset
        self.split_ratios = split_ratios
        self.window_size = window_size
        self.horizon = horizon
        self.batch_size = batch_size
        self.date_col: str = "Date"
        self.time_col: str = "Time"
        self.test_mode = test_mode

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

        # If test_mode is enabled, keep only the first test_subset rows
        if self.test_mode:
            self.df = self.df.iloc[:self.test_subset]
            logger.info(f"[LoadAndPrepareData] test_mode: DataFrame shape after slicing: {self.df.shape}")

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

        # Print number of batches in each dataset for debug
        def count_batches(ds):
            return sum(1 for _ in ds)
        n_train = count_batches(training_ds)
        n_val = count_batches(validation_ds)
        n_test = count_batches(testing_ds)
        logger.info(f"[LoadAndPrepareData] Batches - train: {n_train}, val: {n_val}, test: {n_test}")
        if n_train == 0 or n_val == 0 or n_test == 0:
            logger.warning("[LoadAndPrepareData] One or more datasets are empty! Consider increasing test subset or reducing window/horizon/batch size.")

        return training_ds, validation_ds, testing_ds

    def get_scaler(self):
        return self.scaler
