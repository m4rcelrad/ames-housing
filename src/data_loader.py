from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from src.config import Config


class DataLoader:
    def __init__(self, config: Config):
        self.config = config

    def fetch_raw_data(self):
        dataset = fetch_openml(
            name=self.config.DATASET_NAME,
            as_frame=True,
            parser="auto"
        )
        return dataset.frame

    def split_data(self, df):
        X = df.drop(columns=[self.config.TARGET_COLUMN])
        y = df[self.config.TARGET_COLUMN]

        return train_test_split(
            X, y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE
        )
