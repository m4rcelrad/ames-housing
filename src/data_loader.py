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

    def clean_data(self, df):
        df = df.copy()

        limit_sqft = 360 / self.config.SQFT_TO_SQM_FACTOR
        df = df[df["GrLivArea"] < limit_sqft]

        df = df[df[self.config.TARGET_COLUMN] >= 35000]

        nb_col = self.config.NEIGHBORHOOD_COLUMN
        df = df[~df[nb_col].isin(["Landmark", "Green Hills"])]

        return df

    def split_data(self, df):
        X = df.drop(columns=[self.config.TARGET_COLUMN])
        y = df[self.config.TARGET_COLUMN]

        return train_test_split(
            X, y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE
        )
