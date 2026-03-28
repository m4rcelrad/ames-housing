import pandas as pd
from src.config import Config

class DataCleaner:
    def __init__(self, config: Config):
        self.config = config

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        limit_sqft = self.config.AREA_LIMIT_SQM / self.config.SQFT_TO_SQM_FACTOR
        df = df[df["GrLivArea"] < limit_sqft]

        df = df[df[self.config.TARGET_COLUMN] >= self.config.MIN_PRICE_THRESHOLD]

        df = df[~df[self.config.NEIGHBORHOOD_COLUMN].isin(self.config.EXCLUDED_NEIGHBORHOODS)]

        return df
