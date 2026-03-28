from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from src.config import Config


class DataLoader:
    def __init__(self):
        self.config = Config()

    def _unit_converter_logic(self, X):
        X = X.copy()
        for col in self.config.AREA_COLUMNS:
            if col in X.columns:
                X[col] = (X[col] * self.config.SQFT_TO_SQM_FACTOR).round(2)
        return X

    def get_unit_converter_pipeline(self):
        return Pipeline(steps=[
            ("sqft_to_sqm", FunctionTransformer(self._unit_converter_logic))
        ])

    def load_and_split(self):
        dataset = fetch_openml(
            name=self.config.DATASET_NAME,
            as_frame=True,
            parser="auto"
        )
        df = dataset.frame

        X = df.drop(columns=[self.config.TARGET_COLUMN])
        y = df[self.config.TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE
        )

        return X_train, X_test, y_train, y_test
