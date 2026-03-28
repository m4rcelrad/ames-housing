from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, TargetEncoder, FunctionTransformer
from src.config import Config


def _convert_to_sqm(X, factor):
    return X * factor

class Preprocessor:
    def __init__(self, config: Config):
        self.config = config

    def get_column_transformer(self, numeric_features, categorical_features):
        area_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("converter", FunctionTransformer(
                _convert_to_sqm,
                kw_args={"factor": self.config.SQFT_TO_SQM_FACTOR},
                feature_names_out="one-to-one"
            )),
            ("scaler", RobustScaler())
        ])

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler())
        ])

        neighborhood_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("target_enc", TargetEncoder(target_type="continuous", smooth="auto", cv=5))
        ])

        other_cat_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        area_cols = [col for col in numeric_features if col in self.config.AREA_COLUMNS]
        other_num_cols = [col for col in numeric_features if col not in self.config.AREA_COLUMNS]

        nb_col = self.config.NEIGHBORHOOD_COLUMN
        other_cat_cols = [col for col in categorical_features if col != nb_col]

        return ColumnTransformer(
            transformers=[
                ("area", area_transformer, area_cols),
                ("num", numeric_transformer, other_num_cols),
                ("neighborhood", neighborhood_transformer, [nb_col]),
                ("cat_other", other_cat_transformer, other_cat_cols)
            ],
            remainder=self.config.REMAINDER_STRATEGY
        )
