import os
from importlib.metadata import version

import mlflow.sklearn
from mlflow.models import infer_signature

from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.metricscalculator import MetricsCalculator
from src.model_factory import ModelFactory, ModelType
from src.preprocessing import Preprocessor
from src.trainer import ModelTrainer
from src.visualiser import Visualiser


class TrainingPipeline:
    def __init__(self, config):
        self.config = config
        self.loader = DataLoader(config)
        self.cleaner = DataCleaner(config)
        self.preprocessor_gen = Preprocessor(config)
        self.factory = ModelFactory()
        self.metrics_calc = MetricsCalculator()
        self.visualizer = Visualiser()

    def run(self, run_name="Pro_Ridge_Regression"):
        os.makedirs("reports", exist_ok=True)

        raw_df = self.loader.fetch_raw_data()
        clean_df = self.cleaner.clean_data(raw_df)

        for col in clean_df.select_dtypes(include=['int64']).columns:
            clean_df[col] = clean_df[col].astype('float64')

        X_train, X_test, y_train, y_test = self.loader.split_data(clean_df)

        numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()
        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

        mlflow.set_experiment("Ames_Housing_Price_Prediction")

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "area_limit_sqm": self.config.AREA_LIMIT_SQM,
                "min_price_threshold": self.config.MIN_PRICE_THRESHOLD,
                "excluded_neighborhoods": self.config.EXCLUDED_NEIGHBORHOODS,
                "test_size": self.config.TEST_SIZE,
                "random_state": self.config.RANDOM_STATE
            })
            preprocessor = self.preprocessor_gen.get_column_transformer(
                numeric_features, categorical_features
            )
            trainer = ModelTrainer(preprocessor, random_state=self.config.RANDOM_STATE)

            model = self.factory.get_model(ModelType.RIDGE)
            pipeline = trainer.build_pipeline(model, use_log_transform=True)

            mlflow.log_param("features_numeric", numeric_features)
            mlflow.log_param("features_categorical", categorical_features)

            mlflow.log_params({
                "dataset_name": self.config.DATASET_NAME,
                "test_size": self.config.TEST_SIZE,
                "random_state": self.config.RANDOM_STATE,
                "sqft_to_sqm_factor": self.config.SQFT_TO_SQM_FACTOR
            })

            cv_mean, cv_std = trainer.evaluate_with_cv(pipeline, X_train, y_train)
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            metrics = self.metrics_calc.get_metrics(y_test, y_pred)

            mlflow.log_metrics({"CV_RMSE_Mean": cv_mean, "CV_RMSE_Std": cv_std, **metrics})

            self.visualizer.plot_predicted_vs_actual(y_test, y_pred, "reports/pred.png")
            self.visualizer.plot_residuals(y_test, y_pred, "reports/residuals.png")
            mlflow.log_artifact("reports/pred.png")
            mlflow.log_artifact("reports/residuals.png")

            signature = infer_signature(X_train, pipeline.predict(X_train))

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="housing_model_pipeline",
                serialization_format="skops",
                skops_trusted_types=[
                    "numpy.dtype",
                    "src.config.Config",
                    "src.preprocessing.Preprocessor",
                    "src.preprocessing._convert_to_sqm"
                ],
                pip_requirements=[
                    f"mlflow=={version('mlflow')}",
                    f"scikit-learn=={version('scikit-learn')}",
                    f"skops=={version('skops')}",
                    f"numpy=={version('numpy')}",
                    f"pandas=={version('pandas')}",
                    f"scipy=={version('scipy')}",
                ],
                signature=signature,
                input_example=X_train.iloc[:5],
            )
