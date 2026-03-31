import logging
from pathlib import Path
import mlflow.sklearn
from mlflow.models import infer_signature

from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.metricscalculator import MetricsCalculator
from src.model_factory import ModelFactory, ModelType
from src.preprocessing import Preprocessor
from src.trainer import ModelTrainer
from src.visualiser import Visualiser

logger = logging.getLogger(__name__)


class TrainingPipeline:
    def __init__(self, config):
        self.config = config
        self.loader = DataLoader(config)
        self.cleaner = DataCleaner(config)
        self.preprocessor_gen = Preprocessor(config)
        self.factory = ModelFactory()
        self.metrics_calc = MetricsCalculator()
        self.visualizer = Visualiser()

    def run(self, model_type: ModelType, run_name: str):

        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)

        logger.info(f"Initializing data ingestion for model: {model_type.value}")
        raw_df = self.loader.fetch_raw_data()
        clean_df = self.cleaner.clean_data(raw_df)

        for col in clean_df.select_dtypes(include=["int64", "int32", "int16", "int8"]).columns:
            clean_df[col] = clean_df[col].astype("float64")

        X_train, X_test, y_train, y_test = self.loader.split_data(clean_df)

        numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()
        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

        mlflow.set_experiment("Ames_Housing_Tournament")

        with mlflow.start_run(run_name=run_name):
            self._log_metadata(model_type, numeric_features, categorical_features)

            preprocessor = self.preprocessor_gen.get_column_transformer(
                numeric_features, categorical_features
            )
            trainer = ModelTrainer(preprocessor, random_state=self.config.RANDOM_STATE)
            model = self.factory.get_model(model_type)
            pipeline = trainer.build_pipeline(model, use_log_transform=True)

            logger.info(f"Executing Cross-Validation for {model_type.value}...")
            cv_mean, cv_std = trainer.evaluate_with_cv(pipeline, X_train, y_train)

            logger.info(f"Fitting final pipeline for {model_type.value}...")
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            metrics = self.metrics_calc.get_metrics(y_test, y_pred)

            mlflow.log_metrics({"CV_RMSE_Mean": cv_mean, "CV_RMSE_Std": cv_std, **metrics})

            self._generate_artifacts(y_test, y_pred, model_type, reports_dir)

            signature = infer_signature(X_train, pipeline.predict(X_train))
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name=f"model_{model_type.value}",
                serialization_format="skops",
                skops_trusted_types=[
                    "numpy.dtype",
                    "src.config.Config",
                    "src.preprocessing.Preprocessor",
                    "src.preprocessing._convert_to_sqm",
                    "sklearn._loss.link.IdentityLink",
                    "sklearn._loss.link.Interval",
                    "sklearn._loss.loss.HalfSquaredError",
                ],
                signature=signature,
                input_example=X_train.iloc[:5],
                pip_requirements=[
                    "mlflow",
                    "scikit-learn",
                    "pandas",
                    "numpy",
                    "scipy",
                    "matplotlib",
                    "seaborn",
                    "joblib",
                ],
            )
            logger.info(
                f"Model {model_type.value} successfully logged. "
                f"Test RMSE: {metrics['RMSE']:.2f} | R2: {metrics['R2']:.4f}"
            )

    def _log_metadata(self, model_type, num_feats, cat_feats):
        mlflow.log_params({
            "model_type": model_type.value,
            "area_limit_sqm": self.config.AREA_LIMIT_SQM,
            "min_price_threshold": self.config.MIN_PRICE_THRESHOLD,
            "test_size": self.config.TEST_SIZE,
            "random_state": self.config.RANDOM_STATE,
            "features_count": len(num_feats) + len(cat_feats)
        })
        mlflow.log_param("features_numeric", num_feats)
        mlflow.log_param("features_categorical", cat_feats)

    def _generate_artifacts(self, y_test, y_pred, model_type, reports_dir):
        pred_path = reports_dir / f"{model_type.value}_pred.png"
        res_path = reports_dir / f"{model_type.value}_residuals.png"

        self.visualizer.plot_predicted_vs_actual(y_test, y_pred, str(pred_path))
        self.visualizer.plot_residuals(y_test, y_pred, str(res_path))

        mlflow.log_artifact(str(pred_path))
        mlflow.log_artifact(str(res_path))