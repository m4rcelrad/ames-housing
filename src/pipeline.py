import os

import mlflow.sklearn

from src.data_loader import DataLoader
from src.evaluator import ModelEvaluator
from src.model_factory import ModelFactory, ModelType
from src.preprocessing import Preprocessor
from src.trainer import ModelTrainer


class TrainingPipeline:
    def __init__(self, config):
        self.config = config
        self.loader = DataLoader(config)
        self.preprocessor_gen = Preprocessor(config)
        self.factory = ModelFactory()
        self.evaluator = ModelEvaluator()

    def run(self, run_name="Pro_Ridge_Regression"):
        os.makedirs("reports", exist_ok=True)

        raw_df = self.loader.fetch_raw_data()
        clean_df = self.loader.clean_data(raw_df)
        X_train, X_test, y_train, y_test = self.loader.split_data(clean_df)

        numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()
        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

        mlflow.set_experiment("Ames_Housing_Price_Prediction")

        with mlflow.start_run(run_name=run_name):
            preprocessor = self.preprocessor_gen.get_column_transformer(
                numeric_features, categorical_features
            )
            trainer = ModelTrainer(preprocessor, random_state=self.config.RANDOM_STATE)

            model = self.factory.get_model(ModelType.RIDGE)
            pipeline = trainer.build_pipeline(model, use_log_transform=True)

            cv_mean, cv_std = trainer.evaluate_with_cv(pipeline, X_train, y_train)
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            metrics = self.evaluator.get_metrics(y_test, y_pred)

            mlflow.log_metrics({"CV_RMSE_Mean": cv_mean, "CV_RMSE_Std": cv_std, **metrics})
            self.evaluator.plot_predicted_vs_actual(y_test, y_pred, "reports/pred.png", show_plot=False)
            mlflow.log_artifact("reports/pred.png")

            self.evaluator.plot_residuals(y_test, y_pred, "reports/residuals.png", show_plot=False)
            mlflow.log_artifact("reports/residuals.png")

            mlflow.sklearn.log_model(sk_model=pipeline, name="housing_model_pipeline")
