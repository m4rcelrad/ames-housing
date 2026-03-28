import os
import mlflow.sklearn
from src.config import Config
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.model_factory import ModelFactory
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator

if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    config = Config()
    loader = DataLoader()
    preprocessor_gen = Preprocessor()
    factory = ModelFactory()
    evaluator = ModelEvaluator()

    X_train, X_test, y_train, y_test = loader.load_and_split()
    unit_pipe = loader.get_unit_converter_pipeline()
    X_train = unit_pipe.transform(X_train)
    X_test = unit_pipe.transform(X_test)

    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    mlflow.set_experiment("Ames_Housing_Price_Prediction")

    with mlflow.start_run(run_name="Baseline_Ridge_Regression"):
        preprocessor = preprocessor_gen.get_column_transformer(numeric_features, categorical_features)

        model_params = {"alpha": 1.0}
        model = factory.get_model("ridge", **model_params)

        trainer = ModelTrainer(preprocessor)
        pipeline = trainer.build_pipeline(model)
        pipeline = trainer.fit(pipeline, X_train, y_train)

        y_pred = pipeline.predict(X_test)
        metrics = evaluator.get_metrics(y_test, y_pred)

        mlflow.log_params(model_params)
        mlflow.log_metrics(metrics)

        evaluator.plot_predicted_vs_actual(y_test, y_pred, "reports/pred_vs_actual.png")
        evaluator.plot_residuals(y_test, y_pred, "reports/residuals.png")

        mlflow.log_artifact("reports/pred_vs_actual.png")
        mlflow.log_artifact("reports/residuals.png")

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="housing_model_pipeline",
            serialization_format="skops"
        )

        print(f"Run logged successfully. Metrics: {metrics}")
