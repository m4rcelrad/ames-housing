import logging
from typing import Any, Dict, Optional, Protocol, cast

# noinspection PyPackageRequirements
import mlflow
# noinspection PyPackageRequirements
import mlflow.sklearn
import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)


class SupportsPredict(Protocol):
    def predict(self, X: pd.DataFrame) -> Any:
        ...


def get_best_model_uri(experiment_name: str = Config.MLFLOW_EXPERIMENT_NAME) -> str:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(
            f"Experiment '{experiment_name}' not found. Run main.py first to train models."
        )

    runs_result = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.RMSE ASC"],
        max_results=1,
    )

    runs = runs_result if isinstance(runs_result, pd.DataFrame) else pd.DataFrame(runs_result)

    if runs.empty:
        raise ValueError("No runs found in MLflow. Train models first using main.py.")

    first_index = runs.index[0]
    run_id = str(runs.at[first_index, "run_id"])
    model_type = str(runs.at[first_index, "params.model_type"])
    model_uri = f"runs:/{run_id}/model_{model_type}"

    logger.info("Best model selected: %s (run_id=%s)", model_type, run_id)
    return model_uri


def load_model(model_uri: Optional[str] = None) -> SupportsPredict:
    uri = model_uri or get_best_model_uri()
    return cast(SupportsPredict, mlflow.sklearn.load_model(uri))


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in Config.FEATURES_TO_KEEP if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    prepared: pd.DataFrame = df.reindex(columns=Config.FEATURES_TO_KEEP).copy()

    integer_columns = [
        col for col in prepared.columns if pd.api.types.is_integer_dtype(prepared[col].dtype)
    ]
    if integer_columns:
        prepared.loc[:, integer_columns] = prepared.loc[:, integer_columns].astype("float64")

    return prepared


def predict_batch(model: SupportsPredict, df: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_features(df)
    predictions = model.predict(prepared)

    result = df.copy()
    result["PredictedSalePrice"] = predictions
    return result


def predict_single(model: SupportsPredict, payload: Dict[str, object]) -> float:
    sample = pd.DataFrame([payload])
    prepared = prepare_features(sample)
    prediction = model.predict(prepared)
    return float(prediction[0])

