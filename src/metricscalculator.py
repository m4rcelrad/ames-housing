import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class MetricsCalculator:

    @staticmethod
    def get_metrics(y_test, y_pred):
        return {
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MAE": mean_absolute_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred)
        }
