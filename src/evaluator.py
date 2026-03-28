import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


class ModelEvaluator:
    def __init__(self):
        pass

    def get_metrics(self, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        return {
            "RMSE": rmse,
            "R2": r2
        }

    def plot_predicted_vs_actual(self, y_true, y_pred, output_path=None):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)

        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

        plt.xlabel("Actual Price ($)")
        plt.ylabel("Predicted Price ($)")
        plt.title("Actual vs. Predicted Housing Prices")

        if output_path:
            plt.savefig(output_path)
        plt.show()

    def plot_residuals(self, y_true, y_pred, output_path=None):
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
        plt.axhline(y=0, color='red', linestyle='--')

        plt.xlabel("Predicted Price ($)")
        plt.ylabel("Residuals ($)")
        plt.title("Residual Plot")

        if output_path:
            plt.savefig(output_path)
        plt.show()