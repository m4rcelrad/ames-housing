import matplotlib.pyplot as plt
import seaborn as sns


class Visualiser:
    @staticmethod
    def plot_predicted_vs_actual(y_true, y_pred, output_path=None, show_plot=False):
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

        if show_plot:
            plt.show()

        plt.close()

    @staticmethod
    def plot_residuals(y_true, y_pred, output_path=None, show_plot=False):
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
        plt.axhline(y=0, color='red', linestyle='--')

        plt.xlabel("Predicted Price ($)")
        plt.ylabel("Residuals ($)")
        plt.title("Residual Plot")

        if output_path:
            plt.savefig(output_path)

        if show_plot:
            plt.show()

        plt.close()
