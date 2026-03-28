from datetime import datetime
from src.config import Config
from src.pipeline import TrainingPipeline

if __name__ == "__main__":
    config = Config()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"Ridge_Regression_{timestamp}"

    pipeline = TrainingPipeline(config)
    pipeline.run(run_name=experiment_name)
