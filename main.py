import logging
import sys
from datetime import datetime

from src.config import Config
from src.pipeline import TrainingPipeline
from src.model_factory import ModelType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("experiment_logs.log")
    ]
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    config = Config()
    pipeline = TrainingPipeline(config)

    tournament_models = [
        ModelType.RIDGE,
        ModelType.LASSO,
        ModelType.RANDOM_FOREST,
        ModelType.GRADIENT_BOOSTING
    ]

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info(f"--- Tournament Initialized | Session: {session_id} ---")

    for model_type in tournament_models:
        run_name = f"{model_type.value.upper()}_{session_id}"

        logger.info(f"Dispatching training job for: {model_type.value}")

        try:
            pipeline.run(model_type=model_type, run_name=run_name)

        except:
            logger.error(
                f"Pipeline failed for {model_type.value}. skipping to next model.",
                exc_info=True
            )
            continue

    logger.info(f"--- Tournament Completed Successfully | Session: {session_id} ---")
