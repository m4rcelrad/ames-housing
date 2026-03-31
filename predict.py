import argparse
import logging

from src.inference_service import load_model, predict_single

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict Ames house price from sample features")
    parser.add_argument(
        "--model-uri",
        type=str,
        default=None,
        help="Optional MLflow model URI (default: best model by RMSE)",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    try:
        model = load_model(args.model_uri)

        sample_house = {
            "GrLivArea": 1800.0,
            "TotalBsmtSF": 1000.0,
            "LotArea": 9500.0,
            "GarageArea": 500.0,
            "OverallQual": 7.0,
            "FullBath": 2.0,
            "YearBuilt": 2005.0,
            "Neighborhood": "CollgCr",
            "HouseStyle": "2Story",
        }

        prediction = predict_single(model, sample_house)
        logger.info("Predicted Price for sample house: $%s", format(prediction, ",.2f"))

    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
