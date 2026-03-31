import unittest

import pandas as pd

from src.inference_service import prepare_features


class TestInferenceService(unittest.TestCase):
    def test_prepare_features_orders_and_keeps_required_columns(self):
        df = pd.DataFrame(
            [
                {
                    "HouseStyle": "2Story",
                    "Neighborhood": "CollgCr",
                    "YearBuilt": 2005,
                    "FullBath": 2,
                    "OverallQual": 7,
                    "GarageArea": 500,
                    "LotArea": 9500,
                    "TotalBsmtSF": 1000,
                    "GrLivArea": 1800,
                    "Ignored": "x",
                }
            ]
        )

        prepared = prepare_features(df)

        self.assertEqual(
            list(prepared.columns),
            [
                "GrLivArea",
                "TotalBsmtSF",
                "LotArea",
                "GarageArea",
                "OverallQual",
                "FullBath",
                "YearBuilt",
                "Neighborhood",
                "HouseStyle",
            ],
        )
        self.assertNotIn("Ignored", prepared.columns)

    def test_prepare_features_raises_on_missing_columns(self):
        df = pd.DataFrame([{"GrLivArea": 1800}])

        with self.assertRaises(ValueError):
            prepare_features(df)


if __name__ == "__main__":
    unittest.main()

