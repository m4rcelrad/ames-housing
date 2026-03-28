class Config:
    DATASET_NAME = "house_prices"
    TARGET_COLUMN = "SalePrice"
    NEIGHBORHOOD_COLUMN = "Neighborhood"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    SQFT_TO_SQM_FACTOR = 0.092903

    AREA_COLUMNS = [
        "GrLivArea",
        "TotalBsmtSF",
        "LotArea",
        "GarageArea"
    ]

    REMAINDER_STRATEGY = "passthrough"
