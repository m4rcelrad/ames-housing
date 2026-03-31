import io
from typing import Optional

import pandas as pd
import streamlit as st
from pandas.io.parsers.readers import TextFileReader

from src.inference_service import get_best_model_uri, load_model, predict_batch, predict_single

st.set_page_config(page_title="Ames Housing Predictor", page_icon="🏠", layout="wide")


# noinspection SpellCheckingInspection
DEFAULT_NEIGHBORHOODS = [
    "NAmes",
    "CollgCr",
    "OldTown",
    "Edwards",
    "Somerst",
    "NridgHt",
]

DEFAULT_HOUSE_STYLES = ["1Story", "2Story", "1.5Fin", "SLvl", "SFoyer"]


def _to_float(value: Optional[int | float], *, fallback: float = 0.0) -> float:
    return float(value if value is not None else fallback)


@st.cache_resource
def _load_model(uri: str):
    return load_model(uri)


def _single_prediction_tab(model):
    st.subheader("Single Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        # noinspection SpellCheckingInspection
        gr_liv_area = st.number_input("GrLivArea", min_value=100.0, value=1800.0, step=10.0)
        # noinspection SpellCheckingInspection
        total_basement_sf = st.number_input("TotalBsmtSF", min_value=0.0, value=1000.0, step=10.0)
        lot_area = st.number_input("LotArea", min_value=100.0, value=9500.0, step=50.0)

    with col2:
        garage_area = st.number_input("GarageArea", min_value=0.0, value=500.0, step=10.0)
        overall_qual = st.slider("OverallQual", min_value=1, max_value=10, value=7)
        full_bath = st.slider("FullBath", min_value=0, max_value=4, value=2)

    with col3:
        year_built = st.number_input("YearBuilt", min_value=1872, max_value=2026, value=2005, step=1)
        neighborhood = st.selectbox("Neighborhood", options=DEFAULT_NEIGHBORHOODS, index=1)
        house_style = st.selectbox("HouseStyle", options=DEFAULT_HOUSE_STYLES, index=1)

    if st.button("Predict Price", type="primary"):
        payload = {
            "GrLivArea": _to_float(gr_liv_area),
            "TotalBsmtSF": _to_float(total_basement_sf),
            "LotArea": _to_float(lot_area),
            "GarageArea": _to_float(garage_area),
            "OverallQual": _to_float(overall_qual),
            "FullBath": _to_float(full_bath),
            "YearBuilt": _to_float(year_built),
            "Neighborhood": neighborhood,
            "HouseStyle": house_style,
        }
        prediction = predict_single(model, payload)
        st.success(f"Predicted Sale Price: ${prediction:,.2f}")


def _batch_prediction_tab(model):
    st.subheader("Batch Prediction from CSV")
    # noinspection SpellCheckingInspection
    st.caption("CSV must include required model columns: GrLivArea, TotalBsmtSF, LotArea, GarageArea, OverallQual, FullBath, YearBuilt, Neighborhood, HouseStyle")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if not uploaded:
        return

    if isinstance(uploaded, list):
        st.error("Please upload a single CSV file.")
        return

    uploaded_file = uploaded

    read_result = pd.read_csv(uploaded_file)
    if isinstance(read_result, pd.DataFrame):
        input_df = read_result
    elif isinstance(read_result, TextFileReader):
        input_df = read_result.read()
    else:
        st.error("Could not parse CSV into a DataFrame.")
        return
    st.write("Preview of uploaded data")
    st.dataframe(input_df.head(), use_container_width=True)

    if st.button("Run Batch Prediction"):
        output_df = predict_batch(model, input_df)
        st.success(f"Generated {len(output_df)} predictions")
        st.dataframe(output_df.head(), use_container_width=True)

        csv_buffer = io.StringIO()
        output_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "Download predictions CSV",
            data=csv_buffer.getvalue(),
            file_name="ames_predictions.csv",
            mime="text/csv",
        )


def main():
    st.title("Ames Housing Price Prediction")
    st.write("Predict house prices using the best tracked MLflow model or a custom model URI.")

    source = st.radio("Model source", options=["Best MLflow model", "Custom model URI"], horizontal=True)

    if source == "Best MLflow model":
        model_uri: Optional[str] = get_best_model_uri()
        st.code(model_uri)
    else:
        model_uri = st.text_input("MLflow model URI", value="runs:/<run_id>/model_<model_type>")

    if not model_uri or "<run_id>" in model_uri:
        st.info("Provide a valid model URI or choose best model source.")
        return

    try:
        model = _load_model(model_uri)
    except Exception as exc:
        st.error(f"Model loading failed: {exc}")
        return

    tab1, tab2 = st.tabs(["Single", "Batch CSV"])
    with tab1:
        _single_prediction_tab(model)
    with tab2:
        _batch_prediction_tab(model)


if __name__ == "__main__":
    # noinspection SpellCheckingInspection
    main()

