# app.py
import io
from datetime import datetime
import streamlit as st
import pandas as pd

from src.predict import predict_week  # local import

import os, time
st.write(f"App file: `{__file__}` • CWD: `{os.getcwd()}` • Loaded: {time.strftime('%H:%M:%S')}")

st.set_page_config(page_title="NFL Predictor", layout="wide")
st.title("🏈 NFL Score & Probabilities by Walid Hardan")
st.caption("Demo: nflverse + simple Ridge features + Poisson sims")

with st.sidebar:
    st.header("Inputs")
    season = st.number_input("Season", min_value=2009, max_value=2100, value=2024, step=1)
    weeks = st.multiselect("Weeks (empty = all 1–18)", list(range(1, 19)), default=[1, 2])
    sims = st.number_input("Simulations per game", min_value=1000, max_value=100000, value=10000, step=1000)
    alpha = st.number_input("Ridge alpha", min_value=0.0, value=2.0, step=0.5, format="%.1f")
    go = st.button("Run prediction")

@st.cache_data(show_spinner=True)
def _predict_cached(season: int, weeks_tuple, sims: int, alpha: float) -> pd.DataFrame:
    weeks_list = list(weeks_tuple) if weeks_tuple else []
    return predict_week(season, weeks=weeks_list, sims=int(sims), alpha=float(alpha))

if go:
    try:
        df = _predict_cached(season, tuple(weeks), sims, alpha)
        st.success(f"Predicted {len(df)} games.")
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download CSV",
            data=csv_bytes,
            file_name=f"preds_{season}_{'-'.join(map(str, weeks)) or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Pick your inputs on the left, then click **Run prediction**.")
