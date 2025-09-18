# app.py
from datetime import datetime
import os, time
import streamlit as st
import pandas as pd
from joblib import load
from src.predict import predict_week, predict_week_pregame

st.set_page_config(page_title="NFL Predictor", layout="wide")
st.title("🏈 walid's nfl predictions")
st.caption("always a work in progress and can very much be wrong")

# Debug banner so you know you're editing the right file
st.write(f"App file: `{__file__}` • CWD: `{os.getcwd()}` • Loaded: {time.strftime('%H:%M:%S')}")

# --- Sidebar ---
with st.sidebar:
    st.header("Inputs")
    season = st.number_input("Season", min_value=2009, max_value=2100, value=2025, step=1)
    all_weeks = st.checkbox("Predict all regular-season weeks (1–18)", value=True)
    weeks_input = st.multiselect("Weeks (ignored if 'all' checked)", list(range(1, 19)), default=[1])

    alpha = st.number_input("Ridge alpha", min_value=0.0, value=2.0, step=0.5, format="%.1f")

    model_choice = st.radio(
        "Model",
        ["Pre-game (rolling past games)", "Post-game (same-game PBP)"],
        index=0,
    )

    blend = False
    market_w = 0.7
    if model_choice.startswith("Pre"):
        blend = st.checkbox("Blend with market lines (pregame)", value=True)
        market_w = st.slider("Model weight vs market", 0.0, 1.0, 0.7, 0.05)

    use_calib = st.checkbox("Apply calibration (if available)", value=False)

    go = st.button("Run prediction", type="primary")

holder = st.empty()

# --- Single RUN + Single DISPLAY ---
if go:
    try:
        weeks = list(range(1, 19)) if all_weeks else list(weeks_input or [])
        with st.spinner("Crunching numbers…"):
            if model_choice.startswith("Pre"):
                df = predict_week_pregame(
                    season=int(season), weeks=weeks, alpha=float(alpha),
                    sims=0, rng_seed=0,
                    blend_with_market=bool(blend), market_weight=float(market_w)
                )
                note = "pre-game (rolling features)"
            else:
                df = predict_week(
                    season=int(season), weeks=weeks, alpha=float(alpha),
                    sims=0, rng_seed=0
                )
                note = "post-game (same-game features)"

        if df is None or df.empty:
            holder.warning("No games returned for that season/week selection."); st.stop()

        # Optional calibration
        applied = ""
        if use_calib:
            try:
                cal = load("data/cal_win_cover.joblib")
                df = cal.apply(df)
                applied = " • calibrated"
            except Exception:
                st.info("No calibration file at data/cal_win_cover.joblib. Skipping calibration.")

        weeks_str = "-".join(map(str, weeks)) or "all"
        with holder.container():
            st.subheader(f"Predictions — {note}{applied}")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download CSV",
                df.to_csv(index=False).encode(),
                file_name=f"preds_{season}_{weeks_str}.csv",
                mime="text/csv",
            )
        st.stop()

    except Exception as e:
        import traceback
        holder.error(f"{type(e).__name__}: {e}")
        holder.code(traceback.format_exc())
else:
    holder.info("Pick inputs on the left, then click **Run prediction**.")
