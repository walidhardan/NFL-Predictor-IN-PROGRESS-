# src/predict.py
import argparse
from datetime import datetime
import os
import numpy as np
import polars as pl
import pandas as pd
import nflreadpy as nfl
from sklearn.linear_model import Ridge

def predict_week(season: int, weeks=None, sims: int = 10000, alpha: float = 2.0, rng_seed: int = 7) -> pd.DataFrame:
    if weeks is None or len(weeks) == 0:
        weeks = list(range(1, 19))  # regular season default

    train_seasons = [s for s in [season - 2, season - 1] if s >= 2009]
    seasons = train_seasons + [season]

    pbp = nfl.load_pbp(seasons)
    sched = nfl.load_schedules(seasons)

    team_form = (
        pbp.group_by(["game_id", "posteam"])
           .agg([
               pl.col("epa").mean().alias("epa_pp"),
               (pl.col("epa") > 0).cast(pl.Int8).mean().alias("success_rate"),
           ])
    )

    X = (
        sched
        .join(team_form.rename({"posteam":"home_team"}), on=["game_id","home_team"], how="left")
        .join(team_form.rename({"posteam":"away_team"}), on=["game_id","away_team"], how="left", suffix="_opp")
        .with_columns([
            (pl.col("epa_pp") - pl.col("epa_pp_opp")).alias("epa_diff"),
            (pl.col("success_rate") - pl.col("success_rate_opp")).alias("sr_diff"),
            (pl.col("home_score") - pl.col("away_score")).alias("margin"),
            (pl.col("home_score") + pl.col("away_score")).alias("total"),
        ])
        .drop_nulls(["epa_diff","sr_diff","season"])
    ).to_pandas()

    feats = ["epa_diff", "sr_diff"]
    train = X[X["season"] < season].copy()
    test  = X[(X["season"] == season) & (X["week"].isin(weeks))].copy()
    if train.empty or test.empty:
        raise SystemExit("No data for the requested season or weeks. Check inputs.")

    m_margin = Ridge(alpha=alpha).fit(train[feats], train["margin"])
    m_total  = Ridge(alpha=alpha).fit(train[feats], train["total"])

    pm = m_margin.predict(test[feats])
    pt = m_total.predict(test[feats])

    pred_home = (pt + pm) / 2
    pred_away = (pt - pm) / 2
    lam_home = np.clip(pred_home, 0.1, None)
    lam_away = np.clip(pred_away, 0.1, None)

    sched_season = nfl.load_schedules([season]).select(
        "season","week","home_team","away_team","spread_line","total_line"
    ).to_pandas()

    base = test[["season","week","home_team","away_team"]].copy()
    base["lam_home"] = lam_home
    base["lam_away"] = lam_away
    df = base.merge(sched_season, on=["season","week","home_team","away_team"], how="left")

    rng = np.random.default_rng(rng_seed)
    home_win_prob = []
    home_cover_prob = []
