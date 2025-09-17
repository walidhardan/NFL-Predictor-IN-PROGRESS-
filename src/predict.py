# src/predict.py
import numpy as np
import polars as pl
import pandas as pd
import nflreadpy as nfl
from sklearn.linear_model import Ridge


def predict_week(
    season: int,
    weeks=None,
    sims: int = 10000,
    alpha: float = 2.0,
    rng_seed: int = 7,
) -> pd.DataFrame:
    """Train on prior seasons and predict the requested season/week(s)."""
    # 0) Weeks handling: empty/None => all 1–18; ensure int list
    if not weeks:
        weeks = list(range(1, 19))
    else:
        weeks = [int(w) for w in weeks]

    # 1) Load seasons (prev 1–2 seasons + target season)
    train_seasons = [s for s in [season - 2, season - 1] if s >= 2009]
    seasons = train_seasons + [season]

    pbp = nfl.load_pbp(seasons)
    sched = nfl.load_schedules(seasons)

    # 2) Per-game team features
    team_form = (
        pbp.group_by(["game_id", "posteam"])
        .agg(
            [
                pl.col("epa").mean().alias("epa_pp"),
                (pl.col("epa") > 0).cast(pl.Int8).mean().alias("success_rate"),
            ]
        )
    )

    X = (
        sched
        .join(
            team_form.rename({"posteam": "home_team"}),
            on=["game_id", "home_team"],
            how="left",
        )
        .join(
            team_form.rename({"posteam": "away_team"}),
            on=["game_id", "away_team"],
            how="left",
            suffix="_opp",
        )
        .with_columns(
            [
                (pl.col("epa_pp") - pl.col("epa_pp_opp")).alias("epa_diff"),
                (pl.col("success_rate") - pl.col("success_rate_opp")).alias("sr_diff"),
                (pl.col("home_score") - pl.col("away_score")).alias("margin"),
                (pl.col("home_score") + pl.col("away_score")).alias("total"),
            ]
        )
        .drop_nulls(["epa_diff", "sr_diff", "season"])
    ).to_pandas()

    feats = ["epa_diff", "sr_diff"]
    train = X[X["season"] < season].copy()
    test = X[(X["season"] == season) & (X["week"].isin(weeks))].copy()

    if train.empty or test.empty:
        raise ValueError(
            f"No data for requested season/weeks. train={len(train)}, test={len(test)}"
        )

    # 3) Train models
    m_margin = Ridge(alpha=alpha).fit(train[feats], train["margin"])
    m_total = Ridge(alpha=alpha).fit(train[feats], train["total"])

    pm = m_margin.predict(test[feats])  # predicted margin
    pt = m_total.predict(test[feats])   # predicted total

    # 4) Convert to Poisson means and simulate
    pred_home = (pt + pm) / 2
    pred_away = (pt - pm) / 2
    lam_home = np.clip(pred_home, 0.1, None)
    lam_away = np.clip(pred_away, 0.1, None)

    sched_season = nfl.load_schedules([season]).select(
        "season", "week", "home_team", "away_team", "spread_line", "total_line"
    ).to_pandas()

    base = test[["season", "week", "home_team", "away_team"]].copy()
    base["lam_home"] = lam_home
    base["lam_away"] = lam_away
    df = base.merge(
        sched_season,
        on=["season", "week", "home_team", "away_team"],
        how="left",
    )

    rng = np.random.default_rng(rng_seed)
    home_win_prob, home_cover_prob, over_prob = [], [], []
    mean_home, mean_away = [], []

    for lh, la, spr, ttl in zip(df.lam_home, df.lam_away, df.spread_line, df.total_line):
        h = rng.poisson(lh, size=sims)
        a = rng.poisson(la, size=sims)
        margin = h - a
        total = h + a
        home_win_prob.append((h > a).mean())
        home_cover_prob.append(np.nan if pd.isna(spr) else (margin + spr > 0).mean())
        over_prob.append(np.nan if pd.isna(ttl) else (total > ttl).mean())
        mean_home.append(h.mean())
        mean_away.append(a.mean())

    out = pd.DataFrame(
        {
            "season": df["season"],
            "week": df["week"],
            "home_team": df["home_team"],
            "away_team": df["away_team"],
            "spread_line": df["spread_line"].round(1),
            "total_line": df["total_line"].round(1),
            "pred_home_mean": np.round(mean_home, 1),
            "pred_away_mean": np.round(mean_away, 1),
            "home_win_prob": np.round(home_win_prob, 3),
            "home_cover_prob": np.round(home_cover_prob, 3),
            "over_prob": np.round(over_prob, 3),
        }
    )

    assert isinstance(out, pd.DataFrame)
    return out.sort_values(["week", "home_team"]).reset_index(drop=True)
