# src/predict.py
# Uses exact (non-simulated) probabilities and is robust to missing values in pre-game features.

from __future__ import annotations

from math import floor
from typing import Optional, Tuple

import nflreadpy as nfl
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import poisson, skellam
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from scipy.stats import skellam, poisson


# ---------------------------- helpers ---------------------------------
def _weeks_to_list(weeks) -> list[int]:
    """Empty/None => weeks 1–18; otherwise ensure list[int]."""
    if not weeks:
        return list(range(1, 19))
    return [int(w) for w in weeks]


def _clean_matrix(
    df: pd.DataFrame, med: Optional[pd.Series] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Coerce to numeric, replace inf with NaN, then fill:
    - if med is None: learn medians from df
    - fill NaN with medians; if any remain, fill with 0.0
    Returns (clean_df, medians_used)
    """
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    if med is None:
        med = df.median(numeric_only=True)
    df = df.fillna(med).fillna(0.0)
    return df, med


def exact_probs(
    lam_home: float, lam_away: float, spread: Optional[float] = None, total_line: Optional[float] = None
) -> tuple[float, float, float]:
    """
    Exact probabilities given Poisson means for home/away points.
    - Win: Skellam (Home - Away), strict >
    - Cover: Skellam thresholded by spread, pushes excluded (strict >)
    - Over: Poisson on total, pushes excluded (strict >)
    """
    # Home win: P(D > 0), D ~ Skellam(lam_home, lam_away)
    home_win = 1.0 - skellam.cdf(0, mu1=lam_home, mu2=lam_away)

    # Home cover: D + spread > 0  ->  D > -spread
    if spread is None or np.isnan(spread):
        home_cover = float("nan")
    else:
        thresh = -float(spread)
        home_cover = 1.0 - skellam.cdf(float(spread), mu1=lam_home, mu2=lam_away)

    # Over total_line (strict)
    if total_line is None or np.isnan(total_line):
        over_p = float("nan")
    else:
        lam_tot = lam_home + lam_away
        over_p = 1.0 - poisson.cdf(floor(float(total_line)), mu=lam_tot)

    return home_win, home_cover, over_p


# -------------------- POST-GAME (completed games only) -----------------
def predict_week(
    season: int,
    weeks=None,
    sims: int = 0,       # ignored (kept for CLI compatibility)
    alpha: float = 2.0,
    rng_seed: int = 0,   # ignored (kept for CLI compatibility)
) -> pd.DataFrame:
    """
    POST-GAME model: uses same-game PBP features (only available after a game is played).
    Probabilities are exact (Skellam/Poisson). Use this for analysis/backtesting, not pre-game picks.
    """
    weeks = _weeks_to_list(weeks)

    train_seasons = [s for s in [season - 2, season - 1] if s >= 2009]
    seasons = train_seasons + [season]

    pbp = nfl.load_pbp(seasons)
    sched = nfl.load_schedules(seasons)

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
        sched.join(team_form.rename({"posteam": "home_team"}), on=["game_id", "home_team"], how="left")
        .join(team_form.rename({"posteam": "away_team"}), on=["game_id", "away_team"], how="left", suffix="_opp")
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
        raise ValueError(f"No data for requested season/weeks. train={len(train)}, test={len(test)}")

    m_margin = Ridge(alpha=alpha).fit(train[feats], train["margin"])
    m_total = Ridge(alpha=alpha).fit(train[feats], train["total"])

    pm = m_margin.predict(test[feats])  # predicted margin
    pt = m_total.predict(test[feats])  # predicted total

    lam_home = np.clip((pt + pm) / 2, 0.1, None)
    lam_away = np.clip((pt - pm) / 2, 0.1, None)

    sched_season = (
        nfl.load_schedules([season])
        .select("season", "week", "home_team", "away_team", "spread_line", "total_line")
        .to_pandas()
    )

    base = test[["season", "week", "home_team", "away_team"]].copy()
    base["lam_home"] = lam_home
    base["lam_away"] = lam_away
    df = base.merge(sched_season, on=["season", "week", "home_team", "away_team"], how="left")

    home_win_prob, home_cover_prob, over_prob = [], [], []
    for lh, la, spr, ttl in zip(df["lam_home"], df["lam_away"], df["spread_line"], df["total_line"]):
        w, c, o = exact_probs(float(lh), float(la),
                              spread=float(spr) if pd.notna(spr) else None,
                              total_line=float(ttl) if pd.notna(ttl) else None)
        home_win_prob.append(w)
        home_cover_prob.append(c)
        over_prob.append(o)

    out = pd.DataFrame(
        {
            "season": df["season"],
            "week": df["week"],
            "home_team": df["home_team"],
            "away_team": df["away_team"],
            "spread_line": df["spread_line"].round(1),
            "total_line": df["total_line"].round(1),
            "pred_home_mean": np.round(df["lam_home"], 1),
            "pred_away_mean": np.round(df["lam_away"], 1),
            "home_win_prob": np.round(home_win_prob, 3),
            "home_cover_prob": np.round(home_cover_prob, 3),
            "over_prob": np.round(over_prob, 3),
        }
    )
    return out.sort_values(["week", "home_team"]).reset_index(drop=True)


# --------------- PRE-GAME (future games; rolling features) ---------------
def predict_week_pregame(
    season: int,
    weeks=None,
    sims: int = 0,                # ignored (kept for CLI compatibility)
    alpha: float = 2.0,
    rng_seed: int = 0,            # ignored (kept for CLI compatibility)
    roll_window: int = 4,
    blend_with_market: bool = True,
    market_weight: float = 0.7,   # 0.0 = only market, 1.0 = only model
) -> pd.DataFrame:
    """
    PRE-GAME model with added context features:
      - Rolling EPA/SR (offense) pass/rush/all (prior games only)
      - Rest/bye/travel: days_rest, short_week, off_bye, back-to-back road
      - Venue buckets: indoor, grass
      - QB participation flag: qb_starter_90 (snap% >= 90 last week)
    Uses exact Skellam/Poisson probabilities (no Monte Carlo).
    `spread_line` is the nflverse home-perspective line (home favored => positive).
    """
    # ---- tiny helpers (local; no other files needed) -----------------
    def _weeks_to_list(ws):
        return list(range(1, 19)) if not ws else [int(w) for w in ws]

    def _clean_matrix(df: pd.DataFrame, med: pd.Series | None = None):
        df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if med is None:
            med = df.median(numeric_only=True)
        df = df.fillna(med).fillna(0.0)
        return df, med

    def _exact_probs(lh: float, la: float, spread: float | None, total_line: float | None):
        # Home win: P(D > 0), D ~ Skellam(lh, la)
        home_win = 1.0 - skellam.cdf(0, mu1=lh, mu2=la)
        # Home cover (nflverse): D > spread_line
        if spread is None or np.isnan(spread):
            home_cover = float("nan")
        else:
            home_cover = 1.0 - skellam.cdf(float(spread), mu1=lh, mu2=la)
        # Over total (strict >, pushes excluded)
        if total_line is None or np.isnan(total_line):
            over_p = float("nan")
        else:
            over_p = 1.0 - poisson.cdf(int(np.floor(float(total_line))), mu=lh + la)
        return home_win, home_cover, over_p

    def _make_rest_travel(sched_pl: pl.DataFrame) -> pl.DataFrame:
        s = (
            sched_pl.select(["game_id","season","week","gameday","home_team","away_team","roof","surface"])
            .with_columns(pl.col("gameday").str.strptime(pl.Date, "%Y-%m-%d").alias("date"))
            .melt(
                id_vars=["game_id","season","week","date","roof","surface"],
                value_vars=["home_team","away_team"],
                variable_name="ha", value_name="team"
            )
            .sort(["team","date"])
            # days_rest: robust across Polars versions
            .with_columns(
                (pl.col("date").cast(pl.Int32) - pl.col("date").cast(pl.Int32).shift(1).over("team"))
                .alias("days_rest")
            )
            # flags derived from days_rest
            .with_columns([
                (pl.col("days_rest") <= 4).fill_null(False).cast(pl.Int8).alias("short_week"),
                (pl.col("days_rest") >= 12).fill_null(False).cast(pl.Int8).alias("off_bye"),
            ])
        #    create is_road first…
            .with_columns((pl.col("ha") == "away_team").cast(pl.Int8).alias("is_road"))
            # …then prev_road in a separate step
            .with_columns(pl.col("is_road").shift(1).over("team").alias("prev_road"))
            # …then b2b_road using both
            .with_columns(((pl.col("is_road") == 1) & (pl.col("prev_road") == 1)).cast(pl.Int8).alias("b2b_road"))
        )
        feats = (
            s.select(["game_id","team","days_rest","short_week","off_bye","b2b_road"])
            .group_by(["game_id","team"]).tail(1)
        )
        return feats


    def _make_roof_surface_buckets(sched_pl: pl.DataFrame) -> pl.DataFrame:
        return (
            sched_pl.select(["game_id","roof","surface"])
            .with_columns([
                pl.when(pl.col("roof").is_in(["dome","closed"])).then(1).otherwise(0).cast(pl.Int8).alias("indoor"),
                pl.when(pl.col("surface").str.contains("grass", literal=True, strict=False))
                  .then(1).otherwise(0).cast(pl.Int8).alias("grass"),
            ])
            .unique(keep="first")
        )

    def _make_qb_flags(seasons_list: list[int]) -> pl.DataFrame:
    # QB last-week snap >= 90% (proxy for stable starter)
        try:
            parts = nfl.load_participation(seasons_list)
        except Exception:
            # empty but with explicit schema so joins work
            return pl.DataFrame(
                schema={
                    "season": pl.Int64,
                    "week": pl.Int64,
                    "team": pl.Utf8,
                    "qb_starter_90": pl.Int8,
                }
            )
        if parts.height == 0:
            return pl.DataFrame(
                schema={
                    "season": pl.Int64,
                    "week": pl.Int64,
                    "team": pl.Utf8,
                    "qb_starter_90": pl.Int8,
                }
            )

        qb = (
            parts.filter(pl.col("position") == "QB")
                .group_by(["season","week","team","player"])
                .agg(pl.col("offense_pct").mean().alias("snap_pct"))
        )
        qb1 = (
            qb.sort(["team","season","week","snap_pct"], descending=[False, False, False, True])
            .group_by(["season","week","team"]).head(1)
            .with_columns((pl.col("snap_pct") >= 90).cast(pl.Int8).alias("qb_starter_90"))
            .select(["season","week","team","qb_starter_90"])
        )
        # ensure schema even if result happens to be empty
        return qb1.with_columns([
            pl.col("season").cast(pl.Int64),
            pl.col("week").cast(pl.Int64),
            pl.col("team").cast(pl.Utf8),
            pl.col("qb_starter_90").cast(pl.Int8),
        ])


    # ---- assemble data ------------------------------------------------
    weeks = _weeks_to_list(weeks)
    seasons = [s for s in [season-2, season-1, season] if s >= 2009]

    pbp = nfl.load_pbp(seasons)
    sched_all = nfl.load_schedules(seasons)

    # per-team per-game (offense) split pass/rush + all
    pass_flag = pl.col("pass").cast(pl.Int8)
    rush_flag = pl.col("rush").cast(pl.Int8)
    game_feats = (
        pbp.group_by(["game_id","posteam"])
           .agg([
               pl.col("epa").mean().alias("epa_all"),
               (pl.col("epa") > 0).cast(pl.Int8).mean().alias("sr_all"),
               pl.when(pass_flag==1).then(pl.col("epa")).otherwise(None).mean().alias("epa_pass"),
               pl.when(pass_flag==1).then((pl.col("epa") > 0).cast(pl.Int8)).otherwise(None).mean().alias("sr_pass"),
               pl.when(rush_flag==1).then(pl.col("epa")).otherwise(None).mean().alias("epa_rush"),
               pl.when(rush_flag==1).then((pl.col("epa") > 0).cast(pl.Int8)).otherwise(None).mean().alias("sr_rush"),
           ])
           .join(sched_all.select(["game_id","season","week"]), on="game_id", how="left")
    )

    # rolling (prior games only)
    win = int(roll_window)
    rolling = (
        game_feats.sort(["posteam","season","week"])
        .with_columns([
            pl.col("epa_all").rolling_mean(win).shift(1).over("posteam").alias("epa_all_roll"),
            pl.col("sr_all").rolling_mean(win).shift(1).over("posteam").alias("sr_all_roll"),
            pl.col("epa_pass").rolling_mean(win).shift(1).over("posteam").alias("epa_pass_roll"),
            pl.col("sr_pass").rolling_mean(win).shift(1).over("posteam").alias("sr_pass_roll"),
            pl.col("epa_rush").rolling_mean(win).shift(1).over("posteam").alias("epa_rush_roll"),
            pl.col("sr_rush").rolling_mean(win).shift(1).over("posteam").alias("sr_rush_roll"),
        ])
    )

    # context features (rest/travel, roof/surface, QB flags)
    rest = _make_rest_travel(sched_all)
    roof = _make_roof_surface_buckets(sched_all)
    qb_flags = _make_qb_flags([season-1, season])

    qb_flags = qb_flags.cast({
        "season": pl.Int64,
        "week": pl.Int64,
        "team": pl.Utf8,
        "qb_starter_90": pl.Int8,
    })


    # ---- TRAIN matrix (past games, with targets) ----------------------
    train_mat = (
        sched_all
        .join(rolling.rename({"posteam":"home_team"}), on=["game_id","home_team"], how="left")
        .join(rolling.rename({"posteam":"away_team"}), on=["game_id","away_team"], how="left", suffix="_opp")
        .join(rest.rename({"team":"home_team"}), on=["game_id","home_team"], how="left")
        .rename({"days_rest":"home_days_rest","short_week":"home_short_week","off_bye":"home_off_bye","b2b_road":"home_b2b_road"})
        .join(rest.rename({"team":"away_team"}), on=["game_id","away_team"], how="left")
        .rename({"days_rest":"away_days_rest","short_week":"away_short_week","off_bye":"away_off_bye","b2b_road":"away_b2b_road"})
        .join(roof, on="game_id", how="left")
        .join(qb_flags.rename({"team":"home_team"}), on=["season","week","home_team"], how="left")
        .rename({"qb_starter_90":"home_qb_starter90"})
        .join(qb_flags.rename({"team":"away_team"}), on=["season","week","away_team"], how="left")
        .rename({"qb_starter_90":"away_qb_starter90"})
        .drop_nulls(["season","week","home_score","away_score"])  # keep rows with outcomes
        .with_columns([
            (pl.col("epa_pass_roll") - pl.col("epa_pass_roll_opp")).alias("pass_epa_diff"),
            (pl.col("epa_rush_roll") - pl.col("epa_rush_roll_opp")).alias("rush_epa_diff"),
            (pl.col("sr_pass_roll")  - pl.col("sr_pass_roll_opp")).alias("pass_sr_diff"),
            (pl.col("sr_rush_roll")  - pl.col("sr_rush_roll_opp")).alias("rush_sr_diff"),
            (pl.col("epa_all_roll")  - pl.col("epa_all_roll_opp")).alias("all_epa_diff"),
            (pl.col("sr_all_roll")   - pl.col("sr_all_roll_opp")).alias("all_sr_diff"),
            (pl.col("home_score") - pl.col("away_score")).alias("margin"),
            (pl.col("home_score") + pl.col("away_score")).alias("total"),
        ])
        .to_pandas()
    )

    # same extra features for TRAIN
    train_extra = pd.DataFrame({
        "rest_diff":  train_mat["home_days_rest"] - train_mat["away_days_rest"],
        "bye_edge":   train_mat["home_off_bye"].fillna(0) - train_mat["away_off_bye"].fillna(0),
        "short_wk_d": train_mat["home_short_week"].fillna(0) - train_mat["away_short_week"].fillna(0),
        "b2b_road_d": train_mat["home_b2b_road"].fillna(0) - train_mat["away_b2b_road"].fillna(0),
        "indoor":     train_mat["indoor"].fillna(0),
        "grass":      train_mat["grass"].fillna(0),
        "qb_start_d": train_mat["home_qb_starter90"].fillna(0) - train_mat["away_qb_starter90"].fillna(0),
    })

    base_feats = ["pass_epa_diff","rush_epa_diff","pass_sr_diff","rush_sr_diff","all_epa_diff","all_sr_diff"]
    extra_feats = ["rest_diff","bye_edge","short_wk_d","b2b_road_d","indoor","grass","qb_start_d"]
    feats = base_feats + extra_feats

    mask_tr = train_mat["season"] < season
    Xtr = pd.concat([train_mat.loc[mask_tr, base_feats].reset_index(drop=True),
                     train_extra.loc[mask_tr, extra_feats].reset_index(drop=True)], axis=1)
    y_margin = train_mat.loc[mask_tr, "margin"].copy()
    y_total  = train_mat.loc[mask_tr, "total"].copy()
    if Xtr.empty:
        raise ValueError("Not enough past data to train.")

    # ---- PRED rows (target weeks with same features) ------------------
    sched_target = nfl.load_schedules([season]).filter(pl.col("week").is_in(weeks))
    pred_rows_pl = (
        sched_target
        .join(rolling.rename({"posteam":"home_team"}), on=["game_id","home_team"], how="left")
        .join(rolling.rename({"posteam":"away_team"}), on=["game_id","away_team"], how="left", suffix="_opp")
        .join(_make_rest_travel(nfl.load_schedules([season])).rename({"team":"home_team"}), on=["game_id","home_team"], how="left")
        .rename({"days_rest":"home_days_rest","short_week":"home_short_week","off_bye":"home_off_bye","b2b_road":"home_b2b_road"})
        .join(_make_rest_travel(nfl.load_schedules([season])).rename({"team":"away_team"}), on=["game_id","away_team"], how="left")
        .rename({"days_rest":"away_days_rest","short_week":"away_short_week","off_bye":"away_off_bye","b2b_road":"away_b2b_road"})
        .join(_make_roof_surface_buckets(nfl.load_schedules([season])), on="game_id", how="left")
        .join(qb_flags.rename({"team":"home_team"}), on=["season","week","home_team"], how="left")
        .rename({"qb_starter_90":"home_qb_starter90"})
        .join(qb_flags.rename({"team":"away_team"}), on=["season","week","away_team"], how="left")
        .rename({"qb_starter_90":"away_qb_starter90"})
    )
    pred_rows = pred_rows_pl.to_pandas()

    Xte_base = pd.DataFrame({
        "pass_epa_diff": pred_rows["epa_pass_roll"] - pred_rows["epa_pass_roll_opp"],
        "rush_epa_diff": pred_rows["epa_rush_roll"] - pred_rows["epa_rush_roll_opp"],
        "pass_sr_diff":  pred_rows["sr_pass_roll"]  - pred_rows["sr_pass_roll_opp"],
        "rush_sr_diff":  pred_rows["sr_rush_roll"]  - pred_rows["sr_rush_roll_opp"],
        "all_epa_diff":  pred_rows["epa_all_roll"]  - pred_rows["epa_all_roll_opp"],
        "all_sr_diff":   pred_rows["sr_all_roll"]   - pred_rows["sr_all_roll_opp"],
    })
    Xte_extra = pd.DataFrame({
        "rest_diff":  pred_rows["home_days_rest"] - pred_rows["away_days_rest"],
        "bye_edge":   pred_rows["home_off_bye"].fillna(0) - pred_rows["away_off_bye"].fillna(0),
        "short_wk_d": pred_rows["home_short_week"].fillna(0) - pred_rows["away_short_week"].fillna(0),
        "b2b_road_d": pred_rows["home_b2b_road"].fillna(0) - pred_rows["away_b2b_road"].fillna(0),
        "indoor":     pred_rows["indoor"].fillna(0),
        "grass":      pred_rows["grass"].fillna(0),
        "qb_start_d": pred_rows["home_qb_starter90"].fillna(0) - pred_rows["away_qb_starter90"].fillna(0),
    })
    Xte = pd.concat([Xte_base, Xte_extra], axis=1)

    # ---- robust cleaning + fit ---------------------------------------
    Xtr, med = _clean_matrix(Xtr, med=None)
    Xte, _   = _clean_matrix(Xte,  med=med)

    m_margin = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=alpha)).fit(Xtr, y_margin)
    m_total  = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=alpha)).fit(Xtr, y_total)

    pm = m_margin.predict(Xte)   # model margin
    pt = m_total.predict(Xte)    # model total

    # ---- optional market blend (NOTE sign: home margin = +spread_line) -
    sched_lines = nfl.load_schedules([season]).select(
        "season","week","home_team","away_team","spread_line","total_line"
    ).to_pandas()

    base = pred_rows[["season","week","home_team","away_team"]].copy()
    base["model_margin"] = pm
    base["model_total"]  = pt
    df = base.merge(sched_lines, on=["season","week","home_team","away_team"], how="left")

    if blend_with_market:
        w = float(market_weight)
        implied_margin = df["spread_line"].astype(float)
        implied_total  = df["total_line"]
        pm_final = np.where(implied_margin.notna(), w*df["model_margin"] + (1-w)*implied_margin, df["model_margin"])
        pt_final = np.where(implied_total.notna(),  w*df["model_total"]  + (1-w)*implied_total,  df["model_total"])
    else:
        pm_final = df["model_margin"].values
        pt_final = df["model_total"].values

    # ---- Poisson means + exact probabilities -------------------------
    lam_home = np.clip((pt_final + pm_final) / 2, 0.1, None)
    lam_away = np.clip((pt_final - pm_final) / 2, 0.1, None)

    home_win_prob, home_cover_prob, over_prob = [], [], []
    for lh, la, spr, ttl in zip(lam_home, lam_away, df["spread_line"], df["total_line"]):
        wprob, cprob, oprob = _exact_probs(float(lh), float(la),
                                           float(spr) if pd.notna(spr) else None,
                                           float(ttl) if pd.notna(ttl) else None)
        home_win_prob.append(wprob); home_cover_prob.append(cprob); over_prob.append(oprob)

    out = pd.DataFrame({
        "season": df["season"], "week": df["week"],
        "home_team": df["home_team"], "away_team": df["away_team"],
        "spread_line": df["spread_line"].round(1), "total_line": df["total_line"].round(1),
        "pred_home_mean": np.round(lam_home, 1), "pred_away_mean": np.round(lam_away, 1),
        "home_win_prob": np.round(home_win_prob, 3),
        "home_cover_prob": np.round(home_cover_prob, 3),
        "over_prob": np.round(over_prob, 3),
        # debug/inspection
        "model_margin": np.round(base["model_margin"], 2),
        "model_total":  np.round(base["model_total"], 2),
        "blended_margin": np.round(pm_final, 2),
        "blended_total":  np.round(pt_final, 2),
    }).sort_values(["week","home_team"]).reset_index(drop=True)

    return out
