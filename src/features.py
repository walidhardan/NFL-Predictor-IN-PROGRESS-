# src/features.py
from __future__ import annotations
import polars as pl
import nflreadpy as nfl

def make_rest_travel(sched: pl.DataFrame) -> pl.DataFrame:
    # sched needs: game_id, season, week, gameday (YYYY-MM-DD), home_team, away_team, roof, surface
    s = (sched
         .with_columns(pl.col("gameday").str.strptime(pl.Date, "%Y-%m-%d").alias("date"))
         .melt(id_vars=["game_id","season","week","date","roof","surface"],
               value_vars=["home_team","away_team"],
               variable_name="ha", value_name="team")
         .sort(["team","date"])
         .with_columns([
             pl.col("date").diff().over("team").dt.days().alias("days_rest"),
         ])
         .with_columns([
             (pl.col("days_rest") <= 4).fill_null(False).cast(pl.Int8).alias("short_week"),
             (pl.col("days_rest") >= 12).fill_null(False).cast(pl.Int8).alias("off_bye"),
         ])
    )
    # back-to-back road flag
    s = (s
         .with_columns([
             (pl.col("ha") == "away_team").cast(pl.Int8).alias("is_road"),
             pl.col("is_road").shift(1).over("team").alias("prev_road")
         ])
         .with_columns(((pl.col("is_road") == 1) & (pl.col("prev_road") == 1))
                       .cast(pl.Int8).alias("b2b_road"))
    )
    # compact team-row table to join per side
    feats = (s.select(["game_id","team","days_rest","short_week","off_bye","b2b_road"])
               .group_by(["game_id","team"]).tail(1))
    return feats

def make_roof_surface_buckets(sched: pl.DataFrame) -> pl.DataFrame:
    # simple binary buckets: indoor vs outdoor; grass vs turf
    return (sched.select(["game_id","roof","surface"])
                 .with_columns([
                     pl.when(pl.col("roof").is_in(["dome","closed"]))
                       .then(1).otherwise(0).cast(pl.Int8).alias("indoor"),
                     pl.when(pl.col("surface").str.contains("grass", literal=True, strict=False))
                       .then(1).otherwise(0).cast(pl.Int8).alias("grass"),
                 ]))

def make_participation_flags(seasons: list[int]) -> pl.DataFrame:
    # Basic: last-week QB snap% >= 90% (proxy for stable starter)
    parts = nfl.load_participation(seasons)  # play-level on-field participation
    qb = (parts.filter(pl.col("position") == "QB")
               .group_by(["season","week","team","player"])
               .agg(pl.col("offense_pct").mean().alias("snap_pct")))
    qb1 = (qb.sort(["team","season","week","snap_pct"], descending=[False, False, False, True])
             .group_by(["season","week","team"]).head(1)  # top QB by snap%
             .with_columns((pl.col("snap_pct") >= 90).cast(pl.Int8).alias("qb_starter_90")))
    return qb1.select(["season","week","team","qb_starter_90"])
