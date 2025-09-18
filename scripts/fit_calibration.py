import os, sys, traceback
import pandas as pd
import polars as pl
import nflreadpy as nfl
from joblib import dump
from src.predict import predict_week_pregame
from src.calibrate import ProbCalibrator

def main():
    print("=== Fit calibration ===")
    print("CWD:", os.getcwd())
    print("Will save to:", os.path.join(os.getcwd(), "data", "cal_win_cover.joblib"))

    SEASON = 2024
    MODEL_WEIGHT = 0.7   # adjust later if you tune it
    ROLL = 4

    try:
        # 1) Collect pre-game predictions week-by-week (these include spread_line already)
        preds = []
        for w in range(1, 19):
            print(f"Predicting season={SEASON} week={w} ...", flush=True)
            dfw = predict_week_pregame(
                season=SEASON, weeks=[w],
                roll_window=ROLL,
                blend_with_market=True,
                market_weight=MODEL_WEIGHT
            )
            preds.append(dfw)
        pred = pd.concat(preds, ignore_index=True)
        print("Predictions rows:", len(pred))
        # pred has: ... home_win_prob, home_cover_prob, spread_line, ...

        # 2) Join actual outcomes, but DO NOT include spread_line from schedule (avoid _x/_y)
        sched = nfl.load_schedules([SEASON]).select(
            "season","week","home_team","away_team","home_score","away_score"
        ).to_pandas()
        sched["margin"] = sched["home_score"] - sched["away_score"]

        df_hist = pred.merge(
            sched[["season","week","home_team","away_team","margin"]],
            on=["season","week","home_team","away_team"],
            how="inner"
        )
        print("Historical rows for fit:", len(df_hist))

        # 2b) Extra safety: ensure 'spread_line' exists (should come from pred)
        if "spread_line" not in df_hist.columns:
            # try to recover if a prior merge ever created suffixes
            for c in ("spread_line_x", "spread_line_y"):
                if c in df_hist.columns:
                    df_hist["spread_line"] = df_hist[c]
                    break

        if df_hist.empty:
            raise RuntimeError("No historical rows found to fit. Check season/data.")

        # 3) Fit isotonic calibration
        cal = ProbCalibrator().fit(df_hist)

        # 4) Quick Brier sanity check (raw vs calibrated)
        def brier(p, y): return float(((p - y)**2).mean())
        y_win  = (df_hist["margin"] > 0).astype(int)
        raw_w  = df_hist["home_win_prob"].astype(float)
        cal_w  = cal.apply(df_hist)["home_win_prob"].astype(float)

        mask_cov  = df_hist["spread_line"].notna()
        y_cov  = (df_hist.loc[mask_cov, "margin"] + df_hist.loc[mask_cov, "spread_line"] > 0).astype(int)
        raw_c  = df_hist.loc[mask_cov, "home_cover_prob"].astype(float)
        cal_c  = cal.apply(df_hist.loc[mask_cov])["home_cover_prob"].astype(float)

        print(f"Brier(win) raw={brier(raw_w,y_win):.4f}  cal={brier(cal_w,y_win):.4f}")
        print(f"Brier(cov) raw={brier(raw_c,y_cov):.4f}  cal={brier(cal_c,y_cov):.4f}")

        # 5) Save
        out_path = os.path.join(os.getcwd(), "data", "cal_win_cover.joblib")
        dump(cal, out_path)
        print("Saved calibrator ->", out_path)
        print("Done.")

    except Exception as e:
        print("ERROR:", type(e).__name__, e)
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
