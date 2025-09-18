# src/calibrate.py
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

class ProbCalibrator:
    """
    Fit isotonic calibration for home_win_prob and home_cover_prob on a
    historical dev set, then apply to fresh predictions.
    """
    def __init__(self):
        self.win_iso = IsotonicRegression(out_of_bounds="clip")
        self.cov_iso = IsotonicRegression(out_of_bounds="clip")
        self._fitted = False

    def fit(self, df: pd.DataFrame):
        # df needs: home_win_prob, home_cover_prob, margin, spread_line
        y_win = (df["margin"] > 0).astype(int).values
        self.win_iso.fit(df["home_win_prob"].values, y_win)

        m = df["spread_line"].notna().values
        y_cov = (df["margin"] + df["spread_line"] > 0).astype(int).values
        self.cov_iso.fit(df.loc[m, "home_cover_prob"].values, y_cov[m])

        self._fitted = True
        return self

    def apply(self, df: pd.DataFrame):
        assert self._fitted, "Calibrator not fitted."
        out = df.copy()
        out["home_win_prob"] = self.win_iso.predict(out["home_win_prob"].values)
        m = out["spread_line"].notna()
        out.loc[m, "home_cover_prob"] = self.cov_iso.predict(out.loc[m, "home_cover_prob"].values)
        return out
