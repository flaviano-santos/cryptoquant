"""Meta-labeled machine learning.

The architecture matters more than the model. A primary rule-based signal
decides *direction*; a gradient-boosted classifier decides only *whether to take
the bet and how large*. This turns a low signal-to-noise regression problem into
a supervised binary problem where precision is something you can actually
improve, and it degrades gracefully: a useless classifier just means every bet
is taken, which is the baseline.

Training uses the full Lopez de Prado apparatus - triple-barrier labels, CUSUM
event sampling, sample-uniqueness weighting and purged, embargoed
cross-validation. See :mod:`cryptoquant.research.labeling` and
:mod:`cryptoquant.research.validation` for why each of those is necessary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ...research import labeling
from ...research.validation import PurgedKFold
from .base import Strategy
from .trend_carry import TrendCarry

__all__ = ["MetaLabelML"]


@dataclass
class MetaLabelML(Strategy):
    """Primary signal decides direction; a gradient-boosted classifier decides whether to take the bet and how large, via its probability that the bet would have hit its profit-take before its stop.

    Crucially, a bet is *held* from its entry event until its triple-barrier
    exit, rather than the position being recomputed every bar. Re-deriving the
    position each bar is the standard mistake: it produces a position that
    flickers with model noise, and the turnover eats the edge alive.

    Trained with:
      * triple-barrier labels (path-aware, volatility-scaled)
      * CUSUM event sampling (removes redundant, near-duplicate bars)
      * sample-uniqueness + time-decay weights (kills the overlapping-label
        illusion of a large sample)
      * purged, embargoed cross-validation (no leakage)
    """

    primary: Strategy = field(default_factory=TrendCarry)
    pt_sl: tuple[float, float] = (2.0, 1.0)
    horizon_bars: int = 24
    cusum_mult: float = 1.0
    prob_threshold: float = 0.52
    n_splits: int = 6
    embargo_pct: float = 0.01
    model_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "min_child_samples": 80,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.7,
            "reg_lambda": 5.0,
            "verbose": -1,
        }
    )
    name: str = "meta_ml"
    models_: dict = field(default_factory=dict, init=False)
    events_: dict = field(default_factory=dict, init=False)
    feature_importance_: pd.DataFrame | None = field(default=None, init=False)

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _feature_cols(X: pd.DataFrame) -> list[str]:
        return [c for c in X.columns if c != "close"]

    def _build_dataset(self, X: pd.DataFrame, prim: pd.Series):
        """CUSUM events -> triple-barrier labels -> weighted feature matrix."""
        close = X["close"].dropna()
        vol = labeling.daily_vol(close, span=100, bars_per_day=24)
        thr = (self.cusum_mult * vol).fillna(vol.median())
        events = labeling.cusum_events(close, thr)
        events = events[events.isin(close.index)]
        if len(events) < 250:
            return None

        side = np.sign(prim.reindex(events)).replace(0, np.nan).dropna()
        if side.empty:
            return None
        lab = labeling.triple_barrier(
            close, side.index, vol, pt_sl=self.pt_sl, num_bars=self.horizon_bars, side=side
        )
        if lab.empty or lab["bin"].nunique() < 2:
            return None

        cols = self._feature_cols(X)
        feat = X.loc[lab.index, cols].copy()
        feat["primary_side"] = side.reindex(lab.index)
        feat["primary_strength"] = prim.reindex(lab.index).abs()
        keep = feat.notna().mean(axis=1) > 0.8
        feat, lab = feat[keep].fillna(0.0), lab[keep]

        uniq = labeling.sample_uniqueness(close.index, lab)
        w = (uniq * labeling.time_decay(uniq, last_weight=0.6)).reindex(feat.index).fillna(0.0)
        return feat, lab, w

    @staticmethod
    def _held_positions(index: pd.DatetimeIndex, events: pd.DataFrame) -> pd.Series:
        """Expand discrete bets into a bar-by-bar position. A new bet overwrites the previous one; otherwise the position is held until its barrier."""
        pos = pd.Series(np.nan, index=index, dtype="float64")
        starts = index.searchsorted(events.index)
        ends = index.searchsorted(events["t1"].to_numpy())
        vals = (events["side"] * events["size"]).to_numpy(dtype="float64")
        arr = np.full(len(index), np.nan)
        cur, cur_end = 0.0, -1
        order = np.argsort(starts)
        ptr = 0
        for i in range(len(index)):
            while ptr < len(order) and starts[order[ptr]] == i:
                j = order[ptr]
                if np.isfinite(vals[j]):
                    cur, cur_end = vals[j], ends[j]
                ptr += 1
            if i > cur_end:
                cur = 0.0
            arr[i] = cur
        pos.iloc[:] = arr
        return pos.fillna(0.0)

    # -- training -----------------------------------------------------------
    def fit(self, data: Mapping[str, pd.DataFrame]) -> MetaLabelML:
        """Train the meta-model and record honest out-of-sample probabilities.

        For each symbol this samples events with a CUSUM filter, labels them
        with triple barriers around the primary signal's direction, weights them
        by uniqueness and recency, and fits a classifier under purged, embargoed
        cross-validation. A second model is fitted on the full sample for live
        use only.

        Args:
            data: Feature frames keyed by symbol.

        Returns:
            ``self``.
        """
        from lightgbm import LGBMClassifier

        prim_all = self.primary.signal(data)
        self.models_, self.events_ = {}, {}
        importances = {}

        for sym, X in data.items():
            if sym not in prim_all:
                continue
            built = self._build_dataset(X, prim_all[sym])
            if built is None:
                continue
            feat, lab, w = built

            # purged, embargoed CV -> honest out-of-sample probabilities
            cv = PurgedKFold(n_splits=self.n_splits, t1=lab["t1"], embargo_pct=self.embargo_pct)
            proba = pd.Series(np.nan, index=feat.index)
            for tr, te in cv.split(feat):
                if len(tr) < 250 or len(te) < 20 or lab["bin"].iloc[tr].nunique() < 2:
                    continue
                m = LGBMClassifier(**self.model_params)
                m.fit(feat.iloc[tr], lab["bin"].iloc[tr], sample_weight=w.iloc[tr].to_numpy())
                proba.iloc[te] = np.asarray(m.predict_proba(feat.iloc[te]))[:, 1]

            ev = pd.DataFrame(
                {
                    "side": feat["primary_side"],
                    "t1": lab["t1"],
                    "proba": proba,
                }
            )
            ev["size"] = self._size_from_proba(ev["proba"])
            self.events_[sym] = ev

            final = LGBMClassifier(**self.model_params)
            final.fit(feat, lab["bin"], sample_weight=w.to_numpy())
            self.models_[sym] = (final, list(feat.columns))
            importances[sym] = pd.Series(final.feature_importances_, index=feat.columns)

        if importances:
            self.feature_importance_ = (
                pd.DataFrame(importances).mean(axis=1).sort_values(ascending=False).to_frame("gain")
            )
        return self

    def _size_from_proba(self, proba: pd.Series) -> pd.Series:
        return ((proba - self.prob_threshold) / (1 - self.prob_threshold)).clip(0, 1)

    # -- signal generation --------------------------------------------------
    def signal(self, data: Mapping[str, pd.DataFrame], use_oos: bool = True) -> pd.DataFrame:
        """use_oos=True  -> purged-CV probabilities. This is the only version you are allowed to judge the strategy on.

        use_oos=False -> the full-sample model. This is what runs live.
        """
        prim = self.primary.signal(data)
        if not self.events_:
            raise RuntimeError("call fit() first")

        out = {}
        for sym in prim.columns:
            ev = self.events_.get(sym)
            if ev is None or ev.empty:
                out[sym] = pd.Series(0.0, index=prim.index)
                continue
            ev = ev.copy()
            if not use_oos:
                model, cols = self.models_[sym]
                X = data[sym]
                feat = X.reindex(index=ev.index, columns=[c for c in cols if c in X.columns]).copy()
                feat["primary_side"] = ev["side"]
                feat["primary_strength"] = prim[sym].reindex(ev.index).abs()
                feat = feat.reindex(columns=cols).fillna(0.0)
                ev["proba"] = np.asarray(model.predict_proba(feat))[:, 1]
                ev["size"] = self._size_from_proba(ev["proba"])
            ev = ev.dropna(subset=["size", "t1"])
            out[sym] = self._held_positions(prim.index, ev)

        return pd.DataFrame(out).reindex(columns=prim.columns).fillna(0.0)
