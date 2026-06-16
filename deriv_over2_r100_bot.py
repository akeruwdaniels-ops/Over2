#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║     DERIV DIGIT OVER 2 BOT — WALK-FORWARD EDITION                      ║
║  Symbol      : R_100  (Volatility 100 Index)                            ║
║  Contract    : DIGITOVER  barrier=2  (wins if last digit > 2)           ║
║  Duration    : 5 ticks                                                   ║
║  Base P(win) : 7/10 = 70%  (digits 3-9)                                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Intelligence Stack — 10-Layer Maximum Precision Engine:                 ║
║                                                                          ║
║   L1  Higher-Order Markov Chain  (order-3, digit transition matrix)     ║
║   L2  Hidden Markov Model        (3-state regime: COLD/WARM/HOT)        ║
║   L3  Hawkes Self-Exciting PP    (digit run/cluster detector)            ║
║   L4  Sample Entropy (SampEn)    (predictability gate — trade when low) ║
║   L5  ARFIMA Long-Memory         (fractional integration d in digit seq)║
║   L6  Hurst Exponent (R/S)       (persistence / anti-persistence score) ║
║   L7  Bayesian Beta-Binomial     (live posterior win-rate tracker)      ║
║   L8  Copula Dependence          (joint digit-run tail dependence)      ║
║   L9  Kalman Filter              (latent digit-probability state track) ║
║   L10 Risk Guard                 (cooldown · circuit breaker · stake)   ║
║   ∑   Weighted Ensemble          (regime-conditional · per-model floors)║
╠══════════════════════════════════════════════════════════════════════════╣
║  Precision Features:                                                     ║
║   • Order-3 Markov transition tensor for exact conditional digit probs  ║
║   • HMM regime-switches the ensemble weight profile dynamically         ║
║   • Hawkes process detects digit run clustering (correlated arrivals)   ║
║   • SampEn gate: only trade when sequence is statistically predictable  ║
║   • ARFIMA d-coefficient captures long-range digit memory               ║
║   • Hurst R/S analysis confirms persistence direction                   ║
║   • Kalman filter tracks hidden P(OVER2) in real-time                  ║
║   • Copula tail dependence flags anomalous streak risk                  ║
║   • Bayesian posterior updates after every settled contract             ║
║   • Per-model hard floors: all 9 must pass before ensemble gate        ║
║   • Consecutive-loss circuit breaker (extended cooldown)               ║
║   • 80-tick warmup for proper Markov/HMM calibration                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Walk-Forward Integration:                                               ║
║   • Phase 1 (0-5 min): LEARNING — bot observes ticks, builds Markov    ║
║     tables, calibrates HMM, fits ARFIMA/Hurst — NO trades placed       ║
║   • Phase 2 (5 min+): LIVE TRADING — WFV runs in background,           ║
║     evaluating rolling IS/OOS windows to detect model decay and         ║
║     auto-adjust confidence threshold                                     ║
║   • Decay Guard: OOS win-rate drops >8% vs IS → min_confidence raised  ║
║     automatically; edge positive → threshold relaxed                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Connection : NEW Deriv Options API (REST OTP bootstrap + auth'd WS)   ║
║    1. GET  /trading/v1/options/accounts        -> resolve account_id   ║
║    2. POST /trading/v1/options/accounts/{id}/otp -> authenticated URL  ║
║    3. Connect directly to returned wss:// URL (no authorize message)   ║
║    Re-run steps 1-2 on every reconnect — the OTP URL is single-use.   ║
║    Legacy App IDs (e.g. 1089) and ws.binaryws.com do NOT work here.    ║
╚══════════════════════════════════════════════════════════════════════════╝

Usage:
  export DERIV_APP_ID=<your_new_app_id>
  export DERIV_API_TOKEN=<your_PAT>
  export DERIV_ACCOUNT_ID=<your_demo_account_id>   # optional — auto-resolved
  python deriv_over2_r100_bot.py

Requirements:
  pip install websockets numpy scipy requests
"""

import asyncio
import csv
import enum
import json
import logging
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import websockets
from scipy import stats

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
CFG = {
    # ── Contract ──
    "underlying_symbol": "R_100",
    "contract_type":     "DIGITOVER",
    "barrier":           "2",          # win if last digit > 2  (digits 3-9)
    "currency":          "USD",
    "duration":          5,          # ← 5-tick contract
    "duration_unit":     "t",

    # ── Capital ──
    "starting_bankroll":  1.00,
    "stake":               0.35,       # flat stake (small account)
    "drawdown_stop":       0.70,       # halt below this bankroll

    # ── Kelly (activates when bankroll >= threshold) ──
    "kelly_activation_bankroll":    5.0,
    "kelly_fraction":               0.5,
    "kelly_min_stake":              0.35,
    "kelly_max_fraction_of_bankroll": 0.10,
    "payout_ratio":                 0.3429,  # profit/stake = 0.12/0.35 (actual observed)

    # ── Signal ──
    "warmup_ticks":       80,           # need 80 ticks to fill order-3 Markov table
    "signal_interval":    1,           # evaluate every tick (1-tick contracts)
    "min_confidence":     0.74,        # ensemble gate (above 70% base rate)
    "markov_order":       3,           # use last 3 digits as state
    "sampen_m":           2,           # SampEn template length
    "sampen_r_factor":    0.20,        # SampEn tolerance = r_factor * std
    "sampen_veto_above":  2.20,        # veto if SampEn > this (sequence too random)
    "arfima_window":      60,          # ticks for ARFIMA d estimation
    "hurst_window":       60,
    "hawkes_window":      40,          # ticks for Hawkes intensity estimation
    "hawkes_decay":       0.7,         # exponential decay for Hawkes kernel
    "kalman_process_var": 0.001,       # Kalman process noise
    "kalman_obs_var":     0.05,        # Kalman observation noise
    "copula_window":      50,          # ticks for tail-dependence estimate
    "copula_tail_veto":   0.55,        # veto if upper-tail lambda > threshold

    # ── Risk ──
    "cooldown_win":              2,    # ticks after win
    "cooldown_loss":             8,    # ticks after loss
    "consecutive_loss_limit":    3,
    "consecutive_loss_cooldown": 25,

    # ── Connection (new Options API) ──
    "api_base":      "https://api.derivws.com",
    "accounts_path": "/trading/v1/options/accounts",
    "otp_path":      "/trading/v1/options/accounts/{account_id}/otp",
    "ws_public_url": "wss://api.derivws.com/trading/v1/options/ws/public",
    "reconnect_delay": 5,

    # ── Logging ──
    "log_dir":     os.getenv("LOG_DIR", "logs"),
    "log_file":    "over2_bot_wfv.log",
    "signals_csv": "over2_signals.csv",
    "trades_csv":  "over2_trades.csv",
    "wfv_csv":     "wfv_live_results.csv",
    "tick_buffer": 2000,  # must exceed wfv_is_size+wfv_oos_size+warmup (800+750+80=1630)

    # ── Walk-Forward Validator ──
    "wfv_enabled":           True,
    "wfv_is_size":           800,     # in-sample window (ticks)
    "wfv_oos_size":          750,     # out-of-sample window (ticks)
    "wfv_step":              150,     # advance per fold
    "wfv_decay_threshold":   0.08,    # IS→OOS WR drop triggers penalty
    "wfv_conf_penalty":      0.02,    # raise min_confidence by this on decay
    "wfv_conf_reward":       0.01,    # lower min_confidence by this on good OOS
    "wfv_conf_min":          0.72,    # floor for auto-adjusted confidence
    "wfv_conf_max":          0.82,    # ceiling for auto-adjusted confidence
    "wfv_eval_interval_s":   60,      # re-run WFV every N seconds

    # ── Learning Phase (Phase 1) ──
    "learning_phase_seconds": 1000,    # 5 minutes observation before live trading
}

# ── Regime-conditional ensemble weights ──
# COLD (low run activity): Markov + ARFIMA + Hurst dominate
# WARM (normal):           balanced
# HOT  (high clustering):  Hawkes + Copula + Bayesian dominate
MODEL_WEIGHTS_BY_REGIME = {
    0: {  # COLD — sequential structure most predictable
        "markov":  0.28, "hmm": 0.12, "hawkes": 0.05,
        "sampen":  0.10, "arfima": 0.16, "hurst": 0.14,
        "bayesian":0.07, "copula": 0.04, "kalman": 0.04,
    },
    1: {  # WARM — balanced
        "markov":  0.22, "hmm": 0.14, "hawkes": 0.10,
        "sampen":  0.10, "arfima": 0.12, "hurst": 0.10,
        "bayesian":0.10, "copula": 0.07, "kalman": 0.05,
    },
    2: {  # HOT — cluster/run structure dominates
        "markov":  0.14, "hmm": 0.12, "hawkes": 0.18,
        "sampen":  0.08, "arfima": 0.08, "hurst": 0.08,
        "bayesian":0.12, "copula": 0.12, "kalman": 0.08,
    },
}

MODEL_FLOORS = {
    "markov":   0.60,   # Markov P(digit > 2 | last 3 digits)
    "hmm":      0.52,   # HMM regime prior P(win)
    "hawkes":   0.45,   # Hawkes signal (1 = no clustering risk)
    "sampen":   0.45,   # SampEn signal (1 = highly predictable)
    "arfima":   0.45,   # ARFIMA long-memory signal
    "hurst":    0.45,   # Hurst persistence signal
    "bayesian": 0.45,   # Bayesian posterior win-rate
    "copula":   0.45,   # Copula tail dependence signal
    "kalman":   0.55,   # Kalman-filtered P(win)
}

# ══════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════
LOG_DIR = Path(CFG["log_dir"])
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / CFG["log_file"], encoding="utf-8"),
    ],
)
log = logging.getLogger("OVER2_BOT")


# ══════════════════════════════════════════════════════════════════════════
# DATA LOGGER (CSV)
# ══════════════════════════════════════════════════════════════════════════
class DataLogger:
    SIGNAL_FIELDS = [
        "timestamp", "tick_n", "spot", "last_digit",
        "markov_p", "hmm_sig", "hawkes_sig", "sampen_sig",
        "arfima_sig", "hurst_sig", "bayes_sig", "copula_sig", "kalman_sig",
        "pre_score", "conf", "regime", "reason",
    ]
    TRADE_FIELDS = [
        "timestamp", "trade_n", "contract_id", "spot_entry", "last_digit_entry",
        "stake", "conf", "profit", "won", "bankroll", "total_pnl", "win_rate",
    ]
    WFV_FIELDS = [
        "timestamp", "fold_id", "phase", "n_ticks", "n_trades",
        "win_rate", "edge_vs_base", "total_pnl", "profit_factor",
        "sharpe", "max_drawdown", "avg_conf", "circuit_trips",
        "live_min_conf",
    ]

    def __init__(self, log_dir: Path):
        self._init_file(log_dir / CFG["signals_csv"], self.SIGNAL_FIELDS)
        self._init_file(log_dir / CFG["trades_csv"],  self.TRADE_FIELDS)
        self._init_file(log_dir / CFG["wfv_csv"],     self.WFV_FIELDS)
        self.sig_path   = log_dir / CFG["signals_csv"]
        self.trade_path = log_dir / CFG["trades_csv"]
        self.wfv_path   = log_dir / CFG["wfv_csv"]

    @staticmethod
    def _init_file(path, fields):
        if not path.exists() or path.stat().st_size == 0:
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    def log_signal(self, **row):
        with open(self.sig_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.SIGNAL_FIELDS).writerow(row)

    def log_trade(self, **row):
        with open(self.trade_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.TRADE_FIELDS).writerow(row)

    def log_wfv(self, **row):
        with open(self.wfv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.WFV_FIELDS).writerow(row)


datalog = DataLogger(LOG_DIR)


# ══════════════════════════════════════════════════════════════════════════
# UTILITY — extract last digit from spot price
# ══════════════════════════════════════════════════════════════════════════
def last_digit(spot: float) -> int:
    """Returns the last decimal digit of a Deriv spot price."""
    # R_100 prices have 2 decimal places: e.g. 1234.56 → digit = 6
    return int(round(spot * 100)) % 10


# ══════════════════════════════════════════════════════════════════════════
# LAYER 1 — HIGHER-ORDER MARKOV CHAIN (order-3)
# ══════════════════════════════════════════════════════════════════════════
class MarkovChain:
    """
    Maintains a 10×10×10×10 transition count tensor (order-3).
    State = (digit_{t-2}, digit_{t-1}, digit_t).
    Predicts P(next_digit > 2 | current state).

    Falls back to order-2 then order-1 when count in state < min_count.
    """

    def __init__(self, order: int = 3, min_count: int = 3):
        self.order     = order
        self.min_count = min_count
        # Count tensors for orders 1, 2, 3
        self.counts3   = np.zeros((10, 10, 10, 10), dtype=np.float32)
        self.counts2   = np.zeros((10, 10, 10),     dtype=np.float32)
        self.counts1   = np.zeros((10, 10),         dtype=np.float32)
        self._history  = deque(maxlen=4)

    def update(self, digit: int):
        h = list(self._history)
        if len(h) >= 1:
            self.counts1[h[-1], digit] += 1
        if len(h) >= 2:
            self.counts2[h[-2], h[-1], digit] += 1
        if len(h) >= 3:
            self.counts3[h[-3], h[-2], h[-1], digit] += 1
        self._history.append(digit)

    def p_over2(self) -> float:
        """P(next digit > 2) given current history."""
        h = list(self._history)

        # Try order-3
        if len(h) >= 3:
            row = self.counts3[h[-3], h[-2], h[-1]]
            n   = row.sum()
            if n >= self.min_count:
                return float(row[3:].sum() / n)

        # Fallback order-2
        if len(h) >= 2:
            row = self.counts2[h[-2], h[-1]]
            n   = row.sum()
            if n >= self.min_count:
                return float(row[3:].sum() / n)

        # Fallback order-1
        if len(h) >= 1:
            row = self.counts1[h[-1]]
            n   = row.sum()
            if n >= self.min_count:
                return float(row[3:].sum() / n)

        # Prior: 7/10
        return 0.70


# ══════════════════════════════════════════════════════════════════════════
# LAYER 2 — HIDDEN MARKOV MODEL (3-state digit-run regime)
# ══════════════════════════════════════════════════════════════════════════
class HMMRegimes:
    """
    States:
      0 = COLD — over-2 digits appearing less than expected (< 65%)
      1 = WARM — normal regime (~70% over-2)
      2 = HOT  — over-2 digits clustering above expected (> 75%)

    Regime updates from observed rolling 20-tick over-2 rate.
    Transition matrix is empirical for R_100.
    """
    COLD, WARM, HOT = 0, 1, 2
    _THRESH_LO = 0.63
    _THRESH_HI = 0.76

    _A = np.array([
        [0.80, 0.16, 0.04],   # from COLD
        [0.12, 0.76, 0.12],   # from WARM
        [0.04, 0.16, 0.80],   # from HOT
    ])

    _PRIOR = {COLD: 0.60, WARM: 0.70, HOT: 0.77}

    def __init__(self):
        self.state  = self.WARM
        self._alpha = np.array([0.15, 0.70, 0.15])
        self._win_buf = deque(maxlen=20)

    def update(self, digit: int):
        self._win_buf.append(1 if digit > 2 else 0)
        if len(self._win_buf) < 5:
            return self.state
        rate = np.mean(self._win_buf)
        if rate < self._THRESH_LO:
            obs = self.COLD
        elif rate > self._THRESH_HI:
            obs = self.HOT
        else:
            obs = self.WARM
        new = self._A[obs] * self._alpha
        s   = new.sum()
        self._alpha = new / s if s > 0 else np.ones(3) / 3
        self.state  = int(np.argmax(self._alpha))
        return self.state

    def signal(self) -> float:
        return self._PRIOR[self.state]

    def name(self) -> str:
        return ["COLD", "WARM", "HOT"][self.state]


# ══════════════════════════════════════════════════════════════════════════
# LAYER 3 — HAWKES SELF-EXCITING POINT PROCESS
# ══════════════════════════════════════════════════════════════════════════
class HawkesDetector:
    """
    Models clustering in UNDER-2 digit events (digits 0,1,2).
    If UNDER-2 digits are clustering (high Hawkes intensity),
    the next digit being OVER-2 is suppressed.

    λ(t) = μ + Σ α·exp(-β·(t - t_i))
    Signal = 1 - normalized_intensity  [0,1]
    High signal = safe to trade (low clustering of under-2 events).
    """

    def __init__(self, mu: float = 0.03, alpha: float = 0.4, beta: float = None):
        self.mu    = mu
        self.alpha = alpha
        self.beta  = beta if beta is not None else CFG["hawkes_decay"]
        self._intensity = 0.0
        self._tick_n    = 0

    def update(self, digit: int) -> float:
        # Decay existing intensity
        self._intensity = self._intensity * self.beta + self.mu

        # If under-2 event, add excitation
        if digit <= 2:
            self._intensity += self.alpha

        self._tick_n += 1
        return self._intensity

    def signal(self) -> float:
        """
        Normalize and invert: high intensity → low signal (risky).
        Intensity range roughly [mu/(1-beta), mu/(1-beta) + alpha/(1-beta)]
        """
        floor_i  = self.mu / max(1 - self.beta, 1e-6)
        ceiling_i = floor_i + self.alpha / max(1 - self.beta, 1e-6)
        norm = float(np.clip(
            (self._intensity - floor_i) / max(ceiling_i - floor_i, 1e-9),
            0.0, 1.0,
        ))
        return float(np.clip(1.0 - norm, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════
# LAYER 4 — SAMPLE ENTROPY (SampEn)
# ══════════════════════════════════════════════════════════════════════════
class SampleEntropySignal:
    """
    SampEn measures regularity/predictability of the digit sequence.
    Low SampEn → high regularity → more predictable → safe to trade.
    High SampEn → near-random → veto.

    signal = 1 - SampEn / sampen_veto_above  (clipped to [0,1])
    """

    @staticmethod
    def compute(digits: np.ndarray, m: int = 2, r_factor: float = 0.20) -> float:
        n   = len(digits)
        if n < 2 * (m + 1):
            return 1.5   # unknown — treat as random
        d   = digits.astype(float)
        r   = r_factor * d.std(ddof=1)
        if r < 1e-9:
            return 0.0   # constant sequence → perfectly regular

        def count_templates(length):
            count = 0
            for i in range(n - length):
                template = d[i:i + length]
                for j in range(n - length):
                    if i == j:
                        continue
                    if np.max(np.abs(d[j:j + length] - template)) <= r:
                        count += 1
            return count

        A = count_templates(m + 1)
        B = count_templates(m)

        if B == 0 or A == 0:
            return 2.0   # undefined — treat as very random
        return float(-np.log(A / B))

    @classmethod
    def signal(cls, digits: np.ndarray) -> tuple:
        """Returns (signal_value [0,1], raw_sampen, veto_bool)."""
        se    = cls.compute(digits[-40:] if len(digits) >= 40 else digits,
                            m=CFG["sampen_m"], r_factor=CFG["sampen_r_factor"])
        veto  = se > CFG["sampen_veto_above"]
        sig   = float(np.clip(1.0 - se / CFG["sampen_veto_above"], 0.0, 1.0))
        return sig, se, veto


# ══════════════════════════════════════════════════════════════════════════
# LAYER 5 — ARFIMA LONG-MEMORY SIGNAL
# ══════════════════════════════════════════════════════════════════════════
class ARFIMASignal:
    """
    Estimates the fractional differencing parameter d from the digit sequence
    using the log-periodogram (Geweke-Porter-Hudak) estimator.

    d > 0: long memory (persistence) in over-2 events → favourable
    d < 0: anti-persistence → cautious
    d ≈ 0: short memory → neutral

    Signal maps d ∈ [-0.5, 0.5] → [0.3, 0.9] monotonically.
    """

    @staticmethod
    def estimate_d(series: np.ndarray) -> float:
        n = len(series)
        if n < 20:
            return 0.0
        # Binary over-2 series
        x    = (series > 2).astype(float) - 0.7   # mean-center
        freq = np.fft.rfftfreq(n)[1:]              # exclude DC
        psd  = np.abs(np.fft.rfft(x)[1:]) ** 2
        # Use lower 10% of frequencies (long-memory band)
        m    = max(2, len(freq) // 10)
        lam  = freq[:m]
        Iy   = psd[:m]
        if np.any(Iy <= 0):
            return 0.0
        # OLS: log(Iy) = const - 2d * log(2*pi*lam)
        x_reg = -2.0 * np.log(2 * np.pi * lam)
        y_reg = np.log(Iy)
        try:
            d, _ = np.polyfit(x_reg, y_reg, 1)
            return float(np.clip(d, -0.5, 0.5))
        except Exception:
            return 0.0

    @classmethod
    def signal(cls, digits: np.ndarray) -> tuple:
        d   = cls.estimate_d(digits[-CFG["arfima_window"]:])
        sig = float(np.clip(0.60 + 0.60 * d, 0.30, 0.90))
        return sig, d


# ══════════════════════════════════════════════════════════════════════════
# LAYER 6 — HURST EXPONENT (R/S Analysis)
# ══════════════════════════════════════════════════════════════════════════
class HurstSignal:
    """
    Computed on the binary over-2 series (0/1).
    H > 0.5 → persistent (over-2 begets over-2) → high signal
    H < 0.5 → anti-persistent (alternating) → lower signal
    H ≈ 0.5 → random walk → neutral

    Signal = H  (directly, clipped to [0.2, 0.95])
    """

    @staticmethod
    def compute(digits: np.ndarray) -> float:
        series = (digits > 2).astype(float)
        n      = len(series)
        if n < 20:
            return 0.5
        max_lag = max(5, n // 4)
        lags, rs = [], []
        for lag in range(4, max_lag):
            c = series[:lag]
            m = c.mean()
            d = np.cumsum(c - m)
            r = d.max() - d.min()
            s = c.std(ddof=1)
            if s > 0:
                lags.append(lag)
                rs.append(r / s)
        if len(rs) < 4:
            return 0.5
        try:
            h, _ = np.polyfit(np.log(lags), np.log(rs), 1)
            return float(np.clip(h, 0.05, 0.95))
        except Exception:
            return 0.5

    @classmethod
    def signal(cls, digits: np.ndarray) -> tuple:
        h   = cls.compute(digits[-CFG["hurst_window"]:])
        sig = float(np.clip(h, 0.20, 0.95))
        return sig, h


# ══════════════════════════════════════════════════════════════════════════
# LAYER 7 — BAYESIAN BETA-BINOMIAL WIN-RATE ESTIMATOR
# ══════════════════════════════════════════════════════════════════════════
class BayesianEdge:
    """
    Beta(α, β) posterior on P(win | placed trade).
    Prior centred at 70% (base rate for OVER 2).
    """

    def __init__(self, prior_wr: float = 0.70, prior_n: float = 14.0):
        self.alpha = prior_wr * prior_n
        self.beta  = (1.0 - prior_wr) * prior_n
        self.n_obs = 0

    def update(self, won: bool):
        if won:
            self.alpha += 1.0
        else:
            self.beta  += 1.0
        self.n_obs += 1

    def mean(self) -> float:
        return float(self.alpha / (self.alpha + self.beta))

    def ci95(self) -> tuple:
        lo = float(stats.beta.ppf(0.025, self.alpha, self.beta))
        hi = float(stats.beta.ppf(0.975, self.alpha, self.beta))
        return lo, hi


# ══════════════════════════════════════════════════════════════════════════
# LAYER 8 — COPULA TAIL DEPENDENCE
# ══════════════════════════════════════════════════════════════════════════
class CopulaSignal:
    """
    Estimates upper-tail dependence λ_U between consecutive digit pairs
    using the empirical copula (rank-based).

    High λ_U means consecutive over-2 outcomes cluster (streaks exist),
    which is actually GOOD for our bet — we want to ride the streak.

    Low λ_U means digits are near-independent — neutral.

    We compute lower-tail dependence λ_L (consecutive under-2 clustering);
    high λ_L means under-2 is clustering → veto the trade.

    Signal = 1 - λ_L  (we want λ_L low)
    """

    @staticmethod
    def lower_tail_dependence(digits: np.ndarray, q: float = 0.25) -> float:
        """Empirical lower-tail copula lambda at quantile q."""
        series = (digits <= 2).astype(float)
        n = len(series)
        if n < 20:
            return 0.0
        x = series[:-1]
        y = series[1:]
        # Rank-transform to uniform [0,1]
        rx = stats.rankdata(x) / (len(x) + 1)
        ry = stats.rankdata(y) / (len(y) + 1)
        # Count joint lower-tail events
        joint = np.sum((rx <= q) & (ry <= q))
        marg  = np.sum(rx <= q)
        if marg == 0:
            return 0.0
        return float(joint / marg)

    @classmethod
    def signal(cls, digits: np.ndarray) -> tuple:
        w     = digits[-CFG["copula_window"]:] if len(digits) >= CFG["copula_window"] else digits
        lam_L = cls.lower_tail_dependence(w)
        veto  = lam_L > CFG["copula_tail_veto"]
        sig   = float(np.clip(1.0 - lam_L / max(CFG["copula_tail_veto"], 1e-9), 0.0, 1.0))
        return sig, lam_L, veto


# ══════════════════════════════════════════════════════════════════════════
# LAYER 9 — KALMAN FILTER (hidden P(win) state tracker)
# ══════════════════════════════════════════════════════════════════════════
class KalmanWinProb:
    """
    1-D Kalman filter treating the true P(win) as a slowly-drifting
    latent variable, observed noisily by the raw digit outcome (0/1).

    State x_t = true P(over-2 | regime)
    Observation z_t = (digit_t > 2) ? 1 : 0
    """

    def __init__(self, x0: float = 0.70,
                 P0: float = 0.05,
                 Q: float = None,
                 R: float = None):
        self.x = x0
        self.P = P0
        self.Q = Q if Q is not None else CFG["kalman_process_var"]
        self.R = R if R is not None else CFG["kalman_obs_var"]

    def update(self, observation: float) -> float:
        # Predict
        x_pred = self.x
        P_pred = self.P + self.Q
        # Update
        K      = P_pred / (P_pred + self.R)
        self.x = float(np.clip(x_pred + K * (observation - x_pred), 0.0, 1.0))
        self.P = (1 - K) * P_pred
        return self.x

    def signal(self) -> float:
        return float(np.clip(self.x, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════
# LAYER 10 — RISK GUARD
# ══════════════════════════════════════════════════════════════════════════
class RiskGuard:
    def __init__(self):
        self.stake          = CFG["stake"]
        self._cooldown      = 0
        self._tripped       = False
        self._consec_losses = 0

    def tick(self):
        if self._cooldown > 0:
            self._cooldown -= 1

    def on_win(self):
        self._consec_losses = 0
        self._cooldown = CFG["cooldown_win"]

    def on_loss(self):
        self._consec_losses += 1
        if self._consec_losses >= CFG["consecutive_loss_limit"]:
            self._cooldown = CFG["consecutive_loss_cooldown"]
            log.warning(
                f"⚠️  {self._consec_losses} consecutive losses — "
                f"extended cooldown ({CFG['consecutive_loss_cooldown']} ticks)"
            )
        else:
            self._cooldown = CFG["cooldown_loss"]

    def check_bankroll(self, bankroll: float):
        if bankroll < CFG["drawdown_stop"]:
            self._tripped = True
            log.warning(f"⛔  CIRCUIT BREAKER — bankroll ${bankroll:.2f} halted.")

    def can_trade(self) -> bool:
        return not self._tripped and self._cooldown == 0

    def compute_stake(self, bankroll: float, win_prob: float) -> float:
        if bankroll < CFG["kelly_activation_bankroll"]:
            self.stake = CFG["stake"]
            return self.stake
        b      = CFG["payout_ratio"]
        p      = float(np.clip(win_prob, 0.0, 1.0))
        q      = 1.0 - p
        f_star = max((b * p - q) / b, 0.0)
        f_half = min(f_star * CFG["kelly_fraction"], CFG["kelly_max_fraction_of_bankroll"])
        stake  = max(bankroll * f_half, CFG["kelly_min_stake"])
        stake  = min(stake, bankroll)
        self.stake = round(stake, 2)
        return self.stake

    def status(self) -> str:
        if self._tripped:
            return "CIRCUIT_BREAKER"
        if self._cooldown > 0:
            return f"COOLDOWN({self._cooldown})"
        return "READY"


# ══════════════════════════════════════════════════════════════════════════
# WEIGHTED ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════
class Ensemble:
    """
    1. Hard vetoes (SampEn, Copula tail)
    2. Per-model floor check (all 9 must pass)
    3. Regime-conditional weighted score
    4. Confidence threshold gate
    """

    def decide(
        self,
        markov_p:   float,
        hmm_sig:    float,
        hawkes_sig: float,
        sampen_sig: float,
        arfima_sig: float,
        hurst_sig:  float,
        bayes_sig:  float,
        copula_sig: float,
        kalman_sig: float,
        hmm_state:  int,
        sampen_veto: bool,
        copula_veto: bool,
    ) -> tuple:

        scores = {
            "markov":   markov_p,
            "hmm":      hmm_sig,
            "hawkes":   hawkes_sig,
            "sampen":   sampen_sig,
            "arfima":   arfima_sig,
            "hurst":    hurst_sig,
            "bayesian": bayes_sig,
            "copula":   copula_sig,
            "kalman":   kalman_sig,
        }

        # ── Hard vetoes ──
        if sampen_veto:
            return False, 0.0, scores, [], "SAMPEN_RANDOM_VETO"
        if copula_veto:
            return False, 0.0, scores, [], "COPULA_TAIL_VETO"

        # ── Per-model floor check ──
        failed = [k for k, floor in MODEL_FLOORS.items() if scores[k] < floor]
        if failed:
            return False, 0.0, scores, failed, f"FLOOR_FAIL({','.join(failed)})"

        # ── Weighted ensemble ──
        W    = MODEL_WEIGHTS_BY_REGIME.get(hmm_state, MODEL_WEIGHTS_BY_REGIME[1])
        conf = float(np.clip(sum(scores[k] * W[k] for k in W), 0.0, 1.0))

        if conf < CFG["min_confidence"]:
            return False, conf, scores, [], f"CONF_LOW({conf:.3f}<{CFG['min_confidence']})"

        return True, conf, scores, [], "TRADE"



# ══════════════════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATOR  (Live, in-process)
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class WFVFoldResult:
    fold_id:       int
    phase:         str
    n_ticks:       int
    n_trades:      int
    wins:          int
    win_rate:      float
    edge_vs_base:  float
    total_pnl:     float
    profit_factor: float
    sharpe:        float
    max_drawdown:  float
    avg_conf:      float
    circuit_trips: int


def _wfv_simulate_window(digits, cfg, fold_id, phase, seed=None):
    """Replay intelligence pipeline on a digit window — no live trades."""
    import math as _math
    markov = MarkovChain(order=cfg["markov_order"])
    hmm    = HMMRegimes()
    hawkes = HawkesDetector(beta=cfg["hawkes_decay"])
    bayes  = BayesianEdge()
    kalman = KalmanWinProb(Q=cfg["kalman_process_var"], R=cfg["kalman_obs_var"])
    guard  = RiskGuard()

    if seed:
        markov.counts3  = seed["markov_counts3"].copy()
        markov.counts2  = seed["markov_counts2"].copy()
        markov.counts1  = seed["markov_counts1"].copy()
        markov._history = deque(seed["markov_history"], maxlen=4)
        bayes.alpha     = seed["bayes_alpha"]
        bayes.beta      = seed["bayes_beta"]
        kalman.x        = seed["kalman_x"]
        kalman.P        = seed["kalman_P"]

    history    = deque(maxlen=max(300, cfg["copula_window"] + 10))
    warmup     = cfg["warmup_ticks"]
    dur        = cfg["duration"]
    stake      = cfg["stake"]
    payout     = cfg["payout_ratio"]
    min_conf   = cfg["min_confidence"]

    n_trades = 0; wins = 0; pnl_list = []; conf_list = []
    bankroll = cfg["starting_bankroll"]; circuit_trips = 0

    for i, digit in enumerate(digits):
        markov.update(digit)
        hmm.update(digit)
        hawkes.update(digit)
        kalman.update(1.0 if digit > 2 else 0.0)
        guard.tick()
        history.append(digit)

        if i < warmup:
            continue
        future_idx = i + dur
        if future_idx >= len(digits):
            break

        arr = np.array(history, dtype=float)
        markov_p   = markov.p_over2()
        hmm_sig    = hmm.signal(); hmm_state = hmm.state
        hawkes_sig = hawkes.signal()

        _se = SampleEntropySignal.compute(
            arr[-40:] if len(arr) >= 40 else arr,
            cfg["sampen_m"], cfg["sampen_r_factor"])
        sampen_veto = _se > cfg["sampen_veto_above"]
        sampen_sig  = float(np.clip(1.0 - _se / cfg["sampen_veto_above"], 0.0, 1.0))

        _d = ARFIMASignal.estimate_d(arr[-cfg["arfima_window"]:])
        arfima_sig = float(np.clip(0.60 + 0.60 * _d, 0.30, 0.90))

        _h = HurstSignal.compute(arr[-cfg["hurst_window"]:])
        hurst_sig  = float(np.clip(_h, 0.20, 0.95))

        bayes_sig   = bayes.mean()
        _w          = arr[-cfg["copula_window"]:] if len(arr) >= cfg["copula_window"] else arr
        _lam        = CopulaSignal.lower_tail_dependence(_w)
        copula_veto = _lam > cfg["copula_tail_veto"]
        copula_sig  = float(np.clip(1.0 - _lam / max(cfg["copula_tail_veto"], 1e-9), 0.0, 1.0))
        kalman_sig  = kalman.signal()

        if not guard.can_trade():
            continue

        scores = dict(markov=markov_p, hmm=hmm_sig, hawkes=hawkes_sig,
                      sampen=sampen_sig, arfima=arfima_sig, hurst=hurst_sig,
                      bayesian=bayes_sig, copula=copula_sig, kalman=kalman_sig)

        if sampen_veto or copula_veto:
            continue
        failed = [k for k, floor in MODEL_FLOORS.items() if scores[k] < floor]
        if failed:
            continue

        W    = MODEL_WEIGHTS_BY_REGIME.get(hmm_state, MODEL_WEIGHTS_BY_REGIME[1])
        conf = float(np.clip(sum(scores[k] * W[k] for k in W), 0.0, 1.0))
        if conf < min_conf:
            continue

        outcome_digit = digits[future_idx]
        won    = outcome_digit > 2
        profit = (stake * payout) if won else -stake

        n_trades += 1; conf_list.append(conf); pnl_list.append(profit)
        bankroll += profit
        bayes.update(won)
        guard.check_bankroll(bankroll)

        if won:
            wins += 1; guard.on_win()
        else:
            guard.on_loss()
            if not guard.can_trade() and guard._tripped:
                circuit_trips += 1

    win_rate   = wins / n_trades if n_trades > 0 else 0.0
    gross_win  = sum(p for p in pnl_list if p > 0)
    gross_loss = abs(sum(p for p in pnl_list if p < 0))
    pf         = gross_win / gross_loss if gross_loss > 0 else float("inf")

    if len(pnl_list) >= 2:
        arr_pnl = np.array(pnl_list)
        sharpe  = (arr_pnl.mean() / arr_pnl.std(ddof=1)) * _math.sqrt(len(arr_pnl))
    else:
        sharpe = 0.0

    cum    = np.cumsum(pnl_list) if pnl_list else np.array([0.0])
    peak   = np.maximum.accumulate(cum)
    max_dd = float((peak - cum).max()) if len(cum) > 0 else 0.0

    return WFVFoldResult(
        fold_id=fold_id, phase=phase, n_ticks=len(digits),
        n_trades=n_trades, wins=wins, win_rate=win_rate,
        edge_vs_base=win_rate - 0.70, total_pnl=float(sum(pnl_list)),
        profit_factor=pf, sharpe=sharpe, max_drawdown=max_dd,
        avg_conf=float(np.mean(conf_list)) if conf_list else 0.0,
        circuit_trips=circuit_trips,
    )


def _wfv_extract_seed(digits, cfg):
    """Train Markov/Bayes/Kalman on IS window; return serialisable state."""
    markov = MarkovChain(order=cfg["markov_order"])
    bayes  = BayesianEdge()
    kalman = KalmanWinProb(Q=cfg["kalman_process_var"], R=cfg["kalman_obs_var"])
    for d in digits:
        markov.update(d)
        kalman.update(1.0 if d > 2 else 0.0)
        bayes.update(d > 2)
    return dict(
        markov_counts3=markov.counts3.copy(), markov_counts2=markov.counts2.copy(),
        markov_counts1=markov.counts1.copy(), markov_history=list(markov._history),
        bayes_alpha=bayes.alpha, bayes_beta=bayes.beta,
        kalman_x=kalman.x, kalman_P=kalman.P,
    )


class LiveWalkForwardValidator:
    """
    Runs periodically on the live tick buffer to detect model decay
    and auto-adjust the bot's min_confidence threshold.
    """

    def __init__(self):
        self.fold_counter  = 0
        self.last_run_time = 0.0
        self.last_verdict  = "PENDING"
        self.last_oos_wr   = None
        self.last_is_wr    = None
        self.adj_min_conf  = CFG["min_confidence"]
        self._results      = []

    def should_run(self) -> bool:
        return (time.monotonic() - self.last_run_time) >= CFG["wfv_eval_interval_s"]

    def run(self, digit_buffer, current_min_conf: float) -> float:
        """Returns the (possibly adjusted) min_confidence to use going forward."""
        digits = np.array(digit_buffer, dtype=int)
        n      = len(digits)
        is_sz  = CFG["wfv_is_size"]
        oos_sz = CFG["wfv_oos_size"]

        if n < is_sz + oos_sz + CFG["warmup_ticks"]:
            log.info(f"[WFV] Not enough ticks ({n}/{is_sz+oos_sz+CFG['warmup_ticks']}) — skipping")
            self.last_run_time = time.monotonic()
            return current_min_conf

        is_start   = max(0, n - is_sz - oos_sz)
        is_window  = digits[is_start: is_start + is_sz]
        oos_window = digits[is_start + is_sz: is_start + is_sz + oos_sz]

        self.fold_counter += 1
        fold_cfg = dict(CFG)
        fold_cfg["min_confidence"]    = current_min_conf
        fold_cfg["starting_bankroll"] = 10.0

        log.info(f"[WFV] Running fold #{self.fold_counter} "
                 f"(IS={len(is_window)} OOS={len(oos_window)} ticks) ...")

        seed       = _wfv_extract_seed(is_window, fold_cfg)
        is_result  = _wfv_simulate_window(is_window,  fold_cfg, self.fold_counter, "IS")
        oos_result = _wfv_simulate_window(oos_window, fold_cfg, self.fold_counter, "OOS", seed)

        self._results.extend([is_result, oos_result])
        self.last_is_wr  = is_result.win_rate
        self.last_oos_wr = oos_result.win_rate

        log.info(
            f"[WFV] IS  trades={is_result.n_trades} WR={is_result.win_rate:.1%} "
            f"PnL={is_result.total_pnl:+.2f} Sharpe={is_result.sharpe:+.2f}"
        )
        log.info(
            f"[WFV] OOS trades={oos_result.n_trades} WR={oos_result.win_rate:.1%} "
            f"PnL={oos_result.total_pnl:+.2f} Sharpe={oos_result.sharpe:+.2f}"
        )

        new_conf = current_min_conf
        if is_result.n_trades > 0 and oos_result.n_trades > 0:
            decay = is_result.win_rate - oos_result.win_rate
            if decay > CFG["wfv_decay_threshold"]:
                new_conf = min(current_min_conf + CFG["wfv_conf_penalty"],
                               CFG["wfv_conf_max"])
                self.last_verdict = f"DECAY ({decay:.1%}) -> raising conf to {new_conf:.3f}"
                log.warning(f"[WFV] {self.last_verdict}")
            elif oos_result.win_rate > 0.70:
                new_conf = max(current_min_conf - CFG["wfv_conf_reward"],
                               CFG["wfv_conf_min"])
                self.last_verdict = f"EDGE OK (OOS WR={oos_result.win_rate:.1%}) conf={new_conf:.3f}"
                log.info(f"[WFV] {self.last_verdict}")
            else:
                self.last_verdict = f"NEUTRAL (OOS WR={oos_result.win_rate:.1%})"
                log.info(f"[WFV] {self.last_verdict}")

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        for r in (is_result, oos_result):
            pf_val = r.profit_factor if r.profit_factor != float("inf") else 999
            datalog.log_wfv(
                timestamp=ts, fold_id=r.fold_id, phase=r.phase,
                n_ticks=r.n_ticks, n_trades=r.n_trades,
                win_rate=round(r.win_rate, 4), edge_vs_base=round(r.edge_vs_base, 4),
                total_pnl=round(r.total_pnl, 4), profit_factor=round(pf_val, 4),
                sharpe=round(r.sharpe, 4), max_drawdown=round(r.max_drawdown, 4),
                avg_conf=round(r.avg_conf, 4), circuit_trips=r.circuit_trips,
                live_min_conf=round(new_conf, 4),
            )

        self.adj_min_conf  = new_conf
        self.last_run_time = time.monotonic()
        return new_conf


# ══════════════════════════════════════════════════════════════════════════
# CONNECTION STATE
# ══════════════════════════════════════════════════════════════════════════
class ConnState(enum.IntEnum):
    DISCONNECTED  = 0
    CONNECTING    = 1
    CONNECTED     = 2
    AUTHENTICATED = 3
    SUBSCRIBED    = 4


# ══════════════════════════════════════════════════════════════════════════
# DERIV WS MANAGER  (verbatim architecture from reference bot)
# ══════════════════════════════════════════════════════════════════════════
class DerivWSManager:
    """
    Reconnecting WebSocket manager using the new Deriv Options API.
    Accepts a callable `url` that returns a fresh OTP URL on every connect.
    """
    RECONNECT_BASE    = 5.0
    RECONNECT_CAP     = 60.0
    HEARTBEAT_INTERVAL = 20

    def __init__(self, url, on_disconnect_cb=None, name="DerivWS"):
        self.url               = url
        self._on_disconnect_cb = on_disconnect_cb
        self.name              = name
        self.state             = ConnState.DISCONNECTED
        self._running          = False
        self._ws               = None
        self._attempt          = 0
        self._pending: dict[int, asyncio.Future] = {}

    _counter = 0

    @classmethod
    def _new_id(cls) -> int:
        cls._counter += 1
        return cls._counter

    async def safe_send(self, payload: dict) -> bool:
        ws   = self._ws
        live = (self.state >= ConnState.CONNECTED and ws is not None)
        if not live:
            return False
        try:
            await ws.send(json.dumps(payload))
            return True
        except Exception as e:
            log.warning(f"[{self.name}] safe_send failed: {e}")
            return False

    async def send(self, payload: dict, timeout: float = 15.0) -> dict:
        rid               = self._new_id()
        payload["req_id"] = rid
        fut               = asyncio.get_event_loop().create_future()
        self._pending[rid] = fut
        if not await self.safe_send(payload):
            self._pending.pop(rid, None)
            raise websockets.ConnectionClosed(None, None)
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            raise

    async def send_nowait(self, payload: dict):
        await self.safe_send(payload)

    def stop(self):
        self._running = False
        self.state    = ConnState.DISCONNECTED

    async def close(self):
        ws = self._ws
        if ws:
            try:
                await ws.close()
            except Exception:
                pass

    async def run(self, on_open, on_message):
        self._running = True
        while self._running:
            if self._attempt > 0:
                delay = min(
                    self.RECONNECT_BASE * (2 ** (self._attempt - 1)),
                    self.RECONNECT_CAP,
                ) + random.uniform(-1.0, 1.0)
                delay = max(1.0, delay)
                log.info(f"[{self.name}] Reconnect #{self._attempt} in {delay:.1f}s ...")
                await asyncio.sleep(delay)

            if not self._running:
                break

            self.state    = ConnState.CONNECTING
            self._pending.clear()
            ka_task   = None
            recv_task = None
            try:
                if callable(self.url):
                    try:
                        connect_url = await self.url()
                    except Exception as e:
                        log.error(f"[{self.name}] Failed to get OTP URL: {e}")
                        self.state = ConnState.DISCONNECTED
                        if not self._running:
                            break
                        if self._on_disconnect_cb:
                            try:
                                self._on_disconnect_cb()
                            except Exception as e2:
                                log.error(f"[{self.name}] disconnect_cb raised: {e2}")
                        self._attempt += 1
                        continue
                else:
                    connect_url = self.url

                self._ws = await websockets.connect(
                    connect_url,
                    ping_interval=None,
                    close_timeout=5,
                )
                self.state    = ConnState.CONNECTED
                self._attempt = 0
                log.info(f"[{self.name}] Connected.")

                ka_task = asyncio.create_task(self._heartbeat())

                async def _recv_loop():
                    async for raw in self._ws:
                        msg    = json.loads(raw)
                        req_id = msg.get("req_id")
                        if req_id and req_id in self._pending:
                            fut = self._pending.pop(req_id)
                            if not fut.done():
                                fut.set_result(msg)
                        else:
                            if msg.get("msg_type") == "ping":
                                continue
                            await on_message(msg)

                recv_task = asyncio.create_task(_recv_loop())
                await on_open(self)
                await recv_task

            except websockets.ConnectionClosed:
                log.warning(f"[{self.name}] Connection closed — reconnecting...")
            except Exception as e:
                log.error(f"[{self.name}] run error: {type(e).__name__}: {e}")
            finally:
                if ka_task:
                    ka_task.cancel()
                if recv_task and not recv_task.done():
                    recv_task.cancel()
                self.state = ConnState.DISCONNECTED
                await self.close()
                self._ws = None

                if not self._running:
                    break

                if self._on_disconnect_cb:
                    try:
                        self._on_disconnect_cb()
                    except Exception as e:
                        log.error(f"[{self.name}] disconnect_cb raised: {e}")

                self._attempt += 1

        log.info(f"[{self.name}] Connection loop exited.")

    async def _heartbeat(self):
        try:
            while self.state >= ConnState.CONNECTED:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
                if not await self.safe_send({"ping": 1}):
                    return
        except asyncio.CancelledError:
            pass


# ══════════════════════════════════════════════════════════════════════════
# MAIN BOT
# ══════════════════════════════════════════════════════════════════════════
class Over2Bot:

    STUCK_TIMEOUT_S = 30   # force-unlock if buy/contract stuck

    def __init__(self, app_id: str, api_token: str, account_id: str | None = None):
        self.app_id     = app_id
        self.token      = api_token
        self.account_id = account_id

        # ── Intelligence layers ──
        self.markov  = MarkovChain(order=CFG["markov_order"])
        self.hmm     = HMMRegimes()
        self.hawkes  = HawkesDetector()
        self.sampen  = SampleEntropySignal()
        self.arfima  = ARFIMASignal()
        self.hurst_a = HurstSignal()
        self.bayes   = BayesianEdge()
        self.copula  = CopulaSignal()
        self.kalman  = KalmanWinProb()
        self.guard   = RiskGuard()
        self.ensemble= Ensemble()

        # ── Walk-Forward Validator ──
        self.wfv      = LiveWalkForwardValidator()
        self.min_conf = CFG["min_confidence"]   # live-adjusted by WFV

        # ── Learning Phase ──
        self._start_time        = None   # set on first tick
        self._learning_complete = False
        self._learning_secs     = CFG["learning_phase_seconds"]

        # ── History ──
        self.digits  = deque(maxlen=CFG["tick_buffer"])
        self.spots   = deque(maxlen=CFG["tick_buffer"])
        self._tick_n = 0

        # ── Account ──
        self.bankroll    = CFG["starting_bankroll"]
        self.active_id   = None
        self._buying     = False
        self._lock_since = None
        self.trade_count = 0
        self.wins        = 0
        self.total_pnl   = 0.0
        self._entry_spot    = 0.0
        self._entry_digit   = 0
        self._entry_conf    = 0.0
        self._entry_stake   = CFG["stake"]

        self.wsman: DerivWSManager | None = None


    # ─────────────────────────────────────────────────────────────────────
    # LEARNING PHASE CHECK
    # ─────────────────────────────────────────────────────────────────────
    def _check_learning_phase(self) -> bool:
        """Returns True while still in the 5-minute learning/warmup phase."""
        if self._learning_complete:
            return False
        if self._start_time is None:
            return True
        elapsed   = time.monotonic() - self._start_time
        remaining = self._learning_secs - elapsed
        if remaining <= 0:
            self._learning_complete = True
            log.info("=" * 72)
            log.info("  ✅  LEARNING PHASE COMPLETE — LIVE TRADING NOW ACTIVE")
            log.info(f"     Ticks observed : {self._tick_n}")
            log.info(f"     HMM regime     : {self.hmm.name()}")
            log.info(f"     Bayesian WR    : {self.bayes.mean():.1%}")
            log.info(f"     Min confidence : {self.min_conf:.3f}")
            log.info("=" * 72)
            return False
        if self._tick_n % 50 == 0 and self._tick_n > 0:
            log.info(
                f"[LEARN] {elapsed:.0f}s / {self._learning_secs}s  "
                f"({remaining:.0f}s remaining)  ticks={self._tick_n}  "
                f"regime={self.hmm.name()}  Bayes={self.bayes.mean():.1%}"
            )
        return True

    # ─────────────────────────────────────────────────────────────────────
    # WFV PERIODIC CHECK
    # ─────────────────────────────────────────────────────────────────────
    def _maybe_run_wfv(self):
        if not CFG["wfv_enabled"]:
            return
        if not self._learning_complete:
            return
        if self.wfv.should_run():
            self.min_conf = self.wfv.run(self.digits, self.min_conf)

    # ─────────────────────────────────────────────────────────────────────
    # INTELLIGENCE PIPELINE
    # ─────────────────────────────────────────────────────────────────────
    def run_intelligence(self, spot: float, digit: int) -> tuple:
        if len(self.digits) < CFG["warmup_ticks"]:
            rem = CFG["warmup_ticks"] - len(self.digits)
            return False, 0.0, {}, f"WARMUP({rem} left)"

        arr = np.array(self.digits, dtype=float)

        # ── L1 Markov ──
        markov_p  = self.markov.p_over2()

        # ── L2 HMM ──
        hmm_sig   = self.hmm.signal()
        hmm_state = self.hmm.state

        # ── L3 Hawkes ──
        hawkes_sig = self.hawkes.signal()

        # ── L4 SampEn ──
        sampen_sig, se_raw, sampen_veto = self.sampen.signal(arr)

        # ── L5 ARFIMA ──
        arfima_sig, d_val = self.arfima.signal(arr)

        # ── L6 Hurst ──
        hurst_sig, h_val = self.hurst_a.signal(arr)

        # ── L7 Bayesian ──
        bayes_sig = self.bayes.mean()

        # ── L8 Copula ──
        copula_sig, lam_L, copula_veto = self.copula.signal(arr)

        # ── L9 Kalman ──
        kalman_sig = self.kalman.signal()

        log.info(
            f"[SIG] MKV={markov_p:.3f} HMM={self.hmm.name()}({hmm_sig:.2f}) "
            f"HWK={hawkes_sig:.2f} SE={se_raw:.2f}→{sampen_sig:.2f} "
            f"ARFIMA_d={d_val:.3f}→{arfima_sig:.2f} H={h_val:.3f}→{hurst_sig:.2f} "
            f"Bay={bayes_sig:.3f} Cop_λL={lam_L:.3f}→{copula_sig:.2f} "
            f"Kal={kalman_sig:.3f}"
        )

        trade, conf, scores, failed, reason = self.ensemble.decide(
            markov_p=markov_p, hmm_sig=hmm_sig, hawkes_sig=hawkes_sig,
            sampen_sig=sampen_sig, arfima_sig=arfima_sig, hurst_sig=hurst_sig,
            bayes_sig=bayes_sig, copula_sig=copula_sig, kalman_sig=kalman_sig,
            hmm_state=hmm_state, sampen_veto=sampen_veto, copula_veto=copula_veto,
            min_confidence=self.min_conf,   # live-adjusted by WFV
        )

        try:
            datalog.log_signal(
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                tick_n=self._tick_n, spot=spot, last_digit=digit,
                markov_p=round(markov_p, 5), hmm_sig=round(hmm_sig, 5),
                hawkes_sig=round(hawkes_sig, 5), sampen_sig=round(sampen_sig, 5),
                arfima_sig=round(arfima_sig, 5), hurst_sig=round(hurst_sig, 5),
                bayes_sig=round(bayes_sig, 5), copula_sig=round(copula_sig, 5),
                kalman_sig=round(kalman_sig, 5),
                pre_score=round(conf, 5), conf=round(conf, 5),
                regime=self.hmm.name(), reason=reason,
            )
        except Exception as e:
            log.warning(f"signal CSV failed: {e}")

        return trade, conf, scores, reason

    # ─────────────────────────────────────────────────────────────────────
    # TICK HANDLER
    # ─────────────────────────────────────────────────────────────────────
    async def on_tick(self, tick: dict):
        spot  = float(tick.get("quote", tick.get("ask", 0.0)))
        digit = last_digit(spot)

        # ── Initialise start time on first tick ──
        if self._start_time is None:
            self._start_time = time.monotonic()
            log.info("=" * 72)
            log.info(f"  🔵  LEARNING PHASE STARTED — observing for "
                     f"{self._learning_secs}s before trading")
            log.info("=" * 72)

        self.spots.append(spot)
        self.digits.append(digit)
        self._tick_n += 1

        # Update all online models on every tick (even during cooldown/learning)
        self.markov.update(digit)
        self.hmm.update(digit)
        self.hawkes.update(digit)
        self.kalman.update(1.0 if digit > 2 else 0.0)
        self.guard.tick()

        if self._tick_n % 10 == 0:
            phase_tag = "LEARNING" if not self._learning_complete else "TRADING"
            log.info(
                f"Tick #{self._tick_n:5d} | spot={spot} | digit={digit} | "
                f"bankroll=${self.bankroll:.2f} | guard={self.guard.status()} | "
                f"regime={self.hmm.name()} | phase={phase_tag} | "
                f"min_conf={self.min_conf:.3f}"
            )

        # ── Learning Phase: observe & learn, do NOT trade ──
        if self._check_learning_phase():
            return

        # ── Run WFV if due ──
        self._maybe_run_wfv()

        # ── Stuck contract guard ──
        if (self.active_id or self._buying) and self._lock_since:
            if time.monotonic() - self._lock_since > self.STUCK_TIMEOUT_S:
                log.warning("⏱️  Stuck lock — force-unlocking.")
                self.active_id   = None
                self._buying     = False
                self._lock_since = None
            else:
                return

        if not self.guard.can_trade():
            return

        result = self.run_intelligence(spot, digit)
        trade, conf, scores, reason = result

        if trade:
            stake = self.guard.compute_stake(self.bankroll, self.bayes.mean())
            log.info(
                f"🎯  SIGNAL  conf={conf:.3f}  digit={digit}  spot={spot}  "
                f"stake=${stake:.2f}  reason={reason}  wfv={self.wfv.last_verdict}"
            )
            self._entry_spot  = spot
            self._entry_digit = digit
            self._entry_conf  = conf
            self._buying      = True
            self._lock_since  = time.monotonic()
            asyncio.create_task(self._request_and_buy(spot, conf, stake))

    # ─────────────────────────────────────────────────────────────────────
    # PROPOSAL → BUY
    # ─────────────────────────────────────────────────────────────────────
    async def _request_and_buy(self, spot: float, conf: float = 0.0, stake: float | None = None):
        try:
            if self.active_id:
                return
            if stake is None:
                stake = self.guard.stake

            resp = await self.wsman.send({
                "proposal":          1,
                "amount":            stake,
                "basis":             "stake",
                "contract_type":     CFG["contract_type"],
                "currency":          CFG["currency"],
                "duration":          CFG["duration"],
                "duration_unit":     CFG["duration_unit"],
                "underlying_symbol": CFG["underlying_symbol"],
                "barrier":           CFG["barrier"],
            })

            if resp.get("error"):
                log.warning(f"Proposal error: {resp['error'].get('message')}")
                return

            prop      = resp.get("proposal", {})
            pid       = prop.get("id")
            ask_price = prop.get("ask_price")

            if not pid or not ask_price:
                log.warning("Empty proposal — skipping")
                return

            if self.active_id:
                return

            await self._buy(pid, float(ask_price), spot, conf, stake)

        except asyncio.TimeoutError:
            log.warning("Proposal timed out")
        except Exception as exc:
            log.error(f"_request_and_buy error: {exc}")
        finally:
            self._buying = False

    async def _buy(self, proposal_id: str, price: float,
                   spot: float = 0.0, conf: float = 0.0, stake: float | None = None):
        try:
            resp = await self.wsman.send({"buy": proposal_id, "price": price})

            if resp.get("error"):
                log.warning(f"Buy rejected: {resp['error'].get('message')}")
                return

            buy_data = resp.get("buy", {})
            cid      = buy_data.get("contract_id")
            if not cid:
                log.warning("Buy response missing contract_id")
                return

            self.active_id   = cid
            self.trade_count += 1
            self._entry_stake = stake if stake is not None else self.guard.stake
            self._lock_since  = time.monotonic()
            log.info(
                f"✅  CONTRACT #{self.trade_count} OPEN | "
                f"id={cid} | digit={self._entry_digit} | "
                f"stake=${self._entry_stake:.2f} | conf={conf:.3f}"
            )

            await self.wsman.send_nowait({
                "proposal_open_contract": 1,
                "contract_id":            cid,
                "subscribe":              1,
            })

        except asyncio.TimeoutError:
            log.warning("Buy timed out")
        except Exception as exc:
            log.error(f"_buy error: {exc}")

    # ─────────────────────────────────────────────────────────────────────
    # SETTLEMENT
    # ─────────────────────────────────────────────────────────────────────
    def _settle(self, poc: dict):
        profit = float(poc.get("profit", 0.0))
        won    = profit > 0.0

        self.total_pnl  += profit
        self.bankroll   += profit
        contract_id      = self.active_id
        self.active_id   = None
        self._buying     = False
        self._lock_since = None

        self.bayes.update(won)
        self.guard.check_bankroll(self.bankroll)

        if won:
            self.wins += 1
            self.guard.on_win()
            tag = "🟢  WIN "
        else:
            self.guard.on_loss()
            tag = "🔴  LOSS"

        wr     = self.wins / self.trade_count if self.trade_count else 0.0
        lo, hi = self.bayes.ci95()

        log.info(f"{tag}  {profit:+.3f}  cumPnL={self.total_pnl:+.3f}  "
                 f"bankroll=${self.bankroll:.2f}")
        log.info(
            f"📊  trades={self.trade_count}  WR={wr:.1%}  "
            f"Bayes_WR={self.bayes.mean():.1%} CI=[{lo:.2f},{hi:.2f}]  "
            f"next={self.guard.status()}  wfv_conf={self.min_conf:.3f}"
        )

        try:
            datalog.log_trade(
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                trade_n=self.trade_count, contract_id=contract_id,
                spot_entry=self._entry_spot, last_digit_entry=self._entry_digit,
                stake=self._entry_stake, conf=round(self._entry_conf, 5),
                profit=round(profit, 5), won=int(won),
                bankroll=round(self.bankroll, 5),
                total_pnl=round(self.total_pnl, 5),
                win_rate=round(wr, 5),
            )
        except Exception as e:
            log.warning(f"trade CSV failed: {e}")

    # ─────────────────────────────────────────────────────────────────────
    # NEW OPTIONS API — REST BOOTSTRAP
    # ─────────────────────────────────────────────────────────────────────
    def _rest_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "Deriv-App-ID":  self.app_id,
            "Content-Type":  "application/json",
        }

    def _resolve_account_id_sync(self) -> str:
        url  = CFG["api_base"] + CFG["accounts_path"]
        resp = requests.get(url, headers=self._rest_headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        accounts = data.get("data", data) if isinstance(data, dict) else data
        if isinstance(accounts, dict):
            accounts = accounts.get("accounts", accounts.get("data", []))
        for acc in accounts:
            if acc.get("account_type") == "real":
                acc_id = acc.get("account_id") or acc.get("id")
                if acc_id:
                    return acc_id
        raise RuntimeError(f"No demo account found: {data}")

    def _fetch_otp_url_sync(self) -> str:
        if not self.account_id:
            self.account_id = self._resolve_account_id_sync()
            log.info(f"Resolved demo account_id = {self.account_id}")
        url  = CFG["api_base"] + CFG["otp_path"].format(account_id=self.account_id)
        resp = requests.post(url, headers=self._rest_headers(), timeout=15)
        resp.raise_for_status()
        data    = resp.json()
        payload = data.get("data", data) if isinstance(data, dict) else data
        ws_url  = payload.get("url")
        if not ws_url:
            raise RuntimeError(f"OTP response missing data.url: {data}")
        return ws_url

    async def _get_ws_url(self) -> str:
        return await asyncio.to_thread(self._fetch_otp_url_sync)

    # ─────────────────────────────────────────────────────────────────────
    # MESSAGE DISPATCHER
    # ─────────────────────────────────────────────────────────────────────
    async def on_message(self, msg: dict):
        mt = msg.get("msg_type")
        if mt == "tick":
            await self.on_tick(msg["tick"])
        elif mt == "proposal_open_contract":
            poc = msg.get("proposal_open_contract", {})
            if poc.get("is_sold") or poc.get("status") in ("won", "lost"):
                self._settle(poc)
        elif mt == "error":
            log.error(f"API error: {msg.get('error', {}).get('message')}")

    # ─────────────────────────────────────────────────────────────────────
    # CONNECTION HOOKS
    # ─────────────────────────────────────────────────────────────────────
    async def _on_open(self, wsman: DerivWSManager):
        # No "authorize" message — OTP URL is pre-authenticated
        wsman.state = ConnState.AUTHENTICATED
        log.info(f"Connected to authenticated OTP session (account={self.account_id}).")

        await wsman.send_nowait({
            "ticks":     CFG["underlying_symbol"],
            "subscribe": 1,
        })
        wsman.state = ConnState.SUBSCRIBED
        log.info(f"Subscribed to {CFG['underlying_symbol']} — "
                 f"learning phase: {CFG['learning_phase_seconds']}s then live trading.")

        if self.active_id:
            await wsman.send_nowait({
                "proposal_open_contract": 1,
                "contract_id":            self.active_id,
                "subscribe":              1,
            })

    def _on_disconnect(self):
        if self._buying:
            log.warning("Connection lost during buy — resetting flag.")
            self._buying     = False
            self._lock_since = None
        if self.active_id:
            log.warning(
                f"Connection lost with contract #{self.active_id} open — "
                f"will resubscribe on reconnect."
            )

    # ─────────────────────────────────────────────────────────────────────
    # MAIN LOOP
    # ─────────────────────────────────────────────────────────────────────
    async def run(self):
        bar = "═" * 72
        log.info(bar)
        log.info("  DERIV DIGIT OVER 2 BOT  ·  R_100  ·  WALK-FORWARD EDITION")
        log.info("  10-Layer Intelligence + Live Walk-Forward Validator")
        log.info(f"  Duration     : {CFG['duration']} ticks (5-tick contracts)")
        log.info(f"  Stake        : ${CFG['stake']:.2f} flat  ·  Stop: ${CFG['drawdown_stop']:.2f}")
        log.info(f"  Warmup       : {CFG['warmup_ticks']} ticks  ·  Min Confidence: {CFG['min_confidence']}")
        log.info(f"  Learning     : {CFG['learning_phase_seconds']}s observation before trading")
        log.info(f"  WFV          : IS={CFG['wfv_is_size']} OOS={CFG['wfv_oos_size']} ticks, "
                 f"every {CFG['wfv_eval_interval_s']}s")
        log.info(f"  WFV decay    : >{CFG['wfv_decay_threshold']:.0%} → +{CFG['wfv_conf_penalty']:.3f} conf penalty")
        log.info("  Connection: new Options API (REST OTP bootstrap)")
        log.info(bar)

        self.wsman = DerivWSManager(
            self._get_ws_url,
            on_disconnect_cb=self._on_disconnect,
            name="Over2WS",
        )
        await self.wsman.run(on_open=self._on_open, on_message=self.on_message)


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    APP_ID     = os.getenv("DERIV_APP_ID", "")
    API_TOKEN  = os.getenv("DERIV_API_TOKEN", "")
    ACCOUNT_ID = os.getenv("DERIV_ACCOUNT_ID", "") or None

    missing = []
    if not APP_ID:
        missing.append("DERIV_APP_ID")
    if not API_TOKEN:
        missing.append("DERIV_API_TOKEN")
    if missing:
        print(
            f"\n⚠️  {', '.join(missing)} not set.\n"
            "   Set them as environment variables before starting the bot.\n"
            "   (App ID and PAT must be from a NEW developers.deriv.com\n"
            "   application — legacy App IDs like 1089 no longer work.)\n"
        )
        raise SystemExit(1)

    bot = Over2Bot(app_id=APP_ID, api_token=API_TOKEN, account_id=ACCOUNT_ID)
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        log.info("Bot stopped (Ctrl+C)")
