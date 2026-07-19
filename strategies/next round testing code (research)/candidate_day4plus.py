#!/usr/bin/env python3
"""day4-plus: the testRound day-4 book with proven upgrades + MR gate.

Base = submissions/testRound/day4/sharpeShooters4.py (board leader on
hidden 751-950: score 1004.59, 1023.13/day). Changes:

  1. LL sleeve upgraded: WF-IC-masked sign-at-limits (+pair exclusion)
     -> cross-sectionally demeaned tanh(z/0.35) on the full universe
     (candidate_next's sleeve, live ~920/day on 751-950).
  2. ALGO fade dropped (6.39 true-OOS on 501-750 = noise).
  3. MR gate (toggle): the MR core trades only while its own trailing
     GATE_WIN-day walk-forward paper PnL is positive. Aim: keep the
     +1023/day regime (751-950) while cutting the -86/day one (501-750).
  4. Optional cross-sectional demeaning of the MR signal (toggle).

Stateless regime guards, frozen pairs, hysteresis and dead-band kept
exactly as day4 had them. Class-based so the runner can ablate sleeves.
"""

import numpy as np

N_INST = 51
LIMITS = np.array([100000.0] + [10000.0] * 50)
EPS = 1e-12

# --- MR core (day4 values) ---
FAST_LB = 8
SLOW_LB = 30
FAST_WEIGHT = 0.75
SIGNAL_SCALE = 1.2
POSITION_MULT = 2.0
MIN_ABS_SIGNAL = 0.25
EXIT_ABS_SIGNAL = 0.20
DEAD_BAND_FRAC = 0.15
CORE_ASSETS = np.array([
    0, 35, 40, 5, 37, 29, 22, 14, 10, 44,
    36, 21, 41, 13, 17, 16, 19, 39, 27, 32
], dtype=int)

# --- MR gate ---
GATE_WIN = 60
GATE_MIN_OBS = 20

# --- stateless regime guards (day4 values) ---
RECENT_VOL_WIN = 10
BASE_VOL_WIN = 80
VOL_DANGER = 2.0
VOL_CUT = 0.65
TREND_WIN = 20
TREND_LOOKBACK = 60
TREND_DANGER = 2.5
TREND_CUT = 0.50

# --- adaptive pair selection (cv2 procedure) ---
RESCAN = 50
MIN_SEL_HIST = 250
PREFILTER = 200
HL_MAX = 20.0
SR_MIN = 1.5
SR_KEEP = 1.0
MAX_PAIRS = 15
MAX_PER_NAME = 2

# --- pairs (day4 frozen set) ---
PAIRS = (
    (7, 40, 0.7086), (25, 37, 0.9352), (1, 20, 0.9836),
    (13, 45, 1.0132), (33, 40, 0.2577), (10, 46, 1.0331),
    (33, 42, 0.8358), (31, 43, 0.9692), (18, 28, 0.5642),
    (41, 50, 0.4977), (8, 27, 1.0222), (18, 35, 0.8471),
    (37, 46, 0.4059), (36, 41, 0.9137), (35, 42, 0.9163),
)
PAIR_LEG = 9_000.0
PAIR_ROLL = 60
PAIR_ENTRY = 1.5
PAIR_EXIT = 0.5
PAIR_OWNED = sorted({i for i, _, _ in PAIRS} | {j for _, j, _ in PAIRS})

# --- lead-lag (demeaned tanh, cnext values) ---
LL_LAM = 400.0
LL_RETRAIN = 50
LL_MIN_HIST = 120
LL_TEMP = 0.35


def _half_life(s):
    x, y = s[:-1] - s.mean(), s[1:] - s.mean()
    denom = (x * x).sum()
    if denom < EPS:
        return np.inf
    phi = (x * y).sum() / denom
    if not (0.0 < phi < 1.0):
        return np.inf
    return -np.log(2.0) / np.log(phi)


def _spread_sr_halves(s):
    n = len(s)
    if n < PAIR_ROLL + 20:
        return -np.inf, False
    pos = 0
    pnl = np.zeros(n)
    for t in range(PAIR_ROLL, n - 1):
        win = s[t - PAIR_ROLL:t]
        z = (s[t] - win.mean()) / (win.std() + EPS)
        if pos == 0:
            if z > PAIR_ENTRY:
                pos = -1
            elif z < -PAIR_ENTRY:
                pos = 1
        elif pos == 1 and z > -PAIR_EXIT:
            pos = 0
        elif pos == -1 and z < PAIR_EXIT:
            pos = 0
        pnl[t + 1] = pos * (s[t + 1] - s[t])
    pnl = pnl[PAIR_ROLL:]
    sd = pnl.std()
    if sd < EPS:
        return -np.inf, False
    sr = np.sqrt(250.0) * pnl.mean() / sd
    h = len(pnl) // 2
    return sr, (pnl[:h].sum() > 0 and pnl[h:].sum() > 0)


def _score_pair(lpx, a, b):
    x, y = lpx[b], lpx[a]
    vx = x.var()
    if vx < EPS:
        return None
    g = np.cov(y, x)[0, 1] / vx
    if not (0.1 < g < 3.0):
        return None
    s = y - g * x
    if _half_life(s) > HL_MAX:
        return None
    sr, ok = _spread_sr_halves(s)
    if not ok:
        return None
    return sr, g


def _select_pairs(lpx, incumbents):
    L = lpx[1:]
    n = L.shape[0]
    Ld = L - L.mean(axis=1, keepdims=True)
    norm = np.sqrt((Ld * Ld).sum(axis=1)) + EPS
    C = (Ld @ Ld.T) / np.outer(norm, norm)
    iu = np.triu_indices(n, k=1)
    order = np.argsort(-np.abs(C[iu]))[:PREFILTER]

    inc = set(incumbents)
    cands = {(iu[0][idx] + 1, iu[1][idx] + 1) for idx in order} | inc
    scored = []
    for a, b in cands:
        res = _score_pair(lpx, a, b)
        if res is None:
            continue
        sr, g = res
        if sr >= (SR_KEEP if (a, b) in inc else SR_MIN):
            scored.append((sr + (0.5 if (a, b) in inc else 0.0), a, b, g))

    scored.sort(reverse=True)
    used = {}
    chosen = []
    for _, a, b, g in scored:
        if used.get(a, 0) >= MAX_PER_NAME or used.get(b, 0) >= MAX_PER_NAME:
            continue
        chosen.append((a, b, round(g, 4)))
        used[a] = used.get(a, 0) + 1
        used[b] = used.get(b, 0) + 1
        if len(chosen) >= MAX_PAIRS:
            break
    return chosen


class Day4Plus:
    def __init__(self, mr_on=True, mr_gate=True, mr_demean=False,
                 pairs_on=True, pairs_adaptive=False, ll_on=True,
                 guards_on=True):
        self.mr_on = mr_on
        self.mr_gate = mr_gate
        self.mr_demean = mr_demean
        self.pairs_on = pairs_on
        self.pairs_adaptive = pairs_adaptive
        self.ll_on = ll_on
        self.guards_on = guards_on
        self.reset()

    def reset(self):
        self.prev_book = np.zeros(N_INST)      # MR+pairs book (dead-band)
        self.prev_mr_ungated = np.zeros(N_INST)
        self.mr_pnl = []                       # ungated MR paper PnL
        self.pair_pos = [0] * len(PAIRS)
        self.apairs = []                       # adaptive: [(i, j, g)]
        self.apair_pos = {}
        self.last_scan = -1
        # names excluded from MR (dynamic in adaptive mode)
        self.owned = [] if self.pairs_adaptive else list(PAIR_OWNED)
        self.ll_W = None
        self.ll_mu = None
        self.ll_sd = None
        self.ll_resid_sd = None
        self.ll_last_fit = -1
        self.prev_nt = -1

    # ------------------------------------------------------------ MR
    def _mr_dollars(self, prc):
        nInst, nt = prc.shape
        if nt < SLOW_LB + 1:
            return np.zeros(N_INST)
        cur = np.maximum(prc[:, -1], 1.0)

        def z_to_past(lb):
            hist = prc[:, -lb - 1:-1]
            mu = hist.mean(axis=1)
            sig = hist.std(axis=1)
            sig = np.where(sig > 1e-8, sig, 1.0)
            return (mu - cur) / sig

        signal = (FAST_WEIGHT * z_to_past(FAST_LB)
                  + (1.0 - FAST_WEIGHT) * z_to_past(SLOW_LB))

        if self.mr_demean:
            signal = signal - signal[CORE_ASSETS].mean()

        holding = np.sign(self.prev_mr_ungated)
        keep = ((holding != 0) & (np.sign(signal) == holding)
                & (np.abs(signal) >= EXIT_ABS_SIGNAL))
        active = (np.abs(signal) >= MIN_ABS_SIGNAL) | keep
        signal = np.where(active, signal, 0.0)

        tgt = LIMITS * np.tanh(SIGNAL_SCALE * signal) * POSITION_MULT
        mask = np.zeros(N_INST)
        mask[CORE_ASSETS] = 1.0
        if self.owned:
            mask[self.owned] = 0.0
        tgt *= mask

        if self.guards_on:
            tgt = tgt * self._guard_scales(prc, signal)
        return tgt

    def _guard_scales(self, prc, signal):
        nInst, nt = prc.shape
        scale = np.ones(N_INST)
        if nt <= BASE_VOL_WIN + RECENT_VOL_WIN + 1:
            return scale
        lp = np.log(np.maximum(prc, EPS))
        lr = np.diff(lp, axis=1)

        recent_mkt = lr[0, -RECENT_VOL_WIN:].std()
        base_mkt = lr[0, -(BASE_VOL_WIN + RECENT_VOL_WIN):-RECENT_VOL_WIN].std()
        if base_mkt > 1e-8 and recent_mkt / base_mkt > VOL_DANGER:
            scale *= VOL_CUT

        recent_v = lr[:, -RECENT_VOL_WIN:].std(axis=1)
        base_v = lr[:, -(BASE_VOL_WIN + RECENT_VOL_WIN):-RECENT_VOL_WIN].std(axis=1)
        ratio = recent_v / np.where(base_v > 1e-8, base_v, 1.0)
        scale *= np.where(ratio > VOL_DANGER, VOL_CUT, 1.0)

        if nt > TREND_LOOKBACK + TREND_WIN:
            trend = lp[:, -1] - lp[:, -TREND_WIN - 1]
            dvol = lr[:, -TREND_LOOKBACK:].std(axis=1)
            tz = trend / np.where(dvol > 1e-8, dvol * np.sqrt(TREND_WIN), 1.0)
            fighting = signal * tz < 0
            scale *= np.where(fighting & (np.abs(tz) > TREND_DANGER),
                              TREND_CUT, 1.0)
        return scale

    # ------------------------------------------------------------ pairs
    def _pair_dollars(self, prc):
        if self.pairs_adaptive:
            return self._adaptive_pair_dollars(prc)
        nt = prc.shape[1]
        tgt = np.zeros(N_INST)
        if nt <= PAIR_ROLL + 1:
            return tgt
        lpx = np.log(np.maximum(prc, EPS))
        for k, (i, j, g) in enumerate(PAIRS):
            s = lpx[i] - g * lpx[j]
            win = s[-PAIR_ROLL - 1:-1]
            z = (s[-1] - win.mean()) / (win.std() + EPS)
            pos = self.pair_pos[k]
            if pos == 0:
                if z > PAIR_ENTRY:
                    pos = -1
                elif z < -PAIR_ENTRY:
                    pos = 1
            elif pos == 1 and z > -PAIR_EXIT:
                pos = 0
            elif pos == -1 and z < PAIR_EXIT:
                pos = 0
            self.pair_pos[k] = pos
            if pos != 0:
                tgt[i] += pos * PAIR_LEG
                tgt[j] -= pos * g * PAIR_LEG
        return tgt

    def _adaptive_pair_dollars(self, prc):
        nt = prc.shape[1]
        tgt = np.zeros(N_INST)
        if nt < MIN_SEL_HIST:
            return tgt
        lpx = np.log(np.maximum(prc, EPS))

        if self.last_scan < 0 or (nt - self.last_scan) >= RESCAN:
            incumbents = tuple((i, j) for i, j, _ in self.apairs)
            self.apairs = _select_pairs(lpx, incumbents)
            keys = {(i, j) for i, j, _ in self.apairs}
            self.apair_pos = {k: v for k, v in self.apair_pos.items()
                              if k in keys}
            self.owned = sorted({i for i, _, _ in self.apairs}
                                | {j for _, j, _ in self.apairs})
            self.last_scan = nt

        for i, j, g in self.apairs:
            s = lpx[i] - g * lpx[j]
            win = s[-PAIR_ROLL - 1:-1]
            z = (s[-1] - win.mean()) / (win.std() + EPS)
            pos = self.apair_pos.get((i, j), 0)
            if pos == 0:
                if z > PAIR_ENTRY:
                    pos = -1
                elif z < -PAIR_ENTRY:
                    pos = 1
            elif pos == 1 and z > -PAIR_EXIT:
                pos = 0
            elif pos == -1 and z < PAIR_EXIT:
                pos = 0
            self.apair_pos[(i, j)] = pos
            if pos != 0:
                tgt[i] += pos * PAIR_LEG
                tgt[j] -= pos * g * PAIR_LEG
        return tgt

    # ------------------------------------------------------------ LL
    def _ll_dollars(self, prc):
        nt = prc.shape[1]
        if nt < LL_MIN_HIST:
            return np.zeros(N_INST)
        r = np.diff(np.log(np.maximum(prc, EPS)), axis=1)
        if self.ll_W is None or (nt - self.ll_last_fit) >= LL_RETRAIN:
            X = r[:, :-1].T
            Y = r[:, 1:].T
            self.ll_mu, self.ll_sd = X.mean(0), X.std(0)
            self.ll_sd = np.where(self.ll_sd > 1e-12, self.ll_sd, 1.0)
            Xs = (X - self.ll_mu) / self.ll_sd
            self.ll_W = np.linalg.solve(
                Xs.T @ Xs + LL_LAM * np.eye(N_INST), Xs.T @ Y)
            self.ll_resid_sd = np.maximum(Y.std(0), 1e-8)
            self.ll_last_fit = nt
        pred = ((r[:, -1] - self.ll_mu) / self.ll_sd) @ self.ll_W
        z = pred / self.ll_resid_sd
        z = z - z.mean()
        z = z / (z.std() + 1e-9)
        return LIMITS * np.tanh(z / LL_TEMP)

    # ------------------------------------------------------------ book
    def target_dollars(self, prc):
        nt = prc.shape[1]
        if nt <= self.prev_nt:
            self.reset()
        self.prev_nt = nt

        # walk-forward paper PnL of the UNGATED MR sleeve (for the gate)
        if nt >= 2 and self.prev_mr_ungated.any():
            r = prc[:, -1] / np.maximum(prc[:, -2], EPS) - 1.0
            self.mr_pnl.append(float(self.prev_mr_ungated @ r))

        mr_ungated = self._mr_dollars(prc) if self.mr_on else np.zeros(N_INST)
        gate_open = True
        if self.mr_gate and len(self.mr_pnl) >= GATE_MIN_OBS:
            gate_open = sum(self.mr_pnl[-GATE_WIN:]) > 0.0
        mr = mr_ungated if gate_open else np.zeros(N_INST)
        self.prev_mr_ungated = mr_ungated

        book = mr
        if self.pairs_on:
            book = book + self._pair_dollars(prc)

        # day4 dollar dead-band on the MR+pairs book
        small = ((np.abs(book - self.prev_book) < DEAD_BAND_FRAC * LIMITS)
                 & (book != 0.0))
        book = np.where(small, self.prev_book, book)
        self.prev_book = book.copy()

        if self.ll_on:
            book = book + self._ll_dollars(prc)
        return np.clip(book, -LIMITS, LIMITS)


_default = Day4Plus(pairs_adaptive=True)


def reset_state():
    _default.reset()


def getMyPosition(prcSoFar):
    prcSoFar = np.asarray(prcSoFar, dtype=float)
    tgt = _default.target_dollars(prcSoFar)
    return (tgt / prcSoFar[:, -1]).astype(int)
