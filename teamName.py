import numpy as np


N_INST = 51
ALGO_IDX = 0
EPS = 1e-12
LIMITS = np.array([100_000.0] + [10_000.0] * 50, dtype=float)

# Chosen from split testing with the exact local evaluator logic.
BETA_WINDOW = 55
ZSCORE_WINDOW = 40
ENTRY_Z = 1.75
EXIT_Z = 0.50
STOP_Z = 4.00
MAX_HOLD = 12
TOP_K = 8
STOCK_DOLLAR_FRACTION = 1.00
ALGO_HEDGE_CAP_FRACTION = 0.18
MIN_ABS_BETA = 0.05
MAX_ABS_BETA = 3.00
MIN_HISTORY = max(BETA_WINDOW + 2, ZSCORE_WINDOW + 2, 80)


class AlgoRelativeValueArb:
    """Trade stock-vs-ALGO residual dislocations with ALGO as the hedge leg."""

    def __init__(self):
        self.reset_state()

    def reset_state(self):
        self.last_nt = 0
        self.direction = np.zeros(N_INST, dtype=int)
        self.hold_days = np.zeros(N_INST, dtype=int)
        self.cached_position = np.zeros(N_INST, dtype=int)

    def _signal_snapshot(self, prices):
        n_inst, nt = prices.shape
        if n_inst != N_INST or nt < MIN_HISTORY:
            return None

        log_prices = np.log(np.maximum(prices, EPS))
        log_returns = np.diff(log_prices, axis=1)
        algo_returns = log_returns[ALGO_IDX, -BETA_WINDOW:]
        algo_var = float(np.dot(algo_returns, algo_returns))
        if algo_var < 1e-12:
            return None

        stock_returns = log_returns[1:, -BETA_WINDOW:]
        betas = (stock_returns @ algo_returns) / (algo_var + EPS)

        # Relative log-price after removing the stock's recent ALGO beta.
        relative_series = (
            (log_prices[1:] - log_prices[1:, [0]])
            - betas[:, None] * (log_prices[ALGO_IDX] - log_prices[ALGO_IDX, 0])
        )

        history = relative_series[:, -ZSCORE_WINDOW - 1 : -1]
        spread_mean = history.mean(axis=1)
        spread_std = history.std(axis=1)
        zscores = np.where(
            spread_std > 1e-10,
            (relative_series[:, -1] - spread_mean) / spread_std,
            np.nan,
        )

        return betas, spread_std, zscores

    def _step(self, prices):
        snapshot = self._signal_snapshot(prices)
        if snapshot is None:
            self.direction[:] = 0
            self.hold_days[:] = 0
            return np.zeros(N_INST, dtype=int)

        betas, spread_std, zscores = snapshot
        desired_dollars = np.zeros(N_INST, dtype=float)
        proposals = []

        for local_idx, inst in enumerate(range(1, N_INST)):
            beta = float(betas[local_idx])
            zscore = float(zscores[local_idx])
            valid = (
                np.isfinite(zscore)
                and MIN_ABS_BETA <= abs(beta) <= MAX_ABS_BETA
                and float(spread_std[local_idx]) > 1e-10
            )

            if not valid:
                self.direction[inst] = 0
                self.hold_days[inst] = 0
                continue

            direction = int(self.direction[inst])
            hold = int(self.hold_days[inst])

            if direction == 0:
                if zscore >= ENTRY_Z:
                    direction, hold = -1, 1
                elif zscore <= -ENTRY_Z:
                    direction, hold = 1, 1
            else:
                hold += 1
                if abs(zscore) <= EXIT_Z or abs(zscore) >= STOP_Z or hold >= MAX_HOLD:
                    direction, hold = 0, 0

            self.direction[inst] = direction
            self.hold_days[inst] = hold

            if direction != 0:
                proposals.append((abs(zscore), inst, beta, direction))

        proposals.sort(key=lambda row: row[0], reverse=True)

        for _, inst, beta, direction in proposals[:TOP_K]:
            stock_dollars = min(
                LIMITS[inst] * STOCK_DOLLAR_FRACTION,
                LIMITS[ALGO_IDX] * ALGO_HEDGE_CAP_FRACTION / max(abs(beta), 1e-6),
            )
            desired_dollars[inst] += direction * stock_dollars
            desired_dollars[ALGO_IDX] -= direction * stock_dollars * beta

        used = np.where(np.abs(desired_dollars) > 1e-8)[0]
        if used.size > 0:
            scale = float(
                np.min(LIMITS[used] / np.maximum(np.abs(desired_dollars[used]), 1e-8))
            )
            desired_dollars *= min(1.0, scale)

        current_prices = np.maximum(prices[:, -1], 1.0)
        return np.rint(desired_dollars / current_prices).astype(int)

    def _rebuild_to_current_day(self, prices):
        self.reset_state()
        n_inst, nt = prices.shape
        if n_inst != N_INST:
            return np.zeros(n_inst, dtype=int)
        if nt < MIN_HISTORY:
            self.last_nt = nt
            self.cached_position = np.zeros(n_inst, dtype=int)
            return self.cached_position.copy()

        for day in range(MIN_HISTORY, nt + 1):
            self.cached_position = self._step(prices[:, :day])
            self.last_nt = day
        return self.cached_position.copy()

    def get_position(self, prcSoFar):
        prices = np.asarray(prcSoFar, dtype=float)
        if prices.ndim != 2:
            raise ValueError("prcSoFar must be a 2D array")

        n_inst, nt = prices.shape
        if n_inst != N_INST:
            self.reset_state()
            return np.zeros(n_inst, dtype=int)

        if nt < MIN_HISTORY:
            if nt < self.last_nt:
                self.reset_state()
            self.last_nt = nt
            self.cached_position = np.zeros(n_inst, dtype=int)
            return self.cached_position.copy()

        if nt == self.last_nt:
            return self.cached_position.copy()

        if nt == self.last_nt + 1:
            self.cached_position = self._step(prices)
            self.last_nt = nt
            return self.cached_position.copy()

        return self._rebuild_to_current_day(prices)


_LIVE_STRATEGY = AlgoRelativeValueArb()


def reset_state():
    _LIVE_STRATEGY.reset_state()


def getMyPosition(prcSoFar):
    return _LIVE_STRATEGY.get_position(prcSoFar)
