import numpy as np
import pandas as pd

N_INST = 51
LIMITS = np.array([100000] + [10000] * 50, dtype=float)
COMM = np.array([0.00002] + [0.0001] * 50, dtype=float)
EPS = 1e-12


def score(mu, sigma, param=1.0):
    if mu <= 0 or sigma < 1e-10:
        return mu
    sr = np.sqrt(250.0) * mu / sigma
    frac = sr**2 / (sr**2 + param**2)
    return mu * frac


def eval_strategy(prices, strategy_fn, num_test_days=250):
    n_inst, nt = prices.shape
    cash = 0.0
    cur_pos = np.zeros(n_inst, dtype=int)
    value = 0.0
    comm = 0.0
    pll = []
    total_volume = 0.0
    start = nt - num_test_days

    for t in range(start, nt + 1):
        hist = prices[:, :t]
        cur_prices = hist[:, -1]

        if t < nt:
            new_pos = np.asarray(strategy_fn(hist), dtype=float)
            pos_limits = (LIMITS / cur_prices).astype(int)
            new_pos = np.clip(new_pos, -pos_limits, pos_limits).astype(int)
        else:
            new_pos = cur_pos.copy()

        delta = new_pos - cur_pos
        cash -= cur_prices.dot(delta) + comm
        dvolumes = cur_prices * np.abs(delta)
        total_volume += float(np.sum(dvolumes))
        comm = float(np.sum(dvolumes * COMM))

        cur_pos = new_pos
        pos_value = float(cur_pos.dot(cur_prices))
        today_pl = cash + pos_value - value
        value = cash + pos_value

        if t > start:
            pll.append(today_pl)

    pll = np.asarray(pll)
    mu = float(np.mean(pll))
    sigma = float(np.std(pll))
    return {
        "mean_pl": mu,
        "std_pl": sigma,
        "score": float(score(mu, sigma)),
        "final_value": float(value),
        "total_volume": total_volume,
    }


def make_kalman_strategy(
    q=1e-4,
    r=4e-3,
    signal_scale=1.0,
    signal_power=1.0,
    top_k=None,
    include_algo=True,
):
    def strategy(prc_so_far):
        n_inst, nt = prc_so_far.shape
        if n_inst != N_INST or nt < 35:
            return np.zeros(n_inst, dtype=int)

        log_prices = np.log(np.maximum(prc_so_far, EPS))
        cur = np.maximum(prc_so_far[:, -1], 1.0)
        target_dollars = np.zeros(n_inst, dtype=float)

        start_idx = 0 if include_algo else 1
        for i in range(start_idx, n_inst):
            obs = log_prices[i]
            x = obs[0]
            p = 1.0
            innovs = np.zeros(nt - 1, dtype=float)

            for t in range(1, nt):
                p = p + q
                innov = obs[t] - x
                k = p / (p + r)
                x = x + k * innov
                p = (1.0 - k) * p
                innovs[t - 1] = innov

            recent = innovs[-30:]
            sigma = recent.std()
            if sigma < 1e-8:
                continue

            z = -(innovs[-1] / sigma)
            mag = np.tanh(signal_scale * np.sign(z) * (abs(z) ** signal_power))
            target_dollars[i] = LIMITS[i] * mag

        if top_k is not None:
            idx = np.argsort(np.abs(target_dollars[1:]))[-top_k:] + 1
            mask = np.zeros(n_inst, dtype=bool)
            if include_algo and abs(target_dollars[0]) > 0:
                mask[0] = True
            mask[idx] = True
            target_dollars = np.where(mask, target_dollars, 0.0)

        return (target_dollars / cur).astype(int)

    return strategy


def make_hybrid_kalman_strategy(
    q=1e-4,
    r=4e-3,
    overlay_scale=2000.0,
    top_k=3,
    base_mult=2.0,
):
    def strategy(prc_so_far):
        n_inst, nt = prc_so_far.shape
        if n_inst != N_INST or nt < 35:
            return np.zeros(n_inst, dtype=int)

        cur = np.maximum(prc_so_far[:, -1], 1.0)

        def zscore_to_past(lb):
            hist = prc_so_far[:, -lb - 1:-1]
            mu = hist.mean(axis=1)
            sig = hist.std(axis=1)
            sig = np.where(sig > 1e-8, sig, 1.0)
            return (mu - cur) / sig

        base_signal = 0.75 * zscore_to_past(8) + 0.25 * zscore_to_past(30)
        target_dollars = LIMITS * np.tanh(0.85 * base_signal) * base_mult

        log_prices = np.log(np.maximum(prc_so_far, EPS))
        kalman_signal = np.zeros(n_inst - 1, dtype=float)
        for i in range(1, n_inst):
            obs = log_prices[i]
            x = obs[0]
            p = 1.0
            innovs = np.zeros(nt - 1, dtype=float)

            for t in range(1, nt):
                p = p + q
                innov = obs[t] - x
                k = p / (p + r)
                x = x + k * innov
                p = (1.0 - k) * p
                innovs[t - 1] = innov

            recent = innovs[-30:]
            sigma = recent.std()
            if sigma < 1e-8:
                continue
            kalman_signal[i - 1] = -(innovs[-1] / sigma)

        strongest = np.argsort(np.abs(kalman_signal))[-top_k:]
        overlay = np.zeros(n_inst - 1, dtype=float)
        overlay[strongest] = overlay_scale * np.sign(kalman_signal[strongest]) * np.tanh(np.abs(kalman_signal[strongest]))
        target_dollars[1:] = np.clip(target_dollars[1:] + overlay, -LIMITS[1:], LIMITS[1:])

        return (target_dollars / cur).astype(int)

    return strategy


def make_blended_kalman_strategy(
    q=1e-4,
    r=4e-3,
    kalman_weight=0.25,
    base_mult=2.0,
    top_k=None,
):
    def strategy(prc_so_far):
        n_inst, nt = prc_so_far.shape
        if n_inst != N_INST or nt < 35:
            return np.zeros(n_inst, dtype=int)

        cur = np.maximum(prc_so_far[:, -1], 1.0)

        def zscore_to_past(lb):
            hist = prc_so_far[:, -lb - 1:-1]
            mu = hist.mean(axis=1)
            sig = hist.std(axis=1)
            sig = np.where(sig > 1e-8, sig, 1.0)
            return (mu - cur) / sig

        raw_signal = 0.75 * zscore_to_past(8) + 0.25 * zscore_to_past(30)
        log_prices = np.log(np.maximum(prc_so_far, EPS))
        kalman_signal = np.zeros(n_inst, dtype=float)

        for i in range(n_inst):
            obs = log_prices[i]
            x = obs[0]
            p = 1.0
            innovs = np.zeros(nt - 1, dtype=float)
            for t in range(1, nt):
                p = p + q
                innov = obs[t] - x
                k = p / (p + r)
                x = x + k * innov
                p = (1.0 - k) * p
                innovs[t - 1] = innov

            recent = innovs[-30:]
            sigma = recent.std()
            if sigma >= 1e-8:
                kalman_signal[i] = -(innovs[-1] / sigma)

        signal = (1.0 - kalman_weight) * raw_signal + kalman_weight * kalman_signal
        target_dollars = LIMITS * np.tanh(0.85 * signal) * base_mult

        if top_k is not None:
            idx = np.argsort(np.abs(signal[1:]))[-top_k:] + 1
            mask = np.zeros(n_inst, dtype=bool)
            mask[0] = True
            mask[idx] = True
            target_dollars = np.where(mask, target_dollars, 0.0)

        return (target_dollars / cur).astype(int)

    return strategy


def main():
    prices = pd.read_csv("prices.txt", sep=r"\s+").values.T.astype(float)
    configs = [
        ("kf_full_a", dict(q=1e-4, r=4e-3, signal_scale=0.8, signal_power=1.0, top_k=None, include_algo=True)),
        ("kf_full_b", dict(q=5e-5, r=3e-3, signal_scale=0.9, signal_power=1.0, top_k=None, include_algo=True)),
        ("kf_full_c", dict(q=2e-4, r=5e-3, signal_scale=0.7, signal_power=1.0, top_k=None, include_algo=True)),
        ("kf_top10", dict(q=1e-4, r=4e-3, signal_scale=0.9, signal_power=1.0, top_k=10, include_algo=True)),
        ("kf_top5", dict(q=1e-4, r=4e-3, signal_scale=1.0, signal_power=1.0, top_k=5, include_algo=True)),
        ("kf_no_algo", dict(q=1e-4, r=4e-3, signal_scale=0.8, signal_power=1.0, top_k=None, include_algo=False)),
        ("kf_convex", dict(q=1e-4, r=4e-3, signal_scale=0.7, signal_power=1.2, top_k=None, include_algo=True)),
    ]

    for name, kwargs in configs:
        result = eval_strategy(prices, make_kalman_strategy(**kwargs))
        print(
            f"{name:12s} score={result['score']:8.2f} "
            f"mean={result['mean_pl']:8.2f} std={result['std_pl']:8.2f} "
            f"final={result['final_value']:10.2f} vol={result['total_volume']:,.0f}"
        )

    hybrid_configs = [
        ("hyb_1k", dict(q=1e-4, r=4e-3, overlay_scale=1000.0, top_k=3, base_mult=2.0)),
        ("hyb_2k", dict(q=1e-4, r=4e-3, overlay_scale=2000.0, top_k=3, base_mult=2.0)),
        ("hyb_3k", dict(q=1e-4, r=4e-3, overlay_scale=3000.0, top_k=3, base_mult=2.0)),
        ("hyb_5k", dict(q=1e-4, r=4e-3, overlay_scale=5000.0, top_k=3, base_mult=2.0)),
        ("hyb_3k_q2", dict(q=2e-4, r=4e-3, overlay_scale=3000.0, top_k=3, base_mult=2.0)),
        ("hyb_3k_r2", dict(q=1e-4, r=2e-3, overlay_scale=3000.0, top_k=3, base_mult=2.0)),
        ("hyb_top5", dict(q=1e-4, r=4e-3, overlay_scale=3000.0, top_k=5, base_mult=2.0)),
    ]

    for name, kwargs in hybrid_configs:
        result = eval_strategy(prices, make_hybrid_kalman_strategy(**kwargs))
        print(
            f"{name:12s} score={result['score']:8.2f} "
            f"mean={result['mean_pl']:8.2f} std={result['std_pl']:8.2f} "
            f"final={result['final_value']:10.2f} vol={result['total_volume']:,.0f}"
        )

    blend_configs = [
        ("blend_10", dict(q=1e-4, r=4e-3, kalman_weight=0.10, base_mult=2.0, top_k=None)),
        ("blend_20", dict(q=1e-4, r=4e-3, kalman_weight=0.20, base_mult=2.0, top_k=None)),
        ("blend_30", dict(q=1e-4, r=4e-3, kalman_weight=0.30, base_mult=2.0, top_k=None)),
        ("blend_10_top15", dict(q=1e-4, r=4e-3, kalman_weight=0.10, base_mult=2.0, top_k=15)),
        ("blend_20_top15", dict(q=1e-4, r=4e-3, kalman_weight=0.20, base_mult=2.0, top_k=15)),
    ]

    for name, kwargs in blend_configs:
        result = eval_strategy(prices, make_blended_kalman_strategy(**kwargs))
        print(
            f"{name:12s} score={result['score']:8.2f} "
            f"mean={result['mean_pl']:8.2f} std={result['std_pl']:8.2f} "
            f"final={result['final_value']:10.2f} vol={result['total_volume']:,.0f}"
        )


if __name__ == "__main__":
    main()
