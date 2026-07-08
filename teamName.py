import numpy as np

nInst = 51


def getMyPosition(prcSoFar):
    nInstLocal, nt = prcSoFar.shape
    if nt < 6 or nInstLocal != nInst:
        return np.zeros(nInstLocal, dtype=int)

    prices = prcSoFar
    cur_price = prices[:, -1]

    # short-term mean-reversion signal: price vs 5-day moving average
    lookback = 5
    ma5 = np.mean(prices[:, -lookback - 1:-1], axis=1)
    ma_vol = np.std(prices[:, -lookback - 1:-1], axis=1)
    ma_vol = np.where(ma_vol > 0, ma_vol, 1.0)
    mr_raw = (ma5 - cur_price) / ma_vol

    # optional momentum fallback for instruments that show persistent trend
    mom_raw = np.zeros(nInst)
    if nt > 10:
        mom_raw = np.log(cur_price / prices[:, -11])
        mom_vol = np.std(np.log(prices[:, -11:-1] / prices[:, -12:-2]), axis=1)
        mom_vol = np.where(mom_vol > 0, mom_vol, 1.0)
        mom_raw = mom_raw / mom_vol

    # estimate current edge direction from recent return structure
    edge = np.zeros(nInst)
    if nt > 22:
        r5 = np.log(prices[:, 5:-1] / prices[:, :-6])
        r1 = np.log(prices[:, 6:] / prices[:, 5:-1])
        for i in range(nInst):
            if r5.shape[1] > 1 and np.std(r5[i]) > 0 and np.std(r1[i]) > 0:
                edge[i] = -np.corrcoef(r5[i], r1[i])[0, 1]

    use_momentum = edge > 0.08
    signal = np.where(use_momentum, mom_raw, mr_raw)

    # damp exposure in high-volatility regimes
    damp = np.ones(nInst)
    if nt > 40:
        short_vol = np.std(np.log(prices[:, -20:] / prices[:, -21:-1]), axis=1)
        long_vol = np.std(np.log(prices[:, -40:-20] / prices[:, -41:-21]), axis=1)
        long_vol = np.where(long_vol > 0, long_vol, 1.0)
        damp = np.clip(1.0 - (short_vol / long_vol - 1.0) * 0.7, 0.4, 1.0)

    raw = np.tanh(signal * 0.85) * damp
    weight = np.where(use_momentum, 0.28, 0.42)

    limits = np.concatenate(([100000], np.full(nInst - 1, 10000)))
    target = raw * weight * limits / cur_price
    return target.astype(int)
