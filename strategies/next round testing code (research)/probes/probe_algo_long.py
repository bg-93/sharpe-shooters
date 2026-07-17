import numpy as np

# PROBE: algo_long
# long ALGO $95k: index drift, squashed side
# Constant-dollar book; commission ~1bp of gross once at entry, then
# only drift-rebalancing. Score inversion: see make_probes.py ledger.
W = np.array([95000.,     0.,     0.,     0.,     0.,     0.,     0.,     0.,
     0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.,
     0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.,
     0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.,
     0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.,
     0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.,
     0.,     0.,     0.])

def getMyPosition(prcSoFar):
    return (W / prcSoFar[:, -1]).astype(int)
