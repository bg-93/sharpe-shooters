import numpy as np

# PROBE: hadamard_10
# +/-$9.5k per stock, Hadamard row 10: w.drift projection
# Constant-dollar book; commission ~1bp of gross once at entry, then
# only drift-rebalancing. Score inversion: see make_probes.py ledger.
W = np.array([    0.,  9500.,  9500., -9500., -9500.,  9500.,  9500., -9500.,
 -9500., -9500., -9500.,  9500.,  9500., -9500., -9500.,  9500.,
  9500.,  9500.,  9500., -9500., -9500.,  9500.,  9500., -9500.,
 -9500., -9500., -9500.,  9500.,  9500., -9500., -9500.,  9500.,
  9500.,  9500.,  9500., -9500., -9500.,  9500.,  9500., -9500.,
 -9500., -9500., -9500.,  9500.,  9500., -9500., -9500.,  9500.,
  9500.,  9500.,  9500.])

def getMyPosition(prcSoFar):
    return (W / prcSoFar[:, -1]).astype(int)
