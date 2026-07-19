#!/usr/bin/env python3
"""EDA for basket discovery on prices.txt.

Produces (saved into visualisations/):
  1. eda_normalized_prices.png   - all instruments, normalized to 100
  2. eda_corr_heatmap.png        - return correlation heatmap, cluster-ordered
  3. eda_clusters.png            - normalized prices per correlation cluster
  4. eda_pca.png                 - PCA explained variance at horizons {1, 5, 20}
  5. eda_stats.csv + stdout      - per-asset stats (vol, trend, half-life, ALGO beta)

Run from repo root: python visualisations/eda_baskets.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "visualisations"

prices = pd.read_csv(REPO_ROOT / "prices.txt", sep=r"\s+", header=0).values.T.astype(float)
n_inst, n_days = prices.shape
rets = np.diff(np.log(prices), axis=1)
norm = 100.0 * prices / prices[:, [0]]

# ---------------------------------------------------------------- 1. spaghetti
fig, ax = plt.subplots(figsize=(14, 7))
for i in range(n_inst):
    if i == 0:
        ax.plot(norm[i], color="black", lw=2.0, label="ALGO (inst 0)", zorder=5)
    else:
        ax.plot(norm[i], lw=0.7, alpha=0.6)
ax.set_title(f"Normalized prices (start=100), {n_inst} instruments, {n_days} days")
ax.set_xlabel("day")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "eda_normalized_prices.png", dpi=120)
plt.close(fig)

# ------------------------------------------------- 2. correlation + clustering
corr = np.corrcoef(rets)
dist = 1.0 - corr
np.fill_diagonal(dist, 0.0)
link = linkage(dist[np.triu_indices(n_inst, k=1)], method="average")
order = leaves_list(link)
cluster_ids = fcluster(link, t=0.7, criterion="distance")

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(n_inst))
ax.set_yticks(range(n_inst))
ax.set_xticklabels(order, fontsize=6, rotation=90)
ax.set_yticklabels(order, fontsize=6)
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Daily log-return correlation (hierarchically ordered)")
fig.tight_layout()
fig.savefig(OUT / "eda_corr_heatmap.png", dpi=120)
plt.close(fig)

# --------------------------------------------------- 3. cluster member plots
clusters = {}
for i, c in enumerate(cluster_ids):
    clusters.setdefault(c, []).append(i)
multi = {c: m for c, m in clusters.items() if len(m) >= 2}
singles = sorted(i for c, m in clusters.items() if len(m) == 1 for i in m)

n_plots = len(multi)
ncols = 3
nrows = int(np.ceil(n_plots / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.2 * nrows), squeeze=False)
for ax, (c, members) in zip(axes.flat, sorted(multi.items(), key=lambda kv: -len(kv[1]))):
    for i in members:
        ax.plot(norm[i], lw=0.9, label=str(i))
    ax.set_title(f"cluster {c} (n={len(members)})", fontsize=9)
    ax.legend(fontsize=6, ncol=4)
for ax in axes.flat[n_plots:]:
    ax.axis("off")
fig.suptitle("Correlation clusters (candidate baskets), normalized prices", y=1.0)
fig.tight_layout()
fig.savefig(OUT / "eda_clusters.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------------- 4. PCA
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, h in zip(axes, [1, 5, 20]):
    # h-day overlapping log returns of the 50 stocks (exclude ALGO)
    lp = np.log(prices[1:])
    hr = lp[:, h:] - lp[:, :-h]
    hr = (hr - hr.mean(axis=1, keepdims=True)) / hr.std(axis=1, keepdims=True)
    eigvals = np.linalg.eigvalsh(np.cov(hr))[::-1]
    ev = eigvals / eigvals.sum()
    ax.bar(range(1, 11), ev[:10])
    ax.set_title(f"h={h}d returns: PC1={ev[0]:.1%}, PC2={ev[1]:.1%}")
    ax.set_xlabel("component")
axes[0].set_ylabel("explained variance")
fig.suptitle("PCA of stock returns at multiple horizons (ALGO excluded)")
fig.tight_layout()
fig.savefig(OUT / "eda_pca.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------------ 5. stats
algo_ret = rets[0]
rows = []
for i in range(n_inst):
    p = prices[i]
    roll = pd.Series(p).rolling(30).mean().values
    res = (p - roll)[30:]
    r0, r1 = res[:-1], res[1:]
    rho = np.corrcoef(r0, r1)[0, 1]
    hl = np.log(0.5) / np.log(rho) if 0 < rho < 1 else np.inf
    beta = np.cov(rets[i], algo_ret)[0, 1] / algo_ret.var()
    rows.append(
        {
            "inst": i,
            "price_start": p[0],
            "price_end": p[-1],
            "total_ret_%": 100 * (p[-1] / p[0] - 1),
            "ann_vol_%": 100 * rets[i].std() * np.sqrt(250),
            "mr_half_life": hl,
            "beta_to_ALGO": beta,
            "corr_to_ALGO": np.corrcoef(rets[i], algo_ret)[0, 1],
            "cluster": cluster_ids[i],
        }
    )
stats = pd.DataFrame(rows)
stats.to_csv(OUT / "eda_stats.csv", index=False)

pd.set_option("display.width", 160)
print("==== Per-asset stats (sorted by cluster) ====")
print(stats.sort_values(["cluster", "inst"]).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print()
print("==== Multi-member correlation clusters (candidate baskets) ====")
for c, members in sorted(multi.items(), key=lambda kv: -len(kv[1])):
    sub = corr[np.ix_(members, members)]
    off = sub[np.triu_indices(len(members), k=1)]
    print(f"cluster {c}: members {members} | mean pairwise corr {off.mean():.3f}")
print(f"singletons (no close partner): {singles}")
print()
print("==== Top 15 most correlated pairs ====")
iu = np.triu_indices(n_inst, k=1)
pair_corrs = corr[iu]
top = np.argsort(pair_corrs)[::-1][:15]
for k in top:
    i, j = iu[0][k], iu[1][k]
    print(f"({i:2d},{j:2d})  corr={pair_corrs[k]:.3f}")
print()
print("Saved plots to visualisations/: eda_normalized_prices.png, eda_corr_heatmap.png, eda_clusters.png, eda_pca.png, eda_stats.csv")
