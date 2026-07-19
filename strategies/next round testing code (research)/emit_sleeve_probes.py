#!/usr/bin/env python3
"""Emit self-contained sleeve-isolation probe files.

Each probe is a full strategy file (drop-in teamName.py) with all but
one sleeve disabled, so its leaderboard score decomposes the hidden
window sleeve by sleeve.

From candidate_next.py:   demeaned-LL only, pairs only
From Score-1k book:       MR core (+regime FSMs) only, ALGO fade only

Disabling levers (text substitution on constants):
  pairs off : PAIR_LEG = 0.0
  LL off    : LL_MIN_HIST = 10**9   (sleeve never activates)
  MR off    : CORE_ASSETS = np.array([], dtype=int)
  fade off  : ALGO_FADE_CAP = 0.0
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
PROBE_DIR = HERE / "probes"

CAND = (HERE / "candidate_next.py").read_text()
BOOK = (REPO_ROOT / "strategies" / "Score - 1k" / "sharpeShooters7.py").read_text()

MR_OFF = ("CORE_ASSETS = np.array([\n"
          "    0, 35, 40, 5, 37, 29, 22, 14, 10, 44,\n"
          "    36, 21, 41, 13, 17, 16, 19, 39, 27, 32\n"
          "], dtype=int)",
          "CORE_ASSETS = np.array([], dtype=int)")
PAIRS_OFF = ("PAIR_LEG = 9_000.0", "PAIR_LEG = 0.0")
LL_OFF = ("LL_MIN_HIST = 120", "LL_MIN_HIST = 10**9")
FADE_OFF = ("ALGO_FADE_CAP = 60_000.0", "ALGO_FADE_CAP = 0.0")


def emit(fname, src, subs, header):
    for old, new in subs:
        assert old in src, f"{fname}: lever not found: {old[:40]}"
        src = src.replace(old, new)
    (PROBE_DIR / fname).write_text(f"# PROBE: {header}\n" + src)
    print(f"{fname:32s} {header}")


def main():
    PROBE_DIR.mkdir(exist_ok=True)
    emit("probe_sleeve_ll_demeaned.py", CAND, [PAIRS_OFF],
         "demeaned tanh lead-lag only (candidate minus pairs)")
    emit("probe_sleeve_pairs.py", CAND, [LL_OFF],
         "pairs only (candidate minus lead-lag)")
    emit("probe_sleeve_mr.py", BOOK, [PAIRS_OFF, LL_OFF, FADE_OFF],
         "MR core + regime FSMs only (Score-1k minus pairs/LL/fade)")
    emit("probe_sleeve_fade.py", BOOK, [MR_OFF, PAIRS_OFF, LL_OFF],
         "ALGO fade only (Score-1k minus MR/pairs/LL)")


if __name__ == "__main__":
    main()
