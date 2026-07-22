#!/usr/bin/env python3
"""Algothon 2026 leaderboard dashboard.

Scrapes the public API behind https://www.algothon.au/leaderboard
(https://algothon-backend-26.vercel.app/api/leaderboard), appends every
new snapshot to leaderboard_history.csv, and renders a dashboard PNG
showing how team ranks/scores move across snapshots.

Usage:
    python dashboard/leaderboard_dashboard.py            # scrape + plot
    python dashboard/leaderboard_dashboard.py --no-scrape  # re-plot only
    python dashboard/leaderboard_dashboard.py --watch 30   # poll every 30 min

Panels: score distribution | top-10 rank movements by day | daily score
gains/losses per team (descending).
"""

import argparse
import csv
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

BASE_URL = "https://algothon-backend-26.vercel.app"
API_URL = f"{BASE_URL}/api/leaderboard"
MINE_URL = f"{BASE_URL}/api/submissions/mine"
HERE = Path(__file__).resolve().parent
HISTORY_CSV = HERE / "leaderboard_history.csv"
MINE_CSV = HERE / "submissions_history.csv"
CREDS_JSON = HERE / "team_creds.json"  # gitignored: {"teamId":..., "password":...}
OUT_PNG = HERE / "leaderboard_dashboard.png"
OUR_TEAM = "Sharpe Shooters"

FIELDS = [
    "fetched_at", "snapshot_generated_at", "window_name",
    "start_day", "end_day",
    "rank", "team_id", "team_name", "score", "mean_pl", "std_pl",
    "trade_count", "runtime_ms", "submission_code", "submitted_at",
]


# ---------------------------------------------------------------- scrape

def fetch_api():
    req = urllib.request.Request(API_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def scrape():
    """Fetch API and append rows to history CSV (deduped per snapshot)."""
    data = fetch_api()
    snap = data.get("snapshot", {}) or {}
    win = data.get("evaluationWindow", {}) or {}
    snap_ts = snap.get("generatedAt", "")
    win_name = snap.get("name", win.get("name", ""))
    start_day = win.get("startDayIndex", "")
    end_day = win.get("endDayIndex", "")
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    seen = set()
    if HISTORY_CSV.exists():
        old = pd.read_csv(HISTORY_CSV, usecols=["snapshot_generated_at", "team_id"])
        seen = set(zip(old["snapshot_generated_at"], old["team_id"]))

    rows = []
    for r in data["leaderboard"]:
        if (snap_ts, r["teamId"]) in seen:
            continue
        rows.append({
            "fetched_at": fetched_at,
            "snapshot_generated_at": snap_ts,
            "window_name": win_name,
            "start_day": start_day,
            "end_day": end_day,
            "rank": r["rank"],
            "team_id": r["teamId"],
            "team_name": r["teamName"],
            "score": r["score"],
            "mean_pl": r["meanPl"],
            "std_pl": r["stdPl"],
            "trade_count": r.get("tradeCount"),
            "runtime_ms": r.get("runtimeMs"),
            "submission_code": r.get("submissionCode"),
            "submitted_at": r.get("submittedAt"),
        })

    if rows:
        new_file = not HISTORY_CSV.exists()
        with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                w.writeheader()
            w.writerows(rows)
        print(f"[scrape] snapshot '{win_name}' ({snap_ts}): "
              f"appended {len(rows)} rows -> {HISTORY_CSV.name}")
    else:
        print(f"[scrape] snapshot '{win_name}' ({snap_ts}) already recorded; "
              f"nothing new")
    return len(rows) > 0


def scrape_mine():
    """Fetch our team's private submission list; log every (evaluatedAt,
    submission) state change to submissions_history.csv and print a table."""
    if not CREDS_JSON.exists():
        print(f"[mine] {CREDS_JSON.name} missing — skipping private scrape")
        return
    creds = json.loads(CREDS_JSON.read_text())
    req = urllib.request.Request(MINE_URL, headers={
        "User-Agent": "Mozilla/5.0",
        "Authorization": f"Bearer {creds['teamId']}:{creds['password']}",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        subs = json.load(resp)["submissions"]

    fields = ["fetched_at", "public_code", "original_filename", "status",
              "score", "mean_pl", "std_pl", "trade_count", "runtime_ms",
              "submitted_at", "evaluated_at", "is_active", "error_message"]
    seen = set()
    if MINE_CSV.exists():
        old = pd.read_csv(MINE_CSV, usecols=["public_code", "evaluated_at"])
        seen = set(zip(old["public_code"], old["evaluated_at"]))

    new_rows = []
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for s in subs:
        if (s["publicCode"], s["evaluatedAt"]) in seen:
            continue
        new_rows.append({
            "fetched_at": fetched_at,
            "public_code": s["publicCode"],
            "original_filename": s["originalFilename"],
            "status": s["status"],
            "score": s["score"],
            "mean_pl": s["meanPl"],
            "std_pl": s["stdPl"],
            "trade_count": s["tradeCount"],
            "runtime_ms": s["runtimeMs"],
            "submitted_at": s["submittedAt"],
            "evaluated_at": s["evaluatedAt"],
            "is_active": s["isActive"],
            "error_message": s["errorMessage"],
        })
    if new_rows:
        new_file = not MINE_CSV.exists()
        with open(MINE_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new_file:
                w.writeheader()
            w.writerows(new_rows)
        print(f"[mine] {len(new_rows)} new/changed evaluation(s) -> "
              f"{MINE_CSV.name}")
    else:
        print("[mine] no new evaluations")

    print(f"\n  our submissions ({len(subs)}):")
    for s in sorted(subs, key=lambda x: x["submittedAt"]):
        flag = " <<< ACTIVE" if s["isActive"] else ""
        print(f"    {s['publicCode']} {s['originalFilename'][:26]:26s} "
              f"sub {s['submittedAt'][5:16]} eval {s['evaluatedAt'][5:16]} "
              f"score {s['score']:8.2f}  mu {s['meanPl']:8.1f}  "
              f"sd {s['stdPl']:8.1f}{flag}")


# ------------------------------------------------------------------ plot

def load_history():
    df = pd.read_csv(HISTORY_CSV)
    df["snapshot_generated_at"] = pd.to_datetime(df["snapshot_generated_at"])
    df = df.sort_values(["snapshot_generated_at", "rank"])
    return df


def daily_frames(df):
    """One frame per calendar day (last snapshot of each day)."""
    df = df.copy()
    df["day"] = df["snapshot_generated_at"].dt.date
    days = sorted(df["day"].unique())
    frames = {}
    for d in days:
        sub = df[df["day"] == d]
        last_ts = sub["snapshot_generated_at"].max()
        frames[d] = sub[sub["snapshot_generated_at"] == last_ts]
    return days, frames


def plot():
    df = load_history()
    days, frames = daily_frames(df)
    latest = frames[days[-1]]
    win_name = latest["window_name"].iloc[0]
    ours = latest[latest["team_name"] == OUR_TEAM]

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle(
        f"Algothon 2026 — {win_name} | {len(days)} day(s) tracked | "
        f"latest {days[-1].strftime('%d/%m')}",
        fontsize=14, fontweight="bold")

    # 1. score distribution (latest day)
    ax = axes[0]
    ax.hist(latest["score"].clip(lower=-400), bins=60, color="steelblue",
            alpha=0.85)
    if not ours.empty:
        our_score = ours["score"].iloc[0]
        ax.axvline(our_score, color="red", lw=2,
                   label=f"{OUR_TEAM}: {our_score:.0f} "
                         f"(#{int(ours['rank'].iloc[0])}/{len(latest)})")
        ax.legend(fontsize=9)
    ax.set_title("Score distribution (latest day)")
    ax.set_xlabel("score")
    ax.set_ylabel("teams")
    ax.grid(alpha=0.3)

    # 2. top-10 rank movements by day (current top 10 + us)
    ax = axes[1]
    teams = list(latest.nsmallest(10, "rank")["team_name"])
    if OUR_TEAM not in teams and not ours.empty:
        teams.append(OUR_TEAM)
    colors = cm.tab10(np.linspace(0, 1, 10))
    day_x = list(range(len(days)))
    for i, team in enumerate(teams):
        ranks = []
        for d in days:
            row = frames[d][frames[d]["team_name"] == team]
            ranks.append(int(row["rank"].iloc[0]) if not row.empty else np.nan)
        if team == OUR_TEAM:
            kw = dict(color="red", lw=2.5, zorder=10, marker="o", ms=7)
        else:
            kw = dict(color=colors[i % 10], lw=1.5, alpha=0.85,
                      marker="o", ms=5)
        ax.plot(day_x, ranks, **kw)
        # name label at the right edge
        if not all(np.isnan(r) for r in ranks):
            last_r = [r for r in ranks if not np.isnan(r)][-1]
            ax.annotate(f" {team[:20]} (#{int(last_r)})",
                        (day_x[-1], last_r), fontsize=7,
                        va="center", color=kw["color"])
    ax.invert_yaxis()
    ax.set_xticks(day_x)
    ax.set_xticklabels([d.strftime("%d/%m") for d in days])
    ax.set_xlim(-0.3, len(days) - 1 + 1.6)  # room for labels
    ax.set_title("Top-10 rank movements by day (lower = better)")
    ax.set_ylabel("rank")
    ax.grid(alpha=0.3)

    # 3. score gains/losses since previous day, descending
    ax = axes[2]
    if len(days) >= 2:
        prev = frames[days[-2]]
        m = latest.merge(prev[["team_name", "score"]], on="team_name",
                         suffixes=("", "_prev"))
        m["d_score"] = m["score"] - m["score_prev"]
        m = m.sort_values("d_score", ascending=False)
        n_show = 15
        show = m if len(m) <= 2 * n_show else pd.concat(
            [m.head(n_show), m.tail(n_show)])
        labels = [f"{t[:22]}" for t in show["team_name"]]
        vals = show["d_score"].values
        bar_colors = ["red" if t == OUR_TEAM else
                      ("seagreen" if v >= 0 else "indianred")
                      for t, v in zip(show["team_name"], vals)]
        y = np.arange(len(show))[::-1]  # biggest gain on top
        ax.barh(y, vals, color=bar_colors, alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        for yi, v in zip(y, vals):
            ax.annotate(f"{v:+.0f}", (v, yi), fontsize=7, va="center",
                        ha="left" if v >= 0 else "right")
        if len(m) > 2 * n_show:
            title = (f"Score change {days[-2].strftime('%d/%m')} -> "
                     f"{days[-1].strftime('%d/%m')} "
                     f"(top {n_show} gains / {n_show} losses)")
        else:
            title = (f"Score change {days[-2].strftime('%d/%m')} -> "
                     f"{days[-1].strftime('%d/%m')} (all teams)")
        ax.set_title(title)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("delta score")
        ax.grid(alpha=0.3, axis="x")
        if not ours.empty and (m["team_name"] == OUR_TEAM).any():
            dv = m.loc[m["team_name"] == OUR_TEAM, "d_score"].iloc[0]
            ax.annotate(f"{OUR_TEAM}: {dv:+.1f}", xy=(0.98, 0.02),
                        xycoords="axes fraction", ha="right", fontsize=9,
                        color="red")
    else:
        ax.text(0.5, 0.5, "Need >= 2 days of history\nfor gains/losses",
                ha="center", va="center", fontsize=12, color="gray")
        ax.set_title("Score change vs previous day")
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_PNG, dpi=110)
    plt.close(fig)
    print(f"[plot] wrote {OUT_PNG.name} ({len(days)} days tracked)")


# --------------------------------------------------------------- summary

def summary():
    df = load_history()
    snaps = sorted(df["snapshot_generated_at"].unique())
    latest = df[df["snapshot_generated_at"] == snaps[-1]]
    print(f"\n=== {latest['window_name'].iloc[0]} "
          f"(snapshot {pd.Timestamp(snaps[-1]).strftime('%d/%m %H:%M UTC')}, "
          f"{len(latest)} teams) ===")
    top = latest.nsmallest(10, "rank")
    for _, r in top.iterrows():
        print(f"  #{int(r['rank']):3d} {r['team_name'][:32]:32s} "
              f"score {r['score']:8.2f}  mu {r['mean_pl']:8.2f}  "
              f"sd {r['std_pl']:8.2f}")
    ours = latest[latest["team_name"] == OUR_TEAM]
    if not ours.empty:
        r = ours.iloc[0]
        print(f"  ---\n  #{int(r['rank']):3d} {r['team_name']:32s} "
              f"score {r['score']:8.2f}  mu {r['mean_pl']:8.2f}  "
              f"sd {r['std_pl']:8.2f}")

    # movers vs previous snapshot
    if len(snaps) >= 2:
        prev = df[df["snapshot_generated_at"] == snaps[-2]]
        merged = latest.merge(prev[["team_name", "rank", "score"]],
                              on="team_name", suffixes=("", "_prev"))
        merged["d_rank"] = merged["rank_prev"] - merged["rank"]
        movers = merged.reindex(
            merged["d_rank"].abs().sort_values(ascending=False).index)
        print("\n  biggest movers since previous snapshot:")
        for _, r in movers.head(8).iterrows():
            arrow = "^" if r["d_rank"] > 0 else "v"
            print(f"    {arrow}{abs(int(r['d_rank'])):3d} "
                  f"{r['team_name'][:32]:32s} "
                  f"#{int(r['rank_prev'])} -> #{int(r['rank'])}  "
                  f"score {r['score_prev']:.1f} -> {r['score']:.1f}")


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-scrape", action="store_true",
                    help="skip API fetch, re-plot from history")
    ap.add_argument("--watch", type=float, metavar="MIN",
                    help="poll every MIN minutes forever")
    args = ap.parse_args()

    def cycle():
        if not args.no_scrape:
            scrape()
            scrape_mine()
        if HISTORY_CSV.exists():
            plot()
            summary()
        else:
            print("no history yet — run without --no-scrape first")

    if args.watch:
        while True:
            try:
                cycle()
            except Exception as e:  # keep the watcher alive on flaky network
                print(f"[watch] error: {e}")
            time.sleep(args.watch * 60)
    else:
        cycle()


if __name__ == "__main__":
    main()
