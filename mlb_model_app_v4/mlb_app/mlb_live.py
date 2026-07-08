"""
MLB Live Win Probability — v4 module
====================================
Turns live game state (from the official MLB play-by-play feed) into an
updating win probability, anchored on the pregame model.

How it works — same "explainable statistics" philosophy as the blueprint:
  1. Pregame expected-runs rates (from your morning log) give each team's
     scoring rate per inning.
  2. The live feed gives: score, inning, half, outs, runners on base.
  3. Runners + outs convert to expected runs THIS inning via the classic
     RE24 run-expectancy matrix.
  4. Remaining innings convert to Poisson-distributed remaining runs.
  5. P(home win) = P(current lead + home remaining - away remaining > 0),
     with ties sent to extra innings at a small home edge.

Every snapshot is logged to mlb_live_log.jsonl so you can later measure
how good the live probabilities were at each stage of games.
"""

from datetime import date, datetime
import json
import os

import numpy as np
import statsapi
from scipy.stats import poisson

LIVE_LOG = "mlb_live_log.jsonl"
LEAGUE_RUNS_PER_INNING = 4.5 / 9
HOME_EXTRAS_EDGE = 0.52          # home win share of games tied after 9

# RE24: expected runs for rest of inning, by (runners, outs).
# Runner key: (first, second, third) as 0/1. Standard modern-era values.
RE24 = {
    (0,0,0): [0.48, 0.25, 0.10], (1,0,0): [0.85, 0.50, 0.22],
    (0,1,0): [1.06, 0.64, 0.31], (0,0,1): [1.30, 0.90, 0.36],
    (1,1,0): [1.44, 0.88, 0.43], (1,0,1): [1.75, 1.10, 0.48],
    (0,1,1): [1.96, 1.35, 0.57], (1,1,1): [2.25, 1.54, 0.75],
}

def _base_state(offense):
    """offense block from the live feed -> (first, second, third) 0/1."""
    return (int("first" in offense and offense["first"] is not None),
            int("second" in offense and offense["second"] is not None),
            int("third" in offense and offense["third"] is not None))

def get_live_state(game_pk):
    """Pull one in-progress game's state from the official live feed."""
    g = statsapi.get("game", {"gamePk": game_pk})
    ls = g["liveData"]["linescore"]
    teams = g["gameData"]["teams"]
    state = {
        "game_pk": game_pk,
        "home": teams["home"]["name"], "away": teams["away"]["name"],
        "home_runs": ls["teams"]["home"].get("runs", 0),
        "away_runs": ls["teams"]["away"].get("runs", 0),
        "inning": ls.get("currentInning", 1),
        "is_top": ls.get("isTopInning", True),
        "outs": min(ls.get("outs", 0), 2),
        "bases": _base_state(ls.get("offense", {})),
        "status": g["gameData"]["status"]["abstractGameState"],  # Live/Final/Preview
    }
    # last play description — the "commentary", straight from the feed
    plays = g["liveData"].get("plays", {}).get("allPlays", [])
    state["last_play"] = plays[-1]["result"].get("description", "") if plays else ""
    return state

def live_win_prob(state, lam_home=4.5, lam_away=4.5, max_runs=20):
    """P(home win) from game state + pregame per-game run rates."""
    inn, top, outs = state["inning"], state["is_top"], state["outs"]
    rh = lam_home / 9.0   # per-inning rates from the pregame model
    ra = lam_away / 9.0
    scale_a = ra / LEAGUE_RUNS_PER_INNING
    scale_h = rh / LEAGUE_RUNS_PER_INNING

    # expected remaining runs, current inning handled via RE24
    if inn <= 9:
        if top:   # away batting now
            exp_away = RE24[state["bases"]][outs] * scale_a + ra * (9 - inn)
            exp_home = rh * (9 - inn + 1)      # all of bottoms inn..9 remain
        else:     # home batting now
            exp_away = ra * (9 - inn)
            exp_home = RE24[state["bases"]][outs] * scale_h + rh * (9 - inn)
    else:         # extra innings: value the current half, then coin-ish
        diff = state["home_runs"] - state["away_runs"]
        if top:
            exp_away = RE24[state["bases"]][outs] * scale_a
            exp_home = rh  # home still bats this inning
        else:
            exp_away = 0.0
            exp_home = RE24[state["bases"]][outs] * scale_h
        # fall through to the same Poisson comparison below
        return _p_from_remaining(diff, exp_home, exp_away, max_runs)

    diff = state["home_runs"] - state["away_runs"]
    # walk-off shortcut: home leading going into/during the 9th bottom half
    if inn == 9 and not top and diff > 0:
        return 1.0
    return _p_from_remaining(diff, exp_home, exp_away, max_runs)

def _p_from_remaining(diff, exp_home, exp_away, max_runs):
    ph = poisson.pmf(np.arange(max_runs + 1), max(exp_home, 1e-6))
    pa = poisson.pmf(np.arange(max_runs + 1), max(exp_away, 1e-6))
    m = np.outer(ph, pa)          # m[i,j] = P(home scores i more, away j more)
    i, j = np.indices(m.shape)
    final = diff + i - j
    p_win = m[final > 0].sum()
    p_tie = m[final == 0].sum()
    return float(min(max(p_win + HOME_EXTRAS_EDGE * p_tie, 0.0), 1.0))

def pregame_rates_from_log(log_file="mlb_log.jsonl"):
    """Read today's morning predictions so live WP starts from your model."""
    rates = {}
    if os.path.exists(log_file):
        today = date.today().isoformat()
        with open(log_file) as f:
            for line in f:
                r = json.loads(line)
                if r.get("date") == today:
                    rates[(r["away"], r["home"])] = (
                        r["exp_runs_home"], r["exp_runs_away"], r["p_home"])
    return rates

def snapshot_live_games():
    """One refresh pass over every in-progress game. Returns display rows
    and appends a snapshot line per game to the live log."""
    rates = pregame_rates_from_log()
    rows = []
    for g in statsapi.schedule(date=date.today().strftime("%Y-%m-%d")):
        if g["status"] not in ("In Progress", "Live"):
            continue
        try:
            s = get_live_state(g["game_id"])
        except Exception:
            continue
        lam_h, lam_a, p_pre = rates.get((s["away"], s["home"]), (4.5, 4.5, None))
        p_live = live_win_prob(s, lam_h, lam_a)
        row = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "game_pk": s["game_pk"], "away": s["away"], "home": s["home"],
            "score": f'{s["away_runs"]}-{s["home_runs"]}',
            "inning": f'{"T" if s["is_top"] else "B"}{s["inning"]}',
            "outs": s["outs"], "bases": "".join(str(b) for b in s["bases"]),
            "p_home_live": round(p_live, 3),
            "p_home_pregame": p_pre,
            "movement": round(p_live - p_pre, 3) if p_pre is not None else None,
            "last_play": s["last_play"][:120],
            "final_winner": None, "graded_at": None,   # filled by grade pass
        }
        rows.append(row)
        with open(LIVE_LOG, "a") as f:
            f.write(json.dumps(row) + "\n")
    return rows

def grade_live_log():
    """Night pass: stamp each snapshot with the game's actual winner so you
    can measure live-WP accuracy by inning later."""
    if not os.path.exists(LIVE_LOG):
        return 0
    with open(LIVE_LOG) as f:
        snaps = [json.loads(l) for l in f if l.strip()]
    ungraded_pks = {s["game_pk"] for s in snaps if s["graded_at"] is None}
    winners = {}
    for pk in ungraded_pks:
        try:
            g = statsapi.get("game", {"gamePk": pk})
            if g["gameData"]["status"]["abstractGameState"] != "Final":
                continue
            ls = g["liveData"]["linescore"]["teams"]
            winners[pk] = (g["gameData"]["teams"]["home"]["name"]
                           if ls["home"]["runs"] > ls["away"]["runs"]
                           else g["gameData"]["teams"]["away"]["name"])
        except Exception:
            continue
    n = 0
    for s in snaps:
        if s["graded_at"] is None and s["game_pk"] in winners:
            s["final_winner"] = winners[s["game_pk"]]
            s["graded_at"] = datetime.now().isoformat(timespec="seconds")
            n += 1
    with open(LIVE_LOG, "w") as f:
        for s in snaps:
            f.write(json.dumps(s) + "\n")
    return n
