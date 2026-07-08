"""
MLB Prediction Engine — v3
==========================
Roadmap status:
  [done] days 1-3   Elo + Poisson baseline, daily loop, frozen logs
  [done] days 4-6   starter FIP adjustment
  [done] days 7-9   park factors on expected runs & totals   <- NEW
  [done] days 10-12 market comparison / fade edge — optional, needs a free
                    key from the-odds-api.com pasted into ODDS_API_KEY  <- NEW
  [ready] days 13-14 calibrate the knobs below using mlb_log.jsonl

Log format: mlb_log.jsonl — one JSON object per line, one line per game.
Morning run writes the prediction + every input feature; night run fills in
the actual result and error metrics on the same line. Paste the file (or a
chunk of it) into a Claude chat and every field needed to diagnose accuracy
is there: ratings, FIPs, park factor, probabilities, Brier, total error.
"""

from datetime import date, datetime
import json
import math
import os

import numpy as np
import pandas as pd
import statsapi
from scipy.stats import poisson

try:
    import requests
except ImportError:
    requests = None

# ---------------- knobs (recalibrate on days 13-14 using the log) ----------------
SEASON_START    = f"{date.today().year}-03-20"
ELO_K           = 4
HOME_ELO_EDGE   = 24
REGRESS_GAMES   = 20
LEAGUE_FIP      = 4.20
FIP_CONSTANT    = 3.15
STARTER_SHARE   = 0.60
FIP_REGRESS_IP  = 60
ELO_WEIGHT      = 0.40        # blend: 40% Elo view, 60% run-model view
LOG_FILE        = "mlb_log.jsonl"
ODDS_API_KEY    = ""          # optional: free key from the-odds-api.com

# Park run factors (approximate multi-year; 1.00 = neutral). Tunable.
PARK_FACTORS = {
    "Coors Field": 1.22, "Fenway Park": 1.08, "Great American Ball Park": 1.10,
    "Chase Field": 1.06, "Kauffman Stadium": 1.04, "Citizens Bank Park": 1.05,
    "Yankee Stadium": 1.04, "Wrigley Field": 1.02, "Rogers Centre": 1.02,
    "Rate Field": 1.03, "Guaranteed Rate Field": 1.03, "Truist Park": 1.00,
    "Globe Life Field": 1.00, "Target Field": 1.00, "Nationals Park": 1.00,
    "American Family Field": 1.00, "Daikin Park": 1.01, "Minute Maid Park": 1.01,
    "Oriole Park at Camden Yards": 1.00, "Angel Stadium": 0.98,
    "Dodger Stadium": 0.98, "Progressive Field": 0.98, "Busch Stadium": 0.97,
    "PNC Park": 0.97, "Comerica Park": 0.97, "Citi Field": 0.96,
    "loanDepot park": 0.95, "Petco Park": 0.94, "Oracle Park": 0.92,
    "T-Mobile Park": 0.92, "Sutter Health Park": 1.05,
    "George M. Steinbrenner Field": 1.06,
}

_fip_cache = {}

# ================= data & ratings (unchanged core from v2) =================
def get_season_results():
    today = date.today().strftime("%Y-%m-%d")
    sched = statsapi.schedule(start_date=SEASON_START, end_date=today)
    rows = [{"date": g["game_date"], "home": g["home_name"], "away": g["away_name"],
             "home_runs": g["home_score"], "away_runs": g["away_score"]}
            for g in sched if g["status"] == "Final" and g["game_type"] == "R"]
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

def elo_win_prob(elo_a, elo_b):
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400))

def fit_ratings(df):
    teams = pd.unique(df[["home", "away"]].values.ravel())
    elo = {t: 1500.0 for t in teams}
    rs = {t: [] for t in teams}; ra = {t: [] for t in teams}
    for _, g in df.iterrows():
        h, a = g["home"], g["away"]
        p = elo_win_prob(elo[h] + HOME_ELO_EDGE, elo[a])
        won = 1.0 if g["home_runs"] > g["away_runs"] else 0.0
        mov = math.log(abs(g["home_runs"] - g["away_runs"]) + 1)
        d = ELO_K * mov * (won - p)
        elo[h] += d; elo[a] -= d
        rs[h].append(g["home_runs"]); ra[h].append(g["away_runs"])
        rs[a].append(g["away_runs"]); ra[a].append(g["home_runs"])
    league_avg = df[["home_runs", "away_runs"]].values.mean()
    ratings = {}
    for t in teams:
        n = len(rs[t]); w = min(n / REGRESS_GAMES, 1.0)
        wins = ((df["home"] == t) & (df["home_runs"] > df["away_runs"])).sum() + \
               ((df["away"] == t) & (df["away_runs"] > df["home_runs"])).sum()
        ratings[t] = {"elo": elo[t],
                      "off": w * (np.mean(rs[t]) if n else league_avg) + (1 - w) * league_avg,
                      "def": w * (np.mean(ra[t]) if n else league_avg) + (1 - w) * league_avg,
                      "games": n, "wins": int(wins)}
    return ratings, league_avg

# ================= starter FIP (from v2) =================
def _parse_ip(ip_str):
    try:
        whole, _, frac = str(ip_str).partition(".")
        return int(whole) + {"": 0, "0": 0, "1": 1, "2": 2}.get(frac, 0) / 3.0
    except (ValueError, TypeError):
        return 0.0

def get_starter_fip(name):
    if not name or name in ("", "TBD"):
        return LEAGUE_FIP, 0.0
    if name in _fip_cache:
        return _fip_cache[name]
    fip, ip = LEAGUE_FIP, 0.0
    try:
        matches = statsapi.lookup_player(name)
        if matches:
            data = statsapi.player_stat_data(matches[0]["id"], group="pitching", type="season")
            for block in data.get("stats", []):
                s = block.get("stats", {})
                ip = _parse_ip(s.get("inningsPitched", 0))
                if ip > 0:
                    raw = (13 * int(s.get("homeRuns", 0))
                           + 3 * (int(s.get("baseOnBalls", 0)) + int(s.get("hitByPitch", 0)))
                           - 2 * int(s.get("strikeOuts", 0))) / ip + FIP_CONSTANT
                    w = min(ip / FIP_REGRESS_IP, 1.0)
                    fip = w * raw + (1 - w) * LEAGUE_FIP
                break
    except Exception:
        pass
    _fip_cache[name] = (round(fip, 2), round(ip, 1))
    return _fip_cache[name]

def starter_factor(opp_fip):
    return STARTER_SHARE * (opp_fip / LEAGUE_FIP) + (1 - STARTER_SHARE)

# ================= prediction =================
RATIO_DAMP = 0.75   # v4.1: shrink each ratio toward 1 so factors can't stack to absurd totals
LAMBDA_MIN, LAMBDA_MAX = 2.6, 6.6

def expected_runs(ratings, league_avg, team, opp, is_home, opp_fip, park):
    r, o = ratings[team], ratings[opp]
    lam = league_avg
    lam *= (r["off"] / league_avg) ** RATIO_DAMP
    lam *= (o["def"] / league_avg) ** RATIO_DAMP
    lam *= starter_factor(opp_fip) ** RATIO_DAMP
    lam *= park
    lam *= 1.04 if is_home else 0.96
    return float(min(max(lam, LAMBDA_MIN), LAMBDA_MAX))

def win_prob_from_runs(lh, la, max_runs=15):
    ph = poisson.pmf(np.arange(max_runs + 1), lh)
    pa = poisson.pmf(np.arange(max_runs + 1), la)
    m = np.outer(ph, pa)
    home, away, tie = np.tril(m, -1).sum(), np.triu(m, 1).sum(), m.trace()
    return home + tie * (home / (home + away))

def fetch_market_odds():
    """Optional days 10-12: home moneyline implied prob + total line per matchup."""
    if not ODDS_API_KEY or requests is None:
        return {}
    out = {}
    try:
        base = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
        for mkt in ("h2h", "totals"):
            r = requests.get(base, params={"apiKey": ODDS_API_KEY, "regions": "us",
                                           "markets": mkt, "oddsFormat": "decimal"}, timeout=15)
            for ev in r.json():
                key = (ev["away_team"], ev["home_team"])
                rec = out.setdefault(key, {})
                book = ev.get("bookmakers") or []
                if not book:
                    continue
                for m in book[0]["markets"]:
                    if m["key"] == "h2h":
                        for oc in m["outcomes"]:
                            if oc["name"] == ev["home_team"]:
                                rec["mkt_home_prob"] = round(1 / oc["price"], 3)
                    if m["key"] == "totals" and m["outcomes"]:
                        rec["mkt_total"] = m["outcomes"][0].get("point")
    except Exception:
        pass
    return out

def predict_slate(target_date=None):
    """Morning run. Returns list of rich prediction records (not yet logged)."""
    d = (target_date or date.today()).strftime("%Y-%m-%d")
    results = get_season_results()
    if results.empty:
        return []
    ratings, league_avg = fit_ratings(results)
    market = fetch_market_odds()
    records = []
    for g in statsapi.schedule(date=d):
        h, a = g["home_name"], g["away_name"]
        if h not in ratings or a not in ratings or g["game_type"] != "R":
            continue
        hs, as_ = g.get("home_probable_pitcher", ""), g.get("away_probable_pitcher", "")
        hs_fip, hs_ip = get_starter_fip(hs)
        as_fip, as_ip = get_starter_fip(as_)
        venue = g.get("venue_name", "")
        park = PARK_FACTORS.get(venue, 1.00)
        p_elo = elo_win_prob(ratings[h]["elo"] + HOME_ELO_EDGE, ratings[a]["elo"])
        lh = expected_runs(ratings, league_avg, h, a, True, as_fip, park)
        la = expected_runs(ratings, league_avg, a, h, False, hs_fip, park)
        p_run = win_prob_from_runs(lh, la)
        p_home = ELO_WEIGHT * p_elo + (1 - ELO_WEIGHT) * p_run
        rec = {
            # identity
            "game_id": g["game_id"], "date": d, "away": a, "home": h,
            "venue": venue, "park_factor": park,
            # inputs the prediction used (this is what makes the log diagnosable)
            "home_elo": round(ratings[h]["elo"], 1), "away_elo": round(ratings[a]["elo"], 1),
            "home_off": round(ratings[h]["off"], 2), "home_def": round(ratings[h]["def"], 2),
            "away_off": round(ratings[a]["off"], 2), "away_def": round(ratings[a]["def"], 2),
            "home_starter": hs or "TBD", "home_starter_fip": hs_fip, "home_starter_ip": hs_ip,
            "away_starter": as_ or "TBD", "away_starter_fip": as_fip, "away_starter_ip": as_ip,
            # outputs
            "exp_runs_home": round(lh, 2), "exp_runs_away": round(la, 2),
            "p_home_elo": round(p_elo, 3), "p_home_run_model": round(p_run, 3),
            "p_home": round(p_home, 3), "pick": h if p_home >= 0.5 else a,
            "proj_total": round(lh + la, 1),
            "predicted_at": datetime.now().isoformat(timespec="seconds"),
            # market (days 10-12, blank without an odds key)
            "mkt_home_prob": None, "mkt_total": None, "edge_vs_market": None,
            # filled by the night run
            "final_home_runs": None, "final_away_runs": None, "actual_total": None,
            "winner": None, "pick_correct": None, "brier": None,
            "total_error": None, "graded_at": None,
        }
        mk = market.get((a, h), {})
        if mk.get("mkt_home_prob") is not None:
            rec["mkt_home_prob"] = mk["mkt_home_prob"]
            rec["edge_vs_market"] = round(p_home - mk["mkt_home_prob"], 3)
        if mk.get("mkt_total") is not None:
            rec["mkt_total"] = mk["mkt_total"]
        records.append(rec)
    return records

# ================= logging & grading =================
def read_log():
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE) as f:
        return [json.loads(line) for line in f if line.strip()]

def write_log(records):
    with open(LOG_FILE, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def log_morning(new_records):
    """Append today's predictions; never overwrite already-logged games."""
    log = read_log()
    existing = {r["game_id"] for r in log}
    added = [r for r in new_records if r["game_id"] not in existing]
    write_log(log + added)
    return len(added), len(new_records) - len(added)

def grade_night(target_date=None):
    """Night run: fill in finals + error metrics for any ungraded games."""
    d = (target_date or date.today()).strftime("%Y-%m-%d")
    log = read_log()
    finals = {g["game_id"]: g for g in statsapi.schedule(date=d) if g["status"] == "Final"}
    graded = still_open = 0
    for r in log:
        if r["graded_at"] is not None or r["date"] != d:
            continue
        g = finals.get(r["game_id"])
        if g is None:
            still_open += 1
            continue
        hr, ar = g["home_score"], g["away_score"]
        home_won = 1.0 if hr > ar else 0.0
        r.update({
            "final_home_runs": hr, "final_away_runs": ar, "actual_total": hr + ar,
            "winner": r["home"] if hr > ar else r["away"],
            "pick_correct": (r["pick"] == (r["home"] if hr > ar else r["away"])),
            "brier": round((r["p_home"] - home_won) ** 2, 4),
            "total_error": round(r["proj_total"] - (hr + ar), 1),
            "graded_at": datetime.now().isoformat(timespec="seconds"),
        })
        graded += 1
    write_log(log)
    return graded, still_open

def summarize_log():
    """Dashboard metrics from every graded game in the log."""
    g = [r for r in read_log() if r["graded_at"] is not None]
    if not g:
        return None
    df = pd.DataFrame(g)
    return {
        "graded_games": len(df),
        "record": f"{int(df['pick_correct'].sum())}-{int((~df['pick_correct']).sum())}",
        "accuracy_%": round(100 * df["pick_correct"].mean(), 1),
        "brier": round(df["brier"].mean(), 4),
        "totals_mae": round(df["total_error"].abs().mean(), 2),
        "totals_bias": round(df["total_error"].mean(), 2),  # + = projecting too high
        "df": df,
    }
