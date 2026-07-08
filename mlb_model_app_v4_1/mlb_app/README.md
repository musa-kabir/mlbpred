# MLB Daily Model — install & run

## One-time install (2 minutes)
1. Install Python 3.10+ from python.org (check "Add to PATH" on Windows)
2. Open a terminal in this folder and run:
   pip install -r requirements.txt

## Daily use
Start the app (it opens in your browser):
   streamlit run app.py

- **Morning** (any time before first pitch): click "Run morning predictions"
- **Night** (after games end): click "Grade tonight's results"
  - West coast games not final yet? Run it again later — already-graded
    games are never touched.

## The log
Everything is written to `mlb_log.jsonl` — one JSON line per game containing
every input the prediction used (Elo, offense/defense rates, starter FIPs,
park factor) and, after grading, the result, Brier score, and totals error.
To improve the model, paste this file into a Claude chat.

## Optional: market odds (roadmap days 10-12)
Get a free API key at the-odds-api.com and paste it into ODDS_API_KEY at the
top of mlb_engine.py. Morning runs will then log the market's win probability
and your model's edge vs. the market per game.

## Live win probability (v4)
Open the "🔴 Live" tab during games. Each refresh reads the official
play-by-play feed for every in-progress game and shows a live home win
probability from: score, inning, outs, runners (RE24), and this morning's
pregame run rates. Toggle auto-refresh to update every 60s. Snapshots are
logged to mlb_live_log.jsonl; click "Grade live log" after games to stamp
each snapshot with the actual winner.
