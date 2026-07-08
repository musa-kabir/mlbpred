"""
MLB Model — twice-a-day control panel.
Run with:  streamlit run app.py
"""
import streamlit as st
import pandas as pd
import mlb_engine as eng
import mlb_live as live

st.set_page_config(page_title="MLB Model", page_icon="⚾", layout="wide")
st.title("⚾ MLB Daily Model")
st.caption("Morning: generate & freeze predictions. Night: grade the results. "
           "Everything lands in mlb_log.jsonl.")

tab_daily, tab_live = st.tabs(["📅 Daily model", "🔴 Live win probability"])

with tab_live:
    st.write("Reads the official play-by-play feed for every in-progress game and "
             "converts game state (score, inning, outs, runners) into a live win "
             "probability anchored on this morning's model. Each refresh logs a "
             "snapshot to mlb_live_log.jsonl.")
    auto = st.toggle("Auto-refresh every 60s", value=False)
    if st.button("Refresh live games", type="primary") or auto:
        rows = live.snapshot_live_games()
        if not rows:
            st.info("No games in progress right now.")
        else:
            df = pd.DataFrame(rows)
            df["live WP (home)"] = (df["p_home_live"] * 100).round(1).astype(str) + "%"
            df["pregame"] = df["p_home_pregame"].apply(
                lambda p: f"{p*100:.1f}%" if p is not None else "—")
            st.dataframe(df[["away", "home", "score", "inning", "outs", "bases",
                             "pregame", "live WP (home)", "last_play"]],
                         use_container_width=True, hide_index=True)
            st.caption("bases = runners on 1st/2nd/3rd (101 = corners). "
                       "'—' pregame means today's morning run wasn't logged for that game.")
        if auto:
            import time as _t
            _t.sleep(60)
            st.rerun()
    if st.button("Grade live log against finals"):
        n = live.grade_live_log()
        st.success(f"Stamped {n} snapshots with final winners.")

with tab_daily:
    col1, col2 = st.columns(2)

    # ---------------- MORNING ----------------
    with col1:
        st.subheader("🌅 Morning run")
        st.write("Re-trains on all completed games, predicts today's slate "
                 "(starter FIP + park adjusted), and freezes picks to the log.")
        if st.button("Run morning predictions", type="primary", use_container_width=True):
            with st.spinner("Training ratings and fetching probable starters…"):
                recs = eng.predict_slate()
            if not recs:
                st.warning("No regular-season games found for today (or season data is empty).")
            else:
                added, skipped = eng.log_morning(recs)
                st.success(f"Logged {added} new predictions"
                           + (f" ({skipped} were already frozen — not overwritten)." if skipped else "."))
                show = pd.DataFrame(recs)[[
                    "away", "home", "away_starter", "away_starter_fip",
                    "home_starter", "home_starter_fip", "park_factor",
                    "p_home", "pick", "proj_total"]]
                show["p_home"] = (show["p_home"] * 100).round(1).astype(str) + "%"
                st.dataframe(show, use_container_width=True, hide_index=True)

    # ---------------- NIGHT ----------------
    with col2:
        st.subheader("🌙 Night run")
        st.write("Pulls final scores and grades every ungraded prediction from today: "
                 "winner, Brier score, and total-runs error.")
        if st.button("Grade tonight's results", use_container_width=True):
            with st.spinner("Fetching finals…"):
                graded, still_open = eng.grade_night()
            if graded == 0 and still_open == 0:
                st.info("Nothing to grade — run the morning predictions first.")
            else:
                st.success(f"Graded {graded} games.")
                if still_open:
                    st.warning(f"{still_open} games not final yet — run again later and "
                               "they'll be graded without touching anything else.")

    st.divider()

    # ---------------- DASHBOARD ----------------
    st.subheader("📊 Model record")
    summary = eng.summarize_log()
    if summary is None:
        st.info("No graded games yet. The dashboard appears after your first night run.")
    else:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Record", summary["record"])
        m2.metric("Accuracy", f"{summary['accuracy_%']}%",
                  help="Reminder: 57% is elite over a real sample. Judge on 300+ games.")
        m3.metric("Brier score", summary["brier"],
                  help="Lower is better. 0.25 = coin flip. Good MLB models sit ~0.24 or below.")
        m4.metric("Totals MAE", summary["totals_mae"],
                  help="Average miss on projected total runs.")
        m5.metric("Totals bias", summary["totals_bias"],
                  help="Positive = the model projects totals too HIGH on average. "
                       "A persistent bias is a calibration fix, not bad luck.")

        df = summary["df"]
        daily = (df.groupby("date")
                   .agg(games=("pick_correct", "size"),
                        wins=("pick_correct", "sum"),
                        brier=("brier", "mean"))
                   .reset_index())
        daily["accuracy"] = (100 * daily["wins"] / daily["games"]).round(1)
        st.line_chart(daily.set_index("date")[["accuracy"]])

        with st.expander("Graded game log (newest first)"):
            cols = ["date", "away", "home", "pick", "p_home", "winner", "pick_correct",
                    "brier", "proj_total", "actual_total", "total_error",
                    "home_starter_fip", "away_starter_fip", "park_factor"]
            st.dataframe(df[cols].iloc[::-1], use_container_width=True, hide_index=True)

    st.divider()
    st.caption("To have the model reviewed: open mlb_log.jsonl next to this app and paste "
               "its contents (or attach the file) into a Claude chat — every prediction "
               "input and outcome is on one line per game.")
