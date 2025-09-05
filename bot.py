import os
import io
import re
import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import poisson

import aiohttp
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ========= Runtime config (Railway: set in Variables) =========
BOT_TOKEN = os.environ.get("BOT_TOKEN") or ""        # Telegram token from @BotFather
ODDS_API_KEY = os.environ.get("ODDS_API_KEY") or ""  # The Odds API key (optional; model still works without)

# ========= App-level constants =========
LEAGUES = {
    "laliga":      ("soccer_spain_la_liga",         "SP1"),
    "premier":     ("soccer_epl",                   "E0"),
    "seriea":      ("soccer_italy_serie_a",         "I1"),
    "bundesliga":  ("soccer_germany_bundesliga",    "D1"),
    "ligue1":      ("soccer_france_ligue_one",      "F1"),
    "eredivisie":  ("soccer_netherlands_eredivisie","N1"),
    "primeira":    ("soccer_portugal_primeira_liga","P1"),
}
SEASONS_BACK = 3          # seasons of history to learn team stats
REGION = "eu"             # Odds API region
MAX_GOALS = 10            # scoreline grid up to 10 goals each side

# ========= Logging =========
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("parlay-bot")

# ========= Small utils =========
def implied_prob(odds: float) -> float:
    return 1.0 / max(odds, 1e-12)

def remove_vig_three(p_home: float, p_draw: float, p_away: float) -> Tuple[float,float,float]:
    s = p_home + p_draw + p_away
    return (p_home/s, p_draw/s, p_away/s) if s > 0 else (0.0,0.0,0.0)

def remove_vig_two(p_yes: float, p_no: float) -> Tuple[float,float]:
    s = p_yes + p_no
    return (p_yes/s, p_no/s) if s > 0 else (0.0,0.0)

def fair_odds(p: float) -> float:
    return round(1.0 / max(p, 1e-9), 2)

def pct(p: float) -> str:
    return f"{p*100:.1f}%"

# ========= football-data.co.uk (historical) =========
def season_codes_for_today(n_back: int) -> List[str]:
    from datetime import datetime
    today = datetime.utcnow()
    y = today.year % 100
    active = f"{y:02d}{(y+1)%100:02d}"
    codes = [active]
    for i in range(1, n_back):
        a = (y - i) % 100
        b = (y - i + 1) % 100
        codes.append(f"{a:02d}{b:02d}")
    return codes

def _looks_like_html(b: bytes) -> bool:
    head = b[:200].lower()
    return b"<html" in head or b"<!doctype html" in head

async def fetch_fd_csv(session: aiohttp.ClientSession, code: str, season_code: str) -> Optional[pd.DataFrame]:
    """
    Robust reader for football-data.co.uk CSVs:
    - Tries comma first, then semicolon
    - Skips malformed lines
    - Skips HTML/error responses
    - Ensures required columns exist
    """
    url = f"http://www.football-data.co.uk/mmz4281/{season_code}/{code}.csv"
    try:
        async with session.get(url, timeout=30) as r:
            if r.status != 200:
                log.info(f"[FD] {season_code}/{code} HTTP {r.status} — skipping")
                return None
            content = await r.read()
    except Exception as e:
        log.info(f"[FD] {season_code}/{code} request error: {e} — skipping")
        return None

    if not content or _looks_like_html(content):
        log.info(f"[FD] {season_code}/{code} looks like HTML/empty — skipping")
        return None

    needed = ["HomeTeam","AwayTeam","FTHG","FTAG","HTHG","HTAG"]

    for sep in (",",";"):
        try:
            df = pd.read_csv(
                io.BytesIO(content),
                sep=sep,
                engine="python",
                on_bad_lines="skip",
                encoding_errors="ignore"
            )
            # normalize headers (remove spaces)
            df.columns = [re.sub(r"\s+", "", str(c)) for c in df.columns]
            missing = [c for c in needed if c not in df.columns]
            if missing:
                continue
            df = df[needed].dropna(how="any")
            if df.empty:
                continue
            return df
        except Exception:
            continue

    log.info(f"[FD] {season_code}/{code} could not parse with ',' or ';' — skipping")
    return None

@dataclass
class TeamStats:
    matches: int
    gf: float
    ga: float
    first_half_share: float  # proportion of goals (in this team’s matches) scored in 1H

@dataclass
class LeagueStats:
    avg_home_goals: float
    avg_away_goals: float
    league_first_half_share: float

def compute_team_league_stats(df: pd.DataFrame) -> Tuple[Dict[str,TeamStats], LeagueStats]:
    total_matches = len(df)
    lg_home_goals = df["FTHG"].sum()
    lg_away_goals = df["FTAG"].sum()
    lg_total_goals = lg_home_goals + lg_away_goals
    lg_first_half_goals = df["HTHG"].sum() + df["HTAG"].sum()
    league_first_half_share = (lg_first_half_goals / lg_total_goals) if lg_total_goals > 0 else 0.45

    league = LeagueStats(
        avg_home_goals = (lg_home_goals / total_matches) if total_matches else 1.4,
        avg_away_goals = (lg_away_goals / total_matches) if total_matches else 1.2,
        league_first_half_share = league_first_half_share
    )

    teams: Dict[str, TeamStats] = {}
    for _, row in df.iterrows():
        h, a = row.HomeTeam, row.AwayTeam
        FTHG, FTAG, HTHG, HTAG = row.FTHG, row.FTAG, row.HTHG, row.HTAG

        for name, gf_add, ga_add in ((h, FTHG, FTAG), (a, FTAG, FTHG)):
            t = teams.get(name, {"m":0,"gf":0,"ga":0,"tg":0,"th":0})
            t["m"] += 1
            t["gf"] += gf_add
            t["ga"] += ga_add
            t["tg"] += (FTHG + FTAG)
            t["th"] += (HTHG + HTAG)
            teams[name] = t

    out: Dict[str, TeamStats] = {}
    for name, t in teams.items():
        raw = (t["th"]/t["tg"]) if t["tg"]>0 else league.league_first_half_share
        # shrink to league mean to reduce noise: 70% team, 30% league
        fh = 0.7*raw + 0.3*league.league_first_half_share
        out[name] = TeamStats(matches=t["m"], gf=t["gf"]/t["m"], ga=t["ga"]/t["m"], first_half_share=fh)

    return out, league

async def load_historical(session: aiohttp.ClientSession, fd_code: str) -> Tuple[Dict[str,TeamStats], LeagueStats]:
    frames = []
    season_list = season_codes_for_today(SEASONS_BACK)
    for sc in season_list:
        df = await fetch_fd_csv(session, fd_code, sc)
        if df is not None and not df.empty:
            frames.append(df)
            log.info(f"[FD] loaded season {sc} ({len(df)} rows)")
        else:
            log.info(f"[FD] skipped season {sc}")

    if not frames:
        raise RuntimeError("football-data CSVs unavailable or unparsable for this league right now.")

    all_df = pd.concat(frames, ignore_index=True)
    return compute_team_league_stats(all_df)

# ========= Odds (The Odds API) =========
async def fetch_market(session: aiohttp.ClientSession, sport_key: str, home: str, away: str, region=REGION) -> Dict:
    if not ODDS_API_KEY:
        raise RuntimeError("No ODDS_API_KEY set; running in model-only mode.")
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {"regions": region, "markets": "h2h,totals,btts", "apiKey": ODDS_API_KEY, "oddsFormat": "decimal"}

    async with session.get(url, params=params, timeout=25) as r:
        if r.status != 200:
            raise RuntimeError(f"Odds API error (HTTP {r.status}).")
        data = await r.json()

    def norm(s: str) -> str:
        return s.lower().replace("fc","").replace("cf","").strip()

    nh, na = norm(home), norm(away)
    event = None
    for ev in data:
        teams = [norm(t) for t in ev.get("teams", [])]
        if nh in teams and na in teams:
            event = ev; break
    if not event:
        raise ValueError("Match not found in odds feed.")

    books = {}
    for bk in event.get("bookmakers", []):
        bname = bk["title"]
        entry = {"h2h": None, "totals": {}, "btts": None}
        for mk in bk.get("markets", []):
            key = mk.get("key"); oc = mk.get("outcomes", [])
            if key == "h2h" and len(oc) == 3:
                mp = {o["name"].lower(): o["price"] for o in oc}
                entry["h2h"] = {
                    "home": mp.get(event["home_team"].lower()),
                    "draw": mp.get("draw"),
                    "away": mp.get(event["away_team"].lower()),
                }
            elif key == "totals":
                for o in oc:
                    line = float(o["point"]); name = o["name"].lower(); price = o["price"]
                    lr = entry["totals"].get(line, {"over": None, "under": None})
                    if "over" in name: lr["over"] = price
                    else: lr["under"] = price
                    entry["totals"][line] = lr
            elif key == "btts" and len(oc) == 2:
                entry["btts"] = {"yes": oc[0]["price"], "no": oc[1]["price"]}
        books[bname] = entry

    return {"home": event["home_team"], "away": event["away_team"], "books": books}

# ========= Model (Poisson) =========
@dataclass
class LeagueTeamView:
    lam_home: float
    lam_away: float
    first_half_share: float

def team_lookup(name: str, teams: Dict[str,TeamStats]) -> TeamStats:
    for k in teams.keys():
        if k.lower() == name.lower():
            return teams[k]
    # fallback: best word overlap
    lo = name.lower().split()
    return max(teams.items(), key=lambda kv: len(set(lo) & set(kv[0].lower().split())))[1]

def build_lambdas_from_history(home_team: str, away_team: str, teams: Dict[str,TeamStats], league: LeagueStats) -> LeagueTeamView:
    th = team_lookup(home_team, teams)
    ta = team_lookup(away_team, teams)

    lg_team_gf = (league.avg_home_goals + league.avg_away_goals) / 2.0
    lg_team_ga = lg_team_gf

    home_attack = th.gf / max(lg_team_gf, 1e-6)
    home_def_weak = th.ga / max(lg_team_ga, 1e-6)
    away_attack = ta.gf / max(lg_team_gf, 1e-6)
    away_def_weak = ta.ga / max(lg_team_ga, 1e-6)

    lam_h = league.avg_home_goals * home_attack * away_def_weak
    lam_a = league.avg_away_goals * away_attack * home_def_weak

    fh_share = 0.5*(th.first_half_share + ta.first_half_share)
    return LeagueTeamView(lam_h, lam_a, fh_share)

def prob_matrix(lh: float, la: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    home = [poisson.pmf(i, lh) for i in range(max_goals+1)]
    away = [poisson.pmf(j, la) for j in range(max_goals+1)]
    return np.outer(home, away)

def result_probs(mat: np.ndarray) -> Tuple[float,float,float]:
    h = d = a = 0.0; g = mat.shape[0]-1
    for i in range(g+1):
        for j in range(g+1):
            if i>j: h += mat[i,j]
            elif i==j: d += mat[i,j]
            else: a += mat[i,j]
    return h,d,a

def over_prob(mat: np.ndarray, line: float) -> float:
    g = mat.shape[0]-1; s=0.0
    for i in range(g+1):
        for j in range(g+1):
            if i+j > line: s += mat[i,j]
    return s

def btts_prob(mat: np.ndarray) -> float:
    g = mat.shape[0]-1; s=0.0
    for i in range(1,g+1):
        for j in range(1,g+1):
            s += mat[i,j]
    return s

def p_no_goal_first_half(lh: float, la: float, fh_share: float) -> float:
    lam_1h = (lh+la) * fh_share
    return math.exp(-lam_1h)

def calibrate_to_market_total(lh: float, la: float, target_over: Optional[float], line: Optional[float]) -> Tuple[float,float]:
    if target_over is None or line is None:
        return lh, la
    s = 1.0
    for _ in range(25):
        M = prob_matrix(lh*s, la*s)
        curr = over_prob(M, line)
        diff = target_over - curr
        s *= (1.0 + diff*0.6)
        s = max(0.2, min(5.0, s))
    return lh*s, la*s

# ========= Parlays =========
def prob_of_combo(mat: np.ndarray, result=None, ou=None, btts=None) -> float:
    g = mat.shape[0]-1; s=0.0
    for i in range(g+1):
        for j in range(g+1):
            ok = True
            if result:
                if result=="home" and not (i>j): ok=False
                if result=="draw" and not (i==j): ok=False
                if result=="away" and not (i<j): ok=False
            if ok and ou:
                side, line = ou
                if side=="over" and not (i+j>line): ok=False
                if side=="under" and not (i+j<line): ok=False
            if ok and (btts is not None):
                has = (i>0 and j>0)
                if btts != has: ok=False
            if ok: s += mat[i,j]
    return s

def select_parlays(book_name: str, book: Dict, mat: np.ndarray, line_pref=2.5) -> List[Dict]:
    legs = []
    h2h = book.get("h2h"); totals = book.get("totals", {}); btts = book.get("btts")
    if h2h and all(h2h.get(k) for k in ("home","draw","away")):
        legs += [("result","home", h2h["home"]), ("result","draw", h2h["draw"]), ("result","away", h2h["away"])]
    if totals:
        best = None
        for ln, kv in totals.items():
            if kv.get("over") and kv.get("under"):
                cand = (ln, kv["over"], kv["under"])
                if best is None or abs(ln-line_pref) < abs(best[0]-line_pref):
                    best = cand
        if best:
            ln, oo, uu = best
            legs += [("ou_over", ln, oo), ("ou_under", ln, uu)]
    if btts and btts.get("yes") and btts.get("no"):
        legs += [("btts","yes", btts["yes"]), ("btts","no", btts["no"])]

    from itertools import combinations
    combos=[]
    def model_p(c):
        res=None; ou=None; bt=None
        for t in c:
            if t[0]=="result": res=t[1]
            elif t[0].startswith("ou_"): ou=("over" if t[0]=="ou_over" else "under", t[1])
            elif t[0]=="btts": bt=(t[1]=="yes")
        return prob_of_combo(mat, result=res, ou=ou, btts=bt)

    valid = list(legs)
    for r in (2,3):
        for pick in combinations(valid, r):
            tags=[p[0] for p in pick]
            if tags.count("result")>1: continue
            if "ou_over" in tags and "ou_under" in tags: continue
            if [t for t in tags if t=="btts"].count("btts")>1:
                y=[p for p in pick if p[0]=="btts"]
                if len(y)>1 and y[0][1]!=y[1][1]: continue
            mp = model_p(pick)
            if mp<=0: continue
            fair = fair_odds(mp)
            market=1.0
            for p in pick: market *= p[2]
            combos.append({
                "legs": pick, "size": r,
                "model_p": round(mp,3),
                "fair_odds": fair,
                "market_odds": round(market,2),
                "value_ratio": round(market/fair,3)
            })
    if not combos: return []

    two=[c for c in combos if c["size"]==2]
    tier1 = max(two, key=lambda x: x["model_p"]) if two else max(combos, key=lambda x: x["model_p"])
    mids=[c for c in combos if 0.25<=c["model_p"]<=0.45]
    tier2 = max(mids, key=lambda x: x["value_ratio"]) if mids else max(combos, key=lambda x: x["value_ratio"])
    three=[c for c in combos if c["size"]==3 and 0.05<=c["model_p"]<=0.20]
    if three:
        tier3=max(three, key=lambda x: x["market_odds"])
    else:
        three_any=[c for c in combos if c["size"]==3]
        tier3=max(three_any or combos, key=lambda x: x["market_odds"])

    def leg_text(code):
        if code[0]=="result": return code[1].upper()
        if code[0].startswith("ou_"): return ("Over" if code[0]=="ou_over" else "Under") + f" {code[1]}"
        if code[0]=="btts": return "BTTS Yes" if code[1]=="yes" else "BTTS No"
        return "?"

    chosen=[]
    for t in [tier1,tier2,tier3]:
        chosen.append({
            "book": book_name,
            "legs_text": " + ".join([leg_text(l) for l in t["legs"]]),
            "model_p": t["model_p"],
            "fair_odds": t["fair_odds"],
            "market_odds": t["market_odds"],
            "value_ratio": t["value_ratio"]
        })
    return chosen

# ========= Core prediction =========
async def predict_once(league_key: str, home: str, away: str) -> str:
    if league_key not in LEAGUES:
        return "Unknown league. Try: " + ", ".join(LEAGUES.keys())
    sport_key, fd_code = LEAGUES[league_key]

    async with aiohttp.ClientSession() as session:
        teams, league = await load_historical(session, fd_code)
        view = build_lambdas_from_history(home, away, teams, league)

        # Try to get odds (optional). If it fails/quota exhausted, proceed model-only.
        market = None
        try:
            market = await fetch_market(session, sport_key, home, away, region=REGION)
        except Exception as e:
            log.info(f"Odds fetch skipped: {e}")

    # Calibrate to market total if available
    target_over=None; line_sel=None
    if market:
        p_over_list=[]; lines=[]
        for b in market["books"].values():
            for ln, kv in b.get("totals", {}).items():
                if kv.get("over") and kv.get("under"):
                    lines.append(ln)
        if lines:
            line_sel = sorted(lines, key=lambda x: abs(x-2.5))[0]
            for b in market["books"].values():
                kv=b.get("totals", {}).get(line_sel)
                if kv and kv.get("over") and kv.get("under"):
                    po, pu = implied_prob(kv["over"]), implied_prob(kv["under"])
                    po, pu = remove_vig_two(po, pu)
                    p_over_list.append(po)
        if p_over_list:
            target_over=float(np.mean(p_over_list))

    lh, la = view.lam_home, view.lam_away
    lh, la = calibrate_to_market_total(lh, la, target_over, line_sel)

    M = prob_matrix(lh, la, MAX_GOALS)
    p_home, p_draw, p_away = result_probs(M)
    p_over25 = over_prob(M, 2.5)
    p_btts = btts_prob(M)
    p_ng1h = p_no_goal_first_half(lh, la, view.first_half_share)

    parts=[]
    parts.append(f"Match: {home} vs {away}")
    parts.append("— Model (Poisson calibrated; team-specific 1H share) —")
    parts.append(f"Home {pct(p_home)}, Draw {pct(p_draw)}, Away {pct(p_away)}")
    parts.append(f"Over 2.5 {pct(p_over25)}, BTTS Yes {pct(p_btts)}, No goal 1H {pct(p_ng1h)}")

    # Build parlays from a single bookmaker if odds available
    if market and market["books"]:
        best=None; cover=-1
        for name,b in market["books"].items():
            c=0
            if b.get("h2h"): c+=1
            if b.get("totals"): c+=1
            if b.get("btts"): c+=1
            if c>cover: best=(name,b); cover=c
        tiers = select_parlays(best[0], best[1], M, line_pref=2.5) if best else []
        if tiers:
            parts.append("— Parlays (book odds vs fair) —")
            for t in tiers:
                parts.append(
                    f"[{t['book']}] {t['legs_text']} | p={t['model_p']:.3f} "
                    f"(fair {t['fair_odds']}) | market {t['market_odds']} | value× {t['value_ratio']}"
                )
        else:
            parts.append("No suitable parlay markets available right now.")
    else:
        parts.append("No live odds available (quota/coverage). Showing model only.")
        # Provide 3 model-only suggestions with fair odds
        candidates = [
            ("HOME + Over 2.5", prob_of_combo(M, result="home", ou=("over",2.5))),
            ("HOME + BTTS Yes", prob_of_combo(M, result="home", btts=True)),
            ("AWAY + Over 2.5 + BTTS Yes", prob_of_combo(M, result="away", ou=("over",2.5), btts=True)),
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        parts.append("— Model-only parlays (fair odds) —")
        for name, p in candidates[:3]:
            parts.append(f"{name} | p={p:.3f} (fair {fair_odds(p)})")

    return "\n".join(parts)

# ========= Telegram handlers =========
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "Usage:\n"
        "/predict <league> ; <Home> ; <Away>\n"
        "Leagues: " + ", ".join(LEAGUES.keys()) + "\n"
        "Example: /predict laliga ; Barcelona ; Sevilla"
    )
    await update.message.reply_text(msg)

async def predict_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        payload = update.message.text.split(" ", 1)[1]
        parts = [p.strip() for p in payload.replace(",", ";").split(";") if p.strip()]
        league, home, away = parts[0], parts[1], parts[2]
    except Exception:
        await update.message.reply_text("Usage: /predict <league> ; <Home> ; <Away>")
        return

    try:
        text = await predict_once(league, home, away)
    except Exception as e:
        log.exception("Prediction error")
        text = f"Error: {e}"
    await update.message.reply_text(text)

# ========= Startup hook: clear webhook, then poll =========
async def on_startup(app):
    log.info("Clearing webhook and starting polling…")
    try:
        await app.bot.delete_webhook(drop_pending_updates=True)
        log.info("Webhook cleared.")
    except Exception as e:
        log.info(f"No webhook to clear or error clearing webhook: {e}")

def main():
    if not BOT_TOKEN:
        raise RuntimeError("Missing BOT_TOKEN environment variable.")
    app = Application.builder().token(BOT_TOKEN).post_init(on_startup).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("predict", predict_cmd))
    log.info("Running long polling…")
    app.run_polling()

if __name__ == "__main__":
    main()
