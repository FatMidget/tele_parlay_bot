import os
import math
import io
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import poisson

import aiohttp
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ======== ENV (Fly.io: set in Secrets) ========
BOT_TOKEN = os.environ.get("BOT_TOKEN") or ""
ODDS_API_KEY = os.environ.get("ODDS_API_KEY") or ""

# ======== CONFIG ========
LEAGUES = {
    "laliga":      ("soccer_spain_la_liga",       "SP1"),
    "premier":     ("soccer_epl",                 "E0"),
    "seriea":      ("soccer_italy_serie_a",       "I1"),
    "bundesliga":  ("soccer_germany_bundesliga",  "D1"),
    "ligue1":      ("soccer_france_ligue_one",    "F1"),
    "eredivisie":  ("soccer_netherlands_eredivisie","N1"),
    "primeira":    ("soccer_portugal_primeira_liga","P1"),
}
SEASONS_BACK = 3
REGION = "eu"
MAX_GOALS = 10

# ======== UTILS ========
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

# ======== HISTORICAL (football-data.co.uk) ========
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

async def fetch_fd_csv(session: aiohttp.ClientSession, code: str, season_code: str) -> Optional[pd.DataFrame]:
    url = f"http://www.football-data.co.uk/mmz4281/{season_code}/{code}.csv"
    async with session.get(url, timeout=30) as r:
        if r.status != 200:
            return None
        content = await r.read()
    df = pd.read_csv(io.BytesIO(content))
    needed = ["HomeTeam","AwayTeam","FTHG","FTAG","HTHG","HTAG"]
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan
    return df[needed].dropna(how="any")

@dataclass
class TeamStats:
    matches: int
    gf: float
    ga: float
    first_half_share: float

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
        avg_home_goals = lg_home_goals / total_matches if total_matches else 1.4,
        avg_away_goals = lg_away_goals / total_matches if total_matches else 1.2,
        league_first_half_share = league_first_half_share
    )

    teams = {}
    for _, row in df.iterrows():
        h, a = row.HomeTeam, row.AwayTeam
        FTHG, FTAG, HTHG, HTAG = row.FTHG, row.FTAG, row.HTHG, row.HTAG

        for name, gf_add, ga_add in [(h, FTHG, FTAG), (a, FTAG, FTHG)]:
            t = teams.get(name, {"m":0,"gf":0,"ga":0,"tg":0,"th":0})
            t["m"] += 1
            t["gf"] += gf_add
            t["ga"] += ga_add
            t["tg"] += (FTHG + FTAG)
            t["th"] += (HTHG + HTAG)
            teams[name] = t

    out: Dict[str, TeamStats] = {}
    for name, t in teams.items():
        fh_share_raw = (t["th"]/t["tg"]) if t["tg"]>0 else league.league_first_half_share
        fh_share = 0.7*fh_share_raw + 0.3*league.league_first_half_share  # shrink to league mean
        out[name] = TeamStats(matches=t["m"], gf=t["gf"]/t["m"], ga=t["ga"]/t["m"], first_half_share=fh_share)
    return out, league

async def load_historical(session: aiohttp.ClientSession, fd_code: str) -> Tuple[Dict[str,TeamStats], LeagueStats]:
    frames = []
    for c in season_codes_for_today(SEASONS_BACK):
        df = await fetch_fd_csv(session, fd_code, c)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        raise RuntimeError("No football-data CSVs available now.")
    all_df = pd.concat(frames, ignore_index=True)
    return compute_team_league_stats(all_df)

# ======== ODDS (The Odds API) ========
async def fetch_market(session: aiohttp.ClientSession, sport_key: str, home: str, away: str, region="eu") -> Dict:
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {"regions": region, "markets": "h2h,totals,btts", "apiKey": ODDS_API_KEY, "oddsFormat": "decimal"}
    async with session.get(url, params=params, timeout=25) as r:
        if r.status != 200:
            raise RuntimeError("Odds API error or quota exceeded.")
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
                    line_rec = entry["totals"].get(line, {"over": None, "under": None})
                    if "over" in name: line_rec["over"] = price
                    else: line_rec["under"] = price
                    entry["totals"][line] = line_rec
            elif key == "btts" and len(oc) == 2:
                entry["btts"] = {"yes": oc[0]["price"], "no": oc[1]["price"]}
        books[bname] = entry

    return {"home": event["home_team"], "away": event["away_team"], "books": books}

# ======== MODEL ========
@dataclass
class LeagueTeamView:
    lam_home: float
    lam_away: float
    first_half_share: float

def team_lookup(name: str, teams: Dict[str,TeamStats]) -> TeamStats:
    for k in teams.keys():
        if k.lower() == name.lower():
            return teams[k]
    # fallback: best partial match
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
    return np.outer
