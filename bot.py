import os
import io
import re
import math
import unicodedata
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
from scipy.stats import poisson

import aiohttp
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ========= Runtime config (Railway: set in Variables) =========
BOT_TOKEN = os.environ.get("BOT_TOKEN") or ""         # Telegram token from @BotFather
ODDS_API_KEY = os.environ.get("ODDS_API_KEY") or ""   # The Odds API key (optional; model still works)

# ========= App-level constants =========
LEAGUES = {
    "laliga":      ("soccer_spain_la_liga",         "SP1"),
    "premier":     ("soccer_epl",                   "E0"),
    "seriea":      ("soccer_italy_serie_a",         "I1"),
    "bundesliga":  ("soccer_germany_bundesliga",    "D1"),
    "ligue1":      ("soccer_france_ligue_one",      "F1"),
    "eredivisie":  ("soccer_netherlands_eredivisie","N1"),
    "primeira":    ("soccer_portugal_primeira_liga","P1"),
    # internationals/cups (odds-only if no CSV history)
    "worldcup":    ("soccer_fifa_world_cup",        None),
    "euro":        ("soccer_uefa_euro",             None),
    "ucl":         ("soccer_uefa_champs_league",    None),
    "uel":         ("soccer_uefa_europa_league",    None),
    "nations":     ("soccer_uefa_nations_league",   None),
    "friendlies":  ("soccer_international_friendly",None),
}
SEASONS_BACK = 3
REGION = "eu"
MAX_GOALS = 10

# ========= Logging =========
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("parlay-bot")

# ========= Helpers =========
def _strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")

def _norm_team(s: str) -> str:
    s = _strip_accents(s)
    s = s.lower().replace("fc"," ").replace("cf"," ").replace(".", " ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

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
    Robust reader: tries comma/semicolon, skips bad lines, ignores HTML, normalizes headers.
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
        avg_home_goals = (lg_home_goals / total_matches) if total_matches else 1.4,
        avg_away_goals = (lg_away_goals / total_matches) if total_matches else 1.2,
        league_first_half_share = league_first_half_share
    )

    teams: Dict[str, TeamStats] = {}
    for _, row in df.iterrows():
        h, a = str(row.HomeTeam), str(row.AwayTeam)
        FTHG, FTAG, HTHG, HTAG = row.FTHG, row.FTAG, row.HTHG, row.HTAG

        for name, gf_add, ga_add in ((h, FTHG, FTAG), (a, FTAG, FTHG)):
            key = _norm_team(name)
            t = teams.get(key, {"m":0,"gf":0,"ga":0,"tg":0,"th":0})
            t["m"] += 1
            t["gf"] += gf_add
            t["ga"] += ga_add
            t["tg"] += (FTHG + FTAG)
            t["th"] += (HTHG + HTAG)
            teams[key] = t

    out: Dict[str, TeamStats] = {}
    for key, t in teams.items():
        raw = (t["th"]/t["tg"]) if t["tg"]>0 else league.league_first_half_share
        fh = 0.7*raw + 0.3*league.league_first_half_share
        out[key] = TeamStats(matches=t["m"], gf=t["gf"]/t["m"], ga=t["ga"]/t["m"], first_half_share=fh)

    return out, league

async def load_historical(session: aiohttp.ClientSession, fd_code: str) -> Tuple[Dict[str,TeamStats], LeagueStats]:
    frames = []
    for sc in season_codes_for_today(SEASONS_BACK):
        df = await fetch_fd_csv(session, fd_code, sc)
        if df is not None and not df.empty:
            frames.append(df)
            log.info(f"[FD] loaded season {sc} ({len(df)} rows)")
        else:
            log.info(f"[FD] skipped season {sc}")
    if not frames:
        # instead of hard failing, return empty so caller can decide
        return {}, LeagueStats(1.4, 1.2, 0.45)
    all_df = pd.concat(frames, ignore_index=True)
    return compute_team_league_stats(all_df)

# ========= Odds (The Odds API) =========
async def fetch_market(session: aiohttp.ClientSession, sport_key: str, home: str, away: str, region=REGION) -> Optional[Dict]:
    if not ODDS_API_KEY:
        return None
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    # attempt 1 (with region)
    params1 = {"regions": region, "markets": "h2h,totals,btts", "apiKey": ODDS_API_KEY, "oddsFormat": "decimal"}
    # attempt 2 (no region) — reduces HTTP 422 for some sports keys
    params2 = {"markets": "h2h,totals,btts", "apiKey": ODDS_API_KEY, "oddsFormat": "decimal"}

    async def _try(params):
        async with session.get(url, params=params, timeout=25) as r:
            if r.status != 200:
                raise RuntimeError(f"Odds API HTTP {r.status}")
            return await r.json()

    data = None
    try:
        data = await _try(params1)
    except Exception:
        try:
            data = await _try(params2)
        except Exception as e2:
            log.info(f"Odds failed: {e2}")
            return None

    nh, na = _norm_team(home), _norm_team(away)
    event = None
    for ev in data:
        teams = [_norm_team(t) for t in ev.get("teams", [])]
        if nh in teams and na in teams:
            event = ev; break
    if not event:
        log.info("Match not found in odds feed.")
        return None

    books = {}
    for bk in event.get("bookmakers", []):
        bname = bk["title"]
        entry = {"h2h": None, "totals": {}, "btts": None}
        for mk in bk.get("markets", []):
            key = mk.get("key"); oc = mk.get("outcomes", [])
            if key == "h2h" and len(oc) == 3:
                # names: exact team names + "Draw"
                mp = { _norm_team(o["name"]): o["price"] for o in oc }
                home_name = _norm_team(event["home_team"])
                away_name = _norm_team(event["away_team"])
                entry["h2h"] = {
                    "home": mp.get(home_name),
                    "draw": mp.get(_norm_team("Draw")),
                    "away": mp.get(away_name),
                }
            elif key == "totals":
                for o in oc:
                    try:
                        line = float(o["point"])
                    except Exception:
                        continue
                    name = o["name"].lower(); price = o["price"]
                    lr = entry["totals"].get(line, {"over": None, "under": None})
                    if "over" in name: lr["over"] = price
                    else: lr["under"] = price
                    entry["totals"][line] = lr
            elif key == "btts" and len(oc) == 2:
                opt = {o["name"].lower(): o["price"] for o in oc}
                entry["btts"] = {"yes": opt.get("yes"), "no": opt.get("no")}
        books[bname] = entry

    return {"home": event["home_team"], "away": event["away_team"], "books": books}

# ========= Model (Poisson) =========
@dataclass
class LeagueTeamView:
    lam_home: float
    lam_away: float
    first_half_share: float

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

# ----- Build lambdas from historical team strengths -----
@dataclass
class TeamStatsView:
    lam_home: float
    lam_away: float
    first_half_share: float

def build_lambdas_from_history(home_team: str, away_team: str,
                               teams: Dict[str,'TeamStats'], league: 'LeagueStats') -> Optional[TeamStatsView]:
    if not teams:
        return None
    th = teams.get(_norm_team(home_team))
    ta = teams.get(_norm_team(away_team))
    if th is None or ta is None:
        # best word-overlap fallback
        def best_match(name: str) -> TeamStats:
            lo = _norm_team(name).split()
            return max(teams.items(), key=lambda kv: len(set(lo) & set(kv[0].split())))[1]
        if th is None: th = best_match(home_team)
        if ta is None: ta = best_match(away_team)

    lg_team_gf = (league.avg_home_goals + league.avg_away_goals) / 2.0
    lg_team_ga = lg_team_gf

    home_attack = th.gf / max(lg_team_gf, 1e-6)
    home_def_weak = th.ga / max(lg_team_ga, 1e-6)
    away_attack = ta.gf / max(lg_team_gf, 1e-6)
    away_def_weak = ta.ga / max(lg_team_ga, 1e-6)

    lam_h = league.avg_home_goals * home_attack * away_def_weak
    lam_a = league.avg_away_goals * away_attack * home_def_weak

    fh_share = 0.5*(th.first_half_share + ta.first_half_share)
    return TeamStatsView(lam_h, lam_a, fh_share)

# ----- Fit lambdas from market if no history -----
def fit_lambdas_from_market(p_home: float, p_draw: float, p_away: float,
                            p_over: Optional[float], line: Optional[float],
                            fh_share_default: float = 0.45) -> Optional[TeamStatsView]:
    if p_home is None or p_draw is None or p_away is None:
        return None
    if p_over is None or line is None:
        p_over, line = 0.50, 2.5

    best = None
    for k in np.linspace(0.4, 2.6, 45):  # lam_h = k * lam_a
        lam_a = 1.0
        lam_h = k
        s = 1.0
        for _ in range(20):
            M = prob_matrix(lam_h*s, lam_a*s)
            curr = over_prob(M, line)
            diff = p_over - curr
            s *= (1.0 + diff*0.6)
            s = max(0.2, min(5.0, s))
        lam_h_s, lam_a_s = lam_h*s, lam_a*s
        M = prob_matrix(lam_h_s, lam_a_s)
        ph, pd, pa = result_probs(M)
        err = abs(ph - p_home) + 0.5*abs(pd - p_draw)
        if (best is None) or (err < best[0]):
            best = (err, (lam_h_s, lam_a_s))
    if not best:
        return None
    lam_h_s, lam_a_s = best[1]
    return TeamStatsView(lam_h_s, lam_a_s, fh_share_default)

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

def unique_by_legs(items: Iterable[Tuple[str, float]]) -> List[Tuple[str, float]]:
    seen = set(); out=[]
    for name, p in items:
        key = tuple(sorted(name.split(" + ")))
        if key not in seen:
            seen.add(key); out.append((name, p))
    return out

def _format_coef(x: float) -> str:
    return f"{x:.2f}"

def pick_tier_by_range(candidates: List[Dict], low: float, high: float, key: str) -> Optional[Dict]:
    """Pick candidate whose key (market or fair odds) lies in [low, high]; else nearest."""
    if not candidates:
        return None
    in_range = [c for c in candidates if low <= c[key] <= high]
    if in_range:
        # best value among those in range
        return max(in_range, key=lambda c: c.get("value_ratio", 1.0))
    # nearest to target window center
    target = (low + high)/2.0
    return min(candidates, key=lambda c: abs(c[key]-target))

def select_parlays(book_name: str, book: Dict, mat: np.ndarray, line_pref=2.5) -> List[Dict]:
    """Return many combos with both model probability and book odds."""
    legs = []
    h2h = book.get("h2h"); totals = book.get("totals", {}); btts = book.get("btts")
    if h2h and all(h2h.get(k) for k in ("home","draw","away")):
        legs += [("result","home", h2h["home"]), ("result","draw", h2h["draw"]), ("result","away", h2h["away"])]
    if totals:
        # closest total to 2.5
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
            fair = round(1.0/max(mp,1e-9),2)
            market=1.0
            for p in pick: market *= p[2]
            combos.append({
                "book": book_name,
                "legs": pick, "size": r,
                "model_p": round(mp,3),
                "fair_odds": round(fair,2),
                "market_odds": round(market,2),
                "value_ratio": round(market/fair,3)
            })
    return combos

# ========= Core prediction =========
async def predict_once(league_key: str, home: str, away: str) -> str:
    if league_key not in LEAGUES:
        return "❌ Unknown league. Try: " + ", ".join(LEAGUES.keys())
    sport_key, fd_code = LEAGUES[league_key]

    have_any_data = False
    have_history = False
    lam_h = lam_a = None
    fh_share = 0.45

    async with aiohttp.ClientSession() as session:
        teams = league = None
        if fd_code:
            teams, league = await load_historical(session, fd_code)
            if teams:
                view = build_lambdas_from_history(home, away, teams, league)
                if view:
                    lam_h, lam_a, fh_share = view.lam_home, view.lam_away, view.first_half_share
                    have_history = True
                    have_any_data = True

        market = await fetch_market(session, sport_key, home, away, region=REGION)
        if market:
            have_any_data = True

    # If we still have no data at all, stop (don’t reuse generic priors)
    if not have_any_data:
        return ("❌ No data available for this match right now.\n"
                "• Live odds not found (or API quota).\n"
                "• Historical CSVs not available/blocked.\n"
                "Try a supported domestic league like: premier, laliga, seriea, bundesliga, ligue1.")

    # Extract market consensus for O/U & 1X2 if available
    target_over=None; line_sel=None
    pH=pD=pA=None
    best_book = None
    if market and market.get("books"):
        # choose book with most coverage
        best=None; cover=-1
        for name,b in market["books"].items():
            c= (1 if b.get("h2h") else 0) + (1 if b.get("totals") else 0) + (1 if b.get("btts") else 0)
            if c>cover: best=(name,b); cover=c
        best_book = best

        p_over_list=[]; lines=[]
        for b in market["books"].items():
            bname, bd = b
            for ln, kv in bd.get("totals", {}).items():
                if kv.get("over") and kv.get("under"):
                    lines.append(ln)
        if lines:
            line_sel = sorted(lines, key=lambda x: abs(x-2.5))[0]
            for _, bd in market["books"].items():
                kv=bd.get("totals", {}).get(line_sel)
                if kv and kv.get("over") and kv.get("under"):
                    po, pu = implied_prob(kv["over"]), implied_prob(kv["under"])
                    po, pu = remove_vig_two(po, pu)
                    p_over_list.append(po)
        if p_over_list:
            target_over=float(np.mean(p_over_list))

        h2h_probs=[]
        for _, bd in market["books"].items():
            h2h=bd.get("h2h")
            if h2h and all(h2h.get(k) for k in ("home","draw","away")):
                ph, pd, pa = implied_prob(h2h["home"]), implied_prob(h2h["draw"]), implied_prob(h2h["away"])
                ph,pd,pa = remove_vig_three(ph,pd,pa)
                h2h_probs.append((ph,pd,pa))
        if h2h_probs:
            arr=np.array(h2h_probs)
            pH,pD,pA = arr.mean(axis=0)

    # Calibrate/fill lambdas
    if have_history and target_over is not None and line_sel is not None:
        # scale totals to market line
        s = 1.0
        for _ in range(25):
            M = prob_matrix(lam_h*s, lam_a*s)
            curr = over_prob(M, line_sel)
            diff = target_over - curr
            s *= (1.0 + diff*0.6)
            s = max(0.2, min(5.0, s))
        lam_h, lam_a = lam_h*s, lam_a*s
    elif not have_history and (pH is not None):
        fitted = fit_lambdas_from_market(pH, pD, pA, target_over, line_sel, fh_share_default=0.45)
        if fitted:
            lam_h, lam_a, fh_share = fitted.lam_home, fitted.lam_away, fitted.first_half_share

    # If lambdas still None, as a last resort use priors BUT say so explicitly
    if lam_h is None or lam_a is None:
        lam_h, lam_a, fh_share = 1.45, 1.20, 0.45

    # Probability grid
    M = prob_matrix(lam_h, lam_a, MAX_GOALS)

    # Build response header (coefficients only, no percentages)
    header = []
    header.append(f"⚽️ Match: {home} vs {away}")
    if have_history and (pH is not None):
        header.append("📚 Source: History + Market")
    elif have_history:
        header.append("📚 Source: History (no live odds)")
    elif pH is not None:
        header.append("📚 Source: Market fit (no history)")
    else:
        header.append("📚 Source: Priors (no history/odds)")

    # Build candidates with market if possible
    tierA = tierB = tierC = None
    items_for_range: List[Dict] = []

    if best_book:
        all_parlays = select_parlays(best_book[0], best_book[1], M, line_pref=2.5)
        # choose key = market_odds for tiering; if missing, fallback to fair_odds
        for c in all_parlays:
            c["coef_for_tiering"] = c.get("market_odds") or c.get("fair_odds")
        items_for_range = [c for c in all_parlays if c["coef_for_tiering"] > 1.0]
        # pick by requested ranges
        tierA = pick_tier_by_range(items_for_range, 1.45, 2.20, key="coef_for_tiering")
        tierB = pick_tier_by_range(items_for_range, 3.0, 10.0, key="coef_for_tiering")
        tierC = pick_tier_by_range(items_for_range, 20.0, 200.0, key="coef_for_tiering")

    # If no odds/parlays available, build model-only combos & use fair odds for tiering
    if not (tierA and tierB and tierC):
        # Model-only legs
        base_legs = [
            ("result","home"), ("result","draw"), ("result","away"),
            ("ou","over",2.5), ("ou","under",2.5),
            ("btts",True), ("btts",False)
        ]
        from itertools import combinations
        candidates = []
        def mprob(legs):
            res=None; ou=None; bt=None
            for lg in legs:
                if lg[0]=="result": res=lg[1]
                elif lg[0]=="ou": ou=(lg[1], lg[2])
                elif lg[0]=="btts": bt=lg[1]
            return prob_of_combo(M, result=res, ou=ou, btts=bt)
        for r in (2,3):
            for pick in combinations(base_legs, r):
                tags=[p[0] for p in pick]
                if tags.count("result")>1: continue
                if ("ou","over",2.5) in pick and ("ou","under",2.5) in pick: continue
                if ("btts",True) in pick and ("btts",False) in pick: continue
                p = mprob(pick)
                if p<=0: continue
                name=[]
                for lg in pick:
                    if lg[0]=="result": name.append(lg[1].upper())
                    elif lg[0]=="ou": name.append(("Over" if lg[1]=="over" else "Under")+f" {lg[2]}")
                    elif lg[0]=="btts": name.append("BTTS Yes" if lg[1] else "BTTS No")
                fair = round(1.0/max(p,1e-9),2)
                candidates.append({
                    "book": "MODEL",
                    "legs_text": " + ".join(name),
                    "model_p": round(p,3),
                    "fair_odds": fair,
                    "market_odds": None,
                    "value_ratio": 1.0,
                    "coef_for_tiering": fair
                })
        # de-dup
        unique = {}
        for c in candidates:
            key = tuple(sorted(c["legs_text"].split(" + ")))
            if key not in unique:
                unique[key] = c
        items_for_range = list(unique.values())
        tierA = tierA or pick_tier_by_range(items_for_range, 1.45, 2.20, key="coef_for_tiering")
        tierB = tierB or pick_tier_by_range(items_for_range, 3.0, 10.0, key="coef_for_tiering")
        tierC = tierC or pick_tier_by_range(items_for_range, 20.0, 200.0, key="coef_for_tiering")

    # Format three tiers (always produce exactly three)
    tiers = [("🎯 Safer (1.45–2.20)", tierA), ("🎯 Medium (3–10)", tierB), ("🎯 Long shot (20–200)", tierC)]
    lines = []
    lines.extend(header)

    for label, t in tiers:
        if not t:
            lines.append(f"{label}: not available")
            continue
        coef = t.get("market_odds") or t.get("fair_odds")
        src = "book" if t.get("market_odds") else "model"
        legs = t["legs_text"] if "legs_text" in t else " + ".join([
            ("HOME" if l[1]=="home" else "DRAW" if l[1]=="draw" else "AWAY") if l[0]=="result"
            else (f"Over {l[1]}" if l[0]=="ou_over" else f"Under {l[1]}") if l[0].startswith("ou_")
            else ("BTTS Yes" if (l[1]=="yes" or l[1] is True) else "BTTS No")
            for l in t.get("legs", [])
        ])
        lines.append(f"{label}: {legs} — coef {coef:.2f} ({src})")

    return "\n".join(lines)

# ========= Telegram handlers =========
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "Hi! I turn match data into **coefficients** and pick 3 parlays:\n"
        "1) 1.45–2.20  2) 3–10  3) 20–200\n\n"
        "Use:\n"
        "/predict <league> ; <Home> ; <Away>\n"
        "Leagues: " + ", ".join(LEAGUES.keys()) + "\n"
        "Example: /predict premier ; Chelsea ; Tottenham"
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
        text = f"❌ Error: {e}"
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
