import os, io, re, math, unicodedata, logging, json, datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
from scipy.stats import poisson

import aiohttp
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, ContextTypes

# Images
from PIL import Image, ImageDraw, ImageFont

# ========= Runtime config (Railway: set in Variables) =========
BOT_TOKEN      = os.environ.get("BOT_TOKEN") or ""
ODDS_API_KEY   = os.environ.get("ODDS_API_KEY") or ""         # The Odds API (optional)
RAPIDAPI_KEY   = os.environ.get("RAPIDAPI_KEY") or ""         # RapidAPI key (optional)
RAPIDAPI_HOST  = os.environ.get("RAPIDAPI_HOST") or "api-football-v1.p.rapidapi.com"  # keep default

# ========= App constants =========
LEAGUES = {
    "laliga":      ("soccer_spain_la_liga",         "SP1"),
    "premier":     ("soccer_epl",                   "E0"),
    "seriea":      ("soccer_italy_serie_a",         "I1"),
    "bundesliga":  ("soccer_germany_bundesliga",    "D1"),
    "ligue1":      ("soccer_france_ligue_one",      "F1"),
    "eredivisie":  ("soccer_netherlands_eredivisie","N1"),
    "primeira":    ("soccer_portugal_primeira_liga","P1"),
    # Internationals/cups (odds-only if history missing)
    "worldcup":    ("soccer_fifa_world_cup",        None),
    "euro":        ("soccer_uefa_euro",             None),
    "ucl":         ("soccer_uefa_champs_league",    None),
    "uel":         ("soccer_uefa_europa_league",    None),
    "nations":     ("soccer_uefa_nations_league",   None),
    "friendlies":  ("soccer_international_friendly",None),
}
SEASONS_BACK = 3
REGIONS = "eu,uk,us"
MAX_GOALS = 10

# ========= Logging =========
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("parlay-bot")

# ========= Globals for /diag =========
LAST_DIAG = {
    "odds_primary": None,
    "odds_fallback": None,
    "fd_loaded": [],
    "fd_skipped": [],
    "book_used": None,
    "reason_no_odds": None,
}

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

# ========= football-data (history) =========
def season_codes_for_today(n_back: int) -> List[str]:
    today = datetime.datetime.utcnow()
    y = today.year % 100
    codes = [f"{y:02d}{(y+1)%100:02d}"]
    for i in range(1, n_back):
        a = (y - i) % 100
        b = (y - i + 1) % 100
        codes.append(f"{a:02d}{b:02d}")
    return codes

def _looks_like_html(b: bytes) -> bool:
    head = b[:200].lower()
    return b"<html" in head or b"<!doctype html" in head

async def fetch_fd_csv(session: aiohttp.ClientSession, code: str, season_code: str, diag: dict) -> Optional[pd.DataFrame]:
    url = f"http://www.football-data.co.uk/mmz4281/{season_code}/{code}.csv"
    try:
        async with session.get(url, timeout=30) as r:
            if r.status != 200:
                diag["fd_skipped"].append(f"{season_code}:{r.status}")
                return None
            content = await r.read()
    except Exception as e:
        diag["fd_skipped"].append(f"{season_code}:req-error")
        return None
    if not content or _looks_like_html(content):
        diag["fd_skipped"].append(f"{season_code}:html")
        return None

    needed = ["HomeTeam","AwayTeam","FTHG","FTAG","HTHG","HTAG"]
    for sep in (",",";"):
        try:
            df = pd.read_csv(
                io.BytesIO(content),
                sep=sep, engine="python",
                on_bad_lines="skip", encoding_errors="ignore"
            )
            df.columns = [re.sub(r"\s+", "", str(c)) for c in df.columns]
            if any(c not in df.columns for c in needed):
                continue
            df = df[needed].dropna(how="any")
            if df.empty:
                continue
            diag["fd_loaded"].append(season_code)
            return df
        except Exception:
            continue
    diag["fd_skipped"].append(f"{season_code}:parse")
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

async def load_historical(session: aiohttp.ClientSession, fd_code: str, diag: dict) -> Tuple[Dict[str,TeamStats], LeagueStats]:
    frames = []
    for sc in season_codes_for_today(SEASONS_BACK):
        df = await fetch_fd_csv(session, fd_code, sc, diag)
        if df is not None:
            frames.append(df)
    if not frames:
        return {}, LeagueStats(1.4,1.2,0.45)
    all_df = pd.concat(frames, ignore_index=True)
    return compute_team_league_stats(all_df)

# ========= Odds: The Odds API (primary) =========
async def fetch_market_oddsapi(session: aiohttp.ClientSession, sport_key: str, home: str, away: str) -> Optional[Dict]:
    if not ODDS_API_KEY:
        LAST_DIAG["odds_primary"] = "no-key"
        return None
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params_list = [
        {"regions": REGIONS, "markets": "h2h,totals,btts", "apiKey": ODDS_API_KEY, "oddsFormat": "decimal"},
        {"markets": "h2h,totals,btts", "apiKey": ODDS_API_KEY, "oddsFormat": "decimal"}, # fallback (no region)
    ]
    for params in params_list:
        try:
            async with session.get(url, params=params, timeout=25) as r:
                if r.status != 200:
                    LAST_DIAG["odds_primary"] = f"http-{r.status}"
                    continue
                data = await r.json()
        except Exception as e:
            LAST_DIAG["odds_primary"] = f"req-error"
            continue

        nh, na = _norm_team(home), _norm_team(away)
        event = None
        for ev in data:
            teams = [_norm_team(t) for t in ev.get("teams", [])]
            if nh in teams and na in teams:
                event = ev; break
        if not event:
            LAST_DIAG["odds_primary"] = "match-not-found"
            continue

        books = {}
        for bk in event.get("bookmakers", []):
            bname = bk["title"]
            entry = {"h2h": None, "totals": {}, "btts": None}
            for mk in bk.get("markets", []):
                key = mk.get("key"); oc = mk.get("outcomes", [])
                if key == "h2h" and len(oc) == 3:
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
                        try: line = float(o["point"])
                        except: continue
                        name = o["name"].lower(); price = o["price"]
                        lr = entry["totals"].get(line, {"over": None, "under": None})
                        if "over" in name: lr["over"] = price
                        else: lr["under"] = price
                        entry["totals"][line] = lr
                elif key == "btts" and len(oc) == 2:
                    opt = {o["name"].lower(): o["price"] for o in oc}
                    entry["btts"] = {"yes": opt.get("yes"), "no": opt.get("no")}
            books[bname] = entry
        LAST_DIAG["odds_primary"] = "ok"
        return {"home": event["home_team"], "away": event["away_team"], "books": books}
    return None

# ========= Odds: API-Football via RapidAPI (fallback) =========
async def apif_search_team(session, name: str) -> Optional[int]:
    if not RAPIDAPI_KEY: return None
    url = f"https://{RAPIDAPI_HOST}/v3/teams"
    headers = {"X-RapidAPI-Key": RAPIDAPI_KEY, "X-RapidAPI-Host": RAPIDAPI_HOST}
    params = {"search": name}
    try:
        async with session.get(url, headers=headers, params=params, timeout=25) as r:
            if r.status != 200: return None
            data = await r.json()
            res = data.get("response") or []
            if not res: return None
            return res[0]["team"]["id"]
    except: return None

async def apif_head_to_head(session, home_id: int, away_id: int) -> Optional[int]:
    url = f"https://{RAPIDAPI_HOST}/v3/fixtures/headtohead"
    headers = {"X-RapidAPI-Key": RAPIDAPI_KEY, "X-RapidAPI-Host": RAPIDAPI_HOST}
    params = {"h2h": f"{home_id}-{away_id}", "next": 1}
    try:
        async with session.get(url, headers=headers, params=params, timeout=25) as r:
            if r.status != 200: return None
            data = await r.json()
            resp = data.get("response") or []
            if not resp: return None
            return resp[0]["fixture"]["id"]
    except: return None

async def apif_odds_for_fixture(session, fixture_id: int) -> Optional[Dict]:
    # Odds endpoints may be plan-gated; we try H2H + Totals + BTTS if returned.
    url = f"https://{RAPIDAPI_HOST}/v3/odds"
    headers = {"X-RapidAPI-Key": RAPIDAPI_KEY, "X-RapidAPI-Host": RAPIDAPI_HOST}
    params = {"fixture": fixture_id}
    try:
        async with session.get(url, headers=headers, params=params, timeout=25) as r:
            if r.status != 200:
                return None
            data = await r.json()
            resp = data.get("response") or []
            if not resp: return None
            # API-Football returns multiple bookmakers/markets in a nested way
            books = {}
            for book in resp:
                bname = book.get("bookmaker", {}).get("name") or "APIFootball"
                entry = {"h2h": None, "totals": {}, "btts": None}
                for m in book.get("bets", []):
                    mk = m.get("name","").lower()
                    for val in m.get("values", []):
                        label = (val.get("value") or "").lower()
                        odd = float(val.get("odd") or 0)
                        if odd <= 1.0: continue
                        if mk in ("match winner","1x2"):
                            # labels often "Home", "Draw", "Away"
                            entry.setdefault("h2h", {"home":None,"draw":None,"away":None})
                            if "home" in label: entry["h2h"]["home"] = odd
                            elif "draw" in label or label=="x": entry["h2h"]["draw"] = odd
                            elif "away" in label: entry["h2h"]["away"] = odd
                        elif mk.startswith("over/under"): # varies by provider
                            # value looks like "Over 2.5" etc
                            m2 = re.search(r"([0-9]+(\.[0-9]+)?)", label)
                            if m2:
                                line = float(m2.group(1))
                                lr = entry["totals"].get(line, {"over": None, "under": None})
                                if "over" in label: lr["over"] = odd
                                elif "under" in label: lr["under"] = odd
                                entry["totals"][line] = lr
                        elif mk in ("both teams to score","btts"):
                            yn = {"yes":None,"no":None}
                            if "yes" in label: yn["yes"] = odd
                            elif "no" in label: yn["no"] = odd
                            entry["btts"] = {k:v for k,v in yn.items() if v}
                books[bname] = entry
            return {"books": books}
    except:
        return None

async def fetch_market_apifootball(session, home: str, away: str) -> Optional[Dict]:
    if not RAPIDAPI_KEY:
        LAST_DIAG["odds_fallback"] = "no-key"
        return None
    hid = await apif_search_team(session, home)
    aid = await apif_search_team(session, away)
    if not hid or not aid:
        LAST_DIAG["odds_fallback"] = "team-search-failed"
        return None
    fx = await apif_head_to_head(session, hid, aid)
    if not fx:
        LAST_DIAG["odds_fallback"] = "h2h-fixture-not-found"
        return None
    data = await apif_odds_for_fixture(session, fx)
    if not data:
        LAST_DIAG["odds_fallback"] = "odds-endpoint-unavailable"
        return None
    LAST_DIAG["odds_fallback"] = "ok"
    return {"home": home, "away": away, "books": data["books"]}

# ========= Model (Poisson) & utilities =========
@dataclass
class View:
    lam_home: float
    lam_away: float
    fh_share: float

def prob_matrix(lh: float, la: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    home = [poisson.pmf(i, lh) for i in range(max_goals+1)]
    away = [poisson.pmf(j, la) for j in range(max_goals+1)]
    return np.outer(home, away)

def result_probs(mat: np.ndarray) -> Tuple[float,float,float]:
    h=d=a=0.0; g=mat.shape[0]-1
    for i in range(g+1):
        for j in range(g+1):
            if i>j: h+=mat[i,j]
            elif i==j: d+=mat[i,j]
            else: a+=mat[i,j]
    return h,d,a

def over_prob(mat: np.ndarray, line: float) -> float:
    g=mat.shape[0]-1; s=0.0
    for i in range(g+1):
        for j in range(g+1):
            if i+j>line: s+=mat[i,j]
    return s

def btts_prob(mat: np.ndarray) -> float:
    g=mat.shape[0]-1; s=0.0
    for i in range(1,g+1):
        for j in range(1,g+1):
            s+=mat[i,j]
    return s

def build_from_history(home: str, away: str, teams: Dict[str,TeamStats], league: LeagueStats) -> Optional[View]:
    if not teams: return None
    def tl(name):
        k = _norm_team(name)
        if k in teams: return teams[k]
        lo = k.split()
        return max(teams.items(), key=lambda kv: len(set(lo)&set(kv[0].split())))[1]
    th = tl(home); ta = tl(away)
    lg = (league.avg_home_goals + league.avg_away_goals) / 2.0
    home_attack = th.gf / max(lg,1e-6); home_def_weak = th.ga / max(lg,1e-6)
    away_attack = ta.gf / max(lg,1e-6); away_def_weak = ta.ga / max(lg,1e-6)
    lam_h = league.avg_home_goals * home_attack * away_def_weak
    lam_a = league.avg_away_goals * away_attack * home_def_weak
    fh    = 0.5*(th.first_half_share + ta.first_half_share)
    return View(lam_h, lam_a, fh)

def fit_from_market(pH: float, pD: float, pA: float, p_over: Optional[float], line: Optional[float], fh_default=0.45) -> Optional[View]:
    if pH is None or pD is None or pA is None:
        return None
    if p_over is None or line is None: p_over, line = 0.50, 2.5
    best=None
    for k in np.linspace(0.4,2.6,45):
        lam_a=1.0; lam_h=k; s=1.0
        for _ in range(20):
            M=prob_matrix(lam_h*s, lam_a*s); curr=over_prob(M, line)
            s*= (1.0 + (p_over-curr)*0.6); s=max(0.2, min(5.0, s))
        lh,la=lam_h*s, lam_a*s
        ph,pd,pa = result_probs(prob_matrix(lh,la))
        err = abs(ph-pH) + 0.5*abs(pd-pD)
        if (best is None) or (err<best[0]): best=(err,(lh,la))
    lh,la = best[1]
    return View(lh,la,fh_default)

# ========= Parlays & selection =========
def prob_of_combo(mat: np.ndarray, result=None, ou=None, btts=None) -> float:
    g=mat.shape[0]-1; s=0.0
    for i in range(g+1):
        for j in range(g+1):
            ok=True
            if result:
                if (result=="home" and not (i>j)) or (result=="draw" and not (i==j)) or (result=="away" and not (i<j)): ok=False
            if ok and ou:
                side, line = ou
                if (side=="over" and not (i+j>line)) or (side=="under" and not (i+j<line)): ok=False
            if ok and (btts is not None):
                has=(i>0 and j>0)
                if btts != has: ok=False
            if ok: s+=mat[i,j]
    return s

def combos_from_book(book_name: str, book: Dict, mat: np.ndarray, line_pref=2.5) -> List[Dict]:
    legs=[]
    h2h=book.get("h2h"); totals=book.get("totals",{}); btts=book.get("btts")
    if h2h and all(h2h.get(k) for k in ("home","draw","away")):
        legs += [("result","home",h2h["home"]),("result","draw",h2h["draw"]),("result","away",h2h["away"])]
    if totals:
        best=None
        for ln, kv in totals.items():
            if kv.get("over") and kv.get("under"):
                cand=(ln,kv["over"],kv["under"])
                if best is None or abs(ln-line_pref) < abs(best[0]-line_pref):
                    best=cand
        if best:
            ln,oo,uu=best
            legs += [("ou_over",ln,oo),("ou_under",ln,uu)]
    if btts and btts.get("yes") and btts.get("no"):
        legs += [("btts","yes",btts["yes"]),("btts","no",btts["no"])]

    from itertools import combinations
    out=[]
    def model_p(pick):
        res=None; ou=None; bt=None
        for t in pick:
            if t[0]=="result": res=t[1]
            elif t[0].startswith("ou_"): ou=("over" if t[0]=="ou_over" else "under", t[1])
            elif t[0]=="btts": bt=(t[1]=="yes")
        return prob_of_combo(mat, result=res, ou=ou, btts=bt)

    valid=list(legs)
    for r in (2,3):
        for pick in combinations(valid, r):
            tags=[p[0] for p in pick]
            if tags.count("result")>1: continue
            if "ou_over" in tags and "ou_under" in tags: continue
            btags=[t for t in tags if t=="btts"]
            if len(btags)>1:
                vs=[p for p in pick if p[0]=="btts"]
                if len(vs)>1 and vs[0][1]!=vs[1][1]: continue
            mp=model_p(pick)
            if mp<=0: continue
            f=round(1.0/max(mp,1e-9),2)
            market=1.0
            for p in pick: market*=p[2]
            out.append({
                "book": book_name, "legs": pick, "size": r,
                "model_p": round(mp,3), "fair_odds": round(f,2),
                "market_odds": round(market,2), "coef_for_tiering": round(market,2),
                "value_ratio": round(market/f,3),
            })
    return out

def combos_model_only(mat: np.ndarray) -> List[Dict]:
    base = [("result","home"),("result","draw"),("result","away"),
            ("ou","over",2.5),("ou","under",2.5),("btts",True),("btts",False)]
    from itertools import combinations
    out=[]
    def prob(pick):
        res=None; ou=None; bt=None
        for t in pick:
            if t[0]=="result": res=t[1]
            elif t[0]=="ou": ou=(t[1],t[2])
            elif t[0]=="btts": bt=t[1]
        return prob_of_combo(mat, result=res, ou=ou, btts=bt)
    for r in (2,3):
        for pick in combinations(base, r):
            tags=[p[0] for p in pick]
            if tags.count("result")>1: continue
            if ("ou","over",2.5) in pick and ("ou","under",2.5) in pick: continue
            if ("btts",True) in pick and ("btts",False) in pick: continue
            p=prob(pick)
            if p<=0: continue
            f=round(1.0/max(p,1e-9),2)
            name=[]
            for t in pick:
                if t[0]=="result": name.append(t[1].upper())
                elif t[0]=="ou": name.append(("Over" if t[1]=="over" else "Under")+f" {t[2]}")
                elif t[0]=="btts": name.append("BTTS Yes" if t[1] else "BTTS No")
            out.append({
                "book": "MODEL", "legs_text": " + ".join(name), "fair_odds": f,
                "market_odds": None, "coef_for_tiering": f, "model_p": round(p,3),
                "value_ratio": 1.0, "size": r, "legs": pick
            })
    # de-dup by legs
    uniq={}
    for c in out:
        key=tuple(sorted(c.get("legs_text"," + ".join([] if not c.get("legs") else [ ("HOME" if l[1]=="home" else "DRAW" if l[1]=="draw" else "AWAY") if l[0]=="result" else (f"Over {l[1]}" if l[0].startswith("ou_") else ("BTTS Yes" if (l[1]==True) else "BTTS No")) for l in c["legs"] ])).split(" + ")))
        if key not in uniq: uniq[key]=c
    return list(uniq.values())

def pick_by_range(cands: List[Dict], low: float, high: float) -> Optional[Dict]:
    inside=[c for c in cands if low<=c["coef_for_tiering"]<=high]
    if inside:
        # prefer highest value_ratio then closest to center
        center=(low+high)/2
        inside.sort(key=lambda x: (x.get("value_ratio",1.0), -abs(x["coef_for_tiering"]-center)), reverse=True)
        return inside[0]
    # else nearest to window center
    if not cands: return None
    center=(low+high)/2
    return min(cands, key=lambda x: abs(x["coef_for_tiering"]-center))

def legs_to_text(legs: Tuple) -> str:
    out=[]
    for l in legs:
        if l[0]=="result":
            out.append(l[1].upper())
        elif l[0].startswith("ou_"):
            out.append(("Over" if l[0]=="ou_over" else "Under")+f" {l[1]}")
        elif l[0]=="btts":
            out.append("BTTS Yes" if (l[1]=="yes") else "BTTS No")
    return " + ".join(out)

# ========= Card image (Pillow) =========
def render_card(match: str, source: str, tiers: List[Tuple[str,str,float,str]], insights: List[str]) -> bytes:
    # tiers: [(label, legs, coef, src), ...]
    W,H = 1000, 640
    img = Image.new("RGB", (W,H), (14,18,26))
    draw = ImageDraw.Draw(img)
    try:
        font_title = ImageFont.truetype("DejaVuSans.ttf", 42)
        font_sub   = ImageFont.truetype("DejaVuSans.ttf", 28)
        font_body  = ImageFont.truetype("DejaVuSans.ttf", 26)
    except:
        font_title = ImageFont.load_default()
        font_sub   = ImageFont.load_default()
        font_body  = ImageFont.load_default()

    # header
    draw.text((30,30), f"⚽ {match}", fill=(235,241,255), font=font_title)
    draw.text((30,84), f"Source: {source}", fill=(160,175,200), font=font_sub)

    # tiers boxes
    box_y = 140
    for i,(label, legs, coef, src) in enumerate(tiers):
        y = box_y + i*140
        draw.rounded_rectangle((30,y,970,y+110), radius=18, fill=(28,36,48))
        draw.text((50,y+18), label, fill=(255,255,255), font=font_sub)
        draw.text((50,y+60), legs, fill=(205,215,230), font=font_body)
        draw.text((800,y+18), f"coef {coef:.2f}", fill=(255,219,99), font=font_sub)
        draw.text((800,y+60), f"{src}", fill=(150,170,200), font=font_body)

    # insights
    draw.text((30, H-160), "💡 Insights:", fill=(255,255,255), font=font_sub)
    for j, line in enumerate(insights[:3]):
        draw.text((50, H-120 + j*34), f"• {line}", fill=(200,210,230), font=font_body)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

# ========= Core flow =========
async def predict_flow(league_key: str, home: str, away: str) -> Tuple[str, bytes]:
    LAST_DIAG.update({"fd_loaded": [], "fd_skipped": [], "book_used": None, "reason_no_odds": None})

    if league_key not in LEAGUES:
        text = "❌ Unknown league. Try: " + ", ".join(LEAGUES.keys())
        return text, None
    sport_key, fd_code = LEAGUES[league_key]

    headers = {"User-Agent": "Mozilla/5.0 (compatible; ParlayBot/1.0)"}
    async with aiohttp.ClientSession(headers=headers) as session:
        teams=league=None
        if fd_code:
            teams, league = await load_historical(session, fd_code, LAST_DIAG)

        # Try odds primary ➜ fallback
        market = await fetch_market_oddsapi(session, sport_key, home, away)
        if not market:
            fb = await fetch_market_apifootball(session, home, away)
            market = fb if fb else None

    # Build lambdas
    view=None
    if teams:
        view = build_from_history(home, away, teams, league)

    # Market consensus
    target_over=None; line_sel=None; pH=pD=pA=None; best_book=None
    if market and market.get("books"):
        # choose book with most markets
        best=None; cover=-1
        for name,b in market["books"].items():
            c=(1 if b.get("h2h") else 0)+(1 if b.get("totals") else 0)+(1 if b.get("btts") else 0)
            if c>cover: best=(name,b); cover=c
        best_book = best
        LAST_DIAG["book_used"] = best[0] if best else None

        p_over_list=[]; lines=[]
        for _, bd in market["books"].items():
            for ln, kv in bd.get("totals", {}).items():
                if kv.get("over") and kv.get("under"): lines.append(ln)
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
            arr=np.array(h2h_probs); pH,pD,pA = arr.mean(axis=0)

    # Use history; calibrate to totals if available; else fit from market; else priors
    if view and (target_over is not None) and (line_sel is not None):
        # calibrate totals
        s=1.0
        for _ in range(25):
            M=prob_matrix(view.lam_home*s, view.lam_away*s)
            curr=over_prob(M, line_sel)
            s *= (1.0 + (target_over-curr)*0.6); s=max(0.2, min(5.0, s))
        lam_h, lam_a, fh = view.lam_home*s, view.lam_away*s, view.fh_share
    elif (not view) and (pH is not None):
        fitted = fit_from_market(pH, pD, pA, target_over, line_sel, fh_default=0.45)
        if fitted: lam_h, lam_a, fh = fitted.lam_home, fitted.lam_away, fitted.fh_share
        else: lam_h, lam_a, fh = 1.45, 1.20, 0.45
    elif view:
        lam_h, lam_a, fh = view.lam_home, view.lam_away, view.fh_share
    else:
        # absolute last resort
        lam_h, lam_a, fh = 1.45, 1.20, 0.45

    # Probability grid
    M = prob_matrix(lam_h, lam_a)

    # Build candidate combos
    book_combos = []
    if best_book:
        book_combos = combos_from_book(best_book[0], best_book[1], M, line_pref=2.5)

    if book_combos:
        pool = book_combos
    else:
        pool = combos_model_only(M)
        LAST_DIAG["reason_no_odds"] = LAST_DIAG.get("odds_primary") or LAST_DIAG.get("odds_fallback") or "no-odds"

    # Pick requested coef ranges
    A = pick_by_range(pool, 1.45, 2.20)
    B = pick_by_range(pool, 3.00, 10.00)
    C = pick_by_range(pool, 20.00, 200.00)

    # Ensure text/legs display
    def pretty(c):
        if "legs_text" in c:
            legs_txt = c["legs_text"]
        else:
            legs_txt = legs_to_text(c["legs"])
        coef = c.get("market_odds") or c.get("fair_odds")
        src  = c["book"] if c.get("market_odds") else "model"
        return legs_txt, coef, src

    tiers = []
    for label, c in [("🎯 Safer 1.45–2.20", A), ("🎯 Medium 3–10", B), ("🎯 Long shot 20–200", C)]:
        if c:
            legs_txt, coef, src = pretty(c)
            tiers.append( (label, legs_txt, float(coef), src) )
        else:
            tiers.append( (label, "not available", float("nan"), "n/a") )

    # Insights (Sofascore-style)
    exp_goals = lam_h + lam_a
    strength = "Home edge" if lam_h>lam_a else "Away edge" if lam_a>lam_h else "Balanced"
    first_half = f"{int(round(fh*100))}% of goals expected in 1H"
    why = [
        f"Expected goals total around {exp_goals:.2f}",
        f"Relative strength: {strength}",
        first_half
    ]
    # If market was used, add a line
    if pH is not None and target_over is not None and line_sel is not None:
        why.insert(0, f"Market anchor: 1X2 + O/U {line_sel}")

    # Compose text
    source_txt = "History+Market" if view and (pH is not None) else "History" if view else "Market" if (pH is not None) else "Priors"
    header = [
        f"⚽️ Match: {home} vs {away}",
        f"📚 Source: {source_txt}",
        "🎛️ Three parlays (decimal odds):",
        f"1) {tiers[0][1]} — coef {tiers[0][2]:.2f} ({tiers[0][3]})",
        f"2) {tiers[1][1]} — coef {tiers[1][2]:.2f} ({tiers[1][3]})",
        f"3) {tiers[2][1]} — coef {tiers[2][2]:.2f} ({tiers[2][3]})",
        "💡 Why:",
        f"• {why[0]}",
        f"• {why[1]}",
        f"• {why[2]}",
    ]
    if LAST_DIAG.get("reason_no_odds"):
        header.append(f"ℹ️ No live odds used: {LAST_DIAG['reason_no_odds']}")

    text = "\n".join(header)

    # Card image
    card_bytes = render_card(f"{home} vs {away}", source_txt,
                             [(t[0], t[1], t[2] if t[1]!="not available" else 0.0, t[3]) for t in tiers],
                             why)
    return text, card_bytes

# ========= Telegram handlers =========
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "Hi! I turn match data into **coefficients** and pick 3 parlays:\n"
        "1) 1.45–2.20  2) 3–10  3) 20–200\n\n"
        "Use:\n"
        "/predict <league> ; <Home> ; <Away>\n"
        "Leagues: " + ", ".join(LEAGUES.keys()) + "\n"
        "Example: /predict premier ; Chelsea ; Tottenham\n\n"
        "Extras: /quick (manual odds), /diag (status)"
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
        text, card = await predict_flow(league, home, away)
    except Exception as e:
        log.exception("Prediction error")
        await update.message.reply_text(f"❌ Error: {e}")
        return

    if card:
        await update.message.reply_photo(InputFile(io.BytesIO(card), filename="insights.png"), caption=text)
    else:
        await update.message.reply_text(text)

async def diag_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    info = {
        "BOT_TOKEN_set": bool(BOT_TOKEN),
        "ODDS_API_KEY_set": bool(ODDS_API_KEY),
        "RAPIDAPI_KEY_set": bool(RAPIDAPI_KEY),
        "primary_odds": LAST_DIAG.get("odds_primary"),
        "fallback_odds": LAST_DIAG.get("odds_fallback"),
        "fd_loaded": LAST_DIAG.get("fd_loaded"),
        "fd_skipped": LAST_DIAG.get("fd_skipped"),
        "book_used": LAST_DIAG.get("book_used"),
        "no_odds_reason": LAST_DIAG.get("reason_no_odds"),
    }
    await update.message.reply_text("🔎 Diagnostics:\n" + json.dumps(info, indent=2))

async def quick_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Manual odds input (works even if all APIs fail).
    Format:
    /quick <league> ; <Home> ; <Away> ; h2h:H/D/A ; ou:LINE O/U ; btts:Y/N
    Example:
    /quick premier ; Chelsea ; Tottenham ; h2h:2.10/3.40/3.50 ; ou:2.5 1.95/1.85 ; btts:1.80/2.00
    """
    try:
        payload = update.message.text.split(" ", 1)[1]
        parts = [p.strip() for p in payload.split(";")]
        league, home, away = parts[0], parts[1], parts[2]
        h2h_part = [p for p in parts if p.strip().lower().startswith("h2h:")]
        ou_part  = [p for p in parts if p.strip().lower().startswith("ou:")]
        btts_part= [p for p in parts if p.strip().lower().startswith("btts:")]
        H=D=A=None; line=None; over=None; under=None; by=None; bn=None
        if h2h_part:
            nums = h2h_part[0].split(":")[1].strip().split("/")
            H,D,A = float(nums[0]), float(nums[1]), float(nums[2])
        if ou_part:
            rest = ou_part[0].split(":")[1].strip().split()
            line = float(rest[0]); ous = rest[1].split("/")
            over, under = float(ous[0]), float(ous[1])
        if btts_part:
            by, bn = [float(x) for x in btts_part[0].split(":")[1].strip().split("/")]
    except Exception:
        await update.message.reply_text(
            "Usage:\n/quick <league> ; <Home> ; <Away> ; h2h:H/D/A ; ou:LINE O/U ; btts:Y/N")
        return

    # Build a tiny synthetic market dict from manual odds
    books = {"Manual": {}}
    if all([H,D,A]): books["Manual"]["h2h"] = {"home":H,"draw":D,"away":A}
    if all([line,over,under]): books["Manual"].setdefault("totals", {})[line] = {"over":over,"under":under}
    if all([by,bn]): books["Manual"]["btts"] = {"yes":by,"no":bn}
    market = {"books": books}

    # History (best effort)
    sport_key, fd_code = LEAGUES.get(league, (None,None))
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ParlayBot/1.0)"}
    async with aiohttp.ClientSession(headers=headers) as session:
        teams=league_stats=None
        if fd_code:
            teams, league_stats = await load_historical(session, fd_code, LAST_DIAG)

    # Lambdas
    view=None
    if teams:
        view = build_from_history(home, away, teams, league_stats)

    # Market consensus from manual odds
    pH=pD=pA=None; target_over=None; line_sel=None
    h2h=books["Manual"].get("h2h")
    if h2h:
        ph,pd,pa = implied_prob(h2h["home"]), implied_prob(h2h["draw"]), implied_prob(h2h["away"])
        pH,pD,pA = remove_vig_three(ph,pd,pa)
    tot=books["Manual"].get("totals",{}); 
    if tot:
        line_sel = list(tot.keys())[0]
        kv = tot[line_sel]
        po,pu = implied_prob(kv["over"]), implied_prob(kv["under"])
        po,pu = remove_vig_two(po,pu)
        target_over = po

    if view and (target_over is not None):
        s=1.0
        for _ in range(25):
            M=prob_matrix(view.lam_home*s, view.lam_away*s)
            curr=over_prob(M, line_sel)
            s *= (1.0 + (target_over-curr)*0.6); s=max(0.2, min(5.0, s))
        lam_h, lam_a, fh = view.lam_home*s, view.lam_away*s, view.fh_share
    elif (not view) and (pH is not None):
        fitted = fit_from_market(pH, pD, pA, target_over, line_sel, fh_default=0.45)
        if fitted: lam_h, lam_a, fh = fitted.lam_home, fitted.lam_away, fitted.fh_share
        else: lam_h, lam_a, fh = 1.45, 1.20, 0.45
    elif view:
        lam_h, lam_a, fh = view.lam_home, view.lam_away, view.fh_share
    else:
        lam_h, lam_a, fh = 1.45, 1.20, 0.45

    M = prob_matrix(lam_h, lam_a)
    pool = combos_from_book("Manual", books["Manual"], M) if books["Manual"] else combos_model_only(M)
    A = pick_by_range(pool, 1.45, 2.20)
    B = pick_by_range(pool, 3.00, 10.00)
    C = pick_by_range(pool, 20.00, 200.00)
    def pretty(c):
        if "legs_text" in c: legs=c["legs_text"]
        else: legs=legs_to_text(c["legs"])
        coef = c.get("market_odds") or c.get("fair_odds")
        src = c["book"] if c.get("market_odds") else "model"
        return legs, coef, src
    tiers=[]
    for label, c in [("🎯 Safer 1.45–2.20", A), ("🎯 Medium 3–10", B), ("🎯 Long shot 20–200", C)]:
        legs, coef, src = pretty(c) if c else ("not available", float("nan"), "n/a")
        tiers.append((label, legs, float(coef) if coef else 0.0, src))
    why=[f"Manual odds used. Expected goals ≈ {(lam_h+lam_a):.2f}", f"Home λ={lam_h:.2f}, Away λ={lam_a:.2f}", f"First-half share ≈ {int(round(fh*100))}%"]
    text = (f"⚽️ Match: {home} vs {away}\n📚 Source: {'History+Manual' if view else 'Manual'}\n"
            f"1) {tiers[0][1]} — coef {tiers[0][2]:.2f} ({tiers[0][3]})\n"
            f"2) {tiers[1][1]} — coef {tiers[1][2]:.2f} ({tiers[1][3]})\n"
            f"3) {tiers[2][1]} — coef {tiers[2][2]:.2f} ({tiers[2][3]})\n"
            f"💡 Why:\n• {why[0]}\n• {why[1]}\n• {why[2]}")
    card = render_card(f"{home} vs {away}", "Manual", tiers, why)
    await update.message.reply_photo(InputFile(io.BytesIO(card), filename="insights.png"), caption=text)

# ========= Startup =========
async def on_startup(app):
    log.info("Clearing webhook and starting polling…")
    try:
        await app.bot.delete_webhook(drop_pending_updates=True)
    except Exception:
        pass
    log.info("Webhook cleared.")

def main():
    if not BOT_TOKEN:
        raise RuntimeError("Missing BOT_TOKEN")
    app = Application.builder().token(BOT_TOKEN).post_init(on_startup).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("predict", predict_cmd))
    app.add_handler(CommandHandler("diag", diag_cmd))
    app.add_handler(CommandHandler("quick", quick_cmd))
    log.info("Running long polling…")
    app.run_polling()

if __name__ == "__main__":
    main()
