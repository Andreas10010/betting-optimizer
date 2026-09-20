"""Match results *and* bookmaker odds, straight from football-data.co.uk.

The mirror behind ``data_loader`` dropped every odds column, so it cannot
answer the only question that matters for betting: did the strategy make
money. This module fetches the upstream files instead, one CSV per league
and season, and returns a superset of the mirror's schema.

Season files are cached under ``data/football-data/`` (gitignored). Finished
seasons never change, so they are fetched once; the ongoing season is
refreshed when the cached copy is older than ``CURRENT_SEASON_TTL``.
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urljoin, urlsplit
import csv
import datetime as dt
import io
import re
import warnings

import pandas as pd
import requests

BASE_URL = "https://football-data.co.uk/mmz4281"
FIRST_SEASON_START = 2000
SEASON_START_MONTH = 7  # A season beginning in August belongs to that year.
CURRENT_SEASON_TTL = dt.timedelta(hours=12)
TIMEOUT = (10, 60)
MAX_REDIRECTS = 5
# www redirects to the apex domain, so redirects must be followed -- but only to
# these hosts. In September 2026 the apex started answering with
# "302 -> http://127.0.0.1/<path>", which a client following redirects blindly
# turns into a request against its own loopback interface.
ALLOWED_HOSTS = {"football-data.co.uk", "www.football-data.co.uk"}

# The mirror's league names, so the two sources stay interchangeable.
DIVISIONS = {
    "E0": "Premier League",
    "D1": "Bundesliga",
    "SP1": "La Liga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}

RESULT_COLUMNS = {
    "Date": "MatchDate",
    "HomeTeam": "HomeTeam",
    "AwayTeam": "AwayTeam",
    "FTHG": "FullTimeHomeGoals",
    "FTAG": "FullTimeAwayGoals",
    "FTR": "FullTimeResult",
    "HTHG": "HalfTimeHomeGoals",
    "HTAG": "HalfTimeAwayGoals",
    "HTR": "HalfTimeResult",
    "HS": "HomeShots",
    "AS": "AwayShots",
    "HST": "HomeShotsOnTarget",
    "AST": "AwayShotsOnTarget",
    "HC": "HomeCorners",
    "AC": "AwayCorners",
    "HF": "HomeFouls",
    "AF": "AwayFouls",
    "HY": "HomeYellowCards",
    "AY": "AwayYellowCards",
    "HR": "HomeRedCards",
    "AR": "AwayRedCards",
    "Time": "KickoffTime",
    "Referee": "Referee",
}

# Upstream renamed its market aggregates in 2019/20 (BbAvH -> AvgH). Mapping the
# old names onto the new ones keeps Avg/Max continuous back to 2005/06.
LEGACY_ODDS_COLUMNS = {
    "BbAvH": "AvgH", "BbAvD": "AvgD", "BbAvA": "AvgA",
    "BbMxH": "MaxH", "BbMxD": "MaxD", "BbMxA": "MaxA",
    "BbAv>2.5": "Avg>2.5", "BbAv<2.5": "Avg<2.5",
    "BbMx>2.5": "Max>2.5", "BbMx<2.5": "Max<2.5",
    "BbAHh": "AHh", "BbAvAHH": "AvgAHH", "BbAvAHA": "AvgAHA",
    "BbMxAHH": "MaxAHH", "BbMxAHA": "MaxAHA",
}

NUMERIC_RESULT_COLUMNS = [name for name in RESULT_COLUMNS.values()
                          if name not in ("MatchDate", "HomeTeam", "AwayTeam",
                                          "FullTimeResult", "HalfTimeResult",
                                          "KickoffTime", "Referee")]

# Named 1X2 books in football-data files. Aggregates (Avg/Max/BbAv/BbMx) are
# kept as columns but are not betting candidates. Betsson is not in this source.
BOOKMAKER_NAMES = {
    "B365": "Bet365",
    "BW": "Bet&Win",
    "GB": "Gamebookers",
    "IW": "Interwetten",
    "LB": "Ladbrokes",
    "PS": "Pinnacle",
    "WH": "William Hill",
    "SJ": "Stan James",
    "VC": "VC Bet",
    "BS": "Blue Square",
    "SO": "Sporting Odds",
    "SB": "Sportingbet",
    "SY": "Stanleybet",
    "BF": "Betfair",
    "BFD": "Betfair Sportsbook",
    "BFE": "Betfair Exchange",
    "BMGM": "BetMGM",
    "BV": "BetVictor",
    "CL": "Coral",
    "PP": "Paddy Power",
    "SKB": "Sky Bet",
    "1XB": "1xBet",
}
AGGREGATE_BOOKS = {"Avg", "Max", "BbAv", "BbMx"}
KNOWN_BOOKS = tuple(sorted({*BOOKMAKER_NAMES, *AGGREGATE_BOOKS}, key=len, reverse=True))
_1X2_CLOSING = re.compile(r"^(.+)C([HDA])$")
_1X2_OPENING = re.compile(r"^(.+)([HDA])$")

# Kept verbatim. "C" marks a closing price; the rest are opening prices.
ODDS_COLUMNS = [
    "B365H", "B365D", "B365A", "B365CH", "B365CD", "B365CA",
    "PSH", "PSD", "PSA", "PSCH", "PSCD", "PSCA",
    "AvgH", "AvgD", "AvgA", "AvgCH", "AvgCD", "AvgCA",
    "MaxH", "MaxD", "MaxA", "MaxCH", "MaxCD", "MaxCA",
    "B365>2.5", "B365<2.5", "B365C>2.5", "B365C<2.5",
    "Avg>2.5", "Avg<2.5", "AvgC>2.5", "AvgC<2.5",
    "Max>2.5", "Max<2.5", "MaxC>2.5", "MaxC<2.5",
    "AHh", "AHCh", "AvgAHH", "AvgAHA", "AvgCAHH", "AvgCAHA",
    "MaxAHH", "MaxAHA", "MaxCAHH", "MaxCAHA",
]


def parse_1x2_column(name: str):
    """Return ``(book_code, outcome, is_closing)`` for a 1X2 odds column.

    Known book codes are matched longest-first so ``VCH`` is VC Bet home,
    not a phantom closing line for a book called ``V``.
    """
    if any(token in name for token in (">", "<", "AH")):
        return None
    for book in KNOWN_BOOKS:
        for outcome in "HDA":
            if name == f"{book}C{outcome}":
                return book, outcome, True
            if name == f"{book}{outcome}":
                return book, outcome, False
    closing = _1X2_CLOSING.fullmatch(name)
    if closing:
        return closing.group(1), closing.group(2), True
    opening = _1X2_OPENING.fullmatch(name)
    if opening:
        return opening.group(1), opening.group(2), False
    return None


def bookmaker_label(code: str) -> str:
    return BOOKMAKER_NAMES.get(code, code)


def opening_book_codes(columns) -> list[str]:
    """Named books with at least one opening 1X2 column. Aggregates excluded."""
    codes = []
    seen = set()
    for name in columns:
        parsed = parse_1x2_column(str(name))
        if parsed is None:
            continue
        book, _outcome, closing = parsed
        if closing or book in AGGREGATE_BOOKS or book in seen:
            continue
        seen.add(book)
        codes.append(book)
    return codes


def best_named_opening(frame: pd.DataFrame, outcomes=("H", "D", "A")) -> pd.DataFrame:
    """Highest named-book *opening* price per row and 1X2 outcome.

    Aggregates (``Max``/``Avg``) are skipped. Ties go to the first book in
    ``opening_book_codes`` order. Closing columns are ignored.
    """
    books = opening_book_codes(frame.columns)
    parts = {}
    for outcome in outcomes:
        columns = [f"{book}{outcome}" for book in books if f"{book}{outcome}" in frame.columns]
        if not columns:
            parts[f"best_price_{outcome}"] = pd.Series(float("nan"), index=frame.index)
            parts[f"best_book_{outcome}"] = pd.Series(pd.NA, index=frame.index, dtype="object")
            continue
        odds = frame[columns].apply(pd.to_numeric, errors="coerce")
        odds = odds.where(odds > 1)
        price = odds.max(axis=1, skipna=True)
        winner = odds.idxmax(axis=1).where(odds.notna().any(axis=1))
        rename = {f"{book}{outcome}": book for book in books}
        parts[f"best_price_{outcome}"] = price
        parts[f"best_book_{outcome}"] = winner.map(rename)
    return pd.DataFrame(parts, index=frame.index)


def current_season_start(today: dt.date | None = None) -> int:
    today = today or dt.date.today()
    return today.year - int(today.month <= SEASON_START_MONTH)


def season_code(start_year: int) -> str:
    """2025 -> '2526', the four digits upstream uses in its paths."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    """2025 -> '2025/26', matching the mirror's Season column."""
    return f"{start_year}/{(start_year + 1) % 100:02d}"


def season_url(start_year: int, division: str) -> str:
    return f"{BASE_URL}/{season_code(start_year)}/{division}.csv"


def cache_root(root: str | Path | None = None) -> Path:
    base = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    return base / "data" / "football-data"


def _is_stale(path: Path, start_year: int, today: dt.date) -> bool:
    if start_year < current_season_start(today):
        return False  # A finished season is final.
    age = dt.datetime.now() - dt.datetime.fromtimestamp(path.stat().st_mtime)
    return age > CURRENT_SEASON_TTL


def fetch_season_csv(start_year: int, division: str, cache_dir: Path,
                     today: dt.date | None = None) -> bytes:
    """Raw CSV bytes for one league-season, from cache when it is still valid."""
    today = today or dt.date.today()
    path = cache_dir / f"{season_code(start_year)}_{division}.csv"
    if path.exists() and not _is_stale(path, start_year, today):
        return path.read_bytes()

    try:
        payload = _get_following_trusted_redirects(season_url(start_year, division))
    except (requests.RequestException, RuntimeError) as error:
        if not path.exists():
            raise
        # A cached copy that is merely out of date beats no data at all.
        warnings.warn(f"{season_code(start_year)}/{division}: using cached copy, "
                      f"refresh failed ({type(error).__name__}: {error})", stacklevel=2)
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _tokenize(payload: bytes) -> pd.DataFrame:
    """Rows of a football-data CSV as strings, tolerating ragged files.

    Some season files carry more fields per row than the header declares
    (Ligue 1 2002/03 has a 30-field header and rows of up to 33), so letting
    pandas skip "bad" lines would silently drop half the season. Tokenize with
    the csv module and pad the header instead.
    """
    records = [row for row in csv.reader(io.StringIO(payload.decode("latin-1")))
               if any(field.strip() for field in row)]
    if not records:
        raise ValueError("CSV contains no rows")
    header = [str(name).lstrip("\ufeff").strip() for name in records[0]]
    width = max(len(row) for row in records)
    header += [""] * (width - len(header))
    header = [name or f"_unnamed_{index}" for index, name in enumerate(header)]
    rows = [row + [""] * (width - len(row)) for row in records[1:]]
    return pd.DataFrame(rows, columns=header)


def _get_following_trusted_redirects(url: str) -> bytes:
    """GET `url`, following redirects only while they stay on a known host."""
    for _ in range(MAX_REDIRECTS):
        with requests.get(url, timeout=TIMEOUT, allow_redirects=False) as response:
            if response.status_code not in (301, 302, 303, 307, 308):
                response.raise_for_status()
                return response.content
            target = urljoin(url, response.headers.get("location", ""))
        host = (urlsplit(target).hostname or "").lower()
        if host not in ALLOWED_HOSTS:
            raise RuntimeError(f"refusing redirect from {url} to untrusted host {host!r}")
        url = target
    raise RuntimeError(f"too many redirects starting at {url}")


def _parse_match_dates(values: pd.Series) -> pd.Series:
    """Parse football-data dates on both pandas 1.x and 2.x.

    Upstream mixes ``dd/mm/yy`` and ``dd/mm/yyyy``, sometimes in one file.
    pandas 2 needs ``format="mixed"`` for that. pandas 1.x treats that string
    as a strftime pattern and, with ``errors="coerce"``, silently NaT's every
    row — which previously dropped entire seasons.
    """
    kwargs = dict(dayfirst=True, errors="coerce")
    try:
        parsed = pd.to_datetime(values, format="mixed", **kwargs)
    except (TypeError, ValueError):
        parsed = pd.to_datetime(values, **kwargs)
    else:
        nonempty = values.astype(str).str.strip().ne("")
        if parsed.isna().all() and nonempty.any():
            parsed = pd.to_datetime(values, **kwargs)
    missing = parsed.isna()
    if missing.any():
        for fmt in ("%d/%m/%Y", "%d/%m/%y"):
            parsed = parsed.fillna(pd.to_datetime(values, format=fmt, errors="coerce"))
            if not parsed.isna().any():
                break
    return parsed


def parse_season_csv(payload: bytes, start_year: int, division: str) -> pd.DataFrame:
    """One league-season's CSV, renamed to this repo's schema.

    Upstream widened from 45 columns in 2000/01 to 132 in 2025/26, so columns
    are selected by name and missing ones become NaN rather than being
    concatenated into a sparse mess.
    """
    frame = _tokenize(payload).rename(columns=LEGACY_ODDS_COLUMNS)
    if "Date" not in frame:
        raise ValueError(f"{division} {season_label(start_year)} has no Date column")
    frame = frame.loc[frame["Date"].str.strip().ne("")]

    wanted = {source: target for source, target in RESULT_COLUMNS.items() if source in frame}
    # Always emit the full schema. team_dataset indexes columns like
    # HomeShotsOnTarget directly, and the pre-2005 files simply lack them.
    out = frame[list(wanted)].rename(columns=wanted).reindex(
        columns=list(RESULT_COLUMNS.values()))
    for column in out.columns:
        if out[column].dtype == object:
            out[column] = out[column].str.strip().replace("", pd.NA)
    for column in NUMERIC_RESULT_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in ODDS_COLUMNS:
        out[column] = pd.to_numeric(frame[column], errors="coerce") if column in frame else pd.NA
    extra = {}
    for column in frame.columns:
        if parse_1x2_column(column) is None or column in out.columns:
            continue
        extra[column] = pd.to_numeric(frame[column], errors="coerce")
    if extra:
        out = pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)

    # Older files use dd/mm/yy, newer ones dd/mm/yyyy, sometimes within one
    # league-season, so parse element-wise rather than inferring one format.
    out["MatchDate"] = _parse_match_dates(out["MatchDate"])
    parsed = out["MatchDate"].notna()
    if not parsed.any() and len(out):
        raise ValueError(f"{division} {season_label(start_year)}: could not parse any Date values")
    out.insert(0, "League", DIVISIONS.get(division, division))
    out.insert(0, "Season", season_label(start_year))
    return out.loc[parsed]


def load_matches(start_year: int | None = None, end_year: int | None = None,
                 divisions=None, cache_root_dir=None, today=None) -> pd.DataFrame:
    """Every league-season as one frame: mirror columns plus odds.

    ``start_year``/``end_year`` are season *start* years, both inclusive.
    """
    today = today or dt.date.today()
    start_year = FIRST_SEASON_START if start_year is None else start_year
    end_year = current_season_start(today) if end_year is None else end_year
    if end_year < start_year:
        raise ValueError("end_year must not precede start_year")
    divisions = list(DIVISIONS) if divisions is None else list(divisions)
    unknown = set(divisions) - set(DIVISIONS)
    if unknown:
        raise ValueError(f"Unknown divisions: {sorted(unknown)}")

    cache_dir = cache_root(cache_root_dir)
    parts = []
    for year in range(start_year, end_year + 1):
        for division in divisions:
            payload = fetch_season_csv(year, division, cache_dir, today)
            parts.append(parse_season_csv(payload, year, division))
    if not parts:
        raise ValueError("No season files were loaded")
    combined = pd.concat(parts, ignore_index=True)
    if combined.empty:
        raise ValueError("No matches were parsed from the season files")
    return combined.sort_values(["MatchDate", "League", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
