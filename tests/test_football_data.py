import datetime as dt
from unittest.mock import MagicMock

import pandas as pd
import pytest
import requests

from src import football_data


MODERN = (b"Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,B365H,B365D,B365A,"
          b"B365CH,B365CD,B365CA,AvgCH,AvgCD,AvgCA\n"
          b"E0,15/08/2025,20:00,Liverpool,Bournemouth,4,2,H,15,9,1.3,6,8.5,"
          b"1.29,6.25,9,1.29,6.1,9.4\n"
          b",,,,,,,,,,,,,,,,,,\n")
LEGACY = (b"\xef\xbb\xbfDiv,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,B365H,B365D,B365A,"
          b"BbAvH,BbMxH\n"
          b"E0,19/08/05,Man United,Aston Villa,1,0,H,12,7,1.5,3.6,7.0,1.52,1.58\n")


def _parse(payload, year=2025, division="E0"):
    return football_data.parse_season_csv(payload, year, division)


def test_season_code_and_label_cover_the_century_rollover():
    assert football_data.season_code(1999) == "9900"
    assert football_data.season_code(2025) == "2526"
    assert football_data.season_label(1999) == "1999/00"
    assert football_data.season_label(2025) == "2025/26"


def test_current_season_starts_in_august_not_january():
    assert football_data.current_season_start(dt.date(2026, 9, 16)) == 2026
    assert football_data.current_season_start(dt.date(2026, 1, 19)) == 2025


def test_parse_renames_to_mirror_schema_and_keeps_odds():
    frame = _parse(MODERN)
    assert len(frame) == 1  # The trailing blank row is dropped.
    row = frame.iloc[0]
    assert row.Season == "2025/26" and row.League == "Premier League"
    assert row.MatchDate == pd.Timestamp("2025-08-15")
    assert row.FullTimeHomeGoals == 4 and row.FullTimeResult == "H"
    assert row.HomeShotsOnTarget is pd.NA or pd.isna(row.HomeShotsOnTarget)
    assert row.B365CH == 1.29 and row.AvgCA == 9.4


MIXED_YEARS = (b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
               b"E0,19/08/00,Charlton,Man City,4,0,H\n"
               b"E0,15/08/2025,Liverpool,Bournemouth,4,2,H\n")


def test_two_digit_and_four_digit_dates_parse_in_the_same_file():
    frame = _parse(MIXED_YEARS, year=2000)
    assert list(frame.MatchDate) == [pd.Timestamp("2000-08-19"), pd.Timestamp("2025-08-15")]


def test_parse_match_dates_recovers_when_mixed_is_a_literal_format():
    # pandas 1.x + format="mixed" + errors="coerce" returns all-NaT.
    values = pd.Series(["19/08/00", "15/08/2025", "05/08/00"])
    parsed = football_data._parse_match_dates(values)
    assert list(parsed) == [
        pd.Timestamp("2000-08-19"),
        pd.Timestamp("2025-08-15"),
        pd.Timestamp("2000-08-05"),
    ]


def test_parse_keeps_every_named_1x2_book():
    payload = (
        b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,"
        b"B365H,B365D,B365A,BWH,BWD,BWA,WHH,WHD,WHA,IWH,IWD,IWA\n"
        b"E0,15/08/2025,Liverpool,Bournemouth,4,2,H,"
        b"1.30,6.00,8.50,1.28,6.25,9.00,1.32,5.80,8.00,1.25,6.50,9.50\n"
    )
    frame = _parse(payload)
    assert frame.BWH.iloc[0] == 1.28 and frame.WHH.iloc[0] == 1.32
    assert football_data.opening_book_codes(frame.columns) == ["B365", "BW", "WH", "IW"]
    assert football_data.parse_1x2_column("PSCH") == ("PS", "H", True)
    assert football_data.parse_1x2_column("B365H") == ("B365", "H", False)
    assert football_data.parse_1x2_column("VCH") == ("VC", "H", False)
    assert football_data.parse_1x2_column("VCCH") == ("VC", "H", True)
    assert football_data.parse_1x2_column("B365AHH") is None


def test_best_named_opening_picks_the_highest_price_per_outcome():
    frame = _parse(
        b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,"
        b"B365H,B365D,B365A,BWH,BWD,BWA,WHH,WHD,WHA,IWH,IWD,IWA,MaxH,AvgH\n"
        b"E0,15/08/2025,Liverpool,Bournemouth,4,2,H,"
        b"1.30,6.00,8.50,1.28,6.25,9.00,1.32,5.80,8.00,1.25,6.50,9.50,1.40,1.31\n"
    )
    best = football_data.best_named_opening(frame)
    assert best.best_price_H.iloc[0] == 1.32 and best.best_book_H.iloc[0] == "WH"
    assert best.best_price_D.iloc[0] == 6.50 and best.best_book_D.iloc[0] == "IW"
    assert best.best_price_A.iloc[0] == 9.50 and best.best_book_A.iloc[0] == "IW"
    # Max/Avg are aggregates, not candidates, even when they are higher.


def test_legacy_market_aggregates_map_onto_the_modern_names():
    frame = _parse(LEGACY, year=2005)
    row = frame.iloc[0]
    assert row.Season == "2005/06"
    assert row.MatchDate == pd.Timestamp("2005-08-19")  # Two-digit year, day first.
    assert row.AvgH == 1.52 and row.MaxH == 1.58
    assert pd.isna(row.B365CH)  # Closing prices only exist from 2019/20.


def test_finished_seasons_are_served_from_cache_without_a_request(tmp_path, monkeypatch):
    path = tmp_path / "2526_E0.csv"
    path.write_bytes(MODERN)
    download = MagicMock(side_effect=AssertionError("Network must not be used"))
    monkeypatch.setattr(football_data.requests, "get", download)
    payload = football_data.fetch_season_csv(2025, "E0", tmp_path, today=dt.date(2026, 9, 16))
    assert payload == MODERN
    download.assert_not_called()


def test_ongoing_season_refetches_once_the_cached_copy_goes_stale(tmp_path, monkeypatch):
    path = tmp_path / "2627_E0.csv"
    path.write_bytes(b"stale")
    stale = (dt.datetime.now() - dt.timedelta(days=2)).timestamp()
    import os
    os.utime(path, (stale, stale))
    response = MagicMock()
    response.content = MODERN
    response.__enter__.return_value = response
    monkeypatch.setattr(football_data.requests, "get", MagicMock(return_value=response))
    payload = football_data.fetch_season_csv(2026, "E0", tmp_path, today=dt.date(2026, 9, 16))
    assert payload == MODERN and path.read_bytes() == MODERN


def test_http_failure_does_not_get_cached_or_parsed_as_csv(tmp_path, monkeypatch):
    response = MagicMock()
    response.__enter__.return_value = response
    response.raise_for_status.side_effect = requests.HTTPError("503")
    monkeypatch.setattr(football_data.requests, "get", MagicMock(return_value=response))
    with pytest.raises(requests.HTTPError, match="503"):
        football_data.fetch_season_csv(2025, "E0", tmp_path)
    assert not list(tmp_path.iterdir())


def test_unknown_division_is_rejected_before_any_request(monkeypatch):
    monkeypatch.setattr(football_data.requests, "get",
                        MagicMock(side_effect=AssertionError("Network must not be used")))
    with pytest.raises(ValueError, match="Unknown divisions"):
        football_data.load_matches(divisions=["E0", "XX"])


RAGGED = (b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
          b"F1,03/11/02,Nantes,Bordeaux,0,0,D,,\n"
          b"F1,07/11/02,Ajaccio,Auxerre,1,0,H,,,\n")


def test_rows_wider_than_the_header_are_kept_not_skipped():
    # Ligue 1 2002/03 ships a 30-field header with rows of up to 33 fields.
    frame = _parse(RAGGED, year=2002, division="F1")
    assert len(frame) == 2
    assert list(frame.HomeTeam) == ["Nantes", "Ajaccio"]
    assert frame.League.eq("Ligue 1").all()


def test_result_columns_come_back_numeric_like_the_mirror():
    frame = _parse(MODERN)
    assert frame.FullTimeHomeGoals.dtype.kind in "fi"
    assert frame.HomeShots.iloc[0] == 15


def _redirect(location, status=302):
    response = MagicMock()
    response.status_code = status
    response.headers = {"location": location}
    response.__enter__.return_value = response
    return response


def _payload(content=MODERN):
    response = MagicMock()
    response.status_code = 200
    response.content = content
    response.__enter__.return_value = response
    return response


def test_redirect_to_the_apex_domain_is_followed(tmp_path, monkeypatch):
    responses = [_redirect("https://football-data.co.uk/mmz4281/2526/E0.csv"), _payload()]
    monkeypatch.setattr(football_data.requests, "get", MagicMock(side_effect=responses))
    assert football_data.fetch_season_csv(2025, "E0", tmp_path) == MODERN


def test_redirect_to_loopback_is_refused(tmp_path, monkeypatch):
    # The live site started doing exactly this in September 2026. Following it
    # would send the request to whatever is listening on the caller's own port 80.
    monkeypatch.setattr(football_data.requests, "get",
                        MagicMock(return_value=_redirect("http://127.0.0.1/mmz4281/2526/E0.csv")))
    with pytest.raises(RuntimeError, match="untrusted host"):
        football_data.fetch_season_csv(2025, "E0", tmp_path)
    assert not list(tmp_path.iterdir())


def test_a_stale_cache_is_used_when_the_refresh_fails(tmp_path, monkeypatch):
    path = tmp_path / "2627_E0.csv"
    path.write_bytes(MODERN)
    stale = (dt.datetime.now() - dt.timedelta(days=2)).timestamp()
    import os
    os.utime(path, (stale, stale))
    monkeypatch.setattr(football_data.requests, "get",
                        MagicMock(side_effect=requests.ConnectionError("site down")))
    with pytest.warns(UserWarning, match="using cached copy"):
        assert football_data.fetch_season_csv(2026, "E0", tmp_path, today=dt.date(2026, 9, 17)) == MODERN


def test_a_failed_fetch_with_no_cache_still_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(football_data.requests, "get",
                        MagicMock(side_effect=requests.ConnectionError("site down")))
    with pytest.raises(requests.ConnectionError):
        football_data.fetch_season_csv(2026, "E0", tmp_path, today=dt.date(2026, 9, 17))
