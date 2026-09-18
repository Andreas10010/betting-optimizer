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
