from unittest.mock import MagicMock

import pytest
import requests

from src import data_loader


def test_download_uses_verified_requests_and_checks_http_status(monkeypatch):
    response = MagicMock()
    response.content = b'League,MatchDate\nL,2023-01-01\nL,2023-01-02\n'
    response.__enter__.return_value = response
    download = MagicMock(return_value=response)
    monkeypatch.setattr(data_loader.requests, 'get', download)
    frame = data_loader.load_raw_data(nrows=1)
    assert len(frame) == 1
    assert frame.MatchDate.iloc[0] == '2023-01-01'
    download.assert_called_once_with(data_loader.DATA_URL, timeout=(10, 60))
    response.raise_for_status.assert_called_once()
    response.__exit__.assert_called_once()


def test_local_file_avoids_network_and_honors_row_limit(tmp_path, monkeypatch):
    path = tmp_path / 'matches.csv'
    path.write_text('League\nL\nOther\n')
    download = MagicMock(side_effect=AssertionError('Network must not be used'))
    monkeypatch.setattr(data_loader.requests, 'get', download)
    assert len(data_loader.load_raw_data(str(path), nrows=1)) == 1
    download.assert_not_called()


def test_http_failure_does_not_get_parsed_as_csv(monkeypatch):
    response = MagicMock()
    response.__enter__.return_value = response
    response.raise_for_status.side_effect = requests.HTTPError('503')
    monkeypatch.setattr(data_loader.requests, 'get', MagicMock(return_value=response))
    with pytest.raises(requests.HTTPError, match='503'):
        data_loader.load_raw_data()
