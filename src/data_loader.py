from __future__ import annotations

from io import BytesIO

import pandas as pd
import requests

DATA_URL = "https://raw.githubusercontent.com/vibedatascience/footballdatacouk_leagues_games_results_big5/refs/heads/main/footballdatacouk_leagues_games_results_2000_ytd.csv"


def load_raw_data(path: str | None = None, nrows: int | None = None) -> pd.DataFrame:
    if path is not None:
        return pd.read_csv(path, nrows=nrows)

    # Requests uses its bundled CA trust store (certifi), unlike urllib's
    # platform-dependent default. Keep HTTPS certificate verification enabled.
    with requests.get(DATA_URL, timeout=(10, 60)) as response:
        response.raise_for_status()
        return pd.read_csv(BytesIO(response.content), nrows=nrows)
