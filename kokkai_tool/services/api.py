"""Small client for the National Diet Library speech API."""

from __future__ import annotations

import time
from collections.abc import Callable

import requests

API_URL = "https://kokkai.ndl.go.jp/api/speech"
LogCallback = Callable[[str], None]


def fetch_records(
    keyword: str,
    start_date: str,
    end_date: str,
    *,
    max_retries: int = 3,
    log: LogCallback | None = None,
) -> list[dict]:
    """Return all API speech records for one keyword."""
    emit = log or (lambda _message: None)
    records: list[dict] = []
    start_record = 1
    total_fetched = 0

    while True:
        params = {
            "any": keyword,
            "from": start_date,
            "until": end_date,
            "startRecord": start_record,
            "maximumRecords": 100,
            "recordPacking": "json",
        }
        response = None
        for retry in range(1, max_retries + 1):
            try:
                response = requests.get(API_URL, params=params, timeout=40)
                response.raise_for_status()
                break
            except requests.RequestException as exc:
                if retry == max_retries:
                    raise RuntimeError(f"Diet API request failed: {exc}") from exc
                emit(f"Request failed; retrying ({retry}/{max_retries})…")
                time.sleep(2)

        assert response is not None
        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError as exc:
            raise RuntimeError("Diet API returned invalid JSON.") from exc

        total_records = int(data.get("numberOfRecords", 0))
        if total_records == 0:
            emit(f"No records found for “{keyword}”.")
            break

        page = data.get("speechRecord", [])
        if not isinstance(page, list):
            page = [page]
        if not page:
            break

        records.extend(page)
        total_fetched += len(page)
        emit(f"Downloaded {total_fetched:,} of {total_records:,} records for “{keyword}”.")

        next_position = data.get("nextRecordPosition")
        if next_position is None:
            break
        next_position = int(next_position)
        if next_position <= start_record or next_position > total_records:
            break
        start_record = next_position
        time.sleep(0.5)

    return records
