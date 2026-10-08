"""Download and save National Diet speeches."""

from __future__ import annotations

import csv
import re
from collections.abc import Callable
from pathlib import Path

from .api import fetch_records

LogCallback = Callable[[str], None]

FIELDNAMES = [
    "comment-id", "meeting-id", "session", "name_of_house",
    "name_of_meeting", "issue", "date", "speech_order", "speaker",
    "speaker_group", "speaker_position", "speaker_role", "comment-body",
    "speech_url",
]


def download_speeches(
    raw_keywords: str,
    start_date: str,
    end_date: str,
    output_path: Path,
    *,
    log: LogCallback | None = None,
) -> Path:
    """Download unique speeches and write a UTF-8 CSV."""
    emit = log or (lambda _message: None)
    keywords = [item for item in re.split(r"[\s\u3000]+", raw_keywords.strip()) if item]
    records: list[dict] = []
    speech_ids: set[str] = set()

    for keyword in keywords:
        emit(f"Searching for “{keyword}”…")
        for record in fetch_records(keyword, start_date, end_date, log=emit):
            speech_id = str(record.get("speechID", ""))
            if speech_id and speech_id in speech_ids:
                continue
            if speech_id:
                speech_ids.add(speech_id)
            records.append(record)

    if not records:
        raise RuntimeError("The Diet API returned no speeches for this search.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    emit(f"Writing {len(records):,} unique records to CSV…")
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for index, record in enumerate(records, start=1):
            writer.writerow({
                "comment-id": index,
                "meeting-id": record.get("issueID", ""),
                "session": record.get("session", ""),
                "name_of_house": record.get("nameOfHouse", ""),
                "name_of_meeting": record.get("nameOfMeeting", ""),
                "issue": record.get("issue", ""),
                "date": record.get("date", ""),
                "speech_order": record.get("speechOrder", ""),
                "speaker": record.get("speaker", ""),
                "speaker_group": record.get("speakerGroup", ""),
                "speaker_position": record.get("speakerPosition", ""),
                "speaker_role": record.get("speakerRole", ""),
                "comment-body": record.get("speech", ""),
                "speech_url": record.get("speechURL", ""),
            })
    emit(f"Download complete: {output_path}")
    return output_path.resolve()
