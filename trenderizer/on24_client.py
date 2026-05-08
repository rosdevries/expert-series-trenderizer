"""
ON24 REST API client for Expert Series Trenderizer.

Endpoints used:
    GET /v2/client/{clientId}/event                              - event listing (paginated)
    GET /v2/client/{clientId}/event/{eventId}/qanda             - per-event Q&A questions
    GET /v2/client/{clientId}/event/{eventId}/registrant        - per-event registrants

ON24 API constraint: the events endpoint rejects date ranges > 180 days.
fetch_events() handles this transparently by chunking into ≤180-day windows.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

log = logging.getLogger(__name__)

# ON24 hard limit: startDate–endDate interval must be ≤ 180 days
ON24_MAX_DATE_RANGE_DAYS = 179

from .config import ON24_CLIENT_ID, ON24_TOKEN_KEY, ON24_TOKEN_SECRET, PAGE_SIZE, REQUEST_TIMEOUT


class ON24APIError(Exception):
    pass


def parse_dt(raw: str | None) -> datetime | None:
    """Parse an ON24 timestamp string to a UTC-aware datetime.

    ON24 returns timestamps like '2026-04-15T09:00:00-07:00' (negative TZ offset).
    Python's fromisoformat handles ±HH:MM offsets natively (3.11+).
    """
    if not raw:
        return None
    s = str(raw).strip()
    try:
        dt = datetime.fromisoformat(s)
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    # Fallback: strip trailing ±HH:MM or Z then try naive formats
    cleaned = s.rstrip("Z")
    if len(cleaned) > 6 and cleaned[-6] in "+-":
        cleaned = cleaned[:-6]
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _extract_list(payload: Any, preferred_key: str) -> list[dict]:
    """Normalize an ON24 response to a list — the API wraps lists inconsistently."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if preferred_key in payload and isinstance(payload[preferred_key], list):
            return payload[preferred_key]
        for value in payload.values():
            if isinstance(value, list):
                return value
    return []


class ON24Client:
    def __init__(
        self,
        client_id: str | int | None = None,
        token_key: str | None = None,
        token_secret: str | None = None,
        timeout: int = REQUEST_TIMEOUT,
        page_size: int = PAGE_SIZE,
    ) -> None:
        self.client_id = str(client_id or ON24_CLIENT_ID)
        self.timeout = timeout
        self.page_size = page_size
        self._session = requests.Session()
        self._session.headers.update({
            "accessTokenKey": token_key or ON24_TOKEN_KEY or "",
            "accessTokenSecret": token_secret or ON24_TOKEN_SECRET or "",
            "accept": "application/json",
        })

    @classmethod
    def from_env(cls) -> "ON24Client":
        missing = [
            name for name, val in [
                ("ON24_CLIENT_ID", ON24_CLIENT_ID),
                ("ON24_TOKEN_KEY", ON24_TOKEN_KEY),
                ("ON24_TOKEN_SECRET", ON24_TOKEN_SECRET),
            ] if not val
        ]
        if missing:
            raise ON24APIError(f"Missing environment variable(s): {', '.join(missing)}")
        return cls()

    def _get(self, path: str, params: dict | None = None) -> Any:
        url = f"https://api.on24.com/v2/client/{self.client_id}/{path.lstrip('/')}"
        try:
            resp = self._session.get(url, params=params or {}, timeout=self.timeout)
        except requests.RequestException as exc:
            raise ON24APIError(f"Network error: {exc}") from exc
        if resp.status_code == 401:
            raise ON24APIError("Authentication failed (401). Check ON24_TOKEN_KEY / ON24_TOKEN_SECRET.")
        if not resp.ok:
            raise ON24APIError(f"ON24 API {resp.status_code} for {url}: {resp.text[:300]}")
        try:
            return resp.json()
        except ValueError as exc:
            raise ON24APIError(f"Non-JSON response from {url}: {resp.text[:200]}") from exc

    def _paginate(
        self, path: str, items_key: str, total_key: str, params: dict | None = None
    ) -> list[dict]:
        p = {"itemsPerPage": self.page_size, "pageOffset": 0, **(params or {})}
        all_items: list[dict] = []
        while True:
            data = self._get(path, params=p)
            page = data.get(items_key, [])
            if not isinstance(page, list):
                break
            all_items.extend(page)
            total = int(data.get(total_key, 0) or 0)
            if len(all_items) >= total or len(page) < self.page_size:
                break
            p["pageOffset"] += 1
        return all_items

    def fetch_events(self, start_date: str, end_date: str) -> list[dict]:
        """
        Fetch all events with livestart in [start_date, end_date].

        Automatically chunks into ≤180-day windows to stay within the ON24
        API limit. Deduplicates on eventid across chunks.
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        all_events: dict[int, dict] = {}
        chunk_start = start_dt

        while chunk_start < end_dt:
            chunk_end = min(chunk_start + timedelta(days=ON24_MAX_DATE_RANGE_DAYS), end_dt)
            s = chunk_start.strftime("%Y-%m-%d")
            e = chunk_end.strftime("%Y-%m-%d")
            log.info(f"  Fetching events window {s} → {e}")
            chunk = self._paginate(
                "event",
                items_key="events",
                total_key="totalevents",
                params={"dateFilterMode": "livestart", "startDate": s, "endDate": e},
            )
            for ev in chunk:
                eid = ev.get("eventid") or ev.get("eventId") or ev.get("id")
                if eid:
                    all_events[int(eid)] = ev
            chunk_start = chunk_end + timedelta(days=1)

        return list(all_events.values())

    def get_attendees(self, event_id: str | int) -> list[dict]:
        """
        Fetch all attendees for a single event.

        ON24 embeds each attendee's Q&A questions under attendee["questions"]
        as a list of objects with a "content" field. This is the correct source
        for historical question text — the /qanda endpoint is real-time only.
        """
        try:
            data = self._get(f"event/{event_id}/attendee")
            return _extract_list(data, "attendees")
        except ON24APIError:
            return []

    def get_qanda(self, event_id: str | int) -> list[dict]:
        """Fetch Q&A from the aggregate /qanda endpoint (real-time; often empty for past events)."""
        try:
            data = self._get(f"event/{event_id}/qanda")
            return _extract_list(data, "qanda")
        except ON24APIError:
            return []

    def get_registrants(self, event_id: str | int) -> list[dict]:
        """Fetch all registrants for a single event."""
        try:
            data = self._get(f"event/{event_id}/registrant", params={"excludeAnonymous": "Y"})
            return _extract_list(data, "registrants")
        except ON24APIError:
            return []
