from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


DEFAULT_LOCAL_TIMEZONE = "America/Los_Angeles"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_timestamp(value: object) -> datetime | None:
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_local_date_str(
    value: object,
    timezone_name: str = DEFAULT_LOCAL_TIMEZONE,
    default_now: bool = False,
) -> str | None:
    dt = parse_timestamp(value)
    if dt is None:
        if not default_now:
            return None
        dt = datetime.now(timezone.utc)
    return dt.astimezone(ZoneInfo(timezone_name)).date().isoformat()


def local_day_bounds(
    local_date: str,
    timezone_name: str = DEFAULT_LOCAL_TIMEZONE,
) -> tuple[datetime, datetime]:
    target_date = date.fromisoformat(str(local_date))
    zone = ZoneInfo(timezone_name)
    start_local = datetime.combine(target_date, time.min, tzinfo=zone)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)
