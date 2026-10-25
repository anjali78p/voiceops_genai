"""Deterministic date maths (LLMs are unreliable at business-day counting)."""
import csv, datetime as dt
from .config import DATASET, TODAY

with open(DATASET / "data" / "holidays.csv", encoding="utf-8") as _f:
    HOLIDAYS = {r["date"] for r in csv.DictReader(_f)}


def is_business_day(d):
    return d.weekday() != 6 and d.isoformat() not in HOLIDAYS  # Mon-Sat, minus holidays


def business_days_between(a, b):
    """Business days after `a` up to and including `b` (negative if b < a)."""
    if b < a:
        return -business_days_between(b, a)
    return sum(is_business_day(a + dt.timedelta(days=i)) for i in range(1, (b - a).days + 1))


def add_business_days(a, n):
    d = a
    while n > 0:
        d += dt.timedelta(days=1)
        n -= is_business_day(d)
    return d


def _p(s):
    return dt.date.fromisoformat(str(s)[:10]) if s else TODAY


def date_calc(op, start=None, end=None, n=None):
    a = _p(start)
    if op == "business_days_between":
        return {"business_days": business_days_between(a, _p(end)), "start": a.isoformat(), "end": _p(end).isoformat()}
    if op == "calendar_days_between":
        return {"calendar_days": (_p(end) - a).days, "start": a.isoformat(), "end": _p(end).isoformat()}
    if op == "add_business_days":
        r = add_business_days(a, int(n)); return {"date": r.isoformat(), "weekday": r.strftime("%A")}
    if op == "add_calendar_days":
        r = a + dt.timedelta(days=int(n)); return {"date": r.isoformat(), "weekday": r.strftime("%A")}
    return {"error": "unknown op"}
