"""Relative dates ("last Sunday", "two weeks ago") worked out in code and written next to the words.

Models count days badly: asked what "last weekend" meant in a chat on Monday 17 July 2023,
they often answer a week off. `with_dates` writes the dates in, so nothing is left to count:

    with_dates("We went camping last weekend.", datetime(2023, 7, 17))
    -> "We went camping last weekend [the weekend before Mon 17 Jul 2023: Sat 15 Jul - Sun 16 Jul]."

Weeks run Monday to Sunday. Only phrases with one clear meaning are dated:
- "yesterday", "last night", "tomorrow", "the day before yesterday", "two days ago";
- "last Friday" (the most recent Friday before that day) and "next Friday" (the first Friday
  after it);
- "last weekend", "this weekend", "two weekends ago": said as "the weekend before <day>", with
  its Saturday and Sunday;
- "last week", "this week", "next week", "two weeks ago", "in three weeks": said as "the week
  before <day>", without days, because "last week" said on a Sunday can mean the calendar week
  before or the seven days just gone;
- "last month", "next month", "last year", "three years ago".
Vague phrases ("recently", "the other day"), seasons (they depend on the hemisphere), bare
weekdays ("on Sunday"), "this Tuesday" and "next weekend" could mean more than one date, so
they are left alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable, Optional

_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
            "a couple of": 2, "a couple": 2}
_N = (r"(?P<n>\d{1,2}|a couple of|a couple|an?|one|two|three|four|five|six|seven|eight|nine|"
      r"ten|eleven|twelve)")
_WEEKDAY = r"(?P<day>" + "|".join(_DAYS) + r")"


@dataclass
class _Span:
    first: date
    last: date
    kind: str = "day"   # day | days | month | year
    lead: str = ""      # "the week before": said relative to the day, before the days
    about: bool = False
    days: bool = True   # False: the lead alone ("the week before Sun 16 Jul 2023")


def _week(today: date, offset: int) -> tuple[date, date]:
    monday = today - timedelta(days=today.weekday()) + timedelta(weeks=offset)
    return monday, monday + timedelta(days=6)


def _weekend(today: date, offset: int) -> tuple[date, date]:
    _, sunday = _week(today, offset)
    return sunday - timedelta(days=1), sunday


def _month(today: date, offset: int) -> _Span:
    index = today.year * 12 + today.month - 1 + offset
    first = date(index // 12, index % 12 + 1, 1)
    nxt = date((index + 1) // 12, (index + 1) % 12 + 1, 1)
    return _Span(first, nxt - timedelta(days=1), "month")


def _year(today: date, offset: int) -> _Span:
    return _Span(date(today.year + offset, 1, 1), date(today.year + offset, 12, 31), "year")


def _day(offset: int) -> Callable[[re.Match, date], Optional[_Span]]:
    return lambda m, today: _Span(today + timedelta(days=offset), today + timedelta(days=offset))


def _weeks(offset: int, lead: str, weekend: bool = False) -> Callable[[re.Match, date], Optional[_Span]]:
    def span(m: re.Match, today: date) -> Optional[_Span]:
        first, last = (_weekend if weekend else _week)(today, offset)
        return _Span(first, last, "days", lead, days=weekend)
    return span


def _count(m: re.Match, today: date, sign: int) -> Optional[_Span]:
    """"two weeks ago", "in 3 days", "a year ago"."""
    word, unit = m.group("n").lower(), m.group("unit").lower()
    n = int(word) if word.isdigit() else _NUMBERS[word]
    if n == 0:
        return None
    word = "a" if n == 1 and unit != "year" else word.replace("a couple of", "two").replace("a couple", "two")
    lead = f"{word} {unit}{'s' if n > 1 else ''} {'before' if sign < 0 else 'after'}"
    if unit == "day":
        d = today + timedelta(days=sign * n)
        return _Span(d, d)
    if unit == "week":
        return _Span(*_week(today, sign * n), "days", lead, days=False)
    if unit == "weekend":
        return _Span(*_weekend(today, sign * n), "days", lead)
    if unit == "month":
        return _month(today, sign * n)
    return _year(today, sign * n)


def _named_day(m: re.Match, today: date) -> Optional[_Span]:
    target = _DAYS.index(m.group("day").lower())
    if m.group("which").lower() == "next":  # the first one after today
        d = today + timedelta(days=(target - today.weekday()) % 7 or 7)
    else:  # last, this past: the most recent one before today
        d = today - timedelta(days=(today.weekday() - target) % 7 or 7)
    return _Span(d, d)


def _this_weekend(m: re.Match, today: date) -> Optional[_Span]:
    first, last = _weekend(today, 0)
    return _Span(first, last, "days", "the weekend of" if today.weekday() >= 5 else "the weekend after")


def _over_the_weekend(m: re.Match, today: date) -> Optional[_Span]:
    if today.weekday() > 3:  # said Friday to Sunday, it may mean this one
        return None
    return _Span(*_weekend(today, -1), "days", "the weekend before")


_RULES: list[tuple[re.Pattern, Callable[[re.Match, date], Optional[_Span]]]] = [
    (re.compile(p, re.I), f) for p, f in [
        (r"\bthe day before yesterday\b", _day(-2)),
        (r"\bthe day after tomorrow\b", _day(2)),
        (r"\byesterday\b", _day(-1)),
        (r"\blast night\b", _day(-1)),
        (r"\btomorrow\b", _day(1)),
        (r"\ba few days ago\b",
         lambda m, t: _Span(t - timedelta(days=4), t - timedelta(days=2), "days", about=True)),
        (rf"\b{_N} (?P<unit>day|week|weekend|month|year)s? ago\b", lambda m, t: _count(m, t, -1)),
        (rf"\bin {_N} (?P<unit>day|week|month|year)s?\b", lambda m, t: _count(m, t, 1)),
        (r"\bthe week before last\b", _weeks(-2, "two weeks before")),
        (r"\bthe week after next\b", _weeks(2, "two weeks after")),
        (rf"\b(?P<which>last|this past|next) {_WEEKDAY}\b", _named_day),
        (r"\b(?:last|this past) weekend\b", _weeks(-1, "the weekend before", weekend=True)),
        (r"\bthis weekend\b", _this_weekend),
        (r"\b(?:over|on) the weekend\b", _over_the_weekend),
        (r"\blast week\b", _weeks(-1, "the week before")),
        (r"\bthis week\b", _weeks(0, "the week of")),
        (r"\bnext week\b", _weeks(1, "the week after")),
        (r"\blast month\b", lambda m, t: _month(t, -1)),
        (r"\bthis month\b", lambda m, t: _month(t, 0)),
        (r"\bnext month\b", lambda m, t: _month(t, 1)),
        (r"\blast year\b", lambda m, t: _year(t, -1)),
        (r"\bthis year\b", lambda m, t: _year(t, 0)),
        (r"\bnext year\b", lambda m, t: _year(t, 1)),
    ]
]


def _day_text(d: date, year: bool = True) -> str:
    return f"{d:%a} {d.day} {d:%b}" + (f" {d.year}" if year else "")


def _format(s: _Span, today: date) -> str:
    if s.kind == "month":
        return f"{s.first:%B %Y}"
    if s.kind == "year":
        return str(s.first.year)
    if s.kind == "day":
        return _day_text(s.first)
    if s.lead and not s.days:  # "the week before Fri 9 Jun 2023"
        return f"{s.lead} {_day_text(today)}"
    if s.lead:  # "the weekend before Mon 17 Jul 2023: Sat 15 Jul - Sun 16 Jul"
        same = s.first.year == s.last.year == today.year
        days = f"{_day_text(s.first, not same)} - {_day_text(s.last, not same)}"
        return f"{s.lead} {_day_text(today)}: {days}"
    days = f"{_day_text(s.first, s.first.year != s.last.year)} - {_day_text(s.last)}"
    return ("about " if s.about else "") + days


def relative_dates(text: str, when: datetime | date) -> list[tuple[int, int, str]]:
    """(start, end, dates) for each relative date phrase in `text`, said on `when`."""
    today = when.date() if isinstance(when, datetime) else when
    found: list[tuple[int, int, str]] = []
    for pattern, rule in _RULES:  # earlier rules win: "the day before yesterday" over "yesterday"
        for m in pattern.finditer(text):
            if any(a < m.end() and m.start() < b for a, b, _ in found):
                continue
            span = rule(m, today)
            if span is not None:
                found.append((m.start(), m.end(), _format(span, today)))
    return sorted(found)


def with_dates(text: str, when: datetime | date) -> str:
    """`text` with the date each relative phrase stands for after it, in brackets."""
    out, last = [], 0
    for start, end, dates in relative_dates(text, when):
        if text[end:end + 2] == " [" or text[end:end + 1] == "[":
            continue  # already dated
        out.append(text[last:end] + f" [{dates}]")
        last = end
    return "".join(out) + text[last:]
