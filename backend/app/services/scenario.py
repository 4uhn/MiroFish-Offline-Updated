"""
Scenario clock and fact sheet for simulation agents.

Run 10 agents had no idea what day it was or what was actually known, so
qwen3:8b filled the gaps: 31 of 59 institutional texts gave invented dates
("October 15th, 2026", "Friday, 10 May 2024"), places ("River Wye") and
links. Two pieces of shared state fix that:

  SimClock   maps each round to a scenario date and time. When a run has fewer
             rounds than the configured period (30 rounds for a 168-hour week),
             rounds are spread over the waking hours of the whole period instead
             of truncating it to its first 30 hours, which in Run 10 fired every
             "first week" event inside day one and gave the evening-active
             residents a single evening.
  facts      short statements from the seed document, each with the time it
             became public. Agents see only the facts already public at the
             current round, so the apology on day four is not known on day one.
             Each fact is checked against the document before it is used
             (ground_fact), so an extraction error cannot plant a new rumour.

No package-relative imports: the simulation scripts import this module by
bare name, as they do round_context.
"""

import random
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# Hours when people post. Compressed runs place rounds only in this window.
WAKING_START_HOUR = 7
WAKING_END_HOUR = 23

_MONTHS = ["january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december"]
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_ALWAYS_OK_WORDS = set(_MONTHS) | set(_WEEKDAYS) | {"i", "uk", "england"}

# Number words a fact may use for a figure the source writes in digits. Run 12
# dropped "Twelve children under 5 were found to have elevated blood lead
# levels" because "Twelve" was checked as a name and the seed says "12".
# "one" is left out: "no one", "one of the sites" are not figures.
_UNITS = ["two", "three", "four", "five", "six", "seven", "eight", "nine"]
_NUMBER_WORDS = {w: i for i, w in enumerate(_UNITS, start=2)}
_NUMBER_WORDS.update({w: i for i, w in enumerate(
    ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
     "seventeen", "eighteen", "nineteen"], start=10)})
_TENS = {w: i * 10 for i, w in enumerate(
    ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"], start=2)}
_NUMBER_WORDS.update(_TENS)
_NUMBER_WORD_RE = re.compile(
    rf"\b(?:({'|'.join(_TENS)})-({'|'.join(['one'] + _UNITS)})|({'|'.join(_NUMBER_WORDS)}))\b",
    re.IGNORECASE,
)


def parse_datetime(value: Any) -> Optional[datetime]:
    """Parse "2026-04-16", "2026-04-16 14:00" or "2026-04-16T14:00"; None if unparseable."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d %H", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[:19], fmt)
        except ValueError:
            continue
    return None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace(",", "").lower())


def date_in_source(dt: datetime, source: str) -> bool:
    """True when the source text gives this calendar date in a common form
    and names its year somewhere (timelines often date "17 April" under an
    "April 2026" heading)."""
    src = _norm(source)
    if str(dt.year) not in src:
        return False
    month = _MONTHS[dt.month - 1]
    forms = (
        f"{dt.day} {month}", f"{month} {dt.day}", dt.strftime("%Y-%m-%d"),
        f"{dt.day:02d}/{dt.month:02d}", f"{dt.day}/{dt.month}",
    )
    # Digit boundaries: "April 2026" must not read as "April 20"
    return any(re.search(rf"(?<![\d/-]){re.escape(f)}(?![\d/])", src) for f in forms)


def ground_fact(text: str, source: str) -> bool:
    """True when every number and proper noun in text also appears in source.

    A cheap check against extraction errors: a fact with a figure or a name
    the seed document does not contain is dropped, not shown to agents.
    """
    src = _norm(source)
    if not numbers_in_source(text, src):
        return False
    words = re.findall(r"[A-Za-z][\w'-]*", text or "")
    for word in words:
        # The first word is checked too: "Wessex Water will..." starts with a name.
        # A capitalised common word ("The", "Lead") passes if the source uses it.
        if not word[0].isupper():
            continue
        # Single quotes around a name ('Do Not Drink') are not part of it.
        w = re.sub(r"'s$", "", word.lower().rstrip("'-"))
        # Number words were checked as figures by numbers_in_source
        if w in _ALWAYS_OK_WORDS or w in _NUMBER_WORDS or _NUMBER_WORD_RE.fullmatch(w):
            continue
        if not re.search(rf"\b{re.escape(w)}", src):
            return False
    return True


def numbers_in_source(text: str, source: str) -> bool:
    """True when every number in text appears in source as a whole number
    ("12" must not match inside "0117 922 2000"). A number word ("Twelve",
    "twenty-five") passes when the source has the word or its digits."""
    src = _norm(source)

    def has_digits(num: str) -> bool:
        return re.search(rf"(?<![\d.]){re.escape(num)}(?![\d])", src) is not None

    for num in re.findall(r"\d+(?:[.,:]\d+)*", text or ""):
        if not has_digits(num.replace(",", "")):
            return False
    for match in _NUMBER_WORD_RE.finditer(text or ""):
        word = match.group(0).lower()
        if match.group(3):
            value = _NUMBER_WORDS[word]
        else:
            value = _TENS[match.group(1).lower()] + (["one"] + _UNITS).index(match.group(2).lower()) + 1
        if not (has_digits(str(value)) or re.search(rf"\b{re.escape(word)}\b", src)):
            return False
    return True


@dataclass
class ScenarioFact:
    text: str
    known_from: Optional[datetime] = None  # None: public before the simulation starts


class SimClock:
    """Scenario date and time for each round.

    Round r (1-based) happens at at(r). Seed posts happen at the start. When
    total_rounds covers the configured period at minutes_per_round, time
    advances by minutes_per_round. Otherwise the waking hours of the whole
    period are divided evenly across the rounds.
    """

    def __init__(
        self,
        start: datetime,
        total_hours: int,
        minutes_per_round: int,
        total_rounds: int,
        dated: bool = True,
    ):
        self.start = start
        self.dated = dated
        self.total_hours = max(1, total_hours)
        self.total_rounds = max(1, total_rounds)
        self.minutes_per_round = max(1, minutes_per_round)
        configured_rounds = max(1, (total_hours * 60) // self.minutes_per_round)
        self.compressed = self.total_rounds < configured_rounds
        # Waking hours of the period, as offsets from the start
        self._slots: List[datetime] = []
        if self.compressed:
            for h in range(max(1, total_hours)):
                t = start + timedelta(hours=h)
                if h == 0 or WAKING_START_HOUR <= t.hour < WAKING_END_HOUR:
                    self._slots.append(t)

    def at(self, round_num: int) -> datetime:
        """Scenario time of round round_num (1-based; 0 = the seed posts)."""
        idx = max(0, round_num - 1)
        if not self.compressed:
            return self.start + timedelta(minutes=idx * self.minutes_per_round)
        pos = idx * len(self._slots) / self.total_rounds
        slot = self._slots[min(int(pos), len(self._slots) - 1)]
        return slot + timedelta(minutes=int((pos - int(pos)) * 60))

    @property
    def end(self) -> datetime:
        """End of the simulated period: the configured hours when compressed,
        otherwise one round after the last."""
        if self.compressed:
            return self.start + timedelta(hours=self.total_hours)
        return self.at(self.total_rounds) + timedelta(minutes=self.minutes_per_round)

    @property
    def step_description(self) -> str:
        if not self.compressed:
            return f"{self.minutes_per_round} min per round"
        hours = len(self._slots) / self.total_rounds
        return (f"{hours:.1f} waking hours per round "
                f"({WAKING_START_HOUR:02d}:00-{WAKING_END_HOUR:02d}:00, "
                f"{len(self._slots)} waking hours over {self.total_rounds} rounds)")

    def round_for(self, when: datetime) -> int:
        """First round (1-based) at or after when, clamped to the run."""
        for r in range(1, self.total_rounds + 1):
            if self.at(r) >= when:
                return r
        return self.total_rounds

    def day_number(self, when: datetime) -> int:
        return (when.date() - self.start.date()).days + 1

    def label(self, when: datetime) -> str:
        """"16 April 2026, 14:00 (day 1)"; without a date, "Day 1, 14:00".

        No weekday: seed documents get them wrong (the Bristol document calls
        16 April 2026 a Wednesday; it is a Thursday), and a weekday that
        contradicts the facts gives the model one more thing to reconcile.
        """
        day = self.day_number(when)
        if not self.dated:
            return f"Day {day}, {when:%H:%M}"
        return f"{when.day} {when:%B %Y}, {when:%H:%M} (day {day})"


def build_clock(config: Dict[str, Any], total_rounds: int) -> SimClock:
    """SimClock from a simulation config's scenario and time_config."""
    time_config = config.get("time_config", {})
    scenario = config.get("scenario") or {}
    total_hours = int(time_config.get("total_simulation_hours", 72))
    minutes_per_round = int(time_config.get("minutes_per_round", 60))
    start = parse_datetime(scenario.get("start"))
    if start is None:
        # Undated: day numbers only, starting at the configured hour
        start_hour = int(time_config.get("start_hour", 8))
        return SimClock(datetime(2000, 1, 1, start_hour), total_hours, minutes_per_round,
                        total_rounds, dated=False)
    return SimClock(start, total_hours, minutes_per_round, total_rounds)


def load_facts(config: Dict[str, Any]) -> List[ScenarioFact]:
    facts = []
    for f in (config.get("scenario") or {}).get("facts") or []:
        text = (f.get("text") or "").strip() if isinstance(f, dict) else ""
        if text:
            facts.append(ScenarioFact(text=text, known_from=parse_datetime(f.get("known_from"))))
    return facts


def known_facts_block(
    facts: List[ScenarioFact],
    now: datetime,
    clock: SimClock,
    max_facts: int,
    rng: Optional[random.Random] = None,
) -> Optional[str]:
    """The facts public at time now, newest developments first, as a short block.

    Facts from the last 24 scenario hours are always shown and marked NEW.
    The rest are sampled (rng), so agents do not all see, and repeat, the same
    list. Returns None when there are no facts.
    """
    known = [f for f in facts if f.known_from is None or f.known_from <= now]
    if not known:
        return None
    rng = rng or random.Random()
    fresh = [f for f in known if f.known_from and now - f.known_from < timedelta(hours=24)]
    fresh.sort(key=lambda f: f.known_from, reverse=True)
    older = [f for f in known if f not in fresh]
    rng.shuffle(older)
    chosen = (fresh + older)[:max(1, max_facts)]
    lines = [f"- {'NEW: ' if f in fresh else ''}{f.text}" for f in chosen]
    return (
        f"NOW: {clock.label(now)}.\n"
        "WHAT IS PUBLICLY KNOWN SO FAR (background, do not recite it):\n" + "\n".join(lines)
    )
