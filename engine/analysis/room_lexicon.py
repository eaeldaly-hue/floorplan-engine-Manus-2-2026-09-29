"""Room-name vocabulary: one place for room terms, their display names and abbreviations.

The lexicon is one signal among several (see engine.analysis.ocr_aggregation and the
analyzer's room-label recognition): it identifies architectural room terms in OCR text and
maps them to a display name. Entries are general architectural terminology, not names taken
from any particular drawing. Longer phrases are matched before their parts ("MASTER BEDROOM"
before "BEDROOM").
"""

from __future__ import annotations

import re

# (display name, phrases). Phrases are upper-case, words separated by single spaces.
ENTRIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Master bedroom", ("MASTER BEDROOM", "PRIMARY BEDROOM", "MASTER SUITE", "PRIMARY SUITE", "MASTER BR")),
    ("Living room", ("LIVING ROOM", "LIVING AREA", "LIVING RM")),
    ("Dining room", ("DINING ROOM", "DINING AREA", "DINING RM")),
    ("Family room", ("FAMILY ROOM", "FAMILY RM")),
    ("Great room", ("GREAT ROOM", "GREAT RM")),
    ("Laundry room", ("LAUNDRY ROOM", "UTILITY ROOM")),
    ("Powder room", ("POWDER ROOM",)),
    ("Walk-in closet", ("WALK IN CLOSET", "WALKIN CLOSET", "WIC", "W I C")),
    ("Breakfast nook", ("BREAKFAST NOOK", "BREAKFAST AREA")),
    ("Sitting area", ("SITTING AREA", "SITTING ROOM")),
    ("Bathroom", ("BATHROOM", "WASHROOM", "ENSUITE", "EN SUITE", "SHOWER ROOM")),
    ("Bedroom", ("BEDROOM", "BED", "BR")),
    ("Kitchen", ("KITCHEN", "KITCHENETTE")),
    ("Living", ("LIVING", "LOUNGE")),
    ("Dining", ("DINING",)),
    ("Family", ("FAMILY",)),
    ("Laundry", ("LAUNDRY", "LDRY", "UTILITY")),
    ("Pantry", ("PANTRY",)),
    ("Hallway", ("HALLWAY", "CORRIDOR")),
    ("Hall", ("HALL",)),
    ("Closet", ("CLOSET", "CL", "WARDROBE", "ROBE")),
    ("Storage", ("STORAGE", "STORE")),
    ("Garage", ("GARAGE", "CARPORT")),
    ("Office", ("OFFICE",)),
    ("Study", ("STUDY",)),
    ("Den", ("DEN",)),
    ("Porch", ("PORCH",)),
    ("Terrace", ("TERRACE", "PATIO", "DECK")),
    ("Balcony", ("BALCONY",)),
    ("Foyer", ("FOYER",)),
    ("Entry", ("ENTRY", "ENTRANCE")),
    ("Mudroom", ("MUDROOM", "MUD ROOM")),
    ("Bath", ("BATH",)),
    ("Toilet", ("TOILET", "WC", "LAVATORY")),
    ("Nook", ("NOOK",)),
    ("Gym", ("GYM",)),
    ("Sauna", ("SAUNA",)),
    ("Theater", ("THEATER", "THEATRE", "MEDIA ROOM")),
    ("Spa", ("SPA",)),
    ("Elevator", ("ELEVATOR", "LIFT")),
    ("Mechanical", ("MECHANICAL", "MECH")),
    ("Stairs", ("STAIRS", "STAIRCASE")),
    ("Sunroom", ("SUNROOM", "SUN ROOM")),
    ("Playroom", ("PLAYROOM", "GAMES ROOM")),
    ("Library", ("LIBRARY",)),
    ("Guest room", ("GUEST ROOM", "GUEST")),
    ("Room", ("ROOM",)),
)

# Drafting abbreviations of CAD sheets (room tags such as "BDRM.#1", "BA-3", "KIT-2", "W/D").
# General drafting conventions; experimental, enabled per analysis with `abbreviations(True)`
# (the structural-layer path) until measured on the development set.
ABBREVIATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Bedroom", ("BDRM", "BDR", "BED RM")),
    ("Bathroom", ("BA", "BTH", "BTHRM")),
    ("Kitchen", ("KIT", "KITCH")),
    ("Laundry", ("W D", "WD", "LAU")),
    ("Closet", ("LIN", "LINEN", "CLO", "CLOS")),
    ("Balcony", ("BAL", "BALC")),
    ("Elevator", ("ELEV",)),
    ("Stairs", ("STAIR",)),
    ("Lobby", ("LOBBY",)),
    ("Trash room", ("TRASH",)),
    ("Electrical", ("ELEC",)),
    ("Storage", ("STOR",)),
    ("Living", ("LIV",)),
    ("Dining", ("DIN",)),
)

_ORDERED = sorted(((phrase, display) for display, phrases in ENTRIES for phrase in phrases),
                  key=lambda item: -len(item[0]))
_ORDERED_ABBR = sorted(_ORDERED + [(phrase, display) for display, phrases in ABBREVIATIONS for phrase in phrases],
                       key=lambda item: -len(item[0]))
_TERMS = {phrase for phrase, _ in _ORDERED}
_SINGLE_WORDS = {phrase for phrase in _TERMS if " " not in phrase}

import contextvars                                                     # noqa: E402

_USE_ABBREVIATIONS: contextvars.ContextVar[bool] = contextvars.ContextVar("room_abbreviations", default=False)


def abbreviations_active() -> bool:
    return _USE_ABBREVIATIONS.get()


class abbreviations:
    """Context manager: drafting abbreviations count as room terms in this context (thread)."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def __enter__(self):
        self._token = _USE_ABBREVIATIONS.set(self.enabled)
        return self

    def __exit__(self, *exc):
        _USE_ABBREVIATIONS.reset(self._token)
        return False


def normalize(text: str) -> str:
    """Upper-case letters and digits; punctuation becomes spaces ("W.I.C" -> "W I C")."""
    t = re.sub(r"[^A-Z0-9\s]", " ", (text or "").upper().replace("'", ""))
    return " ".join(t.split())


def room_name(text: str) -> str | None:
    """Display name of the longest room phrase in `text`, or None."""
    norm = f" {normalize(text)} "
    compact = f" {normalize(text).replace(' ', '')} "
    for phrase, display in (_ORDERED_ABBR if _USE_ABBREVIATIONS.get() else _ORDERED):
        if f" {phrase} " in norm:
            return display
        if " " in phrase and f" {phrase.replace(' ', '')} " in compact:   # W.I.C -> WIC
            return display
    return None


def is_room_word(word: str) -> bool:
    """A single OCR word that is (part of) a room term."""
    w = normalize(word).replace(" ", "")
    if not w:
        return False
    if w in _SINGLE_WORDS and len(w) >= 3:
        return True
    return any(w in phrase.split() and len(w) >= 3 for phrase in _TERMS)


def is_short_abbreviation(word: str) -> bool:
    """Two-letter room abbreviations (BR, WC, CL): room terms only next to other evidence."""
    w = normalize(word).replace(" ", "")
    return w in _SINGLE_WORDS and len(w) <= 2


__all__ = ["ENTRIES", "normalize", "room_name", "is_room_word", "is_short_abbreviation"]
