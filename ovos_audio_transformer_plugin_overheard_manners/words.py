"""Stage 2's verdict on text: was anything foul said?

The word lists are good manners' own (ovos-good-manners-pipeline-plugin), read
from its package without importing it, matched the way it matches them: whole
words and phrases, plus foul prefixes so compounds count ("lortebil",
"Scheißwetter"). English is always included, as people swear in English in every
language. For languages good manners has no lists for yet, this plugin carries a
short fallback list (locale/<lang>/foul_words.voc), meant to move into good
manners.

Whisper and other STT engines censor words as "f**k"; a censored word counts.
"""
import importlib.util
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from ovos_audio_transformer_plugin_overheard_manners.spotter import locale_dir

FOUL, SLUR = "foul", "slur"


def _good_manners_locale() -> Optional[Path]:
    spec = importlib.util.find_spec("ovos_good_manners_pipeline_plugin")
    if not spec or not spec.submodule_search_locations:
        return None
    path = Path(list(spec.submodule_search_locations)[0]) / "locale"
    return path if path.is_dir() else None


def _voc(folder: Optional[Path], name: str) -> List[str]:
    if not folder or not (folder / f"{name}.voc").is_file():
        return []
    lines = (folder / f"{name}.voc").read_text(encoding="utf-8").splitlines()
    return [w.strip().lower() for w in lines if w.strip() and not w.lstrip().startswith("#")]


def _gm_folder(code: str) -> Optional[Path]:
    root = _good_manners_locale()
    if not root:
        return None
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name.split("-")[0].lower() == code:
            return d
    return None


def _alternation(words) -> str:
    return "|".join(re.escape(w) for w in sorted(set(words), key=len, reverse=True))


class WordCheck:
    def __init__(self, codes: List[str], extra_phrases: List[str] = ()):
        foul, slurs, prefixes = set(p.lower() for p in extra_phrases), set(), set()
        for code in dict.fromkeys(codes + ["en"]):
            gm = _gm_folder(code)
            foul |= set(_voc(gm, "foul_language")) | set(_voc(locale_dir(code), "foul_words"))
            slurs |= set(_voc(gm, "slurs"))
            prefixes |= set(_voc(gm, "foul_prefixes"))
        self.has_good_manners = _good_manners_locale() is not None
        self._slur = re.compile(rf"(?<!\w)(?:{_alternation(slurs)})(?!\w)") if slurs else None
        self._foul = re.compile(rf"(?<!\w)(?:{_alternation(foul)})(?!\w)") if foul else None
        self._prefix = re.compile(rf"(?<!\w)(?:{_alternation(prefixes)})\w+") if prefixes else None
        self._censored = re.compile(r"\w\*+\w*")

    def verdict(self, text: str) -> Optional[str]:
        """'slur', 'foul' or None."""
        t = (text or "").lower()
        if self._slur and self._slur.search(t):
            return SLUR
        if (self._foul and self._foul.search(t)) or (self._prefix and self._prefix.search(t)) \
                or self._censored.search(text or ""):
            return FOUL
        return None


@lru_cache(maxsize=8)
def word_check(code: str, extra: tuple = ()) -> WordCheck:
    return WordCheck([code], list(extra))
