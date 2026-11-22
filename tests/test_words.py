"""Stage 2's verdict on text."""
import json
from pathlib import Path

import pytest

from ovos_audio_transformer_plugin_overheard_manners.words import FOUL, SLUR, WordCheck

SRC = json.loads((Path(__file__).parents[1] / "scripts" / "phrases.json").read_text(encoding="utf-8"))
LANGS = [k for k in SRC if not k.startswith("_")]


@pytest.mark.parametrize("lang", LANGS)
def test_every_phrase_is_foul_and_lookalikes_are_not(lang):
    check = WordCheck([lang], SRC[lang]["phrases"])
    for p in SRC[lang]["phrases"] + SRC["en"]["phrases"]:
        assert check.verdict(f"oh {p}!") == FOUL, p
    for s in SRC[lang]["lookalikes"]:
        assert check.verdict(s) is None, s


def test_english_counts_in_every_language():
    assert WordCheck(["da"]).verdict("Nej, what the fuck") == FOUL


def test_censored_words_count():
    assert WordCheck(["en"]).verdict("Oh f**k") == FOUL


def test_whole_words_only():
    check = WordCheck(["en"])
    assert check.verdict("Scunthorpe is a town") is None
    assert check.verdict("a classic assessment") is None


def test_good_manners_lists_are_used():
    check = WordCheck(["da"])
    assert check.has_good_manners
    assert check.verdict("det er en lortebil") == FOUL          # foul prefix from good manners
    assert check.verdict("Skide være med det") == FOUL


def test_slur_is_its_own_verdict():
    from ovos_audio_transformer_plugin_overheard_manners.words import _gm_folder, _voc
    slur = _voc(_gm_folder("en"), "slurs")[0]
    assert WordCheck(["en"]).verdict(f"he said {slur}") == SLUR
