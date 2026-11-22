"""Every Emkay language has phrases the model can spot and replies to give."""
import json
from pathlib import Path

import pytest

LOCALE = Path(__file__).parents[1] / "ovos_audio_transformer_plugin_overheard_manners" / "locale"
LANGS = ["ca-ES", "da-DK", "de-DE", "en-US", "es-ES", "eu-ES", "fr-FR", "gl-ES", "it-IT", "nl-NL", "pt-PT"]


@pytest.fixture(scope="module")
def table():
    from ovos_ww_plugin_wakeforge.pretrained import resolve_pretrained
    from ovos_ww_plugin_wakeforge.zeroshot import FEATURIZER, ipa_symbol_table
    return ipa_symbol_table(resolve_pretrained(FEATURIZER)[1]["vocab"]["symbols"])


@pytest.mark.parametrize("lang", LANGS)
def test_language_is_complete(lang):
    for name in ("overheard.json", "overheard_foul.dialog", "overheard_slur.dialog"):
        assert (LOCALE / lang / name).is_file(), f"{lang}/{name}"
    assert len((LOCALE / lang / "overheard_foul.dialog").read_text().strip().splitlines()) >= 3


@pytest.mark.parametrize("lang", LANGS)
def test_pronunciations_are_model_phones(lang, table):
    from ovos_ww_plugin_wakeforge.zeroshot import tokenize_ipa
    data = json.loads((LOCALE / lang / "overheard.json").read_text(encoding="utf-8"))
    assert data["phrases"]
    for p in data["phrases"]:
        for ipa in p["ipa"] + [i for v in p.get("ipa_local", {}).values() for i in v]:
            assert len(tokenize_ipa(ipa, table)) >= 3, (p["text"], ipa)


def test_english_has_every_local_pronunciation():
    data = json.loads((LOCALE / "en-US" / "overheard.json").read_text(encoding="utf-8"))
    for p in data["phrases"]:
        assert set(p["ipa_local"]) == {lang.split("-")[0] for lang in LANGS} - {"en"}, p["text"]


def test_load_phrases_adds_english_with_local_pronunciation():
    from ovos_audio_transformer_plugin_overheard_manners.spotter import load_phrases
    da = load_phrases("da-DK")
    assert {p.lang for p in da} == {"da", "en"}
    wtf = next(p for p in da if p.text == "what the fuck")
    en_only = next(p for p in load_phrases("en-US") if p.text == "what the fuck")
    assert len(wtf.ipa) > len(en_only.ipa)
