"""The real first ear (WakePhoneHuBERT, fetched on first use) on short Piper clips.

A regression check, not a measure of recall (that is evaluation/): the English
clips are a voice not used for the 'heard' pronunciations, the Danish one is
da_DK-talesyntese, which was."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ovos_audio_transformer_plugin_overheard_manners.spotter import PhraseSpotter, load_phrases

CLIPS = Path(__file__).parent / "clips"


def hits(lang, clip):
    audio, sr = sf.read(CLIPS / f"{clip}.wav", dtype="float32")
    assert sr == 16000
    sp = PhraseSpotter(load_phrases(lang))
    found = []
    for i in range(0, len(audio), 1280):
        p = sp.feed(audio[i:i + 1280])
        if p:
            found.append(p.text)
    return found


@pytest.mark.parametrize("lang,clip,phrase", [
    ("en-US", "en_what_the_fuck", "what the fuck"),
    ("da-DK", "en_what_the_fuck", "what the fuck"),      # English swearing in a Danish home
    ("da-DK", "da_av_for_helvede", "av for helvede"),
])
def test_spots_the_phrase(lang, clip, phrase):
    assert phrase in hits(lang, clip)


@pytest.mark.parametrize("lang", ["en-US", "da-DK"])
def test_quiet_on_ordinary_speech(lang):
    assert hits(lang, "en_lovely_day") == []


def test_silence():
    sp = PhraseSpotter(load_phrases("en-US"))
    assert sp.feed(np.zeros(16000 * 3, np.float32)) is None
