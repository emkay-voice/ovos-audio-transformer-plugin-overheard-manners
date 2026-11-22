"""Build locale/<lang>/overheard.json from scripts/phrases.json.

Each phrase gets its pronunciations as WakePhoneHuBERT phones (space separated):
eSpeak NG's, and the ones the model itself heard ("heard", from heard_phones.py).
The English base phrases also get each language's local pronunciation
("ipa_local"), the way a Dane or a Spaniard says "what the fuck".

eSpeak is only needed here, at build time (CI checks the files are up to date);
the plugin itself has no phonemiser.

    pip install phonemizer==3.4.0 espeakng-loader==0.2.4 ovos-ww-plugin-wakeforge
    python scripts/build_phrases.py [--check]
"""
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "scripts" / "phrases.json"
LOCALE = ROOT / "ovos_audio_transformer_plugin_overheard_manners" / "locale"
#: eSpeak symbols that the model's vocab spells differently (Greek epsilon in Danish)
FIX = str.maketrans({"ε": "ɛ"})

_backends = {}


def _table():
    from ovos_ww_plugin_wakeforge.pretrained import resolve_pretrained
    from ovos_ww_plugin_wakeforge.zeroshot import FEATURIZER, ipa_symbol_table
    _, config = resolve_pretrained(FEATURIZER)
    return ipa_symbol_table(config["vocab"]["symbols"])


def _espeak(text, voice):
    import espeakng_loader
    from phonemizer.backend import EspeakBackend
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    from phonemizer.separator import Separator
    if voice not in _backends:
        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
        _backends[voice] = EspeakBackend(voice, language_switch="remove-flags")
    out = _backends[voice].phonemize([text], separator=Separator(phone=" ", word=" | ", syllable=""), strip=True)[0]
    return [p for p in out.replace("|", " ").split()]


def to_phones(text, voice, table):
    """eSpeak phones -> the model's symbols, space separated (validated with wakeforge's tokenizer)."""
    phones = []
    for p in _espeak(text, voice):
        p = p.translate(FIX)
        if not _ok(p, table):
            # a diacritic the model does not know (Basque s̻): keep the base symbol
            p = unicodedata.normalize("NFC", "".join(
                c for c in unicodedata.normalize("NFD", p) if not unicodedata.combining(c)))
        if not _ok(p, table):
            # a mark it has no symbol for at all (Dutch "?" for a glottal stop): leave it out
            p = "".join(c for c in p if _ok(c, table))
        if p:
            phones.append(p)
    return " ".join(phones)


def _ok(p, table):
    from ovos_ww_plugin_wakeforge.zeroshot import tokenize_ipa
    try:
        tokenize_ipa(p, table)
        return True
    except ValueError:
        return False


def build():
    src = json.loads(SRC.read_text(encoding="utf-8"))
    langs = [k for k in src if not k.startswith("_")]
    table = _table()
    out = {}
    for lang in langs:
        d = src[lang]
        heard = d.get("heard", {})
        phrases = []
        for text in d["phrases"]:
            ipa = [to_phones(text, d["espeak"], table)] + heard.get(text, [])
            entry = {"text": text, "ipa": list(dict.fromkeys(ipa))}
            if lang == "en":
                entry["ipa_local"] = {
                    other: list(dict.fromkeys([to_phones(text, src[other]["espeak"], table)]
                                              + src[other].get("heard", {}).get(text, [])))
                    for other in langs if other != "en"}
            phrases.append(entry)
        out[lang] = {"phrases": phrases}
    return out


def main():
    check = "--check" in sys.argv
    stale = []
    for lang, data in build().items():
        f = LOCALE / lang / "overheard.json"
        text = json.dumps(data, ensure_ascii=False, indent=1) + "\n"
        if check:
            if not f.is_file() or f.read_text(encoding="utf-8") != text:
                stale.append(str(f.relative_to(ROOT)))
        else:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(text, encoding="utf-8")
    if stale:
        sys.exit("out of date, run scripts/build_phrases.py: " + ", ".join(stale))


if __name__ == "__main__":
    main()
