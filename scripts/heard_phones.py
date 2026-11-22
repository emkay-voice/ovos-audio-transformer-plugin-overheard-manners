"""Add "heard" pronunciations to scripts/phrases.json: the phones WakePhoneHuBERT
itself hears when Piper voices say each phrase.

eSpeak's phones can be far from what the model hears (Danish "av for helvede":
eSpeak ɑ u f ʌ h ɛ l ʋ ə ð ə, the model a l f o h i v ə). Listening for both
lifted Danish recall from 20 % to 32 % in tests/overheard. Run it when phrases
change, then build_phrases.py.

    pip install piper-tts ovos-ww-plugin-wakeforge
    python scripts/heard_phones.py /path/to/piper/voices
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "scripts" / "phrases.json"
#: two voices per language, as Piper names them; the English base phrases are also
#: spoken by each language's voices (its local pronunciation)
VOICES = {
    "en": ["en_US-lessac-medium", "en_GB-alan-medium"],
    "da": ["da_DK-talesyntese-medium"],
    "de": ["de_DE-thorsten-medium", "de_DE-kerstin-low"],
    "nl": ["nl_NL-pim-medium", "nl_BE-nathalie-medium"],
    "fr": ["fr_FR-siwis-medium", "fr_FR-tom-medium"],
    "es": ["es_ES-davefx-medium", "es_MX-claude-high"],
    "it": ["it_IT-paola-medium", "it_IT-riccardo-x_low"],
    "ca": ["ca_ES-upc_ona-medium", "ca_ES-upc_pau-x_low"],
    "eu": ["eu_ES-antton-medium", "eu_ES-maider-medium"],
    "pt": ["pt_PT-tugão-medium", "pt_BR-faber-medium"],
}


def resample(a, sr):
    if sr == 16000:
        return a.astype(np.float32)
    t = np.arange(int(len(a) * 16000 / sr)) * sr / 16000
    return np.interp(t, np.arange(len(a)), a).astype(np.float32)


def main(voice_dir):
    from piper import PiperVoice, SynthesisConfig
    from ovos_ww_plugin_wakeforge.zeroshot import WakePhoneHuBERTZeroShotPlugin, WINDOW
    from ovos_ww_plugin_wakeforge.pretrained import resolve_pretrained
    from ovos_ww_plugin_wakeforge.zeroshot import FEATURIZER, featurize, ipa_symbol_table, log_posteriors
    eng = WakePhoneHuBERTZeroShotPlugin("heard", {"ipa": "a"})
    _, config = resolve_pretrained(FEATURIZER)
    symbols = config["vocab"]["symbols"]
    usable = set(ipa_symbol_table(symbols))         # the phones a keyword may use

    def heard(audio):
        x = np.zeros(WINDOW, np.float32)
        x[-min(len(audio), WINDOW):] = audio[-WINDOW:]
        lp = log_posteriors(featurize(eng.session, x[np.newaxis])[0, :, eng.ipa_slice]).copy()
        lp[:, 0] -= 2.0                       # blank penalty, as the model card decodes
        out, prev = [], -1
        for k in lp.argmax(1):
            if k > 3 and k != prev and symbols[k] in usable:
                out.append(symbols[k])
            prev = k
        return " ".join(out)

    src = json.loads(SRC.read_text(encoding="utf-8"))
    sil = np.zeros(4800, np.float32)
    for lang, names in VOICES.items():
        texts = src[lang]["phrases"] + ([] if lang == "en" else src["en"]["phrases"])
        found = {}
        for name in names:
            path = Path(voice_dir) / f"{name}.onnx"
            if not path.exists():
                print("missing voice", path); continue
            voice = PiperVoice.load(str(path))
            for text in texts:
                for ls in (0.9, 1.1):
                    a = np.concatenate([c.audio_float_array for c in voice.synthesize(text, SynthesisConfig(length_scale=ls))])
                    h = heard(np.concatenate([sil, resample(a, voice.config.sample_rate), sil]))
                    if len(h.split()) >= 3:
                        found.setdefault(text, [])
                        if h not in found[text]:
                            found[text].append(h)
        src[lang]["heard"] = found
        print(lang, sum(len(v) for v in found.values()), "heard pronunciations")
    SRC.write_text(json.dumps(src, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
