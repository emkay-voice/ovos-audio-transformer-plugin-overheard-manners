"""Calibrate the plugin's own PhraseSpotter (its phrase files, heard phones included, wakeforge scoring,
confirm_blocks=2): per-block min-over-2 margins on FLEURS negatives and synthetic positives whose voices
were NOT used for the heard phones. -> scores/calibp_<lang>.npz"""
import json, sys
import numpy as np
from piper import PiperVoice, SynthesisConfig
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ovos_audio_transformer_plugin_overheard_manners.spotter import load_phrases, PhraseSpotter
from heard_phones import VOICES as HEARD_VOICES
import run_pos
from fleurs import utterances, FLEURS

lang = sys.argv[1]; hours = float(sys.argv[2])
phrases = load_phrases(lang)
sp = PhraseSpotter(phrases, offset=0.0)
owner = np.array([phrases.index(p) for p in sp.owner])


def margins(audio):
    """per block: for each phrase, best over its pronunciations of (score - default threshold)."""
    win = np.zeros(24000, np.float32); out = []
    for i in range(len(audio) // 1280):
        win = np.concatenate([win[1280:], audio[i * 1280:(i + 1) * 1280]])
        m = sp.score_window(win) - sp.thresholds
        out.append([m[owner == j].max() for j in range(len(phrases))])
    return np.array(out, np.float32)


neg, utt = [], []
pad = np.zeros(8000, np.float32); n = 0; total = 0
if lang in FLEURS:
    for a, _ in utterances(lang):
        m = margins(np.concatenate([pad, a, pad])); neg.append(m); utt.append(np.full(len(m), n)); n += 1
        total += len(m)
        if total * 0.08 / 3600 >= hours:
            break
print(lang, "neg", n, round(total * 0.08 / 3600, 2), "h", flush=True)

rng = np.random.default_rng(11)
babble = [a for a, _ in utterances(lang, last=30)] if lang in FLEURS else []
sil = np.zeros(9600, np.float32)
used = set(HEARD_VOICES.get(lang, []))
cast = [c for c in run_pos.CAST.get(lang, []) if c[0] not in used][:4]
pos, meta = [], []
for model, nspk, override in cast:
    v = PiperVoice.load(run_pos.VOICES + model + ".onnx")
    if override:
        v.config.espeak_voice = override
    spk = None if not nspk else int(rng.integers(v.config.num_speakers))
    for j, p in enumerate(phrases):
        a = np.concatenate([c.audio_float_array for c in v.synthesize(p.text, SynthesisConfig(speaker_id=spk, length_scale=float(rng.uniform(0.85, 1.25))))])
        clip = np.concatenate([sil, run_pos.to16k(a, v.config.sample_rate), sil])
        for cond in run_pos.CONDITIONS:
            x = run_pos.condition(clip, cond, rng, babble) if (babble or cond != "babble10") else clip
            x = (x / (np.abs(x).max() + 1e-9) * rng.uniform(0.2, 0.8)).astype(np.float32)
            pos.append(margins(x)); meta.append(json.dumps(dict(voice=model, phrase=p.text, idx=j, cond=cond)))
    print(lang, model, len(pos), flush=True)
np.savez_compressed(f"scores/calibp_{lang}.npz", neg=np.concatenate(neg), utt=np.concatenate(utt),
                    pos=np.array(pos, dtype=object), meta=np.array(meta), phrases=np.array([p.text for p in phrases]))
print("done", lang)
