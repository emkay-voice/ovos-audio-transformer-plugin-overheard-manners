# <img src='overheard.png' width='50' height='50' style='vertical-align:bottom'/> ovos-audio-transformer-plugin-overheard-manners

An [OpenVoiceOS](https://openvoiceos.org) audio transformer that gives the
assistant good manners about what it **overhears**: someone in the room says
"av for helvede!", and the assistant gently remarks that swearing makes you
sound a bit simple. Think of the friendly grandparent in the corner, not the
moral police.

It is part of [Emkay](https://github.com/emkay-voice/emkay) and works on any
OpenVoiceOS. It is the companion of
[ovos-good-manners-pipeline-plugin](https://github.com/OpenVoiceOS/ovos-good-manners-pipeline-plugin),
which reacts to how you talk *to* the assistant. This plugin listens to the
room, and it uses good manners' word lists. Its icon is the thumbs-up's
sibling: an ear that just heard a "#!@".

## Everything stays on the device

- The room audio is held in a 3-second ring buffer in memory and nothing else.
  Nothing is recorded, saved or sent.
- Speech to text only runs **on the device**. If the only STT available is a
  server (OpenVoiceOS' public servers, a cloud service), the plugin stays off
  and says so in the log.
- The transcript is only used for the verdict. It is not logged and not put on
  the bus; the bus only gets `ovos.overheard_manners.heard` with the kind
  (`foul` or `slur`).
- It is off unless you turn it on (see [Setup](#setup)).

## How it listens: two stages

While the listener waits for the wake word, OVOS hands all room audio to audio
transformers. This one listens in two stages:

1. **A sensitive first ear.** A few long phrases per language ("what the
   fuck", "av for helvede", "verdammte Scheiße", …) are spotted from their
   phones with [WakePhoneHuBERT](https://huggingface.co/TigreGotico/wakephonehubert),
   zero-shot, using the scoring of
   [ovos-ww-plugin-wakeforge](https://github.com/OpenVoiceOS/ovos-ww-plugin-wakeforge)'s
   zero-shot engine. The English phrases are listened for in every language,
   also with the local pronunciation, because people swear in English
   everywhere. A hit here only means "worth a closer look".
2. **Local STT decides.** Only then is the last 2.5 seconds transcribed on the
   device, and the text is checked against good manners' word lists, short
   words like "lort" and "shit" included. Only if something foul was really
   said does the assistant comment, and then not again for 10 minutes.

The first stage alone gives far too many false hits to comment on; the second
removes them. In tests on 23 hours of ordinary speech it made 0 false comments
in English and German and 1 in Danish (STT heard "frugtjuicer" as a swear
word). The numbers and how to rerun them are in [evaluation/](evaluation/).

It is quiet while the device itself speaks, while music or a story plays, and
it can be paused on the bus.

## Setup

```bash
pip install ovos-audio-transformer-plugin-overheard-manners
```

Name it in `mycroft.conf` (that is the opt-in):

```json
{
  "audio_transformers": {
    "ovos-audio-transformer-plugin-overheard-manners": {
      "stt": {"module": "ovos-stt-plugin-fasterwhisper", "model": "small"}
    }
  }
}
```

| key | default | |
|---|---|---|
| `stt` | the listener's STT, if it is a known local one | a local STT plugin and its config; naming one here is your statement that it runs on the device |
| `lang` | the device's `lang` | the room's language |
| `cooldown_sec` | `600` | least time between two comments |
| `sensitivity_offset` | `0.07` | first-ear threshold offset, higher is stricter (fewer STT runs, fewer catches) |
| `confirm_blocks` | `2` | windows in a row a phrase must hold |
| `inline` | on only in Pyodide | run both stages as the audio arrives, without a thread of its own (for Pythons that have none, as in [Emkay](https://github.com/emkay-voice/emkay)) |
| `quiet_db` | `-50` inline, else off | inline, the first ear rests while the room is quieter than this (dBFS) |
| `enabled` | `true` | start listening right away; `"active": false` turns the plugin off entirely |

On the bus:

| message | |
|---|---|
| `ovos.overheard_manners.disable` / `.enable` | pause and resume (for example at night) |
| `ovos.overheard_manners.heard` | emitted with `{"kind": "foul" \| "slur"}` when it comments |
| `ovos.overheard_manners.status` | answers with whether it listens, its state, language and STT |
| `ovos.overheard_manners.reload` | new settings: `{"config": the plugin's section}` |

The model (3.5 MB) is fetched from Hugging Face on first use, as the wakeforge
plugin does.

### Speech to text: shared, or loaded when needed

There has to be a speech-to-text engine on the device that runs locally. It
does not have to be the one OVOS uses for your commands. The plugin never pays
for a second model it does not need:

- **A listener can hand over its own STT** with `use_stt(stt, lock)`. Emkay's
  listener does this, so there is only one model in memory. The lock keeps the
  listener and the plugin from transcribing at the same time.
- **Otherwise the model is loaded on the first hit**, the first time something
  in the room sounds like swearing. A quiet house never loads it, and the first
  comment just comes a few seconds later.

## In the control panel

With [ovos-control-panel](https://github.com/OpenVoiceOS/ovos-control-panel)
on the device, the plugin has its own page, **Overheard manners**, under
**Everyday**. It is described in `controlpanel.json` (strings in
`locale/<lang>/controlpanel.json`, 11 languages) and drawn by the panel; no
code of the plugin runs in the panel. It shows whether it is listening, and
why not if it isn't (no local STT, paused). It can pause and resume, and it
sets `enabled`, the cool-down and the sensitivity, which take effect at once
(`ovos.overheard_manners.reload`), no restart.

## Languages

The phrases and replies cover Emkay's languages: Catalan, Danish, Dutch,
English, French, Galician, German, Basque, Italian, Portuguese and Spanish.
Good manners has word lists for English, Danish and German; for the other
languages this plugin carries a short list (`locale/<lang>/foul_words.voc`),
meant to move into good manners. The lists for Catalan, Basque, Galician and
Portuguese need a native speaker's review.

The phrases live in [`scripts/phrases.json`](scripts/phrases.json).
`scripts/build_phrases.py` turns them into the model's phones (eSpeak NG, at
build time only) and `scripts/heard_phones.py` adds the phones the model itself
hears when Piper voices say them, which helps a lot for Danish. CI checks that
the phrase files are up to date.

## Emkay

In Emkay the browser owns the microphone and Emkay's listener does what
ovos-dinkum-listener does. It does not feed audio transformers yet; that is
[emkay#105](https://github.com/emkay-voice/emkay/issues/105), and this plugin
waits for it there. Emkay's speech to text is on the device by default, so its
listener hands it over (`use_stt`) and the plugin costs no extra model.

## License

Apache-2.0
