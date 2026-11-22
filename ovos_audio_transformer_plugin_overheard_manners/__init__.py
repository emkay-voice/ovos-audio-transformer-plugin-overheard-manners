"""Overheard manners: the assistant gently comments on swearing it overhears.

An OpenVoiceOS audio transformer. While the listener waits for the wake word it
hands every bit of room audio to audio transformers (``on_audio``); this one
listens in two stages, all on the device:

1. a sensitive first ear (WakePhoneHuBERT zero-shot phrase spotting, see
   spotter.py) that picks out moments that might be swearing;
2. only then, local STT on the last few seconds, checked against good manners'
   word lists (words.py). Only if the text really holds a foul word does the
   assistant say something, with a cool-down so it does not nag.

Nothing is recorded, sent or stored: the audio lives in a 3-second ring buffer
in memory, the text is only used for the verdict and is not logged or put on
the bus. Off by default: OVOS only loads it when it is named in the
``audio_transformers`` config, and it can be paused and resumed on the bus
(ovos.overheard_manners.disable / enable).
"""
import queue
import random
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
from ovos_bus_client.message import Message
from ovos_plugin_manager.templates.transformers import AudioTransformer
from ovos_utils.log import LOG

from ovos_audio_transformer_plugin_overheard_manners.local_stt import load_stt, pick_local_stt, transcribe
from ovos_audio_transformer_plugin_overheard_manners.spotter import (
    DEFAULT_OFFSET, PhraseSpotter, lang_code, load_phrases, locale_dir)
from ovos_audio_transformer_plugin_overheard_manners.words import SLUR, WordCheck

PLUGIN_ID = "ovos-audio-transformer-plugin-overheard-manners"
SAMPLE_RATE = 16000
#: audio given to STT: what led up to the hit (the phrase ends just before it fires)
BEFORE_SEC = 2.5
#: OCP player state "playing" (ovos_utils.ocp.PlayerState.PLAYING)
OCP_PLAYING = 1


def dialog_lines(code: str, name: str):
    for c in (code, "en"):
        folder = locale_dir(c)
        f = folder / f"{name}.dialog" if folder else None
        if f and f.is_file():
            return [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return []


class OverheardMannersPlugin(AudioTransformer):
    """Config (``audio_transformers.ovos-audio-transformer-plugin-overheard-manners``):

    Being named in that section is the opt-in; "active": false turns it off again (OVOS' rule).
    enabled (bool): start listening right away, default true; the bus can pause and resume it.
    lang (str): language of the room, default the device's ``lang``.
    stt (dict): a local STT to use, e.g. {"module": "ovos-stt-plugin-fasterwhisper", "model": "small"};
        default the listener's STT if it is a known local one (see local_stt.py), else the plugin stays off.
        A listener can also hand over its own STT instead (use_stt), so no second model is loaded.
    cooldown_sec (float): least time between two comments, default 600.
    sensitivity_offset (float): stage 1 threshold offset, higher is stricter, default spotter.DEFAULT_OFFSET.
    confirm_blocks (int): stage 1 windows in a row, default 2.
    """

    def __init__(self, config=None):
        super().__init__(PLUGIN_ID, 10, config or {})
        self._ring = deque(maxlen=int(3 * SAMPLE_RATE / 1280) + 1)     # ~3 s of 80 ms chunks
        self._queue: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=200)
        self._busy_speaking = False
        self._media_playing = False
        self._last_comment: Optional[float] = None     # monotonic time; None = never commented
        self.spotter = None
        self.words = None
        self.stt = None                 # loaded on the first hit, or handed over by the listener
        self._stt_config: Optional[dict] = None
        self._stt_lock = threading.Lock()
        self._shared_stt = False
        self._worker = None
        self._ready = False
        self._off_reason = ""
        self._apply_config(self.config)

    def _apply_config(self, config: dict):
        self.config = config
        self.enabled = bool(config.get("enabled", True))
        self.cooldown = float(config.get("cooldown_sec", 600))
        self.lang = config.get("lang") or self._device_lang()
        self.code = lang_code(self.lang)

    @staticmethod
    def _device_lang() -> str:
        from ovos_config import Configuration
        return Configuration().get("lang", "en-US")

    # ---- setup --------------------------------------------------------------

    def bind(self, bus=None):
        super().bind(bus)
        self.bus.on("ovos.overheard_manners.enable", self._on_enable)
        self.bus.on("ovos.overheard_manners.disable", self._on_disable)
        self.bus.on("ovos.overheard_manners.status", self._on_status)
        self.bus.on("ovos.overheard_manners.reload", self._on_reload)
        self.bus.on("recognizer_loop:audio_output_start", self._on_speaking)
        self.bus.on("recognizer_loop:audio_output_end", self._on_done_speaking)
        self.bus.on("ovos.common_play.player.state", self._on_player_state)
        if self.enabled:            # the transformer service binds but never calls initialize()
            self._start()

    def use_stt(self, stt, lock: Optional[threading.Lock] = None):
        """For a listener whose STT runs on the device: hand it over, so this plugin
        loads no second model. Pass the lock the listener holds while it transcribes,
        so the two never use the STT at the same time. The listener vouches that it is local."""
        self.stt = stt
        self._stt_lock = lock or self._stt_lock
        self._shared_stt = True
        if self.enabled:
            self._start()

    def _start(self) -> bool:
        if self._ready:
            return True
        if not self._shared_stt:
            self._stt_config = pick_local_stt(self.config.get("stt"), self._listener_stt())
            if self._stt_config is None:
                self._off_reason = "no_local_stt"
                LOG.warning("overheard manners stays off: no STT that runs on this device. "
                            "Room audio must not go to a server; name a local STT in this plugin's 'stt' config.")
                return False
        phrases = load_phrases(self.lang)
        if not phrases:
            self._off_reason = "no_phrases"
            LOG.warning(f"overheard manners stays off: no phrases for {self.lang}")
            return False
        try:
            self.spotter = PhraseSpotter(phrases, float(self.config.get("sensitivity_offset", DEFAULT_OFFSET)),
                                         int(self.config.get("confirm_blocks", 2)))
        except Exception:
            self._off_reason = "error"
            LOG.exception("overheard manners could not start")
            return False
        self.words = WordCheck([self.code], [p.text for p in phrases])
        if not self.words.has_good_manners:
            LOG.warning("ovos-good-manners-pipeline-plugin is not installed: only this plugin's own short lists are used")
        if self._worker is None:
            self._worker = threading.Thread(target=self._work, name="overheard-manners", daemon=True)
            self._worker.start()
        self._ready = True
        self._off_reason = ""
        LOG.info(f"overheard manners listening ({self.lang}, {len(phrases)} phrases)")
        return True

    def _transcribe(self, pcm: bytes, lang: str) -> str:
        """Local STT, loaded the first time something sounds like swearing (so a quiet
        house never pays for the model) unless the listener handed its own over."""
        with self._stt_lock:
            if self.stt is None:
                self.stt = load_stt(self._stt_config)
            return transcribe(self.stt, pcm, lang)

    @staticmethod
    def _listener_stt() -> dict:
        from ovos_config import Configuration
        return Configuration().get("stt", {})

    # ---- bus ----------------------------------------------------------------

    def _on_enable(self, message):
        self.enabled = True
        self._start()

    def _on_disable(self, message):
        self.enabled = False
        self._ring.clear()

    def status(self) -> dict:
        if self._shared_stt:
            stt = "listener"
        else:
            stt = (self._stt_config or {}).get("module", "")
        off = "paused" if not self.enabled else ("" if self._ready else self._off_reason)
        return {"listening": bool(self.enabled and self._ready), "enabled": self.enabled,
                "state": off or "listening", "lang": self.lang, "stt": stt,
                "stt_loaded": self.stt is not None, "off_reason": off}

    def _on_status(self, message):
        self.bus.emit(message.response(self.status()))

    def _on_reload(self, message):
        """New settings from the control panel: {"config": this plugin's section}."""
        config = message.data.get("config")
        if config is None:
            return
        old_stt, old_lang = self.config.get("stt"), self.lang
        self._apply_config(dict(config))
        if not self._shared_stt and (self.config.get("stt") != old_stt):
            with self._stt_lock:
                self.stt = None            # loaded again on the next hit
        self._ready = False
        if self.lang != old_lang:
            self.words = None
        with self._queue.mutex:
            self._queue.queue.clear()
        self._ring.clear()
        if self.enabled:
            self._start()

    def _on_speaking(self, message):
        self._busy_speaking = True

    def _on_done_speaking(self, message):
        self._busy_speaking = False
        self._ring.clear()
        if self.spotter:
            self.spotter.reset()

    def _on_player_state(self, message):
        self._media_playing = message.data.get("state") == OCP_PLAYING

    # ---- audio --------------------------------------------------------------

    def on_audio(self, audio_data):
        """Room audio while waiting for the wake word. Never blocks the listener."""
        if self.enabled and self._ready and not self._busy_speaking and not self._media_playing:
            try:
                self._queue.put_nowait(audio_data)
            except queue.Full:
                pass        # the worker is behind (STT running): drop, never stall the microphone
        return audio_data

    def _work(self):
        while True:
            chunk = self._queue.get()
            if chunk is None:
                return
            if not (self.enabled and self._ready):
                continue
            self._ring.append(chunk)
            audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            phrase = self.spotter.feed(audio)
            if phrase is None or (self._last_comment is not None
                                  and time.monotonic() - self._last_comment < self.cooldown):
                continue
            pcm = b"".join(self._ring)[-int(BEFORE_SEC * SAMPLE_RATE) * 2:]
            # English phrases are transcribed as English, so English swearing is not
            # turned into words of the device's language
            stt_lang = "en" if phrase.lang == "en" else self.lang
            try:
                text = self._transcribe(pcm, stt_lang)
            except Exception:
                LOG.exception("overheard manners: STT failed")
                continue
            verdict = self.words.verdict(text)
            del text, pcm                       # not kept, not logged
            if verdict:
                self._comment(verdict)
            self._ring.clear()
            self.spotter.reset()
            with self._queue.mutex:             # what was heard meanwhile is stale now
                self._queue.queue.clear()

    def _comment(self, verdict: str):
        lines = dialog_lines(self.code, "overheard_slur" if verdict == SLUR else "overheard_foul")
        if not lines:
            return
        self._last_comment = time.monotonic()
        self.bus.emit(Message("ovos.overheard_manners.heard", {"kind": verdict}))
        self.bus.emit(Message("speak", {"utterance": random.choice(lines), "lang": self.lang},
                              {"source": PLUGIN_ID, "destination": ["audio"]}))

    def default_shutdown(self):
        self.enabled = False
        if self._worker:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass                # the worker is a daemon thread and ends with the listener
