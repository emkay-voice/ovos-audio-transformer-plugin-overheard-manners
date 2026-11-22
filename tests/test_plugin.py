"""The plugin's flow, with a fake first ear and a fake STT."""
import threading
import time

import pytest
from ovos_utils.fakebus import FakeBus

import ovos_audio_transformer_plugin_overheard_manners as om
from ovos_audio_transformer_plugin_overheard_manners.spotter import Phrase

HIT = b"\x01\x00" * 1280             # a chunk the fake first ear "recognises"
QUIET = b"\x00\x00" * 1280


class FakeSpotter:
    def __init__(self, phrases, *a, **kw):
        self.phrase = Phrase("what the fuck", "en", ["w ʌ t ð ə f ʌ k"])

    def feed(self, audio):
        return self.phrase if audio[0] > 0 else None

    def reset(self):
        pass


class FakeSTT:
    text = "oh what the fuck"
    calls = []

    def execute(self, audio, language=None):
        FakeSTT.calls.append(language)
        return FakeSTT.text


@pytest.fixture
def plugin(monkeypatch):
    monkeypatch.setattr(om, "PhraseSpotter", FakeSpotter)
    monkeypatch.setattr(om, "load_stt", lambda config: FakeSTT())
    FakeSTT.calls.clear(); FakeSTT.text = "oh what the fuck"
    p = om.OverheardMannersPlugin({"lang": "da-DK", "stt": {"module": "ovos-stt-plugin-fasterwhisper"},
                                   "cooldown_sec": 600})
    bus = FakeBus()
    said = []
    bus.on("speak", lambda m: said.append(m))
    p.bind(bus)
    p.said = said
    yield p
    p.default_shutdown()


def wait_for(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not cond():
        time.sleep(0.01)
    return cond()


def test_comments_on_confirmed_swearing(plugin):
    for c in (QUIET, QUIET, HIT):
        assert plugin.on_audio(c) is c           # audio passes through untouched
    assert wait_for(lambda: plugin.said)
    msg = plugin.said[0]
    assert msg.data["lang"] == "da-DK"
    assert msg.data["utterance"] in om.dialog_lines("da", "overheard_foul")
    assert FakeSTT.calls == ["en"]                # an English phrase is transcribed as English


def test_no_comment_when_the_text_is_clean(plugin):
    FakeSTT.text = "what the fun is that"
    plugin.on_audio(HIT)
    assert wait_for(lambda: FakeSTT.calls)
    time.sleep(0.1)
    assert plugin.said == []


def test_cooldown(plugin):
    plugin.on_audio(HIT)
    assert wait_for(lambda: plugin.said)
    plugin.on_audio(HIT)
    time.sleep(0.2)
    assert len(plugin.said) == 1


def test_quiet_while_the_device_speaks(plugin):
    plugin.bus.emit(om.Message("recognizer_loop:audio_output_start"))
    plugin.on_audio(HIT)
    time.sleep(0.2)
    assert FakeSTT.calls == []
    plugin.bus.emit(om.Message("recognizer_loop:audio_output_end"))
    plugin.on_audio(HIT)
    assert wait_for(lambda: plugin.said)


def test_pause_and_resume_on_the_bus(plugin):
    plugin.bus.emit(om.Message("ovos.overheard_manners.disable"))
    plugin.on_audio(HIT)
    time.sleep(0.2)
    assert FakeSTT.calls == []
    plugin.bus.emit(om.Message("ovos.overheard_manners.enable"))
    plugin.on_audio(HIT)
    assert wait_for(lambda: plugin.said)


def test_heard_text_never_goes_on_the_bus(plugin):
    seen = []
    plugin.bus.on("message", lambda m: seen.append(m))
    plugin.on_audio(HIT)
    assert wait_for(lambda: plugin.said)
    assert "fuck" not in str(seen).replace(plugin.said[0].data["utterance"], "")


def test_stays_off_without_a_local_stt(monkeypatch):
    monkeypatch.setattr(om.OverheardMannersPlugin, "_listener_stt",
                        staticmethod(lambda: {"module": "ovos-stt-plugin-server"}))
    p = om.OverheardMannersPlugin({"lang": "en-US"})
    p.bind(FakeBus())
    assert not p._ready
    assert p.on_audio(HIT) is HIT


def test_never_blocks_the_microphone(plugin, monkeypatch):
    monkeypatch.setattr(FakeSTT, "execute", lambda self, a, language=None: time.sleep(1) or "")
    t = time.monotonic()
    for _ in range(500):
        plugin.on_audio(HIT)
    assert time.monotonic() - t < 0.5


def test_stt_is_loaded_on_the_first_hit_only(monkeypatch):
    loads = []
    monkeypatch.setattr(om, "PhraseSpotter", FakeSpotter)
    monkeypatch.setattr(om, "load_stt", lambda config: loads.append(config) or FakeSTT())
    p = om.OverheardMannersPlugin({"lang": "en-US", "stt": {"module": "ovos-stt-plugin-vosk"}, "cooldown_sec": 0})
    p.bind(FakeBus())
    for _ in range(20):
        p.on_audio(QUIET)
    time.sleep(0.2)
    assert loads == [] and p.stt is None          # a quiet house never pays for the model
    p.on_audio(HIT)
    assert wait_for(lambda: loads)
    p.on_audio(HIT)
    time.sleep(0.2)
    assert len(loads) == 1                        # and it is loaded once
    p.default_shutdown()


def test_listener_can_share_its_stt(monkeypatch):
    monkeypatch.setattr(om, "PhraseSpotter", FakeSpotter)
    monkeypatch.setattr(om, "load_stt", lambda config: pytest.fail("must not load a second STT"))
    monkeypatch.setattr(om.OverheardMannersPlugin, "_listener_stt",
                        staticmethod(lambda: {"module": "ovos-stt-plugin-server"}))
    p = om.OverheardMannersPlugin({"lang": "da-DK"})
    bus = FakeBus()
    said = []
    bus.on("speak", lambda m: said.append(m))
    p.bind(bus)
    assert not p._ready                           # server STT only: off
    shared = FakeSTT()
    lock = threading.Lock()
    FakeSTT.calls.clear(); FakeSTT.text = "what the fuck"
    p.use_stt(shared, lock)
    assert p.status()["listening"] and p.status()["stt"] == "listener"
    with lock:                                    # the listener is transcribing: the plugin waits
        p.on_audio(HIT)
        time.sleep(0.2)
        assert FakeSTT.calls == []
    assert wait_for(lambda: said)
    p.default_shutdown()


def test_status_and_reload(plugin):
    replies = []
    plugin.bus.on("ovos.overheard_manners.status.response", lambda m: replies.append(m.data))
    plugin.bus.emit(om.Message("ovos.overheard_manners.status"))
    assert wait_for(lambda: replies)
    assert replies[0]["listening"] and replies[0]["stt"] == "ovos-stt-plugin-fasterwhisper"
    plugin.bus.emit(om.Message("ovos.overheard_manners.reload",
                               {"config": {"lang": "da-DK", "enabled": False,
                                           "stt": {"module": "ovos-stt-plugin-fasterwhisper"}}}))
    assert plugin.status() == {"listening": False, "enabled": False, "state": "paused", "lang": "da-DK",
                               "stt": "ovos-stt-plugin-fasterwhisper", "stt_loaded": False,
                               "off_reason": "paused"}
