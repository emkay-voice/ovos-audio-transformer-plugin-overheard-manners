"""The page this plugin has in ovos-control-panel (controlpanel.json): the file,
its strings in every language, and the status it reads."""
import json
import re
from pathlib import Path

import pytest

import ovos_audio_transformer_plugin_overheard_manners as om

ROOT = Path(om.__file__).parent
TEXT = (ROOT / "controlpanel.json").read_text(encoding="utf-8")
PAGE = json.loads(TEXT)
LANGS = {"ca", "da", "de", "en", "es", "eu", "fr", "gl", "it", "nl", "pt"}
CONFIG = "audio_transformers.ovos-audio-transformer-plugin-overheard-manners"


def test_the_page_names_files_that_exist_and_its_own_messages():
    assert PAGE["controlpanel"] == 1 and PAGE["id"] == "overheard-manners"
    assert (ROOT / PAGE["icon"]).is_file()
    sent = re.findall(r'"(?:query|send|reload)": "([^"]+)"', TEXT)
    assert sent and all(s.startswith(PAGE["bus"]) for s in sent)
    assert PAGE["blocks"][2]["save"]["config"] == CONFIG


def test_the_page_speaks_every_language():
    files = {p.parent.name.split("-")[0]: json.loads(p.read_text(encoding="utf-8"))
             for p in (ROOT / "locale").glob("*/controlpanel.json")}
    assert set(files) == LANGS
    keys = set(files["en"])
    for lang, table in files.items():
        assert set(table) == keys, lang
        assert all(text.strip() for text in table.values()), lang
    used = set(re.findall(r'"(om\.[\w.]+)"', TEXT)) | set(re.findall(r"@(om\.[\w.]+)\}", TEXT))
    assert used - {"om.state"} <= keys


def test_every_state_has_a_label():
    keys = json.loads((ROOT / "locale" / "en-US" / "controlpanel.json").read_text(encoding="utf-8"))
    for state in ("listening", "paused", "no_local_stt", "no_phrases", "error"):
        assert f"om.state.{state}" in keys


def test_the_panel_reads_it():
    described = pytest.importorskip("ovos_webui.described")
    page, refs = described.check(PAGE, "ovos-audio-transformer-plugin-overheard-manners", config_roots=[CONFIG])
    assert [b["type"] for b in page["blocks"]] == ["text", "status", "form"]


def test_settings_from_the_panel_take_effect():
    """The panel's form sends its select values as text; the plugin must take them."""
    p = om.OverheardMannersPlugin({"lang": "en-US", "enabled": False})
    p._apply_config({"enabled": False, "cooldown_sec": "1800", "sensitivity_offset": "0.1"})
    assert p.cooldown == 1800.0
    assert float(p.config["sensitivity_offset"]) == 0.1
